/* dirtest - what does cross socket cache line sharing cost?
 *
 * Reads one buffer from a set of CPUs and never writes it, so any DRAM write
 * a monitor reports is not from this program.
 *
 * `overlap=<0..100>` splits the CPUs in two groups reading len/2 each and
 * sharing that percentage of their lines. Same footprint per group either way.
 *
 * Each thread has its own slice. Sharing a window, threads drift together and
 * feed each other out of the LLC.
 *
 *   numactl --membind=0 ./dirtest/dirtest 256 30 0,1,2,16,17,18 overlap=50
 *
 * Driven by bench_sharing.py. RESULT reports the measured window, which starts
 * after the memset, which is the buffer written once for real.
 */

#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define LINE 64
/* lines between two checks of `stop`: 64 MB, so a run stops promptly */
#define CHUNK (1u << 20)

static uint8_t *buf;
static size_t len;
static volatile int stop;

struct worker {
    pthread_t th;
    int cpu;
    size_t base; /* window it reads, [base, end) */
    size_t end;
    uint64_t lines;
    uint64_t sum;
};

static double mono(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* same shape as config.get_time(), so the stamps line up with the monitors */
static void wallclock(char *out, size_t n) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    localtime_r(&ts.tv_sec, &tm);
    size_t k = strftime(out, n, "%Y-%m-%dT%H:%M:%S", &tm);
    snprintf(out + k, n - k, ".%06ld", ts.tv_nsec / 1000);
}

static void *reader(void *arg) {
    struct worker *w = arg;

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(w->cpu, &set);
    if (sched_setaffinity(0, sizeof set, &set) != 0) {
        perror("sched_setaffinity");
        exit(1);
    }

    /* one load per line, volatile so the compiler keeps it */
    size_t base = w->base, end = w->end, off = base;
    uint64_t sum = 0, lines = 0;

    while (!stop) {
        for (unsigned i = 0; i < CHUNK; i++) {
            sum += *(volatile uint64_t *)(buf + off);
            off += LINE;
            if (off >= end)
                off = base;
        }
        lines += CHUNK;
    }

    w->sum = sum;
    w->lines = lines;
    return NULL;
}

/* check the buffer landed on the node we asked for */
static void show_placement(void) {
    FILE *f = fopen("/proc/self/numa_maps", "r");
    if (!f)
        return;
    char line[1024];
    while (fgets(line, sizeof line, f))
        if (strstr(line, "anon=") && strstr(line, "heap") == NULL)
            fputs(line, stdout);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "usage: %s <mb> <secs> <cpu,cpu,...> [overlap=0..100]\n",
                argv[0]);
        return 2;
    }

    size_t mb = strtoul(argv[1], NULL, 10);
    int secs = atoi(argv[2]);
    len = mb << 20;

    /* -1: no grouping, the whole buffer is one window */
    int overlap = -1;
    if (argc == 5) {
        if (strncmp(argv[4], "overlap=", 8) != 0) {
            fprintf(stderr, "expected overlap=<0..100>, got '%s'\n",
                    argv[4]);
            return 2;
        }
        overlap = atoi(argv[4] + 8);
        if (overlap < 0 || overlap > 100) {
            fprintf(stderr, "overlap must be 0..100, got %d\n", overlap);
            return 2;
        }
    }

    /* strtok chews up argv[3] in place, keep a copy for the banner */
    char *cpulist = strdup(argv[3]);
    int cpus[512], n = 0;
    for (char *tok = strtok(argv[3], ","); tok && n < 512;
         tok = strtok(NULL, ","))
        cpus[n++] = atoi(tok);
    if (n == 0) {
        fprintf(stderr, "no cpus given\n");
        return 2;
    }

    buf = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
               -1, 0);
    if (buf == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    /* memset, not MAP_POPULATE: pages that are only read all map to the one
     * shared zero page, which would collapse 8 GB into one cache line */
    memset(buf, 0xa5, len);

    printf("buffer %zu MB, %d threads on cpus %s, %d s, overlap %d\n", mb, n,
           cpulist, secs, overlap);
    show_placement();
    fflush(stdout);

    /* group A takes [0, len/2). group B takes the same width, slid so that the
     * two windows share `overlap` percent of their lines: at 0 B sits right
     * after A, at 100 it sits exactly on top of it. */
    size_t span = (overlap < 0) ? len : len / 2;
    /* double, not span/100*overlap: integer division leaves overlap=100 a
     * cache line short, and exact is the point of that endpoint */
    size_t slide = (size_t)((double)span * (overlap < 0 ? 0 : overlap) / 100.0);
    slide &= ~(size_t)(LINE - 1);

    struct worker *w = calloc(n, sizeof *w);
    int group = (overlap < 0) ? n : n / 2;
    size_t slice = (span / (size_t)group) & ~(size_t)(LINE - 1);
    for (int i = 0; i < n; i++) {
        w[i].cpu = cpus[i];
        /* the second group's window slides up by the overlap */
        size_t base = (i >= group) ? span - slide : 0;
        /* a slice each, same slice at the same index in both groups: two
         * threads on a socket never meet, the two sockets always do */
        w[i].base = base + slice * (size_t)(i % group);
        w[i].end = w[i].base + slice;
    }

    for (int i = 0; i < n; i++)
        pthread_create(&w[i].th, NULL, reader, &w[i]);

    /* the window starts here, with every thread already reading and the first
     * touch long finished */
    char start_stamp[64], end_stamp[64];
    wallclock(start_stamp, sizeof start_stamp);
    double t0 = mono();

    sleep(secs);

    stop = 1;
    double dt = mono() - t0;
    wallclock(end_stamp, sizeof end_stamp);

    uint64_t lines = 0, sum = 0;
    for (int i = 0; i < n; i++) {
        pthread_join(w[i].th, NULL);
        lines += w[i].lines;
        sum += w[i].sum;
    }

    printf("RESULT start=%s end=%s read_gb_s=%.2f threads=%d overlap=%d"
           " checksum=%llu\n",
           start_stamp, end_stamp, (double)lines * LINE / dt / (1 << 30), n,
           overlap, (unsigned long long)sum);
    return 0;
}
