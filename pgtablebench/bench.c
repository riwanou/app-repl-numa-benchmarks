#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include <numa.h>
#include <sched.h>
#include <math.h>

#define MAX_NROUNDS 4096
#define MAP_REPL    0x8000000
#define MIN(a, b)   ((a) < (b) ? (a) : (b))
#define MAX(a, b)   ((a) > (b) ? (a) : (b))

static inline uint64_t rdtscp(void) {
    uint32_t lo, hi;
    asm volatile("rdtscp" : "=a"(lo), "=d"(hi) :: "rcx");
    return ((uint64_t)hi << 32) | lo;
}

int first_cpu_of_node(int node) {
    struct bitmask *cpus = numa_allocate_cpumask();
    numa_node_to_cpus(node, cpus);
    for (int i = 0; i < (int)cpus->size; i++)
        if (numa_bitmask_isbitset(cpus, i)) {
            numa_free_cpumask(cpus);
            return i;
        }
    numa_free_cpumask(cpus);
    return -1;
}

void set_affinity(int node) {
    int cpu = first_cpu_of_node(node);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
}

int repl_main_node_id(void) {
    int fd, nid = 0;
    char buf[16];
    ssize_t size;
    fd = open("/sys/kernel/debug/repl_pt/main_node_id", O_RDONLY);
    if (fd < 0) return 0;
    size = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (size <= 0) return 0;
    buf[size] = '\0';
    sscanf(buf, "%d", &nid);
    return nid;
}

void touch_n_nodes(char *addr, size_t size, int nnodes, int repl) {
    for (int n = 0; n < nnodes; n++) {
        int node_id = (nnodes == 1 && repl) ? repl_main_node_id() : n;
        set_affinity(node_id);
        for (size_t i = 0; i < size; i += 4096)
            addr[i] = 0;
    }
    set_affinity(0);
}

char *init_mmap(size_t size, int repl) {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    if (repl)
      flags |= MAP_REPL;
    char *addr = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (addr == MAP_FAILED) { perror("mmap"); exit(1); }
    return addr;
}

uint64_t bench_munmap(char *addr, size_t size) {
    uint64_t ts, te;
    ts = rdtscp();
    // if (munmap(addr, size)) { perror("munmap"); exit(1); }
    for (size_t i = 0; i < size; i += 4096)
      if (munmap(addr + i, 4096)) { perror("munmap"); exit(1); }
    te = rdtscp();
    return (te - ts);
}

int cmp_uint64(const void *a, const void *b) {
    return (*(uint64_t*)a > *(uint64_t*)b) - (*(uint64_t*)a < *(uint64_t*)b);
}

void dump_analysis(uint64_t *data, int nrounds, int npages) {
    qsort(data, nrounds, sizeof(uint64_t), cmp_uint64);

    uint64_t min  = data[0];
    uint64_t max  = data[nrounds - 1];
    uint64_t p95  = data[(int)(nrounds * 0.95)];
    double   avg  = 0;
    for (int i = 0; i < nrounds; i++)
        avg += data[i];
    avg /= nrounds;

    printf("npages=%d | avg/page %.0f min/page %lu max/paged %lu p95/page %lu | avg %.0f min %lu max %lu p95 %lu\n",
           npages,
           avg / npages, min / npages, max / npages, p95 / npages,
           avg, min, max, p95);
}

int main(int argc, char *argv[]) {
    uint64_t elapsed[MAX_NROUNDS];
    char *addr;
    int repl = 0, opt, npages = 1, nrounds = 512, max_nodes;

    while ((opt = getopt(argc, argv, "s:n:r")) != -1)
        switch (opt) {
        case 's': npages  = atoi(optarg); break;
        case 'n': nrounds = atoi(optarg); break;
        case 'r': repl    = 1;            break;
        default:
            fprintf(stderr, "usage: %s [-s npages] [-n nrounds] [-r]\n", argv[0]);
            exit(1);
        }

    if (nrounds > MAX_NROUNDS) {
        fprintf(stderr, "nrounds (%d) > MAX_NROUNDS (%d)\n", nrounds, MAX_NROUNDS);
        exit(1);
    }

    size_t region_size = (size_t)4096 * npages;
    max_nodes = numa_max_node() + 1;
    set_affinity(0);

    printf("bench region %zu KB | nrounds %d | repl %d | max nodes %d\n",
           region_size / 1024, nrounds, repl, max_nodes);

    for (int nn = 1; nn <= max_nodes; nn++) {
        for (int i = 0; i < nrounds; i++) {
            addr = init_mmap(region_size, repl);
            touch_n_nodes(addr, region_size, nn, repl);
            elapsed[i] = bench_munmap(addr, region_size);
        }

        dump_analysis(elapsed, nrounds, npages);
    }
    return 0;
}
