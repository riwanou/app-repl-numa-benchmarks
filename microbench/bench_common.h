#define _GNU_SOURCE
#include <assert.h>
#include <fcntl.h>
#include <numa.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <unistd.h>

#define ARRAY_NB_ENTRIES (1024UL * 1024UL * 32UL)
#define NB_ROUNDS 5

int *array;
int repl_enabled;
size_t size;
pthread_barrier_t barrier;

#define MADV_REPLICATE 63
#define MAP_REPL 0x8000000

#define PAGE_SIZE 4096

#define NO_REPLICATION 0
#define USE_REPLICATION 1

int use_mmap = 0;
int use_madvise = 0;

int nthreads = 0;
int nprocs = 0;
int nsockets = 0;

void print_help(const char *progname) {
  fprintf(stderr, "Usage: %s <mmap|madvise|none> [num_threads]\n", progname);
  fprintf(stderr, "  mmap: Use mmap() for allocation with special flag.\n");
  fprintf(stderr, "  madvise: Use mmap() followed by madvise().\n");
  fprintf(stderr, "  none: Use malloc() for allocation.\n");
  fprintf(stderr, "  num_threads: Number of threads\n");
}

void parse_args(int argc, char *argv[]) {
  if (argc < 2 || argc > 3) {
    print_help(argv[0]);
    exit(EXIT_FAILURE);
  }

  if (strcmp(argv[1], "mmap") == 0) {
    use_mmap = 1;
  } else if (strcmp(argv[1], "madvise") == 0) {
    use_madvise = 1;
  } else {
    fprintf(stderr, "Unknown argument: %s\n", argv[1]);
    print_help(argv[0]);
    exit(EXIT_FAILURE);
  }

  if (argc == 3) {
    long val = strtol(argv[2], NULL, 10);
    if (val <= 0) {
      fprintf(stderr, "Invalid number of threads: %s\n", argv[2]);
      exit(EXIT_FAILURE);
    }
    nthreads = (size_t)val;
  } else {
    nthreads = (size_t)sysconf(_SC_NPROCESSORS_ONLN);
  }
}

void init_platform_parse_args(int argc, char *argv[]) {

  parse_args(argc, argv);
  nprocs = get_nprocs();
  nsockets = numa_num_configured_nodes();
  printf("Running with %d threads\n", nthreads);
}

void common_init(int argc, char **argv) {
  init_platform_parse_args(argc, argv);
  pthread_barrier_init(&barrier, NULL, nthreads);
  size = ARRAY_NB_ENTRIES * sizeof(int);

  char *repl_env = getenv("REPLICATION");
  repl_enabled = (repl_env != NULL && atoi(repl_env) == 1);
}

typedef struct {
  FILE *fp;
  char path[512];
} csv_logger_t;

csv_logger_t *csv_init(const char *tag, const char *metric, int repl_enabled) {
  csv_logger_t *logger = malloc(sizeof(*logger));
  const char *env_dir = getenv("CSV_DIR");
  const char *csv_dir = env_dir ? env_dir : ".";
  const char *method = use_mmap ? "mmap" : "madvise";

  snprintf(logger->path, sizeof(logger->path), "%s/%s_%s_repl_%d_%d.csv",
           csv_dir, tag, method, repl_enabled, nthreads);

  logger->fp = fopen(logger->path, "w");
  if (!logger->fp) {
    perror("fopen csv");
    exit(EXIT_FAILURE);
  }

  fprintf(logger->fp, "round,%s,tag\n", metric);
  fflush(logger->fp);
  return logger;
}

void csv_write(csv_logger_t *logger, int round, double elapsed,
               const char *tag) {
  if (logger && logger->fp) {
    fprintf(logger->fp, "%d,%.6f,%s\n", round, elapsed, tag);
    fflush(logger->fp);
  }
}

void csv_close(csv_logger_t *logger) {
  if (logger) {
    if (logger->fp)
      fclose(logger->fp);
    free(logger);
  }
}

void *allocate_buffer_platform(int repl, size_t size) {
  void *array;

  if (use_mmap) {
    int flags = MAP_ANONYMOUS | MAP_PRIVATE;
    if (repl)
      flags |= MAP_REPL;
    array = mmap(NULL, size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (array == MAP_FAILED) {
      perror("mmap failed");
      exit(EXIT_FAILURE);
    }
  } else if (use_madvise) {
    array = mmap(NULL, size, PROT_READ | PROT_WRITE,
                 MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  }

  return array;
}

void touch_buffer(int repl, char *buffer, size_t size) {
  for (int i = 0; i < size; i++) {
    buffer[i] = (char)i;
  }
  if (use_madvise && repl) {
    assert(madvise(buffer, size, MADV_REPLICATE) == 0);
  }
}

void run_and_join_on_all_threads(void *thread_fn(void *args)) {
  pthread_t *threads = malloc(sizeof(pthread_t) * nthreads);
  int *thread_ids = malloc(sizeof(int) * nthreads);
  assert(threads && thread_ids);

  for (int t = 0; t < nthreads; t++) {
    thread_ids[t] = t;
    int rc = pthread_create(&threads[t], NULL, thread_fn, &thread_ids[t]);
    assert(rc == 0);
  }

  for (int t = 0; t < nthreads; t++) {
    pthread_join(threads[t], NULL);
  }
}

int main_node_id(void) {
  int main_node_id = 0;
  FILE *f = fopen("/sys/kernel/debug/repl_pt/main_node_id", "r");
  if (f) {
    fscanf(f, "%d", &main_node_id);
    fclose(f);
  } else {
    perror("fopen main_node_id");
    exit(EXIT_FAILURE);
  }
  return main_node_id;
}

double elapsed_time(struct timespec start, struct timespec end) {
  time_t sec = end.tv_sec - start.tv_sec;
  long nsec = end.tv_nsec - start.tv_nsec;

  if (nsec < 0) {
    sec -= 1;
    nsec += 1000000000L;
  }

  return (double)sec + (double)nsec / 1e9;
}

pid_t gettid(void) { return syscall(__NR_gettid); }

void set_affinity(unsigned long tid, unsigned long core_id) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(core_id, &mask);
  int ret = sched_setaffinity(tid, sizeof(mask), &mask);
  if (ret < 0) {
    perror("sched_affinity failed");
    exit(EXIT_FAILURE);
  }
}

int get_nthcore_in_numa_socket(int socket, int index) {
  struct bitmask *bm = numa_allocate_cpumask();
  numa_node_to_cpus(socket, bm);

  int count = 0;
  for (int core = 0; core < nprocs; core++) {
    if (numa_bitmask_isbitset(bm, core)) {
      if (count == index)
        return core;
      count++;
    }
  }

  numa_free_nodemask(bm);
  return 0;
}
