#include "bench_common.h"

static csv_logger_t *logger;

static void *alloc_worker(void *arg) {
  unsigned int thread_id = *(unsigned int *)arg;
  unsigned int socket_id = thread_id % nsockets;
  unsigned int index_in_node = thread_id / nsockets;
  unsigned int core_id = get_nthcore_in_numa_socket(socket_id, index_in_node);
  set_affinity(gettid(), core_id);

  touch_buffer(repl_enabled, (char *)array, size);

  return NULL;
}

int main(int argc, char **argv) {
  common_init(argc, argv);
  logger = csv_init("alloc", "elapsed_ms");
  int nb_rounds = NB_ROUNDS;
  if (use_madvise)
    nb_rounds = 10;

  for (int i = 0; i < nb_rounds; i++) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    array = allocate_buffer_platform(repl_enabled, size);
    run_and_join_on_all_threads(alloc_worker);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = elapsed_time(start, end);
    printf("alloc elapsed: %.6f ms\n", elapsed);
    csv_write(logger, i, elapsed, repl_enabled ? "alloc_repl" : "alloc");
  }

  csv_close(logger);
  return 0;
}
