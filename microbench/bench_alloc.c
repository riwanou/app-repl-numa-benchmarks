#include "bench_common.h"

static csv_logger_t *logger;

static void *alloc_worker(void *arg) {
  int thread_id = *(int *)arg;
  int socket_id = thread_id % nsockets;
  int index_in_node = thread_id / nsockets;
  int core_id = get_nthcore_in_numa_socket(socket_id, index_in_node);
  set_affinity(gettid(), core_id);

  touch_buffer(repl_enabled, (char *)array, size);

  return NULL;
}

int main(int argc, char **argv) {
  common_init(argc, argv);
  logger = csv_init("alloc", "elapsed_ms", repl_enabled);

  for (int i = 0; i < NB_ROUNDS; i++) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    array = allocate_buffer_platform(repl_enabled, size);
    run_and_join_on_all_threads(alloc_worker);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = elapsed_time(start, end);
    printf("mem elapsed: %.6f ms\n", elapsed);
    csv_write(logger, i, elapsed, "mem");
  }

  csv_close(logger);
  return 0;
}
