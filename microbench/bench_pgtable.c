#include "bench_common.h"

static csv_logger_t *logger;
static int round;

static void *pgtable_worker(void *arg) {
  unsigned int thread_id = *(unsigned int *)arg;
  unsigned int socket_id = thread_id % nsockets;
  unsigned int index_in_node = thread_id / nsockets;
  unsigned int core_id = get_nthcore_in_numa_socket(socket_id, index_in_node);
  set_affinity(gettid(), core_id);

  touch_buffer_read((char *)array, size);
  pthread_barrier_wait(&barrier);

  if (thread_id == 0) {
    struct timespec t_unmap_start, t_unmap_end;
    clock_gettime(CLOCK_MONOTONIC, &t_unmap_start);

    for (size_t i = 0; i < size; i += PAGE_SIZE)
      munmap((char *)array + i, PAGE_SIZE);

    clock_gettime(CLOCK_MONOTONIC, &t_unmap_end);
    double unmap_elapsed = elapsed_time(t_unmap_start, t_unmap_end);
    printf("unmap elapsed: %.6f ms\n", unmap_elapsed);
    csv_write(logger, round, unmap_elapsed,
              repl_enabled ? "pgtable_repl" : "pgtable_norepl");
  }

  return NULL;
}

int main(int argc, char **argv) {
  common_init(argc, argv);
  logger = csv_init("pgtable", "elapsed_ms");

  for (round = 0; round < NB_ROUNDS; round++) {
    array = allocate_buffer_platform(repl_enabled, size);
    touch_buffer_write(repl_enabled, (char *)array, size);
    run_and_join_on_all_threads(pgtable_worker);
  }

  if (repl_enabled) {
    printf("> mmap without replication after replication\n");
    repl_enabled = 0;
    for (round = 0; round < NB_ROUNDS; round++) {
      array = allocate_buffer_platform(repl_enabled, size);
      touch_buffer_write(repl_enabled, (char *)array, size);
      run_and_join_on_all_threads(pgtable_worker);
    }
  }

  csv_close(logger);
  return 0;
}
