#include "bench_common.h"

static csv_logger_t *logger;
static int i;

static void *pgtable_worker(void *arg) {
  int thread_id = *(int *)arg;
  int socket_id = thread_id % nsockets;
  int index_in_node = thread_id / nsockets;
  int core_id = get_nthcore_in_numa_socket(socket_id, index_in_node);
  set_affinity(gettid(), core_id);

  touch_buffer(repl_enabled, (char *)array, size);
  pthread_barrier_wait(&barrier);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  if (thread_id == 0) {
    struct timespec t_unmap_start, t_unmap_end;
    clock_gettime(CLOCK_MONOTONIC, &t_unmap_start);

    for (size_t i = 0; i < size; i += PAGE_SIZE)
      munmap((char *)array + i, PAGE_SIZE);

    clock_gettime(CLOCK_MONOTONIC, &t_unmap_end);
    double unmap_elapsed = elapsed_time(t_unmap_start, t_unmap_end);
    printf("unmap elapsed: %.6f ms\n", unmap_elapsed);
    csv_write(logger, i, unmap_elapsed, repl_enabled ? "repl" : "norepl");
  }

  return NULL;
}

int main(int argc, char **argv) {
  common_init(argc, argv);
  logger = csv_init("pgtable", "elapsed_ms", repl_enabled);
  
  for (i = 0; i < NB_ROUNDS; i++) {
    array = allocate_buffer_platform(repl_enabled, size);
    run_and_join_on_all_threads(pgtable_worker);
  }

  if (repl_enabled) {
    printf("> mmap without replication after replication\n");
    repl_enabled = 0;
    for (i = 0; i < NB_ROUNDS; i++) {
      array = allocate_buffer_platform(repl_enabled, size);
      run_and_join_on_all_threads(pgtable_worker);
    }
  }

  csv_close(logger);
  return 0;
}
