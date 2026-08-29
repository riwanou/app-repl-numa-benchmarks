#include "bench_common.h"

/* SYNC=1: log the whole unmap averaged over its munmap calls, in ms. VARIANT
 * is baseline on the vanilla kernel, on on the repl one, where the second
 * pass gives the non replicated area. CHUNK_KB is the size of each munmap
 * call, in KiB, default one page. */
static int sync_mode;
static const char *variant;
static char sync_name[64];
static size_t chunk_size;

static csv_logger_t *logger;
static int round;

static char node_tag[32];

static const char *round_tag(void) {
  if (!sync_mode)
    return repl_enabled ? "pgtable_repl" : "pgtable_norepl";
  if (strcmp(variant, "on") != 0)
    return sync_name;
  const char *base = repl_enabled ? "pgtable_sync_repl" : "pgtable_sync_norepl";
  if (active_nodes == nsockets)
    return base;
  snprintf(node_tag, sizeof(node_tag), "%s_%un", base, active_nodes);
  return node_tag;
}

static void *pgtable_worker(void *arg) {
  unsigned int thread_id = *(unsigned int *)arg;
  unsigned int socket_id = thread_id % active_nodes;
  unsigned int index_in_node = thread_id / active_nodes;
  unsigned int core_id = get_nthcore_in_numa_socket(socket_id, index_in_node);
  set_affinity(gettid(), core_id);

  touch_buffer_read((char *)array, size);
  pthread_barrier_wait(&barrier);

  if (thread_id == 0) {
    struct timespec t_unmap_start, t_unmap_end;
    clock_gettime(CLOCK_MONOTONIC, &t_unmap_start);

    for (size_t i = 0; i < size; i += chunk_size)
      munmap((char *)array + i, chunk_size);

    clock_gettime(CLOCK_MONOTONIC, &t_unmap_end);
    double unmap_elapsed = elapsed_time(t_unmap_start, t_unmap_end);
    printf("unmap elapsed: %.6f ms\n", unmap_elapsed);

    if (sync_mode)
      unmap_elapsed = unmap_elapsed * 1e3 / (size / chunk_size);
    csv_write(logger, round, unmap_elapsed, round_tag());
  }

  return NULL;
}

static void run_rounds(void) {
  for (round = 0; round < NB_ROUNDS; round++) {
    array = allocate_buffer_platform(repl_enabled, size);
    touch_buffer_write(repl_enabled, (char *)array, size);
    run_and_join_on_all_threads(pgtable_worker);
  }
}

int main(int argc, char **argv) {
  common_init(argc, argv);

  char *sync_env = getenv("SYNC");
  sync_mode = (sync_env != NULL && atoi(sync_env) == 1);
  variant = getenv("VARIANT") ? getenv("VARIANT") : "baseline";

  char *chunk_env = getenv("CHUNK_KB");
  chunk_size = chunk_env ? (size_t)atol(chunk_env) * 1024UL : PAGE_SIZE;
  snprintf(sync_name, sizeof(sync_name), "pgtable_sync_%s", variant);

  logger = csv_init(sync_mode ? sync_name : "pgtable", "elapsed_ms");
  run_rounds();

  if (repl_enabled) {
    printf("> mmap without replication after replication\n");
    repl_enabled = 0;
    run_rounds();
  }

  csv_close(logger);
  return 0;
}
