
#include "bench_common.h"

static csv_logger_t *logger;
static int mmap_main_alloc;

size_t get_system_anon_kb(void) {
  FILE *fp = fopen("/proc/meminfo", "r");
  if (!fp) {
    perror("fopen /proc/meminfo");
    exit(EXIT_FAILURE);
  }

  char line[256];
  size_t anon = (size_t)-1;

  while (fgets(line, sizeof(line), fp)) {
    if (sscanf(line, "AnonPages: %zu kB", &anon) == 1) {
      break;
    }
  }

  fclose(fp);

  if (anon == (size_t)-1) {
    fprintf(stderr, "Failed to parse AnonPages from /proc/meminfo\n");
    exit(EXIT_FAILURE);
  }

  return anon;
}

static void *mem_worker(void *arg) {
  int thread_id = *(int *)arg;
  int socket_id = thread_id % nsockets;
  int index_in_node = thread_id / nsockets;

  if (nthreads == 1 && use_mmap && repl_enabled) {
    socket_id =
        mmap_main_alloc ? main_node_id() : (main_node_id() + 1) % nsockets;
  }

  int core_id = get_nthcore_in_numa_socket(socket_id, index_in_node);
  set_affinity(gettid(), core_id);

  touch_buffer(repl_enabled, (char *)array, size);

  return NULL;
}

int main(int argc, char **argv) {
  common_init(argc, argv);

  char *env = getenv("MMAP_MAIN_ALLOC");
  mmap_main_alloc = (env != NULL && atoi(env) == 1);

  logger = csv_init("mem", "mem_used", repl_enabled);
  size_t anon_before, anon_after, anon;
  size_t system_before, system_after, system_mem;

  for (int i = 0; i < NB_ROUNDS; i++) {
    system_before = get_system_anon_kb();

    array = allocate_buffer_platform(repl_enabled, size);
    run_and_join_on_all_threads(mem_worker);

    system_after = get_system_anon_kb();
    system_mem = system_after - system_before;

    printf("system anon %zu\n", system_mem);
    munmap(array, size);
    csv_write(logger, i, system_mem, "mem");
  }

  csv_close(logger);
  return 0;
}
