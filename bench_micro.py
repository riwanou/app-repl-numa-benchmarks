import os
from config import sh, RESULT_DIR_MICROBENCH

BENCH_PGTABLE = "microbench/bench_pgtable"
BENCH_ALLOC = "microbench/bench_alloc"
BENCH_MEM = "microbench/bench_mem"


def prepare_dirs(result_dir):
    os.makedirs(result_dir, exist_ok=True)


def run_repl(cmd: str) -> str:
    return f"""(
      echo 1 > /sys/kernel/debug/repl_pt/policy &&
      {cmd};
      echo 0 > /sys/kernel/debug/repl_pt/policy
    )"""


def run_bench(
    bench: str,
    method: str,
    nthreads: int,
    repl_enabled: bool,
    mmap_main_alloc=False,
) -> str:
    # is carrefour
    if method == "madvise":
        result_dir = os.path.join(RESULT_DIR_MICROBENCH, "carrefour")
    else:
        result_dir = RESULT_DIR_MICROBENCH

    prepare_dirs(result_dir)

    cmd = f"""CSV_DIR={result_dir} \
      MMAP_MAIN_ALLOC={"1" if mmap_main_alloc else "0"} \
      REPLICATION={"1" if repl_enabled else "0"} \
      ./{bench} {method} {nthreads}"""
    if repl_enabled and method == "mmap":
        return run_repl(cmd)
    else:
        return cmd


def get_numa_nodes():
    """Return the number of NUMA nodes on this system."""
    nodes = [
        d
        for d in os.listdir("/sys/devices/system/node/")
        if d.startswith("node")
    ]
    return len(nodes)


def run_bench_pgtable(method: str):
    num_nodes = get_numa_nodes()

    for nthreads in range(1, num_nodes + 1):
        sh("echo 3 > /proc/sys/vm/drop_caches")
        sh(
            f"{run_bench(BENCH_PGTABLE, method, nthreads, repl_enabled=False, mmap_main_alloc=False)}"
        )
        sh("echo 3 > /proc/sys/vm/drop_caches")
        sh(
            f"{run_bench(BENCH_PGTABLE, method, nthreads, repl_enabled=True, mmap_main_alloc=False)}"
        )


def run_bench_alloc(method: str):
    num_nodes = get_numa_nodes()

    for nthreads in range(1, num_nodes + 1):
        sh("echo 3 > /proc/sys/vm/drop_caches")
        sh(
            f"{run_bench(BENCH_ALLOC, method, nthreads, repl_enabled=False, mmap_main_alloc=False)}"
        )
        sh("echo 3 > /proc/sys/vm/drop_caches")
        sh(
            f"{run_bench(BENCH_ALLOC, method, nthreads, repl_enabled=True, mmap_main_alloc=False)}"
        )


def run_bench_mem(method: str):
    num_nodes = get_numa_nodes()

    for nthreads in range(1, num_nodes + 1):
        sh("echo 3 > /proc/sys/vm/drop_caches")
        sh(
            f"{run_bench(BENCH_MEM, method, nthreads, repl_enabled=False, mmap_main_alloc=False)}"
        )
        sh("echo 3 > /proc/sys/vm/drop_caches")
        sh(
            f"{run_bench(BENCH_MEM, method, nthreads, repl_enabled=True, mmap_main_alloc=False)}"
        )
        if method == "mmap" and nthreads == 1:
            sh("echo 3 > /proc/sys/vm/drop_caches")
            sh(
                f"{run_bench(BENCH_MEM, method, nthreads, repl_enabled=True, mmap_main_alloc=True)}"
            )
