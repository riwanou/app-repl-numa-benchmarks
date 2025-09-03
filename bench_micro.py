from config import sh, RESULT_DIR_MICROBENCH

BENCH_PGTABLE = "bench_pgtable"


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
    cmd = f"""
    CSV_DIR={RESULT_DIR_MICROBENCH}
    MMAP_MAIN_ALLOC={"1" if mmap_main_alloc else "0"}
    REPLICATION={"1" if repl_enabled else "0"}
    ./{bench} {method} {nthreads}
    """
    if repl_enabled:
        return run_repl(cmd)
    else:
        return cmd


def run_bench_pgtable(method: str):
    sh(f"{run_bench(BENCH_PGTABLE, method, 1, False, False)}")
    sh(f"{run_bench(BENCH_PGTABLE, method, 2, True, False)}")
