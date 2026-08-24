"""duckdb TPC-H and ClickBench under each memory policy."""

import os
import config
from config import sh

RESULT_DIR = config.RESULT_DIR_DUCKDB

SCALE_FACTORS = [10, 30]
THREADS_PER_STREAM = 8
STREAMS = sorted({1, max(1, config.NUM_THREADS // THREADS_PER_STREAM)})

# tag, numactl, numa balancing
ARMS = [
    ("firsttouch", "", False),
    ("imbalanced", "numactl --membind=0", False),
    ("interleaved", "numactl --interleave=all", False),
    ("numa-balancing", "", True),
]

REPL = "/sys/kernel/debug/repl_pt"


def run_tpch(sf, tag, streams):
    return (
        f"uv run python -m duckdb_lib.tpch --sf {sf} --tag {tag}"
        f" --streams {streams}"
    )


def run_clickbench(tag, streams):
    return f"uv run python -m duckdb_lib.clickbench --tag {tag} --streams {streams}"


def _drop_caches():
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")


def _repl_setup():
    sh("echo 0 > /proc/sys/kernel/numa_balancing")
    sh(f"echo 0 > {REPL}/main_placement")
    sh(f"echo 1 > {REPL}/clear_registered")
    sh(f"echo .db > {REPL}/registered")


def _repl_run(cmd):
    # policy is per-pid, so the run must be a child of this shell
    _drop_caches()
    sh(f"""(
      echo 1 > {REPL}/policy &&
      {cmd};
      echo 0 > {REPL}/policy
    )""")
    sh(f"cat {REPL}/registered")


def build_duckdb():
    sh("uv run python -m duckdb_lib.build")


def prepare_dirs():
    os.makedirs(RESULT_DIR, exist_ok=True)


def run_bench_duckdb():
    prepare_dirs()

    for tag, numactl, balancing in ARMS:
        sh(f"echo {int(balancing)} > /proc/sys/kernel/numa_balancing")
        for streams in STREAMS:
            for sf in SCALE_FACTORS:
                _drop_caches()
                sh(f"{numactl} {run_tpch(sf, tag, streams)}")
            _drop_caches()
            sh(f"{numactl} {run_clickbench(tag, streams)}")

    sh("echo 0 > /proc/sys/kernel/numa_balancing")


def run_bench_duckdb_repl():
    prepare_dirs()
    _repl_setup()

    for streams in STREAMS:
        for sf in SCALE_FACTORS:
            _repl_run(run_tpch(sf, "repl", streams))
        _repl_run(run_clickbench("repl", streams))
