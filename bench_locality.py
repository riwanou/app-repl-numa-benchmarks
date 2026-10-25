from config import RESULT_DIR_LOCALITY, repl_cmd, repl_setup, sh

DATASET = "gist-960-euclidean.hdf5"

RUNNER = "--usearch"

WARMUP = 30
WARMUP_BALANCING = 60



def run(tag: str, numactl: str = "", warmup: int = WARMUP, repl: bool = False):
    """No numactl: every CPU, and the kernel default first touch."""
    cmd = (
        f"{numactl}uv run run_ann.py {RUNNER} --bench --datasets {DATASET}"
        f" --tag {tag} --warmup-time {warmup}"
        f" --result-dir {RESULT_DIR_LOCALITY}"
    )
    if repl:
        cmd = repl_cmd(cmd)

    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    sh(cmd)


def register_repl():
    repl_setup(0, [".usearch"])


def run_bench_locality():
    """One node against the whole machine, per policy."""
    sh("echo 0 > /proc/sys/kernel/numa_balancing")

    run("local", "numactl --cpunodebind=0 --membind=0 ")

    run("firsttouch")
    run("imbalanced", "numactl --membind=0 ")
    run("interleaved", "numactl --interleave=all ")

    register_repl()
    run("repl", repl=True)

    sh("echo 1 > /proc/sys/kernel/numa_balancing")
    run("numa-balancing", warmup=WARMUP_BALANCING)
    sh("echo 0 > /proc/sys/kernel/numa_balancing")
