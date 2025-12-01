import os
from config import sh, RESULT_DIR_LLAMA


def run_repl(cmd: str) -> str:
    return f"""(
      echo 1 > /sys/kernel/debug/repl_pt/policy &&
      {cmd};
      echo 0 > /sys/kernel/debug/repl_pt/policy
    )"""


def run_bench(tag: str, repl_enabled: bool, numa_distribute=False):
    os.makedirs(RESULT_DIR_LLAMA, exist_ok=True)
    csv_path = os.path.join(RESULT_DIR_LLAMA, f"{tag}.csv")

    cmd = "./llama.cpp/build/bin/llama-bench -m ./llama.cpp/Llama-3.1-Tulu-3-8B-Q8_0.gguf -t $(nproc --all) --mmap 1 -n 128,256,512"

    if numa_distribute:
        cmd = f"{cmd} --numa distribute"

    cmd = f"{cmd} --output csv | tee {csv_path}"

    if repl_enabled:
        return run_repl(cmd)
    else:
        return cmd


def run_bench_llama():
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"{run_bench(tag='baseline', repl_enabled=False)}")

    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(
        f"{run_bench(tag='distribute', repl_enabled=False, numa_distribute=True)}"
    )


def run_bench_llama_repl():
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"{run_bench(tag='repl', repl_enabled=True)}")
