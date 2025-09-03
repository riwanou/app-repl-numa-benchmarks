import os
from config import sh, RESULT_DIR_FIO

RUNTIME = 45


def run_repl(cmd: str) -> str:
    return f"""(
      echo 1 > /sys/kernel/debug/repl_pt/policy &&
      {cmd};
      echo 0 > /sys/kernel/debug/repl_pt/policy
    )"""


def run_bench(
    section: str,
    tag: str,
    repl_enabled: bool,
    mixread=90,
    mixwrite=0,
    distrib="random",
) -> str:
    os.makedirs(RESULT_DIR_FIO, exist_ok=True)
    json_path = os.path.join(RESULT_DIR_FIO, f"{tag}.json")

    cmd = f"""RUNTIME={RUNTIME} \
        MIXREAD={mixread} \
        MIXWRITE={mixwrite} \
        DISTRIB={distrib} \
        ./fio-3.40/fio \
        --output-format=json \
        bench.fio \
        --output={json_path} \
        --section={section}"""

    if repl_enabled:
        return run_repl(cmd)
    else:
        return cmd


def run_bench_readwrite(mixread, mixwrite):
    base_tag = f"readwrite_{mixread}_{mixwrite}"

    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(
        f"{
            run_bench(
                section='readwrite',
                tag=base_tag,
                repl_enabled=False,
                mixread=mixread,
                mixwrite=mixwrite,
            )
        }"
    )

    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(
        f"{
            run_bench(
                section='readwrite',
                tag=f'{base_tag}_repl',
                repl_enabled=True,
                mixread=mixread,
                mixwrite=mixwrite,
            )
        }"
    )


def run_bench_fio():
    # random read
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"{run_bench(section='read', tag='read', repl_enabled=False)}")
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"{run_bench(section='read', tag='read_repl', repl_enabled=True)}")

    # random read write
    run_bench_readwrite(mixread=95, mixwrite=5)
    run_bench_readwrite(mixread=90, mixwrite=10)
    run_bench_readwrite(mixread=85, mixwrite=15)
