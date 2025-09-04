import os
from config import sh, RESULT_DIR_FIO

RUNTIME = 30


def run_repl(cmd: str) -> str:
    return f"""(
      echo 1 > /sys/kernel/debug/repl_pt/policy &&
      {cmd};
      echo 0 > /sys/kernel/debug/repl_pt/policy
    )"""


def run_bench(
    tag: str,
    repl_enabled: bool,
    readjobs,
    writejobs,
    distrib="random",
) -> str:
    os.makedirs(RESULT_DIR_FIO, exist_ok=True)
    json_path = os.path.join(RESULT_DIR_FIO, f"{tag}.json")

    cmd = f"""RUNTIME={RUNTIME} \
        READJOBS={readjobs} \
        WRITEJOBS={writejobs} \
        DISTRIB={distrib} \
        ./fio-3.40/fio \
        --output-format=json \
        {"--section=readers" if writejobs == 0 else ""} \
        bench.fio \
        --output={json_path}"""

    if repl_enabled:
        return run_repl(cmd)
    else:
        return cmd


def run_bench_readwrite(distrib, total_jobs, read_ratio, write_ratio):
    base_tag = f"readwrite_{distrib}_{read_ratio}_{write_ratio}"

    num_readers = round(total_jobs * read_ratio / 100)
    num_writers = total_jobs - num_readers

    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(
        f"{
            run_bench(
                tag=base_tag,
                repl_enabled=False,
                readjobs=num_readers,
                writejobs=num_writers,
                distrib=distrib,
            )
        }"
    )

    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(
        f"{
            run_bench(
                tag=f'{base_tag}_repl',
                repl_enabled=True,
                readjobs=num_readers,
                writejobs=num_writers,
                distrib=distrib,
            )
        }"
    )


def run_bench_fio_distrib(distrib):
    total_jobs = os.cpu_count()

    # random read write
    run_bench_readwrite(distrib, total_jobs, read_ratio=100, write_ratio=0)
    run_bench_readwrite(distrib, total_jobs, read_ratio=95, write_ratio=5)
    run_bench_readwrite(distrib, total_jobs, read_ratio=90, write_ratio=10)
    run_bench_readwrite(distrib, total_jobs, read_ratio=80, write_ratio=20)
    run_bench_readwrite(distrib, total_jobs, read_ratio=70, write_ratio=30)
    run_bench_readwrite(distrib, total_jobs, read_ratio=60, write_ratio=40)
    run_bench_readwrite(distrib, total_jobs, read_ratio=50, write_ratio=50)
    run_bench_readwrite(distrib, total_jobs, read_ratio=40, write_ratio=60)
    run_bench_readwrite(distrib, total_jobs, read_ratio=30, write_ratio=70)
    run_bench_readwrite(distrib, total_jobs, read_ratio=20, write_ratio=80)
    run_bench_readwrite(distrib, total_jobs, read_ratio=10, write_ratio=90)


def run_bench_fio():
    run_bench_fio_distrib("random")
    # run_bench_fio_distrib("zipf")
