import json
import os
import time
from config import sh, RESULT_DIR_FIO, HYDRA_NUMACTL, MITOSIS_NUMACTL

# same size for every fio bench
SIZE = "1G"
RUNTIME = 30
NB_RUNS = 5
NB_RUNS_PGTABLE = 5
TEMP_JSON = "/tmp/fio_run_tmp.json"


def run_repl(cmd: str) -> str:
    return f"""(
      echo 1 > /sys/kernel/debug/repl_pt/policy &&
      {cmd};
      echo 0 > /sys/kernel/debug/repl_pt/policy
    )"""


def run_bench(
    repl_enabled: bool,
    readjobs,
    writejobs,
    distrib="random",
    prepend="",
    size=SIZE,
) -> str:
    cmd = f"""RUNTIME={RUNTIME} \
        READJOBS={readjobs} \
        WRITEJOBS={writejobs} \
        DISTRIB={distrib} \
        SIZE={size} \
        {prepend} \
        ./fio-3.40/fio \
        --output-format=json \
        {"--section=readers" if writejobs == 0 else ""} \
        bench.fio \
        --output={TEMP_JSON}"""

    meta = {
        "distrib": distrib,
        "readjobs": readjobs,
        "writejobs": writejobs,
        "repl_enabled": repl_enabled,
        "prepend": prepend,
        "size": size,
    }

    return run_repl(cmd) if repl_enabled else cmd, meta


def run_one(json_path, run, tag, cmd, meta):
    ts_start = time.time()
    sh(cmd)
    ts_end = time.time()
    with open(TEMP_JSON) as f:
        record = {
            "run": run,
            "tag": tag,
            **meta,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "data": json.load(f),
        }
    with open(json_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def init_json(filename):
    path = os.path.join(RESULT_DIR_FIO, f"{filename}.jsonl")
    os.makedirs(RESULT_DIR_FIO, exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    return path


def run_bench_readwrite(distrib, base_tag, num_readers, num_writers):
    json_path = init_json(f"{base_tag}-default")
    cmd, meta = run_bench(
        repl_enabled=False,
        readjobs=num_readers,
        writejobs=num_writers,
        distrib=distrib,
    )

    cmd_interleaved, meta_interleaved = run_bench(
        repl_enabled=False,
        readjobs=num_readers,
        writejobs=num_writers,
        distrib=distrib,
        prepend="numactl --interleave=all",
    )

    # default (without NUMA balancing) — all runs first
    sh("echo 0 > /proc/sys/kernel/numa_balancing")
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    for run in range(1, NB_RUNS + 1):
        run_one(json_path, run, "default", cmd, meta)

    # interleaved (still without NUMA balancing)
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    for run in range(1, NB_RUNS + 1):
        run_one(
            json_path, run, "interleaved", cmd_interleaved, meta_interleaved
        )

    # baseline (with NUMA Balancing) — all runs together
    sh("echo 1 > /proc/sys/kernel/numa_balancing")
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    for run in range(1, NB_RUNS + 1):
        run_one(json_path, run, "numabalancing", cmd, meta)
    sh("echo 0 > /proc/sys/kernel/numa_balancing")


def run_bench_readwrite_repl(distrib, base_tag, num_readers, num_writers):
    json_path = init_json(f"{base_tag}-repl")
    cmd, meta = run_bench(
        repl_enabled=True,
        readjobs=num_readers,
        writejobs=num_writers,
        distrib=distrib,
    )

    sh("echo 0 > /sys/kernel/debug/repl_pt/main_placement")

    # replication
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    for run in range(1, NB_RUNS + 1):
        run_one(json_path, run, "repl", cmd, meta)

    # unreplication
    sh("echo 1 > /sys/kernel/debug/repl_pt/write_unreplication")

    # main bound
    sh("echo 0 > /sys/kernel/debug/repl_pt/main_placement")
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    for run in range(1, NB_RUNS + 1):
        run_one(json_path, run, "unrepl-bound", cmd, meta)

    # (main first touch)
    sh("echo 1 > /sys/kernel/debug/repl_pt/main_placement")
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    for run in range(1, NB_RUNS + 1):
        run_one(json_path, run, "unrepl-firsttouch", cmd, meta)

    # (main interleaved)
    sh("echo 2 > /sys/kernel/debug/repl_pt/main_placement")
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    for run in range(1, NB_RUNS + 1):
        run_one(json_path, run, "unrepl-interleaved", cmd, meta)

    sh("echo 0 > /sys/kernel/debug/repl_pt/main_placement")
    sh("echo 0 > /sys/kernel/debug/repl_pt/write_unreplication")


def run_bench_readwrite_fio(repl, distrib, total_jobs, read_ratio, write_ratio):
    base_tag = f"readwrite_{distrib}_{read_ratio}_{write_ratio}"
    num_readers = round(total_jobs * read_ratio / 100)
    num_writers = total_jobs - num_readers

    if repl:
        run_bench_readwrite_repl(distrib, base_tag, num_readers, num_writers)
    else:
        run_bench_readwrite(distrib, base_tag, num_readers, num_writers)


def run_bench_fio_distrib(distrib, repl=False):
    total_jobs = os.cpu_count()

    # random read write
    run_bench_readwrite_fio(
        repl, distrib, total_jobs, read_ratio=100, write_ratio=0
    )
    run_bench_readwrite_fio(
        repl, distrib, total_jobs, read_ratio=95, write_ratio=5
    )
    run_bench_readwrite_fio(
        repl, distrib, total_jobs, read_ratio=90, write_ratio=10
    )
    run_bench_readwrite_fio(
        repl, distrib, total_jobs, read_ratio=85, write_ratio=15
    )
    run_bench_readwrite_fio(
        repl, distrib, total_jobs, read_ratio=80, write_ratio=20
    )
    run_bench_readwrite_fio(
        repl, distrib, total_jobs, read_ratio=70, write_ratio=30
    )
    run_bench_readwrite_fio(
        repl, distrib, total_jobs, read_ratio=60, write_ratio=40
    )
    run_bench_readwrite_fio(
        repl, distrib, total_jobs, read_ratio=50, write_ratio=50
    )
    run_bench_readwrite_fio(
        repl, distrib, total_jobs, read_ratio=40, write_ratio=60
    )
    run_bench_readwrite_fio(
        repl, distrib, total_jobs, read_ratio=30, write_ratio=70
    )
    run_bench_readwrite_fio(
        repl, distrib, total_jobs, read_ratio=20, write_ratio=80
    )
    run_bench_readwrite_fio(
        repl, distrib, total_jobs, read_ratio=10, write_ratio=90
    )


def run_bench_fio():
    run_bench_fio_distrib("random")


def run_bench_fio_repl():
    run_bench_fio_distrib("random", repl=True)


def run_bench_fio_pgtable(json_path, tag, prepend="", spare_repl=False):
    total_jobs = os.cpu_count()

    cmd, meta = run_bench(
        repl_enabled=spare_repl,
        readjobs=total_jobs,
        writejobs=0,
        prepend=prepend,
        size=SIZE,
    )
    for run in range(1, NB_RUNS_PGTABLE + 1):
        sh("sync; echo 3 > /proc/sys/vm/drop_caches")
        run_one(json_path, run, tag, cmd, meta)


def run_bench_fio_pgt_spare():
    json_path = init_json("pgtable_spare")
    run_bench_fio_pgtable(
        json_path, tag="interleave", prepend="numactl --interleave=all"
    )

    # page tables only, main interleaved
    sh("echo 1 > /sys/kernel/debug/repl_pt/pgtable_only")
    sh("echo 2 > /sys/kernel/debug/repl_pt/main_placement")
    run_bench_fio_pgtable(json_path, tag="repl-pt", spare_repl=True)
    sh("echo 0 > /sys/kernel/debug/repl_pt/main_placement")
    sh("echo 0 > /sys/kernel/debug/repl_pt/pgtable_only")

    # page tables + data, main bound
    run_bench_fio_pgtable(json_path, tag="repl", spare_repl=True)


def run_bench_fio_pgt_mitosis():
    json_path = init_json("pgtable_mitosis")
    run_bench_fio_pgtable(
        json_path,
        tag="interleave",
        prepend=f"{MITOSIS_NUMACTL} --interleave=all",
    )
    run_bench_fio_pgtable(
        json_path,
        tag="repl",
        prepend=f"{MITOSIS_NUMACTL} --pgtablerepl=all --interleave=all",
    )


def run_bench_fio_pgt_hydra():
    json_path = init_json("pgtable_hydra")
    run_bench_fio_pgtable(
        json_path, tag="interleave", prepend=f"{HYDRA_NUMACTL} --interleave=all"
    )
    run_bench_fio_pgtable(
        json_path,
        tag="repl",
        prepend=f"{HYDRA_NUMACTL} --pgtablerepl=all --interleave=all",
    )
