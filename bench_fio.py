import json
import os
import time
from config import sh, RESULT_DIR_FIO, HYDRA_NUMACTL, MITOSIS_NUMACTL, ARCH_SUBNAMES, PLATFORM
from config import repl_cmd, repl_knob, repl_setup, repl_write_unreplication

# same size for every fio bench
SIZE = "1G"
# percent a read takes longer once its page is shared, per machine
READ_PENALTY = {"gold": 195, "silver": 61}.get(ARCH_SUBNAMES.get(PLATFORM), 100)
RUNTIME = 30
NB_RUNS = 3
NB_RUNS_PGTABLE = 5
TEMP_JSON = "/tmp/fio_run_tmp.json"


# bench.fio maps this file; replication registers it by name
FIO_FILE = "fio_readwrite"
# a disk filesystem, not tmpfs: writeback is part of what is measured
FIO_DIR = os.environ.get("FIO_DIR", "/tmp")


def run_bench(
    repl_enabled: bool,
    readjobs,
    writejobs,
    distrib="random",
    prepend="",
    size=SIZE,
    wdistrib=None,
) -> str:
    cmd = f"""RUNTIME={RUNTIME} \
        READJOBS={readjobs} \
        WRITEJOBS={writejobs} \
        DISTRIB={distrib} \
        SIZE={size} \
        WDISTRIB={wdistrib or distrib} \
        FIO_DIR={FIO_DIR} \
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
        "wdistrib": wdistrib or distrib,
    }

    return repl_cmd(f"env {cmd}") if repl_enabled else cmd, meta


def run_one(json_path, run, tag, cmd, meta):
    # reset system state between each run
    sh(
        "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory"
    )
    # a fresh file: no page cache left from the previous run
    sh(f"rm -f {FIO_DIR}/{FIO_FILE}; truncate -s {meta['size']} {FIO_DIR}/{FIO_FILE}")
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


# stock kernel: vanilla linux, numa balancing on
def run_bench_readwrite(
    distrib, base_tag, num_readers, num_writers, wdistrib=None, interleaved=False
):
    json_path = init_json(f"{base_tag}-default")
    cmd, meta = run_bench(
        repl_enabled=False,
        readjobs=num_readers,
        writejobs=num_writers,
        distrib=distrib,
        wdistrib=wdistrib,
    )

    sh("echo 1 > /proc/sys/kernel/numa_balancing")
    for run in range(1, NB_RUNS + 1):
        run_one(json_path, run, "numabalancing", cmd, meta)
    sh("echo 0 > /proc/sys/kernel/numa_balancing")

    if interleaved:
        cmd, meta = run_bench(
            repl_enabled=False,
            readjobs=num_readers,
            writejobs=num_writers,
            distrib=distrib,
            prepend="numactl --interleave=all",
            wdistrib=wdistrib,
        )
        for run in range(1, NB_RUNS + 1):
            run_one(json_path, run, "interleaved", cmd, meta)


REPL_VARIANTS = ("interleaved", "repl", "unrepl", "policy")


def run_bench_readwrite_repl(
    distrib,
    base_tag,
    num_readers,
    num_writers,
    wdistrib=None,
    variants=REPL_VARIANTS,
    retry=60,
    suffix="repl",
):
    json_path = init_json(f"{base_tag}-{suffix}")
    cmd, meta = run_bench(
        repl_enabled=True,
        readjobs=num_readers,
        writejobs=num_writers,
        distrib=distrib,
        wdistrib=wdistrib,
    )
    cmd_interleaved, meta_interleaved = run_bench(
        repl_enabled=False,
        readjobs=num_readers,
        writejobs=num_writers,
        distrib=distrib,
        prepend="numactl --interleave=all",
        wdistrib=wdistrib,
    )

    # same kernel as the policy: the reference it should match on writes
    sh("echo 0 > /proc/sys/kernel/numa_balancing")
    if "interleaved" in variants:
        for run in range(1, NB_RUNS + 1):
            run_one(
                json_path, run, "interleaved", cmd_interleaved, meta_interleaved
            )

    # replication, main bound
    repl_setup(0, [FIO_FILE])
    if "repl" in variants:
        for run in range(1, NB_RUNS + 1):
            run_one(json_path, run, "repl", cmd, meta)

    # unreplication: reads lose nothing to sharing, no retry: always shared
    repl_write_unreplication(True)
    if "unrepl" in variants:
        repl_knob("read_penalty", 0)
        repl_knob("write_retry_interval", 0)
        for run in range(1, NB_RUNS + 1):
            run_one(json_path, run, "unrepl", cmd, meta)

    # policy: the defaults
    repl_knob("read_penalty", READ_PENALTY)
    repl_knob("write_retry_interval", retry)
    if "policy" in variants:
        for run in range(1, NB_RUNS + 1):
            run_one(json_path, run, "policy", cmd, meta)

    repl_knob("write_retry_interval", 60)
    repl_write_unreplication(False)


def run_bench_readwrite_fio(repl, distrib, total_jobs, read_ratio, write_ratio):
    base_tag = f"readwrite_{distrib}_{read_ratio}_{write_ratio}"
    num_readers = round(total_jobs * read_ratio / 100)
    num_writers = total_jobs - num_readers

    if repl:
        run_bench_readwrite_repl(distrib, base_tag, num_readers, num_writers)
    else:
        run_bench_readwrite(distrib, base_tag, num_readers, num_writers)


# read ratios (%), the rest writes
READ_RATIOS = [100, 90, 80, 70, 60, 50, 30, 10]


def run_bench_fio_distrib(distrib, repl=False):
    total_jobs = os.cpu_count()

    for read_ratio in READ_RATIOS:
        run_bench_readwrite_fio(
            repl, distrib, total_jobs, read_ratio, 100 - read_ratio
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
    # data placement is fixed by numactl, balancing would move it
    sh("echo 0 > /proc/sys/kernel/numa_balancing")
    for run in range(1, NB_RUNS_PGTABLE + 1):
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


# writers on the first 20% of the file, readers on the other 80%
HOT_PCT = 20
HOT_WRITE = f"zoned:100/{HOT_PCT}:0/{100 - HOT_PCT}"
HOT_READ = f"zoned:0/{HOT_PCT}:100/{100 - HOT_PCT}"
# short enough for the policy to retry inside a run
HOT_RETRY = 10


def run_bench_fio_hot(variants=REPL_VARIANTS, suffix="repl"):
    num_readers = os.cpu_count() // 2
    run_bench_readwrite_repl(
        HOT_READ,
        f"readwrite_split{HOT_PCT}_50_50",
        num_readers,
        os.cpu_count() - num_readers,
        wdistrib=HOT_WRITE,
        variants=variants,
        retry=HOT_RETRY,
        suffix=suffix,
    )


# the same policy on a kernel whose retry copies every page back
def run_bench_fio_hot_trial():
    run_bench_fio_hot(variants=("policy",), suffix="trial")


# 80% readers on the whole file, 20% writers on its first pct%
WRITTEN_PCTS = [5, 20, 50]
WRITTEN_VARIANTS = ("repl", "unrepl", "policy")


def run_bench_fio_written():
    num_readers = round(os.cpu_count() * 0.8)
    for pct in WRITTEN_PCTS:
        run_bench_readwrite_repl(
            "random",
            f"readwrite_written{pct}_80_20",
            num_readers,
            os.cpu_count() - num_readers,
            wdistrib=f"zoned:100/{pct}:0/{100 - pct}",
            variants=WRITTEN_VARIANTS,
        )


# stock kernel: the written parts under numa balancing and interleave
def run_bench_fio_written_default():
    num_readers = round(os.cpu_count() * 0.8)
    for pct in WRITTEN_PCTS:
        run_bench_readwrite(
            "random",
            f"readwrite_written{pct}_80_20",
            num_readers,
            os.cpu_count() - num_readers,
            wdistrib=f"zoned:100/{pct}:0/{100 - pct}",
            interleaved=True,
        )
