import os
from typing import Literal
import config
import shutil
import csv
from config import sh, get_time

# all path are from rocksdb/build
BUILD_DIR = os.path.join("rocksdb", "build")

RESULT_DIR = os.path.abspath(config.RESULT_DIR_ROCKSDB)
CSV_PATH = os.path.join(RESULT_DIR, "results.csv")
NUM_THREADS = config.NUM_THREADS
DB_DIR = os.path.join(config.TMP_DIR_ROCKSDB, "db")
WAL_DIR = os.path.join(config.TMP_DIR_ROCKSDB, "wal")
NUM_KEYS = 2000000
CACHE_SIZE = 6442450944
COMPRESSION_TYPE = "none"
DURATION = 120
STAT_INTERVAL_SECONDS = 15

LOAD_ENV = f"DB_DIR={DB_DIR} WAL_DIR={WAL_DIR} NUM_KEYS={NUM_KEYS} CACHE_SIZE={CACHE_SIZE} COMPRESSION_TYPE={COMPRESSION_TYPE}"
BENCH_ENV = f"{LOAD_ENV} DURATION={DURATION} STATS_INTERVAL_SECONDS={STAT_INTERVAL_SECONDS} NUM_THREADS={NUM_THREADS}"
BENCHMARK_SCRIPT = os.path.join("..", "tools", "benchmark.sh")


def decomment(csvfile):
    for row in csvfile:
        raw = row.split("#")[0].strip()
        if raw:
            yield raw


def run(
    tag: str,
    variant: Literal[
        "bulkload",
        "readrandom",
        "multireadrandom",
        "fwdrange",
        "readwhilewriting",
    ],
    numactl_invoc: str = "",
    repl: bool = False,
):
    output_dir = os.path.join(RESULT_DIR, "outputs", tag)
    output_option = f"OUTPUT_DIR={output_dir}"
    report_path = os.path.join(output_dir, "report.tsv")

    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    repl_start = ""
    repl_end = ""
    if repl:
        repl_start = "echo 1 > /sys/kernel/debug/repl_pt/policy &&"
        repl_end = "echo 0 > /sys/kernel/debug/repl_pt/policy"

    start_time = get_time()

    if variant == "bulkload":
        sh(f"{LOAD_ENV} {output_option} {BENCHMARK_SCRIPT} bulkload")
        return

    if variant == "readrandom":
        sh(
            f"""
            {repl_start}
            {BENCH_ENV} {output_option} {numactl_invoc} {BENCHMARK_SCRIPT} readrandom --mmap_read=1;
            {repl_end}
            """
        )
    elif variant == "multireadrandom":
        sh(
            f"""
            {repl_start}
            {BENCH_ENV} {output_option} {numactl_invoc} {BENCHMARK_SCRIPT} multireadrandom --mmap_read=1 --multiread_batched;
            {repl_end}
            """
        )
    elif variant == "fwdrange":
        sh(
            f"""
            {repl_start}
            {BENCH_ENV} {output_option} {numactl_invoc} {BENCHMARK_SCRIPT} fwdrange --mmap_read=1;
            {repl_end}
            """
        )
    elif variant == "readwhilewriting":
        sh(
            f"""
            {repl_start}
            {BENCH_ENV} MB_WRITE_PER_SEC=1 {output_option} {numactl_invoc} {BENCHMARK_SCRIPT} readwhilewriting --mmap_read=1;
            {repl_end}
            """
        )

    end_time = get_time()

    with open(report_path, mode="r", newline="") as f:
        reader = csv.DictReader(decomment(f), delimiter="\t")
        rows = list(reader)
        result = rows[0]
        result["tag"] = tag
        result["start_time"] = start_time
        result["end_time"] = end_time

    final_rows = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            final_rows = list(reader)

    final_rows = [row for row in final_rows if row.get("tag") != tag]
    final_rows.append(result)

    with open(CSV_PATH, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerows(final_rows)


def prepare_dirs():
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(WAL_DIR, exist_ok=True)
    os.chdir(BUILD_DIR)


def run_bench_rocksdb():
    prepare_dirs()

    # create db, load data
    run("bulkload", "bulkload")

    # run
    sh("echo 3 > /proc/sys/vm/drop_caches")
    run("multireadrandom", "multireadrandom")

    # worst case
    sh("echo 3 > /proc/sys/vm/drop_caches")
    run("imbalanced-multireadrandom", "multireadrandom", "numactl --membind=0")

    # best case
    sh("echo 3 > /proc/sys/vm/drop_caches")
    run(
        "interleaved-multireadrandom",
        "multireadrandom",
        "numactl --interleave=all",
    )


def run_bench_rocksdb_repl():
    prepare_dirs()
    benches = ["readrandom", "multireadrandom", "fwdrange", "readwhilewriting"]

    # create db, load data
    run("patched-bulkload", "bulkload")

    # best case
    for bench in benches:
        sh("echo 3 > /proc/sys/vm/drop_caches")
        run(
            f"patched-interleaved-{bench}",
            bench,
            "numactl --interleave=all",
        )

        sh("echo 1 > /sys/kernel/debug/repl_pt/clear_registered")
        sh("echo .sst > /sys/kernel/debug/repl_pt/registered")

        sh("echo 3 > /proc/sys/vm/drop_caches")
        run(f"patched-repl-{bench}", bench, repl=True)
