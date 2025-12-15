import os
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
NUM_KEYS = 40_000_000
CACHE_SIZE = 32 * 1000 * 1000 * 1000  # 32 GB
MB_WRITE_PER_SEC = 2
COMPRESSION_TYPE = "none"
DURATION = 300
STAT_INTERVAL_SECONDS = 30
NB_RUNS = 10

LOAD_ENV = f"DB_DIR={DB_DIR} WAL_DIR={WAL_DIR} NUM_KEYS={NUM_KEYS} CACHE_SIZE={CACHE_SIZE} COMPRESSION_TYPE={COMPRESSION_TYPE}"
BENCH_ENV = f"{LOAD_ENV} DURATION={DURATION} STATS_INTERVAL_SECONDS={STAT_INTERVAL_SECONDS} NUM_THREADS={NUM_THREADS}"
BENCHMARK_SCRIPT = os.path.join("..", "tools", "benchmark.sh")

BENCHES = [
    "readrandom",
    "multireadrandom",
    "fwdrange",
    "revrange",
    "readwhilewriting",
    "overwrite",
    "fwdrangewhilewriting",
    "revrangewhilewriting",
]


def decomment(csvfile):
    for row in csvfile:
        raw = row.split("#")[0].strip()
        if raw:
            yield raw


def run(
    tag: str,
    variant: str,
    numactl_invoc: str = "",
    repl: bool = False,
    num_runs: int = NB_RUNS,
):
    all_results = []

    output_dir = os.path.join(RESULT_DIR, "outputs", tag)
    output_option = f"OUTPUT_DIR={output_dir}"
    output_option_load = f"OUTPUT_DIR={output_dir}_load"
    report_path = os.path.join(output_dir, "report.tsv")

    repl_start = ""
    repl_end = ""
    if repl:
        repl_start = "echo 1 > /sys/kernel/debug/repl_pt/policy &&"
        repl_end = "echo 0 > /sys/kernel/debug/repl_pt/policy"

    for run_idx in range(num_runs):
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)

        # reset state for each run, avoid strange effect
        sh("echo 3 > /proc/sys/vm/drop_caches")
        sh(f"{LOAD_ENV} {output_option_load} {BENCHMARK_SCRIPT} bulkload")

        start_time = get_time()

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
        elif variant == "revrange":
            sh(
                f"""
                {repl_start}
                {BENCH_ENV} {output_option} {numactl_invoc} {BENCHMARK_SCRIPT} revrange --mmap_read=1;
                {repl_end}
                """
            )
        elif variant == "overwrite":
            sh(
                f"""
                {repl_start}
                {BENCH_ENV} {output_option} {numactl_invoc} {BENCHMARK_SCRIPT} overwrite --mmap_read=1;
                {repl_end}
                """
            )
        elif variant == "readwhilewriting":
            sh(
                f"""
                {repl_start}
                {BENCH_ENV} MB_WRITE_PER_SEC={MB_WRITE_PER_SEC} {output_option} {numactl_invoc} {BENCHMARK_SCRIPT} readwhilewriting --mmap_read=1;
                {repl_end}
                """
            )
        elif variant == "fwdrangewhilewriting":
            sh(
                f"""
                {repl_start}
                {BENCH_ENV} MB_WRITE_PER_SEC={MB_WRITE_PER_SEC} {output_option} {numactl_invoc} {BENCHMARK_SCRIPT} fwdrangewhilewriting --mmap_read=1;
                {repl_end}
                """
            )
        elif variant == "revrangewhilewriting":
            sh(
                f"""
                {repl_start}
                {BENCH_ENV} MB_WRITE_PER_SEC={MB_WRITE_PER_SEC} {output_option} {numactl_invoc} {BENCHMARK_SCRIPT} revrangewhilewriting --mmap_read=1;
                {repl_end}
                """
            )

        end_time = get_time()

        with open(report_path, mode="r", newline="") as f:
            reader = csv.DictReader(decomment(f), delimiter="\t")
            rows = list(reader)
            result = rows[0]
            result["tag"] = tag
            result["nb_runs"] = run_idx
            result["start_time"] = start_time
            result["end_time"] = end_time

            all_results.append(result)

    final_rows = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            final_rows = list(reader)

    final_rows = [row for row in final_rows if row.get("tag") != tag]
    final_rows.extend(all_results)

    if all_results:
        with open(CSV_PATH, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(final_rows)


def prepare_dirs():
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(WAL_DIR, exist_ok=True)
    os.chdir(BUILD_DIR)


def run_bench_rocksdb():
    prepare_dirs()

    # disable numa balancing
    sh("echo 0 > /proc/sys/kernel/numa_balancing")

    for bench in BENCHES:
        # default
        run(
            f"{bench}",
            bench,
        )

        # worst case
        run(
            f"imbalanced-{bench}",
            bench,
            "numactl --membind=0",
        )

        # best case
        run(
            f"interleaved-{bench}",
            bench,
            "numactl --interleave=all",
        )

        # a case
        sh("echo 1 > /proc/sys/kernel/numa_balancing")
        run(
            f"balancing-{bench}",
            bench,
        )
        sh("echo 0 > /proc/sys/kernel/numa_balancing")


def run_bench_rocksdb_repl():
    prepare_dirs()

    # create db, load data

    for bench in BENCHES:
        # best case, debug purpose
        run(
            f"patched-interleaved-{bench}",
            bench,
            "numactl --interleave=all",
        )

        # with normal replication
        sh("echo 1 > /sys/kernel/debug/repl_pt/clear_registered")
        sh("echo .sst > /sys/kernel/debug/repl_pt/registered")
        sh("echo 1 > /sys/kernel/debug/repl_pt/write_unreplication")
        run(f"patched-repl-{bench}", bench, repl=True)
        sh("echo 0 > /sys/kernel/debug/repl_pt/write_unreplication")

        # with unreplication on write pressure
        # but there is no write so not needed
        # sh("echo 3 > /proc/sys/vm/drop_caches")
        # run("patched-bulkload", "bulkload")

        # sh("echo 1 > /sys/kernel/debug/repl_pt/write_unreplication")
        # run(f"patched-repl-unrepl-{bench}", bench, repl=True)
        # sh("echo 0 > /sys/kernel/debug/repl_pt/write_unreplication")
