import os
import csv
import shutil
import config
from config import sh, get_time

# all path are from rocksdb/build
BUILD_DIR = os.path.join("rocksdb", "build")

RESULT_DIR = os.path.abspath(config.RESULT_DIR_ROCKSDB)
CSV_PATH = os.path.join(RESULT_DIR, "results.csv")
NUM_THREADS = config.NUM_THREADS
DB_DIR = os.path.join(config.TMP_DIR_ROCKSDB, "db")
WAL_DIR = os.path.join(config.TMP_DIR_ROCKSDB, "wal")
NUM_KEYS = 40_000_000
CACHE_SIZE = 16_000_000_000  # 16 GB
MB_WRITE_PER_SEC = 2
COMPRESSION_TYPE = "none"
DURATION = 60
RAMP_SECS = 20
STAT_INTERVAL_SECONDS = 5
NB_RUNS = 10

LOAD_ENV = f"DB_DIR={DB_DIR} WAL_DIR={WAL_DIR} NUM_KEYS={NUM_KEYS} CACHE_SIZE={CACHE_SIZE} COMPRESSION_TYPE={COMPRESSION_TYPE}"
BENCH_ENV = f"{LOAD_ENV} DURATION={DURATION} STATS_INTERVAL_SECONDS={STAT_INTERVAL_SECONDS} NUM_THREADS={NUM_THREADS}"
BENCHMARK_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "rocksdb",
    "tools",
    "benchmark.sh",
)

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


def _bench_cmd(
    variant: str, bench_env: str, output_option: str, numactl_invoc: str
) -> str:
    """Build the benchmark.sh shell command for a given variant."""
    base = f"{bench_env} {output_option} {numactl_invoc} {BENCHMARK_SCRIPT}"
    write_env = f"MB_WRITE_PER_SEC={MB_WRITE_PER_SEC}"
    # no_compact = "--disable_auto_compactions=1"
    # extra = f"--mmap_read=1 {no_compact}"
    extra = f"--mmap_read=1"
    cmds = {
        "readrandom": f"{base} readrandom {extra}",
        "multireadrandom": f"{base} multireadrandom --multiread_batched {extra}",
        "fwdrange": f"{base} fwdrange {extra}",
        "revrange": f"{base} revrange {extra}",
        "overwrite": f"{base} overwrite {extra}",
        "readwhilewriting": f"{bench_env} {write_env} {output_option} {numactl_invoc} {BENCHMARK_SCRIPT} readwhilewriting {extra}",
        "fwdrangewhilewriting": f"{bench_env} {write_env} {output_option} {numactl_invoc} {BENCHMARK_SCRIPT} fwdrangewhilewriting {extra}",
        "revrangewhilewriting": f"{bench_env} {write_env} {output_option} {numactl_invoc} {BENCHMARK_SCRIPT} revrangewhilewriting {extra}",
    }
    return cmds[variant]


def _load_db(output_tag: str):
    """Drop caches and load the database. Call once per round before running benches."""
    output_dir = os.path.join(RESULT_DIR, "outputs", f"{output_tag}_load")
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"{LOAD_ENV} OUTPUT_DIR={output_dir} {BENCHMARK_SCRIPT} bulkload")


def _do_bench(
    tag: str,
    variant: str,
    run_idx: int,
    numactl_invoc: str = "",
    repl: bool = False,
):
    """Run a single benchmark and append the result to the CSV."""
    output_dir = os.path.join(RESULT_DIR, "outputs", tag)
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "report.tsv")

    repl_start = ""
    repl_end = ""
    if repl:
        repl_start = "echo 1 > /sys/kernel/debug/repl_pt/policy &&"
        repl_end = "&& echo 0 > /sys/kernel/debug/repl_pt/policy"

    bench_cmd = _bench_cmd(
        variant, BENCH_ENV, f"OUTPUT_DIR={output_dir}", numactl_invoc
    )
    start_time = get_time()
    sh(f"{repl_start} {bench_cmd} {repl_end}")
    end_time = get_time()

    with open(report_path, mode="r", newline="") as f:
        result = list(csv.DictReader(decomment(f), delimiter="\t"))[0]

    # report.tsv averages over the full run including the ~9s ramp-up.
    # Recompute ops_sec / mb_sec from the last STABLE_WINDOW seconds of the
    # per-second CSV, which is always past the ramp-up.
    per_sec_csv = os.path.join(
        output_dir, f"benchmark_{variant}.t{NUM_THREADS}.log.r.csv"
    )
    if os.path.exists(per_sec_csv):
        with open(per_sec_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        stable = [r for r in rows if int(r["secs_elapsed"]) > RAMP_SECS]
        if stable:
            stable_ops = sum(int(r["interval_qps"]) for r in stable) / len(
                stable
            )
            ops_report = float(result["ops_sec"])
            mb_report = float(result["mb_sec"])
            result["ops_sec"] = f"{stable_ops:.0f}"
            # scale mb_sec by the same ratio so bytes-per-op stays consistent
            if ops_report > 0:
                result["mb_sec"] = f"{mb_report * stable_ops / ops_report:.1f}"

    result["tag"] = tag
    result["nb_runs"] = run_idx
    result["start_time"] = start_time
    result["end_time"] = end_time

    # Append result to CSV (replace existing row with same tag if any)
    final_rows = []
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            final_rows = list(reader)

    final_rows = [
        row
        for row in final_rows
        if not (row.get("tag") == tag and row.get("nb_runs") == str(run_idx))
    ]
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
    sh("echo 0 > /proc/sys/kernel/numa_balancing")

    # (variant_tag, numactl_invoc, pre_bench_setup, post_bench_teardown)
    variants = [
        ("default", "", None, None),
        ("imbalanced", "numactl --membind=0", None, None),
        ("interleaved", "numactl --interleave=all", None, None),
        (
            "balancing",
            "",
            lambda: sh("echo 1 > /proc/sys/kernel/numa_balancing"),
            lambda: sh("echo 0 > /proc/sys/kernel/numa_balancing"),
        ),
    ]

    for variant_tag, numactl, setup, teardown in variants:
        for bench in BENCHES:
            for run_idx in range(NB_RUNS):
                if setup:
                    setup()
                _load_db(f"{variant_tag}-{bench}-round{run_idx}")
                _do_bench(f"{variant_tag}-{bench}", bench, run_idx, numactl)
                if teardown:
                    teardown()


def run_bench_rocksdb_repl():
    prepare_dirs()

    # patched-interleaved variant: best case, debug purpose
    for bench in BENCHES:
        for run_idx in range(NB_RUNS):
            _load_db(f"patched-interleaved-{bench}-round{run_idx}")
            _do_bench(
                f"patched-interleaved-{bench}",
                bench,
                run_idx,
                "numactl --interleave=all",
            )

    # patched-repl variant: with normal replication
    for bench in BENCHES:
        for run_idx in range(NB_RUNS):
            _load_db(f"patched-repl-{bench}-round{run_idx}")
            sh("echo 1 > /sys/kernel/debug/repl_pt/clear_registered")
            sh("echo .sst > /sys/kernel/debug/repl_pt/registered")
            sh("echo 1 > /sys/kernel/debug/repl_pt/write_unreplication")
            _do_bench(f"patched-repl-{bench}", bench, run_idx, repl=True)
            sh("echo 0 > /sys/kernel/debug/repl_pt/write_unreplication")
