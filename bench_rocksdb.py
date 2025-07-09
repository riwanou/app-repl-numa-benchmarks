import os
import config
import shutil
from config import sh

# all path are from rocksdb/build
BUILD_DIR = os.path.join("rocksdb", "build")

OUTPUT_DIR = os.path.abspath(config.RESULT_DIR_ROCKSDB)
NUM_THREADS = config.NUM_THREADS
COMPRESSION_TYPE = "none"
DB_DIR = os.path.join("..", "db")
WAL_DIR = os.path.join("..", "wal")
NUM_KEYS = 90000000
CACHE_SIZE = 6442450944
DURATION = 900

LOAD_ENV = f"COMPRESSION_TYPE={COMPRESSION_TYPE} DB_DIR={DB_DIR} WAL_DIR={WAL_DIR} NUM_KEYS={NUM_KEYS} CACHE_SIZE={CACHE_SIZE}"
BENCH_ENV = f"{LOAD_ENV} DURATION={DURATION} NUM_THREADS={NUM_THREADS}"
BENCHMARK_SCRIPT = os.path.join("..", "tools", "benchmark.sh")


def init_output_dir(output_dir: str):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)


def run_load(tag: str):
    output_dir = os.path.join(OUTPUT_DIR, tag)
    init_output_dir(output_dir)
    return f"{LOAD_ENV} OUTPUT_DIR={output_dir} {BENCHMARK_SCRIPT}"


def run_bench(tag: str) -> str:
    output_dir = os.path.join(OUTPUT_DIR, tag)
    init_output_dir(output_dir)
    return f"{BENCH_ENV} OUTPUT_DIR={output_dir} {BENCHMARK_SCRIPT}"


def makedirs():
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(WAL_DIR, exist_ok=True)


def run_bench_rocksdb():
    makedirs()
    os.chdir(BUILD_DIR)

    # create db, load data
    sh(f"{run_load('default')} bulkload")

    # run readrandom, multireadrandom
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"{run_bench('default')} readrandom --mmap_read=1")
    sh(
        f"{run_bench('default')} multireadrandom --mmap_read=1 --multiread_batched"
    )


def run_bench_rocksdb_repl():
    makedirs()
    os.chdir(BUILD_DIR)

    # create db, load data
    sh(f"{run_load('patched')} bulkload")

    # run readrandom, multireadrandom, no repl
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"{run_bench('patched')} readrandom --mmap_read=1")
    sh(
        f"{run_bench('patched')} multireadrandom --mmap_read=1 --multiread_batched"
    )

    # run readrandom, multireadrandom, repl
    sh("echo 1 > /sys/kernel/debug/repl_pt/clear_registered")
    sh("echo .sst > /sys/kernel/debug/repl_pt/registered")
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"""(
      echo 1 > /sys/kernel/debug/repl_pt/policy &&
      {run_bench("patched-repl")} readrandom --mmap_read=1 &&
      {run_bench("patched-repl")} multireadrandom --mmap_read=1 --multiread_batched;
      echo 0 > /sys/kernel/debug/repl_pt/policy
    )""")
