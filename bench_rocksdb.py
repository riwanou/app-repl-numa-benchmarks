import os
import config
import shutil
from config import sh

# all path are from rocksdb/build
BUILD_DIR = os.path.join("rocksdb", "build")

OUTPUT_DIR = os.path.abspath(config.RESULT_DIR_ROCKSDB)
NUM_THREADS = config.NUM_THREADS
DB_DIR = os.path.join("..", "db")
WAL_DIR = os.path.join("..", "wal")
NUM_KEYS = 90000000
CACHE_SIZE = 6442450944
DURATION = 300

LOAD_ENV = f"DB_DIR={DB_DIR} WAL_DIR={WAL_DIR} NUM_KEYS={NUM_KEYS} CACHE_SIZE={CACHE_SIZE}"
BENCH_ENV = f"{LOAD_ENV} DURATION={DURATION} NUM_THREADS={NUM_THREADS}"
BENCHMARK_SCRIPT = os.path.join("..", "tools", "benchmark.sh")


def init_output_dir(output_dir: str):
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)


def env_load(tag: str):
    output_dir = os.path.join(OUTPUT_DIR, tag)
    init_output_dir(output_dir)
    return f"{LOAD_ENV} OUTPUT_DIR={output_dir}"


def env_bench(tag: str) -> str:
    output_dir = os.path.join(OUTPUT_DIR, tag)
    init_output_dir(output_dir)
    return f"{BENCH_ENV} OUTPUT_DIR={output_dir}"


def makedirs():
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(WAL_DIR, exist_ok=True)


def run_bench_rocksdb():
    makedirs()
    os.chdir(BUILD_DIR)

    # create db, load data
    sh(f"{env_load('bulkload')} {BENCHMARK_SCRIPT} bulkload")

    # run
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"{env_bench('readrandom')} {BENCHMARK_SCRIPT} readrandom --mmap_read=1")
    sh(
        f"{env_bench('multireadrandom')} {BENCHMARK_SCRIPT} multireadrandom --mmap_read=1 --multiread_batched"
    )

    # worst case (mem in 1 node)
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(
        f"{env_bench('imbalanced-readrandom')} numactl --membind={0} {BENCHMARK_SCRIPT} readrandom --mmap_read=1"
    )
    sh(
        f"{env_bench('imbalanced-multireadrandom')} numactl --membind={0} {BENCHMARK_SCRIPT} multireadrandom --mmap_read=1 --multiread_batched"
    )


def run_bench_rocksdb_repl():
    makedirs()
    os.chdir(BUILD_DIR)

    # create db, load data
    sh(f"{env_load('patched-bulkload')} {BENCHMARK_SCRIPT} bulkload")

    # worst case (mem in 1 node)
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(
        f"{env_bench('patched-imbalanced-readrandom')} numactl --membind={0} {BENCHMARK_SCRIPT} readrandom --mmap_read=1"
    )
    sh(
        f"{env_bench('patched-imbalanced-multireadrandom')} numactl --membind={0} {BENCHMARK_SCRIPT} multireadrandom --mmap_read=1 --multiread_batched"
    )

    # run readrandom, multireadrandom, repl
    sh("echo 1 > /sys/kernel/debug/repl_pt/clear_registered")
    sh("echo .sst > /sys/kernel/debug/repl_pt/registered")
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"""(
      echo 1 > /sys/kernel/debug/repl_pt/policy &&
      {env_bench("patched-repl-readrandom")} {BENCHMARK_SCRIPT} readrandom --mmap_read=1 &&
      {env_bench("patched-repl-multireadrandom")} {BENCHMARK_SCRIPT} multireadrandom --mmap_read=1 --multiread_batched;
      echo 0 > /sys/kernel/debug/repl_pt/policy
    )""")
