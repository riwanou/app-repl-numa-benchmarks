import multiprocessing
import os
import re
import subprocess
import tempfile
import datetime


def get_safe_platform_string():
    arch = os.uname().machine.upper()
    brand_raw = "unknown-cpu"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    brand_raw = line.split(":", 1)[1].strip()
                    break
    except Exception as e:
        print(f"Warning: Could not read CPU info from /proc. Error: {e}")

    platform_name = re.sub(
        r"\s+",
        "_",
        re.sub(r"[()]", "", brand_raw).strip(),
    )

    return f"{platform_name}_{arch}"


NUM_THREADS = multiprocessing.cpu_count()
PLATFORM = get_safe_platform_string()

TMP_DIR = tempfile.gettempdir()
TMP_DIR_ROCKSDB = os.path.join(TMP_DIR, "rocksdb")

RESULT_DIR = "results"
RESULT_DIR_ANN = os.path.join(RESULT_DIR, PLATFORM, "ann")
RESULT_DIR_ROCKSDB = os.path.join(RESULT_DIR, PLATFORM, "rocksdb")
RESULT_DIR_FIO = os.path.join(RESULT_DIR, PLATFORM, "fio")
RESULT_DIR_MICROBENCH = os.path.join(RESULT_DIR, PLATFORM, "microbench")

PLOT_DIR = "plots"
PLOT_DIR_ANN = os.path.join(PLOT_DIR, "ann")
PLOT_DIR_ROCKSDB = os.path.join(PLOT_DIR, "rocksdb")
PLOT_DIR_FIO = os.path.join(PLOT_DIR, "fio")
PLOT_DIR_MONITORING = os.path.join(PLOT_DIR, "monitoring")
PLOT_DIR_MICROBENCH = os.path.join(PLOT_DIR, "microbench")

MONITOR_DIR = os.path.join(RESULT_DIR, PLATFORM, "monitor")
MONITOR_PCM = os.path.join(MONITOR_DIR, "pcm")
MONITOR_PCM_MEMORY = os.path.join(MONITOR_DIR, "pcm_memory")
MONITOR_MEM = os.path.join(MONITOR_DIR, "mem")


def sh(cmd, cwd=None):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, executable="/bin/bash")


def get_time():
    return datetime.datetime.now().isoformat()
