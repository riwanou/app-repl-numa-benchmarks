import multiprocessing
import os
import re
import cpuinfo
import subprocess
import tempfile
import datetime

NUM_THREADS = multiprocessing.cpu_count()
CPU_INFO = cpuinfo.get_cpu_info()
PLATFORM = (
    re.sub(
        r"\s+",
        "_",
        re.sub(r"[()]", "", CPU_INFO.get("brand_raw", "unknown-cpu")).strip(),
    )
    + "_"
    + CPU_INFO.get("arch", "unknown-arch")
)

TMP_DIR = tempfile.gettempdir()
TMP_DIR_ROCKSDB = os.path.join(TMP_DIR, "rocksdb")

RESULT_DIR = "results"
RESULT_DIR_ANN = os.path.join(RESULT_DIR, PLATFORM, "ann")
RESULT_DIR_ROCKSDB = os.path.join(RESULT_DIR, PLATFORM, "rocksdb")
RESULT_DIR_MICROBENCH = os.path.join(RESULT_DIR, PLATFORM, "microbench")

PLOT_DIR = "plots"
PLOT_DIR_ANN = os.path.join(PLOT_DIR, "ann")
PLOT_DIR_ROCKSDB = os.path.join(PLOT_DIR, "rocksdb")
PLOT_DIR_MONITORING = os.path.join(PLOT_DIR, "monitoring")

MONITOR_DIR = os.path.join(RESULT_DIR, PLATFORM, "monitor")
MONITOR_PCM = os.path.join(MONITOR_DIR, "pcm")
MONITOR_PCM_MEMORY = os.path.join(MONITOR_DIR, "pcm_memory")
MONITOR_MEM = os.path.join(MONITOR_DIR, "mem")


def sh(cmd, cwd=None):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, executable="/bin/bash")


def get_time():
    return datetime.datetime.now().isoformat()
