import multiprocessing
import os
import re
import cpuinfo
import subprocess
import tempfile

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

PLOT_DIR = "plots"
PLOT_DIR_ANN = os.path.join(PLOT_DIR, "ann")
PLOT_DIR_ROCKSDB = os.path.join(PLOT_DIR, "rocksdb")


def sh(cmd, cwd=None):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)
