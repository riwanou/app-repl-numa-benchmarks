import multiprocessing
import os
import re
import subprocess
import tempfile
import datetime

LINUX_COLOR = "Oranges"
CARREFOUR_COLOR = "Greens"
SPARE_COLOR = "Blues"

ARCH_SUBNAMES = {
    "IntelR_XeonR_Gold_6130_CPU_@_2.10GHz_X86_64": "gold",
    "IntelR_XeonR_Silver_4216_CPU_@_2.10GHz_X86_64": "silver",
    "INTELR_XEONR_PLATINUM_8568Y+_X86_64": "plat",
    "IntelR_XeonR_Gold_5320_CPU_@_2.20GHz_X86_64": "gold5320",
}


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

# Anchor every path to the repo, not to the current working directory. The
# benches are launched from just, from run.py, and from subprocesses that
# inherit whatever cwd their caller had, so a bare "results" resolves
# differently depending on who started it, and silently writes (or reads) the
# wrong tree. Deriving from __file__ makes the checkout relocatable: it works
# the same in ~/phd/numa/repl_benches and in /tmp/app-repl-numa-benchmarks.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

HYDRA_DIR = os.path.join(ROOT_DIR, "..", "linux-hydra-6.5")
HYDRA_NUMACTL = os.path.join(HYDRA_DIR, "hydra-numactl", "numactl")

MITOSIS_DIR = os.path.join(ROOT_DIR, "..", "linux-mitosis-4.17")
MITOSIS_NUMACTL = os.path.join(MITOSIS_DIR, "mitosis-numactl", "numactl")

TMP_DIR = tempfile.gettempdir()
TMP_DIR_ROCKSDB = os.path.join(TMP_DIR, "rocksdb")

# the ann bench's inputs: both are gitignored, so they are per machine
ANN_DATA_DIR = os.path.join(ROOT_DIR, "ann", "data")
ANN_INDEX_DIR = os.path.join(ROOT_DIR, "ann", "indices")

RESULT_DIR = os.path.join(ROOT_DIR, "results")
RESULT_DIR_ANN = os.path.join(RESULT_DIR, PLATFORM, "ann")
RESULT_DIR_ROCKSDB = os.path.join(RESULT_DIR, PLATFORM, "rocksdb")
RESULT_DIR_FIO = os.path.join(RESULT_DIR, PLATFORM, "fio")
RESULT_DIR_MICROBENCH = os.path.join(RESULT_DIR, PLATFORM, "microbench")
RESULT_DIR_LLAMA = os.path.join(RESULT_DIR, PLATFORM, "llama")
RESULT_DIR_PRESSURE = os.path.join(RESULT_DIR, PLATFORM, "pressure")
RESULT_DIR_SHARING = os.path.join(RESULT_DIR, PLATFORM, "sharing")

PLOT_DIR = os.path.join(ROOT_DIR, "plots")
PLOT_DIR_ANN = os.path.join(PLOT_DIR, "ann")
PLOT_DIR_ROCKSDB = os.path.join(PLOT_DIR, "rocksdb")
PLOT_DIR_FIO = os.path.join(PLOT_DIR, "fio")
PLOT_DIR_LLAMA = os.path.join(PLOT_DIR, "llama")
PLOT_DIR_MONITORING = os.path.join(PLOT_DIR, "monitoring")
PLOT_DIR_PRESSURE = os.path.join(PLOT_DIR, "pressure")
PLOT_DIR_MICROBENCH = os.path.join(PLOT_DIR, "microbench")
PLOT_DIR_SHARING = os.path.join(PLOT_DIR, "sharing")

MONITOR_DIR = os.path.join(RESULT_DIR, PLATFORM, "monitor")
MONITOR_PCM = os.path.join(MONITOR_DIR, "pcm")
MONITOR_PCM_MEMORY = os.path.join(MONITOR_DIR, "pcm_memory")
MONITOR_MEM = os.path.join(MONITOR_DIR, "mem")
MONITOR_PERF = os.path.join(MONITOR_DIR, "perf")
# the coherence directory counters, only the sharing bench turns them on
MONITOR_PERF_COHERENCE = os.path.join(MONITOR_DIR, "perf_coherence")


def sh(cmd, cwd=None):
    # default to the repo, not the caller's cwd: these commands run the
    # benches by relative script name ("uv run run_ann.py")
    print(f"$ {cmd}")
    subprocess.run(
        cmd,
        shell=True,
        check=True,
        cwd=cwd or ROOT_DIR,
        executable="/bin/bash",
    )


def get_time():
    return datetime.datetime.now().isoformat()
