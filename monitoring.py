import datetime
import subprocess
import signal
import os
import ctypes.util
import shutil
from config import (
    MONITOR_DIR,
    MONITOR_MEM,
    MONITOR_PCM,
    MONITOR_PCM_MEMORY,
    MONITOR_PERF,
    sh,
)

INTERVAL = 1.0
PR_SET_PDEATHSIG = 1

# numa balancing task placement: move = task sent to its preferred node,
# swap = two tasks exchanged, stick = the balancer gave up on a move
NUMA_SCHED_EVENTS = ["sched_move_numa", "sched_swap_numa", "sched_stick_numa"]


def tmp_csv(path: str):
    return f"{path}_tmp.csv"


def label_csv(path: str, label: str):
    return f"{path}_{label}.csv"


def set_pdeathsig():
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
        raise OSError(ctypes.get_errno(), "SET_PDEATHSIG")


def safe_copy(src, dst):
    try:
        shutil.copy(src, dst)
        print(f"[OK] Copied {src} → {dst}")
    except FileNotFoundError:
        print(f"[WARN] File not found, skipping: {src}")
    except Exception as e:
        print(f"[ERROR] Failed to copy {src} → {dst}: {e}")


class Monitoring:
    def __init__(self, label: str, interval: float = INTERVAL):
        self.label = label
        # benches that need to see a transient raise it, the rest stay at 1s
        self.interval = interval
        self.pcm_proc = None
        self.pcm_memory_proc = None
        self.mem_proc = None
        self.perf_proc = None

    def start(self):
        os.makedirs(MONITOR_DIR, exist_ok=True)
        sh("modprobe msr")
        self.pcm_proc = self.start_pcm()
        self.pcm_memory_proc = self.start_pcm_memory()
        self.mem_proc = self.start_mem()
        self.perf_proc = self.start_perf()

    def stop(self):
        for proc in [
            self.pcm_proc,
            self.pcm_memory_proc,
            self.mem_proc,
            self.perf_proc,
        ]:
            if proc:
                proc.terminate()

    def start_pcm(self):
        return subprocess.Popen(
            [
                "pcm",
                str(self.interval),
                f"-csv={tmp_csv(MONITOR_PCM)}",
                "-nc",
            ],
            preexec_fn=set_pdeathsig,
        )

    def start_pcm_memory(self):
        return subprocess.Popen(
            [
                "pcm-memory",
                str(self.interval),
                f"-csv={tmp_csv(MONITOR_PCM_MEMORY)}",
            ],
            preexec_fn=set_pdeathsig,
        )

    def start_mem(self):
        return subprocess.Popen(
            [
                "uv",
                "run",
                "collect_mem.py",
                "-i",
                str(self.interval),
                "-csv",
                f"{tmp_csv(MONITOR_MEM)}",
            ],
            preexec_fn=set_pdeathsig,
        )

    def start_perf(self):
        """Count the numa balancing task placement tracepoints. They have no
        counter file, so unlike the vmstat ones they land in their own CSV,
        cut on the phase windows at plot time like the pcm ones."""
        if not shutil.which("perf"):
            print("[WARN] perf not found, skipping numa sched events")
            return None

        # perf stat writes to stderr, and timestamps each line relative to its
        # own start, so anchor them with a first line
        out = open(tmp_csv(MONITOR_PERF), "w")
        out.write(f"# start {datetime.datetime.now().isoformat()}\n")
        out.flush()

        cmd = [
            "perf", "stat", "-a", "-x,",
            "-e", ",".join(f"sched:{e}" for e in NUMA_SCHED_EVENTS),
            "-I", str(int(self.interval * 1000)),
        ]
        print(f"$ {' '.join(cmd)}")
        try:
            return subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=out,
                preexec_fn=set_pdeathsig,
            )
        except OSError as e:
            print(f"[WARN] could not start perf: {e}")
            return None

    def mv_output_files(self):
        safe_copy(tmp_csv(MONITOR_PERF), label_csv(MONITOR_PERF, self.label))
        safe_copy(tmp_csv(MONITOR_PCM), label_csv(MONITOR_PCM, self.label))
        safe_copy(
            tmp_csv(MONITOR_PCM_MEMORY),
            label_csv(MONITOR_PCM_MEMORY, self.label),
        )
        safe_copy(tmp_csv(MONITOR_MEM), label_csv(MONITOR_MEM, self.label))
