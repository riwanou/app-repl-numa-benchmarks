import subprocess
import signal
import os
import ctypes.util
import shutil
from config import MONITOR_DIR, MONITOR_MEM, MONITOR_PCM, MONITOR_PCM_MEMORY, sh

INTERVAL = 1
PR_SET_PDEATHSIG = 1


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
    def __init__(self, label: str):
        self.label = label
        self.pcm_proc = None
        self.pcm_memory_proc = None
        self.mem_proc = None

    def start(self):
        os.makedirs(MONITOR_DIR, exist_ok=True)
        sh("modprobe msr")
        self.pcm_proc = self.start_pcm()
        self.pcm_memory_proc = self.start_pcm_memory()
        self.mem_proc = self.start_mem()

    def stop(self):
        for proc in [self.pcm_proc, self.pcm_memory_proc, self.mem_proc]:
            if proc:
                proc.send_signal(signal.SIGINT)

    def start_pcm(self):
        return subprocess.Popen(
            ["pcm", "1", f"-csv={tmp_csv(MONITOR_PCM)}", "-nc"],
            preexec_fn=set_pdeathsig,
        )

    def start_pcm_memory(self):
        return subprocess.Popen(
            ["pcm-memory", "1", f"-csv={tmp_csv(MONITOR_PCM_MEMORY)}"],
            preexec_fn=set_pdeathsig,
        )

    def start_mem(self):
        return subprocess.Popen(
            [
                "uv",
                "run",
                "collect_mem.py",
                "-i",
                str(INTERVAL),
                "-csv",
                f"{tmp_csv(MONITOR_MEM)}",
            ],
            preexec_fn=set_pdeathsig,
        )

    def mv_output_files(self):
        safe_copy(tmp_csv(MONITOR_PCM), label_csv(MONITOR_PCM, self.label))
        safe_copy(
            tmp_csv(MONITOR_PCM_MEMORY),
            label_csv(MONITOR_PCM_MEMORY, self.label),
        )
        safe_copy(tmp_csv(MONITOR_MEM), label_csv(MONITOR_MEM, self.label))
