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
    MONITOR_PCM_RAW,
    sh,
)

INTERVAL = 1.0
PR_SET_PDEATHSIG = 1

# Skylake / Cascade Lake CHA (Caching Home Agent) memory directory events. The
# CHA control register packs config = (umask << 8) | event | (1 << 22), the
# last being the enable bit, which is why UNC_CHA_CLOCKTICKS (event 0, umask 0)
# is documented as 0x400000.
#
# DIR_UPDATE is the write traffic itself, one 64 B line per update. DIR_LOOKUP
# splits the reads into those that had to snoop the other socket and those that
# did not, which is the latency half of the same story.
#
# Opt in per bench: pcm and pcm-raw both program uncore boxes, so this is only
# passed for the sharing bench rather than left on for every capture.
CHA_DIR_EVENTS = [
    ("UNC_CHA_DIR_UPDATE.HA", 0x400154),
    ("UNC_CHA_DIR_UPDATE.TOR", 0x400254),
    ("UNC_CHA_DIR_LOOKUP.SNP", 0x400153),
    ("UNC_CHA_DIR_LOOKUP.NO_SNP", 0x400253),
]


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
    def __init__(
        self,
        label: str,
        interval: float = INTERVAL,
        raw_events: list[tuple[str, int]] | None = None,
    ):
        self.label = label
        # pcm / pcm-memory take the delay as a float, so sub-second sampling
        # only costs an extra MSR read per interval. Benches that need to see
        # a transient (ann-pressure) raise it; the rest stay at 1 s.
        self.interval = interval
        # uncore events to capture with pcm-raw, e.g. CHA_DIR_EVENTS
        self.raw_events = raw_events
        self.pcm_proc = None
        self.pcm_memory_proc = None
        self.mem_proc = None
        self.pcm_raw_proc = None

    def start(self):
        os.makedirs(MONITOR_DIR, exist_ok=True)
        sh("modprobe msr")
        self.pcm_proc = self.start_pcm()
        self.pcm_memory_proc = self.start_pcm_memory()
        self.mem_proc = self.start_mem()
        if self.raw_events:
            self.pcm_raw_proc = self.start_pcm_raw()

    def stop(self):
        for proc in [
            self.pcm_proc,
            self.pcm_memory_proc,
            self.mem_proc,
            self.pcm_raw_proc,
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

    def start_pcm_raw(self):
        """pcm-raw with the uncore events of `raw_events`, if it is installed.

        A missing pcm-raw or a rejected event must not take the rest of the
        capture down with it, so this degrades to no file rather than raising.
        """
        if not shutil.which("pcm-raw"):
            print("[WARN] pcm-raw not found, skipping uncore event capture")
            return None

        cmd = ["pcm-raw", str(self.interval), f"-csv={tmp_csv(MONITOR_PCM_RAW)}"]
        for name, config in self.raw_events:
            cmd += ["-e", f"cha/config=0x{config:x},name={name}"]

        print(f"$ {' '.join(cmd)}")
        try:
            return subprocess.Popen(cmd, preexec_fn=set_pdeathsig)
        except OSError as e:
            print(f"[WARN] could not start pcm-raw: {e}")
            return None

    def mv_output_files(self):
        safe_copy(tmp_csv(MONITOR_PCM), label_csv(MONITOR_PCM, self.label))
        if self.raw_events:
            safe_copy(
                tmp_csv(MONITOR_PCM_RAW),
                label_csv(MONITOR_PCM_RAW, self.label),
            )
        safe_copy(
            tmp_csv(MONITOR_PCM_MEMORY),
            label_csv(MONITOR_PCM_MEMORY, self.label),
        )
        safe_copy(tmp_csv(MONITOR_MEM), label_csv(MONITOR_MEM, self.label))
