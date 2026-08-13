"""Run a bench inside a cgroup whose memory.high walks down a staircase, and
sample the replication counters, to see what replication does under pressure.

The bench itself and its variants live in bench_ann.

    uv run run.py pressure         # stock kernel variants
    uv run run.py pressure-repl    # patched kernel variants
"""

import csv
import ctypes
import ctypes.util
import datetime
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass

import bench_ann
import config
from config import sh


@dataclass
class Phase:
    label: str
    limit: str  # memory.high value
    seconds: int


# A staircase: each step inherits the previous one's state, so this reads as a
# dose response.
#
# One copy of the index is 3.84G, so cN sits just under N copies and the Nth
# has to go. c1.5 is where repl-bound's QPS actually collapses, halfway down
# the last copy. normal is short (each variant converges during its settle),
# release long (re-replicating is the slow direction).
PLAN = [
    Phase("normal", "max", 30),
    Phase("c3", "11G", 40),
    Phase("c2", "7G", 40),
    Phase("c1.5", "5G", 40),
    Phase("c1", "3G", 40),
    Phase("release", "max", 60),
]

# the bench outlives the plan: settle before (per variant, see PressureVariant)
# and drain after
TAIL = 10

# our sampling rate, and the monitoring one run.py passes to Monitoring
SAMPLE_INTERVAL = 0.5

CGROUP_ROOT = "/sys/fs/cgroup"
CGROUP = os.path.join(CGROUP_ROOT, "bench")
REPL_STATS = "/sys/kernel/debug/repl_pt/stats"
REPL_PG_STATS = "/sys/kernel/debug/repl_pt/pg_stats"

CGROUP_STAT_KEYS = ["anon", "file", "pgscan", "pgsteal", "pgmajfault"]
PR_SET_PDEATHSIG = 1

# Explanatory only, like the repl_ ones: QPS and bandwidth are what we report,
# these say *why* a curve moved. numa balancing migrates pages (vmstat) and
# threads (their node), and neither is visible in the other's counters.
NODE_DIR = "/sys/devices/system/node"
INDEX_EXTS = (".usearch", ".ivf", ".ann")  # same set repl_pt registers
VMSTAT_KEYS = [
    "numa_pte_updates",  # scanner arming hinting faults, i.e. the tax
    "numa_huge_pte_updates",
    "numa_hint_faults",
    "numa_hint_faults_local",  # vs hint_faults: was the page already home
    "numa_pages_migrated",
    "pgmigrate_success",
    "pgmigrate_fail",
]


def read_text(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def read_int(path: str):
    try:
        return int(read_text(path))
    except ValueError:
        return None


def read_kv(path: str) -> dict[str, str]:
    return dict(
        line.split(maxsplit=1)
        for line in read_text(path).splitlines()
        if " " in line
    )


def cgroup_sample() -> dict:
    stat = read_kv(os.path.join(CGROUP, "memory.stat"))
    psi = read_kv(os.path.join(CGROUP, "memory.pressure")).get("some", "")
    return {
        "current_mb": (read_int(os.path.join(CGROUP, "memory.current")) or 0)
        // (1024 * 1024),
        "high_events": read_kv(os.path.join(CGROUP, "memory.events")).get(
            "high", ""
        ),
        "psi_some_avg10": psi.split()[0].removeprefix("avg10=") if psi else "",
        **{key: stat.get(key, "") for key in CGROUP_STAT_KEYS},
    }


def repl_sample() -> dict:
    """Every scalar counter the module exposes, so a new one needs no edit."""
    if not os.path.isdir(REPL_STATS):
        return {}
    values = {
        name: read_int(os.path.join(REPL_STATS, name))
        for name in sorted(os.listdir(REPL_STATS))
        if name != "clear"
    }
    return {f"repl_{k}": v for k, v in values.items() if v is not None}


def cpu_node() -> dict[int, int]:
    """cpu -> node, from nodeN/cpulist ("0-15,32-47"). Read once at import:
    the topology does not move under us."""
    mapping = {}
    nodes = sorted(os.listdir(NODE_DIR)) if os.path.isdir(NODE_DIR) else []
    for node in nodes:
        if not node.startswith("node") or not node[4:].isdigit():
            continue
        for span in read_text(f"{NODE_DIR}/{node}/cpulist").split(","):
            if not span:
                continue
            lo, _, hi = span.partition("-")
            for cpu in range(int(lo), int(hi or lo) + 1):
                mapping[cpu] = int(node[4:])
    return mapping


CPU_NODE = cpu_node()
NODES = sorted(set(CPU_NODE.values()))


def bench_pids() -> list[str]:
    if not os.path.isdir("/proc"):
        return []
    return [
        pid
        for pid in os.listdir("/proc")
        if pid.isdigit() and "python" in read_text(f"/proc/{pid}/comm")
    ]


def vmstat_sample() -> dict:
    stat = read_kv("/proc/vmstat")
    return {f"vm_{key}": stat.get(key, "") for key in VMSTAT_KEYS}


def threads_sample() -> dict:
    """How many of the bench's threads are running on each node. Thread
    migration is the half of numa balancing that vmstat does not count, and it
    shows up here as the histogram sloshing between nodes."""
    counts = dict.fromkeys(NODES, 0)
    for pid in bench_pids():
        tasks = f"/proc/{pid}/task"
        try:
            tids = os.listdir(tasks)
        except OSError:
            continue
        for tid in tids:
            # field 39 of /proc/<tid>/stat is the cpu it last ran on; comm
            # comes second and can hold spaces and parens, so cut it off first
            stat = read_text(f"{tasks}/{tid}/stat").rpartition(") ")[2]
            fields = stat.split()
            if len(fields) > 36:
                node = CPU_NODE.get(int(fields[36]))
                if node is not None:
                    counts[node] += 1
    return {f"threads_node{node}": n for node, n in counts.items()}


def sample(variant, phase: Phase, elapsed: float) -> dict:
    return {
        "time": datetime.datetime.now().isoformat(timespec="milliseconds"),
        "variant": variant.tag,
        "elapsed": round(elapsed, 3),
        "phase": phase.label,
        "limit": phase.limit,
        **cgroup_sample(),
        **repl_sample(),
        **vmstat_sample(),
        **threads_sample(),
    }


def pg_stats() -> str:
    """Per process replication table. pg_stats/<pid> is generated on open, so
    listing the directory yields nothing: walk the python processes instead."""
    if not os.path.isdir(REPL_PG_STATS) or not os.path.isdir("/proc"):
        return ""
    return "\n".join(
        f"-- pid {pid}\n{out}"
        for pid in sorted(os.listdir("/proc"), key=lambda p: p.zfill(9))
        if pid.isdigit() and "python" in read_text(f"/proc/{pid}/comm")
        if (out := read_text(os.path.join(REPL_PG_STATS, pid)))
        and not out.startswith("replication not enabled")
    )


def numa_maps() -> str:
    """Where the index pages actually sit, per node, for the stock variants:
    the only way to see numa balancing move the mapping. Kept to phase ends,
    it walks the page table. See run_phase for why repl variants skip it."""
    return "\n".join(
        f"-- pid {pid}\n" + "\n".join(lines)
        for pid in sorted(bench_pids(), key=lambda p: p.zfill(9))
        if (
            lines := [
                line
                for line in read_text(f"/proc/{pid}/numa_maps").splitlines()
                # "<addr> default file=<path> mapped=N N0=n N1=n ..."
                if any(
                    tok.startswith("file=") and tok.endswith(INDEX_EXTS)
                    for tok in line.split()
                )
            ]
        )
    )


def start_bench(variant, running_time: int) -> subprocess.Popen:
    cmd = bench_ann.run_bench_pressure(variant, running_time)

    def preexec():
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
            raise OSError(ctypes.get_errno(), "SET_PDEATHSIG")
        with open(os.path.join(CGROUP, "cgroup.procs"), "w") as f:
            f.write(str(os.getpid()))

    print(f"$ {cmd}")
    return subprocess.Popen(
        cmd, shell=True, executable="/bin/bash", preexec_fn=preexec
    )


def save_results(base: str, variant, since, windows):
    """Copy this variant's ann rows next to the samples, tagged with the phase
    they ran in. Phase windows go in their own file, so the monitoring CSVs in
    monitor/ can be cut on the same boundaries at plot time."""
    try:
        import pandas as pd
    except ImportError as e:
        print(f"[WARN] {e}: no ann results copied into {base}-ann.csv")
        return

    pd.DataFrame(
        windows, columns=["phase", "limit", "start_time", "end_time"]
    ).to_csv(f"{base}-phases.csv", index=False)

    dataset = os.path.splitext(bench_ann.PRESSURE_DATASET)[0]
    details = os.path.join(config.RESULT_DIR_ANN, f"{dataset}-details.csv")
    try:
        runs = pd.read_csv(details)
        for col in ("start_time", "end_time"):
            runs[col] = pd.to_datetime(
                runs[col], format="mixed", errors="coerce"
            )
        # the details file is appended to, keep only this run of this variant
        runs = runs[
            (runs.tag == f"pressure-{variant.tag}")
            & (runs.start_time >= pd.Timestamp(since))
        ].sort_values("start_time")
        phase = pd.Series(pd.NA, index=runs.index, dtype=object)
        limit = pd.Series(pd.NA, index=runs.index, dtype=object)
        for label, value, start, end in windows:
            inside = (runs.start_time >= start) & (runs.start_time < end)
            phase[inside], limit[inside] = label, value
        runs.assign(phase=phase, limit=limit).to_csv(
            f"{base}-ann.csv", index=False
        )
        print(f"[OK] {base}-ann.csv ({len(runs)} runs)")
    except Exception as e:
        print(f"[WARN] no ann results copied: {e}")


def run_phase(phase, variant, bench, writer, log, start) -> bool:
    print(f"=== {phase.label}: memory.high={phase.limit} ({phase.seconds}s)")

    # The write blocks until the kernel has reclaimed the cgroup back under
    # the limit, up to 8 s, which is exactly the transient we are here to
    # measure. Off-thread so the loop below samples all the way through it.
    setter = threading.Thread(
        target=sh, args=(f"echo {phase.limit} > {CGROUP}/memory.high",)
    )
    setter.start()

    # fixed grid so we do not drift away from the pcm one
    tick = time.monotonic()
    deadline = tick + phase.seconds
    while tick < deadline:
        if bench.poll() is not None:
            print("[WARN] bench exited early, stopping the plan")
            return False
        writer.writerow(sample(variant, phase, time.monotonic() - start))
        tick += SAMPLE_INTERVAL
        time.sleep(max(0.0, tick - time.monotonic()))

    setter.join()
    log(f"== pg_stats @ {phase.label} end", pg_stats())
    # numa_maps walks the main page table only, so under the repl policy it
    # reports the main copy and is blind to the replicas: pg_stats is what
    # sees those. Keep it for the stock variants, where it is the only
    # placement signal we have, and skip the walk (seconds, on this mapping)
    # everywhere else.
    if variant.main_placement is None:
        log(f"== numa_maps @ {phase.label} end", numa_maps())
    return True


def run_variant(variant):
    running_time = variant.settle + sum(p.seconds for p in PLAN) + TAIL
    print(f"=== {variant.tag}: {len(PLAN)} phases, {running_time}s")

    os.makedirs(config.RESULT_DIR_PRESSURE, exist_ok=True)
    base = os.path.join(
        config.RESULT_DIR_PRESSURE, f"ann-pressure-{variant.tag}"
    )

    sh(f"echo +memory > {CGROUP_ROOT}/cgroup.subtree_control")
    # recreate rather than reuse: memory.events cannot be reset, and a stale
    # cgroup carries the previous variant's (or run's) counters in
    sh(f"rmdir {CGROUP} 2>/dev/null || true")
    sh(f"mkdir -p {CGROUP}")
    sh(f"echo {PLAN[0].limit} > {CGROUP}/memory.high")

    windows = []
    with (
        # the per sample counters are a debugging aid, the files save_results
        # writes are what the plots are built from
        open(f"{base}-cgroup.csv", "w", newline="") as csv_file,
        open(f"{base}.log", "w") as log_file,
    ):
        writer = csv.DictWriter(
            csv_file, fieldnames=list(sample(variant, PLAN[0], 0))
        )
        writer.writeheader()

        def log(*blocks):
            for block in filter(None, blocks):
                print(block)
                log_file.write(block + "\n")
            log_file.flush()

        since = datetime.datetime.now().isoformat()
        bench = start_bench(variant, running_time)
        sh(f"echo 1 > {REPL_STATS}/clear || true")
        time.sleep(variant.settle)

        start = time.monotonic()
        try:
            for phase in PLAN:
                began = datetime.datetime.now()
                ok = run_phase(phase, variant, bench, writer, log, start)
                windows.append(
                    (phase.label, phase.limit, began, datetime.datetime.now())
                )
                if not ok:
                    break
                csv_file.flush()
        finally:
            log(
                "== cgroup summary",
                read_text(os.path.join(CGROUP, "memory.events")),
                f"peak {read_text(os.path.join(CGROUP, 'memory.peak'))}",
                "== repl stats",
                "\n".join(f"{k} {v}" for k, v in repl_sample().items()),
            )
            bench.wait()

    # only once the bench has exited: it writes its details CSV at the end
    if windows:
        save_results(base, variant, since, windows)
    print(f"[OK] {base}-cgroup.csv")


def run_bench_pressure():
    for variant in bench_ann.PRESSURE_VARIANTS:
        run_variant(variant)


def run_bench_pressure_repl():
    for variant in bench_ann.PRESSURE_VARIANTS_REPL:
        run_variant(variant)
