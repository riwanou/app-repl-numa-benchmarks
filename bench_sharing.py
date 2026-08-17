"""Cross socket cache line sharing microbenchmark.

Threads stream reads and never write, so any DRAM write the monitors
record is coherence directory traffic. It decays over a run as the directory
settles.

    local     every reader on one node
    remote    every reader on the other node
    disjoint  half the readers on each node, on different lines
    shared    half the readers on each node, on the same lines

disjoint and shared are the pair that matters: same placement, same remote
fraction, same footprint, differing only in whether the two sockets read common
lines. Every phase runs under both policies.

    uv run run.py sharing
"""

import csv
import os
import re
import subprocess

import config
from config import sh

DIRTEST_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dirtest"
)
BIN = os.path.join(DIRTEST_DIR, "dirtest")
CSV_PATH = os.path.join(config.RESULT_DIR_SHARING, "results.csv")

GB = 8  # must dwarf the LLC, and fit on one node
SECS = 120
THREADS = 16  # total readers, kept equal across phases, even
MEM_NODE = 0  # the buffer goes on these two nodes, the readers sit on them
FAR_NODE = 1
RUNS = 1

# where the buffer goes, both in the same pass
POLICIES = {
    "membind": [f"--membind={MEM_NODE}"],
    "interleaved": [f"--interleave={MEM_NODE},{FAR_NODE}"],
}

FIELDS = [
    "policy",
    "phase",
    "overlap",
    "run_id",
    "threads",
    "gb",
    "secs",
    "read_gb_s",
    "cpus",
    "start_time",
    "end_time",
]


def build():
    """make keeps the compile flags in one place and skips the rebuild itself."""
    sh(f"make -C {DIRTEST_DIR}")


def node_cpus(node: int) -> list[int]:
    out = subprocess.run(
        ["numactl", "-H"], capture_output=True, text=True, check=True
    ).stdout
    for line in out.splitlines():
        if line.startswith(f"node {node} cpus:"):
            return [int(cpu) for cpu in line.split(":", 1)[1].split()]
    return []


def phases() -> list[tuple[str, list[int], int | None]]:
    """(name, cpu list, overlap percent or None for one undivided group)."""
    near = node_cpus(MEM_NODE)
    far = node_cpus(FAR_NODE)
    if len(near) < THREADS or len(far) < THREADS:
        raise RuntimeError(
            f"need {THREADS} cpus on nodes {MEM_NODE} and {FAR_NODE},"
            f" found {len(near)} and {len(far)}"
        )

    half = THREADS // 2
    mixed = near[:half] + far[:half]
    return [
        ("local", near[:THREADS], None),
        ("remote", far[:THREADS], None),
        ("disjoint", mixed, 0),
        ("shared", mixed, 100),
    ]


def run_phase(
    name: str, cpus: list[int], overlap: int | None, run_id: int, policy: str
):
    cmd = [
        "numactl",
        *POLICIES[policy],
        BIN,
        str(GB),
        str(SECS),
        ",".join(str(cpu) for cpu in cpus),
    ]
    if overlap is not None:
        cmd.append(f"overlap={overlap}")

    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(proc.stdout, end="")

    # the window the program actually measured, which excludes the first touch
    match = re.search(r"^RESULT (.*)$", proc.stdout, re.M)
    if not match:
        raise RuntimeError(f"{name}: no RESULT line in dirtest output")
    result = dict(kv.split("=", 1) for kv in match.group(1).split())

    return {
        "policy": policy,
        "phase": name,
        "overlap": "" if overlap is None else overlap,
        "run_id": run_id,
        "threads": len(cpus),
        "gb": GB,
        "secs": SECS,
        "read_gb_s": result["read_gb_s"],
        "cpus": " ".join(str(cpu) for cpu in cpus),
        "start_time": result["start"],
        "end_time": result["end"],
    }


def run_bench_sharing():
    build()
    os.makedirs(config.RESULT_DIR_SHARING, exist_ok=True)

    # autonuma would react to the remote phases by migrating pages, and page
    # migration is itself a write: that would produce the effect under test for
    # entirely the wrong reason
    sh("echo 0 > /proc/sys/kernel/numa_balancing")

    rows = []
    for run_id in range(1, RUNS + 1):
        for policy in POLICIES:
            for name, cpus, overlap in phases():
                rows.append(run_phase(name, cpus, overlap, run_id, policy))
                # let the counters fall back to idle so a phase never bleeds
                # into the next one's window
                sh("sleep 5")

                with open(CSV_PATH, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)

    print(f"[OK] {len(rows)} phases -> {CSV_PATH}")
