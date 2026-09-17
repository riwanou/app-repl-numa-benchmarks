"""Cross socket cache line sharing microbenchmark.

Threads only read, so any DRAM write a monitor sees is the coherence directory
being rewritten. Each thread gets a slice of the buffer to itself.

membind puts the buffer on one node, interleaved stripes it over both.

    local     all 16 readers on one node          (membind only)
    remote    all 16 readers on the other node    (membind only)
    disjoint  8 readers a node, different lines
    shared    8 readers a node, the same lines
    pingpong  the same lines, but the two nodes read them far apart in time

disjoint against shared is the pair that matters: same placement, same remote
fraction, only the sharing differs. Interleaved has no local or remote phase:
with the buffer striped, every reader is half local wherever it sits.

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

# must dwarf the LLC, and fit on one node
MB = 512            # small, so shared settles into S inside a phase
MB_PINGPONG = 8192  # big, so pingpong does not
SECS = 200
THREADS = 16  # readers, same in every phase, even
MEM_NODE = 0  # buffer and readers use these two nodes
FAR_NODE = 1
RUNS = 1

# what to run dirtest under
POLICIES = {
    "membind": ["numactl", f"--membind={MEM_NODE}"],
    "interleaved": ["numactl", f"--interleave={MEM_NODE},{FAR_NODE}"],
}

FIELDS = [
    "policy",
    "phase",
    "run_id",
    "threads",
    "mb",
    "secs",
    "read_gb_s",
    "cpus",
    "start_time",
    "end_time",
]


def build():
    """the Makefile holds the flags and skips the rebuild itself."""
    sh(f"make -C {DIRTEST_DIR}")


def node_cpus(node: int) -> list[int]:
    out = subprocess.run(
        ["numactl", "-H"], capture_output=True, text=True, check=True
    ).stdout
    for line in out.splitlines():
        if line.startswith(f"node {node} cpus:"):
            return [int(cpu) for cpu in line.split(":", 1)[1].split()]
    return []


def phases(policy: str) -> list[tuple[str, list[int], list[str]]]:
    """(name, cpus, the flags dirtest gets)."""
    near = node_cpus(MEM_NODE)
    far = node_cpus(FAR_NODE)
    if len(near) < THREADS or len(far) < THREADS:
        raise RuntimeError(
            f"need {THREADS} cpus on nodes {MEM_NODE} and {FAR_NODE},"
            f" found {len(near)} and {len(far)}"
        )

    half = THREADS // 2
    mixed = near[:half] + far[:half]

    grouped = [("disjoint", mixed, ["overlap=0"]),
               ("shared", mixed, ["overlap=100"]),
               ("pingpong", mixed, ["overlap=100", "pingpong"])]
    if policy == "interleaved":
        return grouped
    return [("local", near[:THREADS], []),
            ("remote", far[:THREADS], [])] + grouped


def run_phase(
    name: str, cpus: list[int], flags: list[str], run_id: int, policy: str
):
    mb = MB_PINGPONG if name == "pingpong" else MB
    cmd = [
        *POLICIES[policy],
        BIN,
        str(mb),
        str(SECS),
        ",".join(str(cpu) for cpu in cpus),
        *flags,
    ]

    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(proc.stdout, end="")

    # the window it measured, which excludes the first touch
    match = re.search(r"^RESULT (.*)$", proc.stdout, re.M)
    if not match:
        raise RuntimeError(f"{name}: no RESULT line in dirtest output")
    result = dict(kv.split("=", 1) for kv in match.group(1).split())

    return {
        "policy": policy,
        "phase": name,
        "run_id": run_id,
        "threads": len(cpus),
        "mb": mb,
        "secs": SECS,
        "read_gb_s": result["read_gb_s"],
        "cpus": " ".join(str(cpu) for cpu in cpus),
        "start_time": result["start"],
        "end_time": result["end"],
    }


def run_bench_sharing():
    build()
    os.makedirs(config.RESULT_DIR_SHARING, exist_ok=True)

    # autonuma migrates pages, and a migration is a write: it would fake the
    # effect under test
    sh("echo 0 > /proc/sys/kernel/numa_balancing")

    rows = []
    for run_id in range(1, RUNS + 1):
        for policy in POLICIES:
            for name, cpus, flags in phases(policy):
                rows.append(run_phase(name, cpus, flags, run_id, policy))
                # let the counters go idle between phases
                sh("sleep 5")

                with open(CSV_PATH, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)

    print(f"[OK] {len(rows)} phases -> {CSV_PATH}")
