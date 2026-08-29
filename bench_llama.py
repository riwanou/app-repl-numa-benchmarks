"""llama.cpp token throughput under each memory policy.

llama-bench runs its default tests (pp512 and tg128) REPS times per arm, via
-r, and dumps its own -o json report per arm. The csv the plot reads is the
mean over the warm repetitions only: the first WARMUP repetitions are
consistently slower (page cache, first-touch settling), so they are dropped.
"""

import json
import os
import shutil
import config
from config import sh

RESULT_DIR = config.RESULT_DIR_LLAMA

MODEL = "./llama.cpp/Llama-3.1-Tulu-3-8B-Q8_0.gguf"
BENCH = "./llama.cpp/build/bin/llama-bench"

REPS = 10
WARMUP = 0

# tag, numactl, llama --numa flag, numa balancing
ARMS = [
    ("baseline", "", "", False),
    ("baseline-warmup", "", "--numa-warmup", False),
    ("baseline-balancing", "", "", True),
    ("baseline-balancing-warmup", "", "--numa-warmup", True),
    ("distribute", "", "--numa distribute", False),
    ("distribute-warmup", "", "--numa distribute --numa-warmup", False),
    (
        "interleaved-distribute",
        "numactl --interleave=all",
        "--numa distribute",
        False,
    ),
    (
        "interleaved-distribute-warmup",
        "numactl --interleave=all",
        "--numa distribute --numa-warmup",
        False,
    ),
]
REPL_ARMS = [
    ("repl-distribute", "", "--numa distribute", False),
    ("repl-distribute-warmup", "", "--numa distribute --numa-warmup", False),
]

# one csv per kernel variant, holding the mean of each of its arms
CSVS = [("llama", ARMS), ("llama-repl", REPL_ARMS)]

REPL = "/sys/kernel/debug/repl_pt"


def _bench_cmd(numactl, numa_flag, out):
    return (
        f"{numactl} {BENCH} -m {MODEL} -t $(nproc --all) -r {REPS}"
        f" {numa_flag} -o json > {out}"
    )


def _drop_caches():
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")


def _repl_setup():
    sh("echo 0 > /proc/sys/kernel/numa_balancing")
    sh(f"echo 0 > {REPL}/main_placement")
    sh(f"echo 1 > {REPL}/clear_registered")
    sh(f"echo Llama-3.1-Tulu-3-8B-Q8_0.gguf > {REPL}/registered")


def _run_arm(tag, numactl, numa_flag, repl_enabled):
    """One llama-bench invocation, its -o json report kept as is."""
    json_path = os.path.join(RESULT_DIR, f"{tag}.json")
    out = os.path.join(config.TMP_DIR, "llama-bench.json")

    _drop_caches()

    cmd = _bench_cmd(numactl, numa_flag, out)
    if repl_enabled:
        # policy is per-pid, so the run must be a child of this shell
        cmd = f"""(
          echo 1 > {REPL}/policy &&
          {cmd};
          echo 0 > {REPL}/policy
        )"""
    sh(cmd)

    shutil.copy(out, json_path)


def _warm_rows(tag):
    """The warm repetitions of an arm, empty when it never ran here."""
    json_path = os.path.join(RESULT_DIR, f"{tag}.json")
    if not os.path.exists(json_path):
        return []

    with open(json_path) as f:
        results = json.load(f)

    rows = []
    for r in results:
        name = f"pp{r['n_prompt']}" if r["n_gen"] == 0 else f"tg{r['n_gen']}"
        for run, (ts, ns) in enumerate(zip(r["samples_ts"], r["samples_ns"])):
            if run < WARMUP:
                continue
            rows.append(
                {
                    "test": name,
                    "run": run,
                    "ts": ts,
                    "ns": ns,
                    "n_prompt": r["n_prompt"],
                    "n_gen": r["n_gen"],
                    "test_time": r["test_time"],
                }
            )
    return rows


def write_csv(name, arms):
    """Mean over the warm repetitions, one row per (arm, test)."""
    csv_path = os.path.join(RESULT_DIR, f"{name}.csv")
    with open(csv_path, "w") as f:
        # test_time and avg_ns are what stats_monitoring slices the run window on
        f.write(
            "tag,test,n_prompt,n_gen,runs,test_time,avg_ns,avg_ts,stddev_ts\n"
        )
        for tag, *_ in arms:
            tests = {}
            for r in _warm_rows(tag):
                tests.setdefault(r["test"], []).append(r)
            for test, rs in tests.items():
                ts = [r["ts"] for r in rs]
                mean = sum(ts) / len(ts)
                var = sum((t - mean) ** 2 for t in ts) / max(1, len(ts) - 1)
                avg_ns = sum(r["ns"] for r in rs) / len(rs)
                f.write(
                    f"{tag},{test},{rs[0]['n_prompt']},{rs[0]['n_gen']},"
                    f"{len(ts)},{rs[0]['test_time']},{avg_ns:.0f},"
                    f"{mean:.6f},{var**0.5:.6f}\n"
                )
    print(f"wrote {csv_path}")


def run_bench_llama():
    os.makedirs(RESULT_DIR, exist_ok=True)

    for tag, numactl, numa_flag, balancing in ARMS:
        sh(f"echo {int(balancing)} > /proc/sys/kernel/numa_balancing")
        _run_arm(tag, numactl, numa_flag, repl_enabled=False)
        write_csv("llama", ARMS)


def run_bench_llama_repl():
    os.makedirs(RESULT_DIR, exist_ok=True)
    _repl_setup()

    for tag, numactl, numa_flag, _ in REPL_ARMS:
        _run_arm(tag, numactl, numa_flag, repl_enabled=True)
        write_csv("llama-repl", REPL_ARMS)
        sh(f"cat {REPL}/registered")


def write_all_csv():
    """Redo both csvs from the jsonl files, without re-running the bench."""
    for name, arms in CSVS:
        write_csv(name, arms)
