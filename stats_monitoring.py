"""Join benchmark results with monitoring stats.

Each benchmark result CSV has a `start_time` / `end_time` column per run.
For a given monitoring label (the one passed to `monitoring.Monitoring`, e.g.
"ann-repl"), this slices the pcm / pcm_memory / mem CSVs on each run window and
computes one row of stats per run.

Output is the original result CSV plus one column per stat, written to
`results/<arch>/stats/<label>/<result file name>`.

    uv run run.py stats-monitoring    # every (arch, bench, label) of JOBS
    uv run stats_monitoring.py results/<arch>/ann/glove-100-angular-details.csv -l ann-repl

Adding a stat is a single function, see the STATS section below.
"""

import argparse
import functools
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from dateutil import tz

import config

# ---------------------------------------------------------------- monitoring

# every monitor writes the naive local time of the machine it ran on
MONITOR_TZ = os.environ.get("MONITOR_TZ") or tz.tzlocal()

PCM_DATE = ("System", "Date")
PCM_TIME = ("System", "Time")
PCM_MEM_DATE = ("Unnamed: 0_level_0", "Date")
PCM_MEM_TIME = ("Unnamed: 1_level_0", "Time")

# perf timestamps relative to its own start, so monitoring.py writes an anchor
COHERENCE_ANCHOR = "# start "


@dataclass
class Window:
    """Monitoring samples covering a single benchmark run."""

    pcm: pd.DataFrame
    pcm_memory: pd.DataFrame
    mem: pd.DataFrame
    # only the benches that ask for it, empty elsewhere and its stats read NaN
    perf_coherence: pd.DataFrame


def monitor_dir(arch: str) -> str:
    return os.path.join(config.RESULT_DIR, arch, "monitor")


# no Xeon socket reads near this, but pcm emits ~1 TB/s counter artifacts
MAX_BW_MBS = 500 * 1024


def drop_bogus_bandwidth(
    df: pd.DataFrame, names: tuple[str, ...], label: str
) -> pd.DataFrame:
    """NaN out impossible bandwidth samples so the stats skip them."""
    if df.empty:
        return df

    columns = [col for col in df.columns if col[1] in names]
    if not columns:
        return df

    values = df[columns].apply(pd.to_numeric, errors="coerce")
    bogus = values > MAX_BW_MBS
    count = int(bogus.to_numpy().sum())
    if count:
        rows = int(bogus.any(axis=1).sum())
        print(
            f"[WARN] {label}: dropped {count} impossible bandwidth samples"
            f" over {rows}/{len(df)} rows"
        )
        df = df.copy()
        df[columns] = values.where(~bogus)

    return df


def read_perf_coherence(path: str) -> pd.DataFrame:
    """A perf stat -x, --per-socket capture as one column per (socket, event),
    in events per second, in the tuple columns the pcm frames use."""
    with open(path) as file:
        lines = file.read().splitlines()

    anchor = None
    rows = []
    for line in lines:
        if line.startswith(COHERENCE_ANCHOR):
            anchor = pd.to_datetime(line[len(COHERENCE_ANCHOR) :].strip())
            continue
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split(",")
        # a monitor killed mid write leaves the last line truncated
        if len(fields) < 6:
            continue
        try:
            offset = float(fields[0])
        except ValueError:
            continue
        rows.append(
            (
                offset,
                fields[1].strip(),
                fields[5].strip(),
                # <not counted> reads NaN, which is not a zero rate
                pd.to_numeric(fields[3].strip(), errors="coerce"),
            )
        )

    if anchor is None or not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["offset", "socket", "event", "value"])
    df = df.sort_values("offset")
    gap = df.groupby(["socket", "event"])["offset"].diff()
    # per interval counts, and the first sample's gap is its own offset
    df["rate"] = df["value"] / gap.fillna(df["offset"])

    wide = df.pivot_table(
        index="offset", columns=["socket", "event"], values="rate"
    )
    # the machine total, so a stat reads the same whatever the socket count
    for event in df["event"].unique():
        columns = [col for col in wide.columns if col[1] == event]
        wide[("System", event)] = wide[columns].sum(axis=1, min_count=1)

    times = anchor + pd.to_timedelta(wide.index.to_series(), unit="s")
    return wide.reset_index(drop=True).assign(time_dt=times.to_numpy())


@functools.lru_cache(maxsize=None)
def load_monitoring(arch: str, label: str) -> Window:
    """Load the whole monitoring run for `label`, with a `time_dt` column."""
    directory = monitor_dir(arch)

    missing = []

    def read(name: str, multi_header: bool) -> pd.DataFrame:
        path = os.path.join(directory, f"{name}_{label}.csv")
        if not os.path.exists(path):
            missing.append(path)
            return pd.DataFrame()
        try:
            return pd.read_csv(path, header=[0, 1] if multi_header else 0)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"[WARN] unreadable monitoring file {path}: {e}")
            return pd.DataFrame()

    pcm = read("pcm", True)
    pcm_memory = read("pcm_memory", True)
    mem = read("mem", False)

    # all three missing is a label never run here, a partial capture is not
    if missing and len(missing) < 3:
        for path in missing:
            print(f"[WARN] missing monitoring file: {path}")

    pcm = drop_bogus_bandwidth(pcm, ("LMB", "RMB", "READ", "WRITE"), label)
    pcm_memory = drop_bogus_bandwidth(
        pcm_memory, ("Read", "Write", "Memory"), label
    )

    def timed(df: pd.DataFrame, times: pd.Series | None) -> pd.DataFrame:
        """Add `time_dt`, dropping the rows a killed monitor left truncated."""
        if df.empty or times is None:
            return df
        df = df.assign(time_dt=pd.to_datetime(times, errors="coerce"))
        return pd.DataFrame(df[df["time_dt"].notna()])

    pcm = timed(
        pcm,
        pcm[PCM_DATE].astype(str) + " " + pcm[PCM_TIME].astype(str)
        if not pcm.empty
        else None,
    )
    pcm_memory = timed(
        pcm_memory,
        pcm_memory[PCM_MEM_DATE].astype(str)
        + " "
        + pcm_memory[PCM_MEM_TIME].astype(str)
        if not pcm_memory.empty
        else None,
    )
    mem = timed(mem, mem["time"] if not mem.empty else None)

    # opt in per bench, so a label without one is normal, not a partial capture
    coherence_path = os.path.join(directory, f"perf_coherence_{label}.csv")
    coherence = pd.DataFrame()
    if os.path.exists(coherence_path):
        try:
            coherence = read_perf_coherence(coherence_path)
        except (OSError, ValueError) as e:
            print(f"[WARN] unreadable monitoring file {coherence_path}: {e}")

    return Window(pcm, pcm_memory, mem, coherence)


def slice_window(full: Window, start, end) -> Window:
    def cut(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        mask = (df["time_dt"] >= start) & (df["time_dt"] <= end)
        return pd.DataFrame(df[mask])

    return Window(
        cut(full.pcm),
        cut(full.pcm_memory),
        cut(full.mem),
        cut(full.perf_coherence),
    )


# --------------------------------------------------------------- stat helpers

STATS: dict[str, callable] = {}


def stat(name: str):
    """Register a stat: a function of a Window returning a scalar."""

    def decorator(fn):
        STATS[name] = fn
        return fn

    return decorator


def mean(df: pd.DataFrame, col, scale: float = 1.0) -> float:
    """Mean of a column, NaN if the column or the window is missing."""
    if df.empty or col not in df.columns:
        return float("nan")
    return pd.to_numeric(df[col], errors="coerce").mean() * scale


def mean_pct(df: pd.DataFrame, col) -> float:
    """Mean of a "42.0%"-style column."""
    if df.empty or col not in df.columns:
        return float("nan")
    values = df[col].astype(str).str.rstrip("%")
    return pd.to_numeric(values, errors="coerce").mean()


MB_TO_GB = 1 / 1024
KB_TO_GB = 1 / (1024 * 1024)
KB_TO_MB = 1 / 1024

# the gold 6130 has 4 sockets, the others 2. Columns a machine lacks are
# empty and get dropped when writing.
MAX_SOCKETS = 4


def per_socket(name: str, column: str, scale: float = 1.0):
    """Register `name`_skt0 .. _skt<MAX_SOCKETS>, one column each, so the
    file reads local_pct_skt0, local_pct_skt1, ... then the next metric."""
    for i in range(MAX_SOCKETS):
        stat(f"{name}_skt{i}")(
            lambda w, s=f"Socket {i}", c=column, k=scale: mean(w.pcm, (s, c), k)
        )


def upi_pct_cols(df: pd.DataFrame, kind: str) -> list:
    """Every UPI link column of the machine, however many sockets it has."""
    if df.empty:
        return []
    return [
        col
        for col in df.columns
        if isinstance(col, tuple)
        and col[0].endswith(f"{kind} (percent)")
        and str(col[1]).startswith("UPI")
    ]


# ---------------------------------------------------------------------- STATS
# Add a stat here and it shows up as a column, nothing else to touch.


@stat("samples")
def _(w: Window) -> float:
    return len(w.pcm)


# locality


@stat("local_pct")
def _(w: Window) -> float:
    """Share of memory traffic served by the accessing socket's own node."""
    return mean(w.pcm, ("System", "LOCAL"))


# memory side: of what this socket's controller served, the share from its own
# cores. Under --membind=0 node 0 serves both sockets and reads 50%.
per_socket("local_pct", "LOCAL")
per_socket("lmb_gb", "LMB", MB_TO_GB)
per_socket("rmb_gb", "RMB", MB_TO_GB)


# cpu


@stat("ipc")
def _(w: Window) -> float:
    return mean(w.pcm, ("System", "IPC"))


@stat("inst")
def _(w: Window) -> float:
    """Work retired in the window: a sample diluted by idle time reports
    fewer instructions, which makes every ratio of that row suspect."""
    return mean(w.pcm, ("System", "INST"))


per_socket("inst", "INST")


@stat("afreq")
def _(w: Window) -> float:
    """Active core frequency over nominal: above 1 is turbo. An arm that keeps
    fewer cores busy boosts higher, which flatters its runtime for reasons
    that have nothing to do with placement."""
    return mean(w.pcm, ("System", "AFREQ"))


@stat("freq")
def _(w: Window) -> float:
    """Same, counting halted cycles: the gap to afreq is idle time."""
    return mean(w.pcm, ("System", "FREQ"))


@stat("cpu_balance")
def _(w: Window) -> float:
    """1.0 is every socket retiring the same, 0 is everything consolidated on
    one of them. A policy that migrates the threads onto one node shows up
    here, and it inflates local_pct as a side effect."""
    per_socket = [
        mean(w.pcm, (f"Socket {i}", "INST")) for i in range(MAX_SOCKETS)
    ]
    present = [value for value in per_socket if not pd.isna(value)]
    if not present or not max(present):
        return float("nan")
    return min(present) / max(present)


# caches


@stat("llc_rd_miss_lat_ns")
def _(w: Window) -> float:
    return mean(w.pcm, ("System", "LLCRDMISSLAT (ns)"))


@stat("l3hit")
def _(w: Window) -> float:
    """Hit ratio, only comparable between rows with a similar l2miss."""
    return mean(w.pcm, ("System", "L3HIT"))


@stat("l2miss")
def _(w: Window) -> float:
    """Requests that reached L3: the denominator behind l3hit."""
    return mean(w.pcm, ("System", "L2MISS"))


@stat("l3miss")
def _(w: Window) -> float:
    return mean(w.pcm, ("System", "L3MISS"))


@stat("l3mpi")
def _(w: Window) -> float:
    """Misses per instruction, the one normalized by work."""
    return mean(w.pcm, ("System", "L3MPI"))


# interconnect


@stat("upi_in_gb")
def _(w: Window) -> float:
    return mean(w.pcm, ("System", "TotalUPIin"), MB_TO_GB)


@stat("upi_out_gb")
def _(w: Window) -> float:
    return mean(w.pcm, ("System", "TotalUPIout"), MB_TO_GB)


@stat("upi_out_pct")
def _(w: Window) -> float:
    """Utilisation averaged over every link of the machine, 4 on a 2 socket
    box, 12 on the 4 socket gold."""
    cols = upi_pct_cols(w.pcm, "trafficOut")
    return pd.Series([mean_pct(w.pcm, col) for col in cols]).mean()


@stat("upi_in_pct")
def _(w: Window) -> float:
    cols = upi_pct_cols(w.pcm, "dataIn")
    return pd.Series([mean_pct(w.pcm, col) for col in cols]).mean()


# bandwidth


@stat("mem_bw_gb")
def _(w: Window) -> float:
    return mean(w.pcm_memory, ("System", "Memory"), MB_TO_GB)


@stat("mem_read_gb")
def _(w: Window) -> float:
    return mean(w.pcm_memory, ("System", "Read"), MB_TO_GB)


@stat("mem_write_gb")
def _(w: Window) -> float:
    return mean(w.pcm_memory, ("System", "Write"), MB_TO_GB)


for _i in range(MAX_SOCKETS):
    stat(f"mem_read_gb_skt{_i}")(
        lambda w, s=f"SKT{_i}": mean(
            w.pcm_memory, (s, "Mem Read (MB/s)"), MB_TO_GB
        )
    )

for _i in range(MAX_SOCKETS):
    stat(f"mem_write_gb_skt{_i}")(
        lambda w, s=f"SKT{_i}": mean(
            w.pcm_memory, (s, "Mem Write (MB/s)"), MB_TO_GB
        )
    )


# coherence directory
# what is left to explain the writes, since dirtest never writes its buffer
DIR_UPDATE = "UNC_M2M_DIRECTORY_UPDATE.ANY"
DIR_SNP = "UNC_CHA_DIR_LOOKUP.SNP"
DIR_NO_SNP = "UNC_CHA_DIR_LOOKUP.NO_SNP"

PER_M = 1e-6  # events per second -> millions per second


def coherence(
    w: Window, event: str, socket: str = "System", scale=1.0
) -> float:
    """Rate of one event, NaN when the capture does not have it."""
    return mean(w.perf_coherence, (socket, event), scale)


def per_socket_coherence(name: str, event: str, scale=1.0):
    """`name`_skt0 and _skt1: the bench uses two nodes."""
    for i in range(2):
        stat(f"{name}_skt{i}")(
            lambda w, s=f"S{i}", e=event, k=scale: coherence(w, e, s, k)
        )


@stat("dir_update_m_s")
def _(w: Window) -> float:
    """Directory state transitions: a flip is a write back to DRAM."""
    return coherence(w, DIR_UPDATE, scale=PER_M)


per_socket_coherence("dir_update_m_s", DIR_UPDATE, PER_M)


@stat("dir_lookup_snp_m_s")
def _(w: Window) -> float:
    """Lookups that had to snoop, at the cache agent."""
    return coherence(w, DIR_SNP, scale=PER_M)


per_socket_coherence("dir_lookup_snp_m_s", DIR_SNP, PER_M)


@stat("dir_lookup_no_snp_m_s")
def _(w: Window) -> float:
    """Lookups that did not."""
    return coherence(w, DIR_NO_SNP, scale=PER_M)


# ----------------------------------------------------------------------- main


Derive = Optional[Callable[[pd.DataFrame], pd.DataFrame]]


def read_result(
    path: str, derive: Derive = None, header_only: bool = False
) -> pd.DataFrame:
    """A result CSV, plus the window columns a bench derives itself. With a
    `derive`, `header_only` still reads the rows: it decides the columns."""
    df = pd.read_csv(path, nrows=0 if header_only and derive is None else None)
    return derive(df) if derive else df


def compute_stats(
    result_csv: str,
    label: str,
    arch: str,
    stats: list[str] | None = None,
    warmup_s: float = 0.0,
    cooldown_s: float = 0.0,
    derive: Derive = None,
) -> pd.DataFrame:
    """Result CSV + one column per stat, computed on each run's time window.

    `warmup_s` / `cooldown_s` drop the head and tail, so a ramp up does not
    drag the mean. The `warmup_s` column reports what was applied.
    """
    df = read_result(result_csv, derive)
    for col in ("start_time", "end_time"):
        if col not in df.columns:
            raise ValueError(f"{result_csv} has no '{col}' column")

    names = stats or list(STATS)
    unknown = [name for name in names if name not in STATS]
    if unknown:
        raise ValueError(f"unknown stats: {unknown}, known: {list(STATS)}")

    full = load_monitoring(arch, label)
    starts = pd.to_datetime(df["start_time"])
    ends = pd.to_datetime(df["end_time"])
    warmup = pd.Timedelta(seconds=warmup_s)
    cooldown = pd.Timedelta(seconds=cooldown_s)

    rows = []
    for start, end in zip(starts, ends):
        window = slice_window(full, start, end)
        trimmed = window
        applied = 0.0
        if start + warmup < end - cooldown:
            trimmed = slice_window(full, start + warmup, end - cooldown)
            # a run shorter than the trim keeps its full window
            if not trimmed.pcm.empty or window.pcm.empty:
                applied = warmup_s
            else:
                trimmed = window
        rows.append(
            {name: STATS[name](trimmed) for name in names}
            | {"warmup_s": applied}
        )

    return pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)


def infer_arch(result_csv: str) -> str:
    """results/<arch>/<bench>/<file>.csv -> <arch>, else the current platform."""
    parts = os.path.normpath(os.path.abspath(result_csv)).split(os.sep)
    if config.RESULT_DIR in parts:
        index = parts.index(config.RESULT_DIR)
        if index + 1 < len(parts):
            return parts[index + 1]
    return config.PLATFORM


def output_path(result_csv: str, arch: str, label: str) -> str:
    """results/<arch>/stats/details/<label>_<result file name>."""
    directory = os.path.join(config.RESULT_DIR, arch, "stats", "details")
    name = os.path.basename(result_csv)
    return os.path.join(directory, f"{label}_{name}")


def write_stats(
    result_csv: str,
    label: str,
    arch: str,
    output: str | None = None,
    stats: list[str] | None = None,
    keep_uncovered: bool = False,
    warmup_s: float = 0.0,
    cooldown_s: float = 0.0,
) -> str | None:
    """Compute and write one result CSV, return the path written."""
    bench = bench_of(result_csv)
    df = compute_stats(
        result_csv,
        label,
        arch,
        stats,
        warmup_s,
        cooldown_s,
        bench.derive_window if bench else None,
    )
    if not keep_uncovered:
        df = pd.DataFrame(df[df["samples"] > 0])
    if df.empty:
        print(f"[SKIP] {result_csv} [{label}]: no run in the monitoring window")
        return None

    output = output or output_path(result_csv, arch, label)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"[OK] {len(df)} runs -> {output}")
    return output


# ------------------------------------------------------------------ batch run
# one summary per bench and arch, labels mixed in and told apart by `label`


@dataclass
class Bench:
    """A bench directory, its monitoring labels, and how to summarize it."""

    labels: list[str]
    # which result CSVs of the bench dir to read, by file name
    keep_file: Callable[[str], bool]
    # group runs into one row per key, None if the CSV is already a summary
    group_by: list[str] | None = None
    # metrics to also report the standard deviation of when grouping
    std_of: tuple[str, ...] = ()
    # per file fixups, e.g. normalizing a column before grouping on it
    prepare: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    # builds start_time / end_time for a bench whose tool reports none
    derive_window: Derive = None
    # which CSVs feed the per run details file, defaults to keep_file
    keep_detail_file: Optional[Callable[[str], bool]] = None
    # seconds dropped at the start / end of every run window
    warmup_s: float = 0.0
    cooldown_s: float = 0.0
    # drop run 1: it loads the index and pays the replication ramp up
    drop_first_run: bool = False


def _rocksdb_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """readrandom.t64 -> readrandom, same as plot_rocksdb."""
    df["test"] = (
        df["test"]
        .str.replace(r"\.t\d+", "", regex=True)
        .str.replace(r"\.s\d+", "", regex=True)
    )
    return df


# llama-bench reports no run window, and its `test_time` is stamped before the
# model load. Anchoring on the *next* test_time keeps that load out of it.
LLAMA_REPS = 5  # llama-bench default, bench_llama.py does not pass -r


def _llama_window(df: pd.DataFrame) -> pd.DataFrame:
    """start_time / end_time of each llama-bench test, in monitor local time."""
    if df.empty or "test_time" not in df.columns:
        return df

    df = df.sort_values("test_time").copy()
    # the one UTC stamp of the pipeline, everything else is naive local
    start = (
        pd.to_datetime(df["test_time"], utc=True)
        .dt.tz_convert(MONITOR_TZ)
        .dt.tz_localize(None)
    )
    span = pd.to_timedelta(df["avg_ns"] * LLAMA_REPS, unit="ns")
    # the last test has no next one to anchor on, and carries no load
    end = start.shift(-1).fillna(start + span)

    df["start_time"] = end - span
    df["end_time"] = end
    return df


def _fio_window(df: pd.DataFrame) -> pd.DataFrame:
    """start_time / end_time of each fio run, in monitor local time. fio stamps
    epoch seconds, and MONITOR_TZ is only known here."""
    if df.empty or "ts_start" not in df.columns:
        return df

    df = df.copy()
    for column, stamp in (("start_time", "ts_start"), ("end_time", "ts_end")):
        df[column] = (
            pd.to_datetime(df[stamp], unit="s", utc=True)
            .dt.tz_convert(MONITOR_TZ)
            .dt.tz_localize(None)
        )
    return df


def _duckdb_window(df: pd.DataFrame) -> pd.DataFrame:
    """One row per pass: a single query is far shorter than a monitor sample,
    so the queries of a pass share one window. Pass 1 is the cold run."""
    if df.empty or "start_time" not in df.columns:
        return df

    keys = [c for c in ("sf", "streams", "tag") if c in df.columns] + ["pass"]
    out = df.groupby(keys, as_index=False).agg(
        elapsed_s=("elapsed_s", "sum"),
        start_time=("start_time", "min"),
        end_time=("end_time", "max"),
    )
    out["phase"] = np.where(out["pass"] == 1, "cold", "warm")
    return out.rename(columns={"pass": "run_id"})


BENCHES = {
    "ann": Bench(
        labels=["ann", "ann-repl", "ann-pressure", "ann-pressure-repl"],
        # summarize the per run details rather than read the bench summary,
        # same as plot_ann: one window per run, and no run 1
        keep_file=lambda name: name.endswith("-details.csv"),
        group_by=["runner_name", "tag"],
        std_of=("qps",),
        drop_first_run=True,
    ),
    "rocksdb": Bench(
        labels=["rocksdb", "rocksdb-repl"],
        keep_file=lambda name: name == "results.csv",
        # results.csv is one row per run, summarize like the ann files
        group_by=["test", "tag"],
        std_of=("ops_sec", "mb_sec"),
        prepare=_rocksdb_prepare,
        # db_bench only reaches its steady locality after 45s
        warmup_s=75,
        # its teardown starts up to 8s before the end it reports, and those
        # samples have no traffic, which reads as ~15% locality
        cooldown_s=10,
    ),
    "fio": Bench(
        labels=[
            "fio",
            "fio-repl",
            "fio-pgt-spare",
            "fio-pgt-mitosis",
            "fio-pgt-hydra",
        ],
        # pgtable.csv is written by plot_fio_pgtable, details.csv by plot_fio
        keep_file=lambda name: name in ("details.csv", "pgtable.csv"),
        group_by=["benchmark", "tag", "readratio", "writeratio"],
        std_of=("read_bw_gb", "write_bw_gb"),
        derive_window=_fio_window,
    ),
    "sharing": Bench(
        labels=["sharing"],
        keep_file=lambda name: name == "results.csv",
        group_by=["policy", "phase", "overlap"],
        std_of=(
            "read_gb_s",
            "mem_write_gb",
            "llc_rd_miss_lat_ns",
            "dir_update_m_s",
        ),
        # dirtest reports its own window, first touch already excluded
    ),
    "duckdb": Bench(
        labels=["duckdb", "duckdb-repl"],
        keep_file=lambda name: name.startswith(("tpch_", "clickbench_")),
        # the cold run is a row of its own, never averaged with the warm ones
        group_by=["sf", "streams", "tag", "phase"],
        std_of=("elapsed_s",),
        derive_window=_duckdb_window,
    ),
    "llama": Bench(
        labels=["llama", "llama-repl"],
        keep_file=lambda name: True,
        # one row per test already, nothing to average over
        group_by=None,
        derive_window=_llama_window,
    ),
}

# meaningless once runs are averaged together
DROP_ON_GROUP = [
    "run",
    "run_id",
    "job_id",
    "start_time",
    "end_time",
    "ts_start",
    "ts_end",
]


def bench_of(result_csv: str) -> Optional[Bench]:
    """results/<arch>/<bench>/<file>.csv -> its BENCHES entry, if there is one."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(result_csv)))
    return BENCHES.get(parent)


def result_csvs(
    directory: str, keep_file: Callable[[str], bool], derive: Derive = None
) -> list[str]:
    """Result CSVs of a bench directory that carry per-run timestamps."""
    paths = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".csv") or not keep_file(name):
            continue
        path = os.path.join(directory, name)
        try:
            header = read_result(path, derive, header_only=True).columns
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        if "start_time" in header and "end_time" in header:
            paths.append(path)
    return paths


def summarize(df: pd.DataFrame, keys: list[str], std_of) -> pd.DataFrame:
    """One row per key: mean of every metric, plus `runs` and the asked stds."""
    keys = [key for key in keys if key in df.columns]
    grouped = df.groupby(keys, dropna=False)

    metrics = [
        col
        for col in df.select_dtypes("number").columns
        if col not in keys and col not in DROP_ON_GROUP
    ]
    out = grouped[metrics].mean()
    for col in std_of:
        if col in df.columns:
            out[f"{col}_std"] = grouped[col].std()
    out["runs"] = grouped.size()

    return out.reset_index()


def bench_stats(
    arch: str,
    name: str,
    bench: Bench,
    keep_uncovered: bool,
    details: bool = False,
) -> pd.DataFrame:
    """Every label of one bench on one arch, concatenated. `details` keeps one
    row per run instead of summarizing."""
    bench_dir = os.path.join(config.RESULT_DIR, arch, name)
    if not os.path.isdir(bench_dir):
        return pd.DataFrame()

    keep_file = bench.keep_file
    if details and bench.keep_detail_file:
        keep_file = bench.keep_detail_file

    paths = result_csvs(bench_dir, keep_file, bench.derive_window)
    if not paths:
        candidates = [
            f for f in os.listdir(bench_dir) if f.endswith(".csv") and keep_file(f)
        ]
        if candidates:
            print(
                f"[WARN] {arch}/{name}: {len(candidates)} result csv carry no "
                "start_time / end_time, no window to slice"
            )
        return pd.DataFrame()

    frames = []
    for label in bench.labels:
        monitoring = load_monitoring(arch, label)
        if monitoring.pcm.empty and monitoring.mem.empty:
            continue

        for path in paths:
            df = compute_stats(
                path,
                label,
                arch,
                warmup_s=bench.warmup_s,
                cooldown_s=bench.cooldown_s,
                derive=bench.derive_window,
            )
            if not keep_uncovered:
                df = pd.DataFrame(df[df["samples"] > 0])
            if df.empty:
                continue

            if bench.prepare:
                df = bench.prepare(df)
            # same dataset name in the summary and the details file
            dataset = os.path.splitext(os.path.basename(path))[0]
            df.insert(0, "dataset", dataset.removesuffix("-details"))
            if not details:
                # the details file keeps run 1, it is the one to look at when
                # the ramp up itself is the question
                if bench.drop_first_run and "run_id" in df.columns:
                    df = pd.DataFrame(df[df["run_id"] != 1])
                if bench.group_by:
                    df = summarize(
                        df, ["dataset", *bench.group_by], bench.std_of
                    )
            df.insert(0, "label", label)
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    # a 2 socket machine has no skt2 / skt3 columns to show
    return df.dropna(axis=1, how="all")


# ----------------------------------------------------------------- comparison
# pick a few rows of a bench summary, one column each, so a stat is one line


@dataclass
class Comparison:
    """A bench, and the summary rows to put side by side, in order."""

    bench: str
    # column name -> the values identifying one summary row
    rows: dict[str, dict]


# the same seven policies at every ratio, so one parameterized row set serves
FIO_VARIANTS = [
    ("firsttouch", "fio", "default"),
    ("balancing", "fio", "numabalancing"),
    ("interleaved", "fio", "interleaved"),
    ("repl-no-unreplication", "fio-repl", "repl"),
    ("repl-main-bound", "fio-repl", "unrepl-bound"),
    ("repl-main-firsttouch", "fio-repl", "unrepl-firsttouch"),
    ("repl-main-interleaved", "fio-repl", "unrepl-interleaved"),
]


def fio_comparison(readratio: int) -> Comparison:
    """The seven policies side by side at one point of the read/write sweep."""
    return Comparison(
        bench="fio",
        rows={
            name: {
                "label": label,
                "dataset": "details",
                "benchmark": "readwrite_random",
                "readratio": readratio,
                "writeratio": 100 - readratio,
                "tag": tag,
            }
            for name, label, tag in FIO_VARIANTS
        },
    )


# each kernel with its own interleaved baseline; repl-pt is SPaRe only
PGT_TAGS = {
    "spare": ("interleave", "repl-pt", "repl"),
    "mitosis": ("interleave", "repl"),
    "hydra": ("interleave", "repl"),
}


def _pgt_row(kernel: str, size: str, tag: str) -> dict:
    """One summary row of the fio page table bench, as written by
    plot_fio_pgtable into pgtable.csv."""
    return {
        "label": f"fio-pgt-{kernel}",
        "dataset": "pgtable",
        "benchmark": f"pgtable_{size}",
        "tag": f"{kernel}-{tag}",
        "readratio": 100,
        "writeratio": 0,
    }


def pgtable_kernels_comparison(size: str) -> Comparison:
    """Every run side by side, for reading the counters across kernels."""
    return Comparison(
        bench="fio",
        rows={
            f"{kernel}-{tag}": _pgt_row(kernel, size, tag)
            for kernel, tags in PGT_TAGS.items()
            for tag in tags
        },
    )


# tag -> the monitoring label that ran it
DUCKDB_VARIANTS = [
    ("firsttouch", "duckdb"),
    ("imbalanced", "duckdb"),
    ("interleaved", "duckdb"),
    ("numa-balancing", "duckdb"),
    ("repl", "duckdb-repl"),
]


def duckdb_comparison(dataset: str) -> Comparison:
    """Every policy side by side on one database, warm passes only."""
    return Comparison(
        bench="duckdb",
        rows={
            tag: {
                "label": label,
                "dataset": f"{dataset}_{tag}",
                "tag": tag,
                "phase": "warm",
            }
            for tag, label in DUCKDB_VARIANTS
        },
    )


COMPARISONS = {
    "duckdb-clickbench-firsttouch-vs-imbalanced-vs-interleaved"
    "-vs-balancing-vs-repl": duckdb_comparison("clickbench_s1"),
    "usearch-gist-imbalanced-vs-balancing-vs-interleaved-vs-repl": Comparison(
        bench="ann",
        rows={
            "imbalanced": {
                "label": "ann",
                "dataset": "gist-960-euclidean",
                "runner_name": "usearch",
                "tag": "imbalanced-memory",
            },
            "balancing": {
                "label": "ann",
                "dataset": "gist-960-euclidean",
                "runner_name": "usearch",
                "tag": "numa-balancing",
            },
            "interleaved": {
                "label": "ann",
                "dataset": "gist-960-euclidean",
                "runner_name": "usearch",
                "tag": "interleaved-memory",
            },
            "patched-repl": {
                "label": "ann-repl",
                "dataset": "gist-960-euclidean",
                "runner_name": "usearch",
                "tag": "patched-repl",
            },
        },
    ),
    "faiss-gist-imbalanced-vs-balancing-vs-interleaved-vs-repl": Comparison(
        bench="ann",
        rows={
            "imbalanced": {
                "label": "ann",
                "dataset": "gist-960-euclidean",
                "runner_name": "faiss",
                "tag": "imbalanced-memory",
            },
            "balancing": {
                "label": "ann",
                "dataset": "gist-960-euclidean",
                "runner_name": "faiss",
                "tag": "numa-balancing",
            },
            "interleaved": {
                "label": "ann",
                "dataset": "gist-960-euclidean",
                "runner_name": "faiss",
                "tag": "interleaved-memory",
            },
            "patched-repl": {
                "label": "ann-repl",
                "dataset": "gist-960-euclidean",
                "runner_name": "faiss",
                "tag": "patched-repl",
            },
        },
    ),
    "rocksdb-readrandom-imbalanced-vs-balancing-vs-interleaved-vs-repl": (
        Comparison(
            bench="rocksdb",
            # _rocksdb_prepare already stripped the .t64 off the test name
            rows={
                "imbalanced": {
                    "label": "rocksdb",
                    "test": "readrandom",
                    "tag": "imbalanced-readrandom",
                },
                "balancing": {
                    "label": "rocksdb",
                    "test": "readrandom",
                    "tag": "balancing-readrandom",
                },
                "interleaved": {
                    "label": "rocksdb",
                    "test": "readrandom",
                    "tag": "interleaved-readrandom",
                },
                "patched-repl": {
                    "label": "rocksdb-repl",
                    "test": "readrandom",
                    "tag": "patched-repl-readrandom",
                },
            },
        )
    ),
    # overwrite is the one bench where replication loses
    "rocksdb-overwrite-imbalanced-vs-balancing-vs-interleaved-vs-repl": (
        Comparison(
            bench="rocksdb",
            rows={
                "imbalanced": {
                    "label": "rocksdb",
                    "test": "overwrite",
                    "tag": "imbalanced-overwrite",
                },
                "balancing": {
                    "label": "rocksdb",
                    "test": "overwrite",
                    "tag": "balancing-overwrite",
                },
                "interleaved": {
                    "label": "rocksdb",
                    "test": "overwrite",
                    "tag": "interleaved-overwrite",
                },
                "patched-repl": {
                    "label": "rocksdb-repl",
                    "test": "overwrite",
                    "tag": "patched-repl-overwrite",
                },
            },
        )
    ),
    # the control pair: same placement, same remote fraction, same footprint,
    # differing only in whether the two sockets read common lines
    "sharing-local-vs-remote-vs-disjoint-vs-shared": Comparison(
        bench="sharing",
        rows={
            "local": {"policy": "membind", "phase": "local"},
            "remote": {"policy": "membind", "phase": "remote"},
            "disjoint": {"policy": "membind", "phase": "disjoint"},
            "shared": {"policy": "membind", "phase": "shared"},
        },
    ),
    # the same four with the buffer spread instead of bound
    "sharing-interleaved-local-vs-remote-vs-disjoint-vs-shared": Comparison(
        bench="sharing",
        rows={
            "local": {"policy": "interleaved", "phase": "local"},
            "remote": {"policy": "interleaved", "phase": "remote"},
            "disjoint": {"policy": "interleaved", "phase": "disjoint"},
            "shared": {"policy": "interleaved", "phase": "shared"},
        },
    ),
    # the pure read point: nothing unreplicates, the ceiling replication reaches
    "fio-read100-firsttouch-vs-balancing-vs-interleaved-vs-repl": fio_comparison(
        100
    ),
    # the even mix: unreplication fires constantly and the policies separate
    "fio-pgtable-4G-spare-vs-mitosis-vs-hydra": pgtable_kernels_comparison("4G"),
    "fio-read50-firsttouch-vs-balancing-vs-interleaved-vs-repl": fio_comparison(
        50
    ),
    "llama-ngen512-baseline-vs-distribute-vs-repl": Comparison(
        bench="llama",
        # the longest generation test, the one that runs long enough to settle
        # and the most memory bound of the four
        rows={
            "baseline": {
                "label": "llama",
                "dataset": "baseline",
                "n_gen": 512,
            },
            "distribute": {
                "label": "llama",
                "dataset": "distribute",
                "n_gen": 512,
            },
            "patched-repl": {
                "label": "llama-repl",
                "dataset": "repl",
                "n_gen": 512,
            },
        },
    ),
}


def pick_row(df: pd.DataFrame, selector: dict, where: str):
    """The one summary row matching every key of `selector`."""
    mask = pd.Series(True, index=df.index)
    for column, value in selector.items():
        if column not in df.columns:
            print(f"[WARN] {where}: no '{column}' column")
            return None
        mask &= df[column] == value

    matched = df[mask]
    if matched.empty:
        print(f"[WARN] {where}: no row matching {selector}")
        return None
    if len(matched) > 1:
        print(f"[WARN] {where}: {len(matched)} rows match {selector}")
    return matched.iloc[0]


def compare(df: pd.DataFrame, comparison: Comparison, where: str):
    """One line per metric, one column per selected row, plus their delta."""
    columns = {}
    for name, selector in comparison.rows.items():
        row = pick_row(df, selector, f"{where}/{name}")
        if row is None:
            return pd.DataFrame()
        # no per row dropna: differing indexes make pandas union them, which
        # sorts the metrics alphabetically instead of keeping the STATS order
        columns[name] = pd.to_numeric(row, errors="coerce")

    # shared index, so the file keeps the STATS order
    out = pd.DataFrame(columns).dropna(how="all")
    if len(out.columns) == 2:
        first, second = out.columns
        out["delta"] = out[second] - out[first]
        out["delta_pct"] = 100 * out["delta"] / out[first].replace(0, pd.NA)

    return out.rename_axis("metric").reset_index()


def make_stats_monitoring(keep_uncovered: bool = False):
    """One summary and one per run details file per bench, per arch."""
    for arch in sorted(os.listdir(config.RESULT_DIR)):
        if not os.path.isdir(os.path.join(config.RESULT_DIR, arch)):
            continue

        stats_dir = os.path.join(config.RESULT_DIR, arch, "stats")
        summaries = {}
        for name, bench in BENCHES.items():
            outputs = {
                # results/<arch>/stats/<bench>.csv
                os.path.join(stats_dir, f"{name}.csv"): False,
                # results/<arch>/stats/details/<bench>.csv
                os.path.join(stats_dir, "details", f"{name}.csv"): True,
            }
            for output, details in outputs.items():
                df = bench_stats(arch, name, bench, keep_uncovered, details)
                if df.empty:
                    continue
                if not details:
                    summaries[name] = df

                os.makedirs(os.path.dirname(output), exist_ok=True)
                df.to_csv(output, index=False)
                print(f"[OK] {len(df)} rows -> {output}")

        for name, comparison in COMPARISONS.items():
            summary = summaries.get(comparison.bench)
            if summary is None:
                continue

            df = compare(summary, comparison, f"{arch}/{name}")
            if df.empty:
                continue

            # results/<arch>/stats/compare/<name>.csv
            output = os.path.join(stats_dir, "compare", f"{name}.csv")
            os.makedirs(os.path.dirname(output), exist_ok=True)
            df.to_csv(output, index=False)
            print(f"[OK] {len(df)} metrics -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_csv", nargs="?", help="a single result CSV, or all of BENCHES"
    )
    parser.add_argument(
        "-l", "--label", default=None, help="monitoring label, e.g. ann-repl"
    )
    parser.add_argument(
        "-a", "--arch", default=None, help="defaults to the result CSV path"
    )
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument(
        "-s",
        "--stats",
        nargs="+",
        default=None,
        help=f"subset of: {' '.join(STATS)}",
    )
    parser.add_argument(
        "--keep-uncovered",
        action="store_true",
        help="keep runs with no monitoring sample (empty stats)",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=float,
        default=0.0,
        help="seconds to drop at the start of every run window",
    )
    parser.add_argument(
        "-c",
        "--cooldown",
        type=float,
        default=0.0,
        help="seconds to drop at the end of every run window",
    )
    args = parser.parse_args()

    if args.result_csv is None:
        make_stats_monitoring(args.keep_uncovered)
        return

    if args.label is None:
        parser.error("--label is required with a result CSV")

    write_stats(
        args.result_csv,
        args.label,
        args.arch or infer_arch(args.result_csv),
        args.output,
        args.stats,
        args.keep_uncovered,
        args.warmup,
        args.cooldown,
    )


if __name__ == "__main__":
    main()
