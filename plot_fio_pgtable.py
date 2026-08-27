"""What SPaRe's full replication buys over page table only replication.

bench_fio.run_bench_fio_pgt_* writes one jsonl per kernel, holding an
`interleave` run set and a `repl` one, plus a `repl-pt` one for SPaRe.
Mitosis and Hydra replicate the page tables and leave the data interleaved,
which is what SPaRe does under `repl-pt`; `repl` replicates the data too.

This reads the jsonl files, writes the per run CSV the stats pipeline slices
on, and plots the three kernels side by side.
"""

import json
import os

import config
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from plot_fio import bw_from_fio_output

RESULT_DIR = config.RESULT_DIR

# jsonl file suffix, in plot order
KERNELS = ["spare", "mitosis", "hydra"]

# repl-pt is SPaRe only; `repl` is the whole mapping for SPaRe and the page
# tables alone for the other two, hence one label per (kernel, tag)
TAGS = ["interleave", "repl-pt", "repl"]
BAR_LABELS = {
    ("spare", "interleave"): "SPaRe Interleaved",
    ("spare", "repl-pt"): "SPaRe Replicated PT",
    ("spare", "repl"): "SPaRe Replicated PT + data",
    ("mitosis", "interleave"): "Mitosis Interleaved",
    ("mitosis", "repl"): "Mitosis Replicated PT",
    ("hydra", "interleave"): "Hydra Interleaved",
    ("hydra", "repl"): "Hydra Replicated PT",
}

SIZE = "4G"
SIZE_LABEL = "4 GB"
BENCHMARK = "pgtable_4G"

# one cluster per metric, top to bottom
METRICS = [
    ("read_bw_gb", "Read Bandwidth (GB/s)"),
    ("local_pct", "Local Accesses (%)"),
    ("upi_out_gb", "UPI Traffic (GB/s)"),
]
# the ones that come from the counters, the rest from fio itself
STAT_METRICS = ["local_pct", "upi_out_gb"]
# less interconnect traffic is the win, so the green goes the other way
LOWER_IS_BETTER = ["upi_out_gb"]

# one column of the paper wide, three clusters tall
FIGSIZE = (3.3, 3.4)

# one ramp per kernel so the three stay apart at a glance
KERNEL_RAMPS = {
    "spare": config.SPARE_COLOR,
    "mitosis": config.LINUX_COLOR,
    "hydra": config.CARREFOUR_COLOR,
}


def _kernel_color(kernel: str, tag: str):
    """One hue per kernel, darker the more the run replicates."""
    ramp = sns.color_palette(KERNEL_RAMPS[kernel], n_colors=9)
    return {"interleave": ramp[3], "repl-pt": ramp[5], "repl": ramp[7]}[tag]


# --- Data loading ---

def _read_jsonl(path: str, kernel: str) -> list:
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping {path}:{lineno}: {e}")
                continue

            # older files also carry a 768m run set, dropped
            if record.get("size") != SIZE:
                continue

            rows.append(
                {
                    "kernel": kernel,
                    "run": int(record.get("run", 1)),
                    # the stats pipeline groups on benchmark / tag, so the
                    # kernel has to ride along in the tag
                    "benchmark": BENCHMARK,
                    "tag": f"{kernel}-{record.get('tag', '')}",
                    "pgt_tag": record.get("tag", ""),
                    # all readers, kept for the fio group_by
                    "readratio": 100,
                    "writeratio": 0,
                    # epoch seconds, stats_monitoring converts them
                    "ts_start": record.get("ts_start"),
                    "ts_end": record.get("ts_end"),
                    **bw_from_fio_output(record.get("data", {})),
                }
            )
    return rows


def get_data(arch: str) -> pd.DataFrame:
    """One row per run of every pgtable_<kernel>.jsonl of an arch."""
    directory = os.path.join(RESULT_DIR, arch, "fio")
    rows = []
    for kernel in KERNELS:
        path = os.path.join(directory, f"pgtable_{kernel}.jsonl")
        if os.path.exists(path):
            rows += _read_jsonl(path, kernel)
    return pd.DataFrame(rows)


def load_stats(arch: str) -> pd.DataFrame:
    """(kernel, pgt_tag) -> the hardware counters of that run set.

    Written by `just stats` from pgtable.csv; absent until that has been run,
    in which case the bars simply carry no strips.
    """
    path = os.path.join(RESULT_DIR, arch, "stats", "fio.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df[df["benchmark"] == BENCHMARK]
    if df.empty:
        return df

    df = df.copy()
    # tag is "<kernel>-<tag>"; split on the first dash, repl-pt has one of its own
    df[["kernel", "pgt_tag"]] = df["tag"].str.split("-", n=1, expand=True)
    keep = ["kernel", "pgt_tag"] + STAT_METRICS
    return df[[c for c in keep if c in df.columns]]


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over the runs of each (kernel, tag)."""
    return (
        df.groupby(["kernel", "pgt_tag"])
        .agg(
            read_bw_gb=("read_bw_gb", "mean"),
            read_bw_std=("read_bw_gb", "std"),
            nb_runs=("run", "count"),
        )
        .fillna({"read_bw_std": 0})
        .reset_index()
    )


def _table(agg: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """(kernel, pgt_tag) -> every metric of that run set."""
    if not stats.empty:
        agg = agg.merge(stats, on=["kernel", "pgt_tag"], how="left")
    return agg.set_index(["kernel", "pgt_tag"])


# --- Plot ---

def _setup_style():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})


def plot_pgtable(arch: str, table: pd.DataFrame):
    """One cluster per metric, one bar per run set, with the difference to
    its own kernel's interleaved run next to it."""
    _setup_style()
    bars = [
        (kernel, BAR_LABELS[(kernel, tag)], tag)
        for kernel in KERNELS
        for tag in TAGS
        if (kernel, tag) in table.index
    ]

    fig, axes = plt.subplots(len(METRICS), 1, figsize=FIGSIZE, sharey=True)
    colors = [_kernel_color(kernel, tag) for kernel, _, tag in bars]

    for ax, (metric, mlabel) in zip(axes, METRICS):
        if metric not in table.columns:
            continue
        values = [table.loc[(k, tag), metric] for k, _, tag in bars]
        std = (
            [table.loc[(k, tag), "read_bw_std"] for k, _, tag in bars]
            if metric == "read_bw_gb"
            else None
        )

        ax.barh(
            range(len(bars)), values, height=0.7,
            color=colors, edgecolor=colors, linewidth=0.25,
            xerr=std, capsize=0.6,
            error_kw=dict(lw=0.3, capthick=0.3, color="gray", alpha=0.5),
        )

        for i, (kernel, _, tag) in enumerate(bars):
            base = table.loc[(kernel, "interleave"), metric]
            if tag == "interleave" or not base:
                continue
            pct = 100 * (values[i] - base) / base
            ax.text(
                values[i] + (std[i] if std else 0), i, f"  {pct:+.1f}%",
                ha="left", va="center", fontsize=4,
                color="green"
                if (pct > 0) != (metric in LOWER_IS_BETTER)
                else "red",
            )

        sns.despine(ax=ax)
        ax.set_yticks(range(len(bars)))
        ax.set_yticklabels([label for _, label, _ in bars], fontsize=4)
        ax.tick_params(axis="x", labelsize=6, length=2)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(mtick.MaxNLocator(nbins=6))
        # room for the percentage at the right of the longest bar
        ax.set_xlim(0, max(values) * 1.25)
        ax.set_xlabel(f"{mlabel}, {SIZE_LABEL} working set", fontsize=6)

    # the kernels read top down, SPaRe first
    axes[0].invert_yaxis()
    fig.tight_layout(pad=0, h_pad=0.8)
    path = os.path.join(
        config.PLOT_DIR_FIO, f"{config.ARCH_SUBNAMES[arch]}_fio_pgtable.pdf"
    )
    plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"[OK] {path}")


def make_plot_fio_pgtable():
    os.makedirs(config.PLOT_DIR_FIO, exist_ok=True)

    for arch in sorted(os.listdir(RESULT_DIR)):
        if arch not in config.ARCH_SUBNAMES:
            continue

        df = get_data(arch)
        if df.empty:
            continue

        # the stats pipeline reads this one, it needs a run window per row
        out = os.path.join(RESULT_DIR, arch, "fio", "pgtable.csv")
        df.to_csv(out, index=False)
        print(f"[OK] {len(df)} runs -> {out}")

        stats = load_stats(arch)
        if stats.empty:
            print(f"[WARN] {arch}: no stats/fio.csv rows, run `just stats`")

        plot_pgtable(arch, _table(_aggregate(df), stats))
