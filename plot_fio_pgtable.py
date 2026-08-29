"""What SPARe's full replication buys over page table only replication.

bench_fio.run_bench_fio_pgt_* writes one jsonl per kernel, holding a baseline
run set and a `repl` one, plus a `repl-pt` one for SPARe. The three baselines
are the same vanilla Linux, so they pool into one bar. Every arm but SPARe's
`repl` leaves the data to the kernel, first touch with NUMA balancing, so the
delta over the baseline is the page table replication alone. Mitosis, Hydra
and SPARe's `repl-pt` replicate the page tables only; `repl` the data too.

This reads the jsonl files, writes the per run CSV the stats pipeline slices
on, and plots the five bars side by side.
"""

import json
import os
from typing import NamedTuple

import config
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from plot_fio import bw_from_fio_output

RESULT_DIR = config.RESULT_DIR

# jsonl file suffix, in read order
KERNELS = ["mitosis", "hydra", "spare"]

HATCH_DATA = "/"
HATCH_COLOR = "0.9"

# Linux orange as always, SPARe blue; Mitosis yellow and Hydra green between
MITOSIS_COLOR = "YlOrBr"
HYDRA_COLOR = config.CARREFOUR_COLOR


class Bar(NamedTuple):
    key: str
    label: str
    ramp: str
    shade: int  # index into the 9 colour ramp
    hatch: str | None


# the five bars, top to bottom
BARS = [
    Bar("linux", "Linux*", config.LINUX_COLOR, 5, None),
    Bar("mitosis", "Mitosis", MITOSIS_COLOR, 3, None),
    Bar("hydra", "Hydra", HYDRA_COLOR, 5, None),
    Bar("spare-pt", "SPARe", config.SPARE_COLOR, 4, None),
    Bar("spare", "SPARe", config.SPARE_COLOR, 7, HATCH_DATA),
]

# (kernel, tag) -> bar. the three kernels' baselines pool into the Linux bar.
# `interleave` is the old baseline, `firsttouch` the one the bench runs now.
BAR_OF = {
    ("mitosis", "interleave"): "linux",
    ("hydra", "interleave"): "linux",
    ("spare", "interleave"): "linux",
    ("mitosis", "firsttouch"): "linux",
    ("hydra", "firsttouch"): "linux",
    ("spare", "firsttouch"): "linux",
    ("mitosis", "repl"): "mitosis",
    ("hydra", "repl"): "hydra",
    ("spare", "repl-pt"): "spare-pt",
    ("spare", "repl"): "spare",
}

SIZE = "1G"
BENCHMARK = "pgtable_1G"

# one column of the paper wide, minimal height for the 5 bars
FIGSIZE = (3.3, 0.65)


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

            # older files carry other sizes
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


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over the runs of each bar, indexed for lookup. The Linux
    bar pools the baseline runs of the three kernels."""
    df = df.copy()
    df["bar"] = [BAR_OF.get((k, t)) for k, t in zip(df.kernel, df.pgt_tag)]
    return (
        df.dropna(subset=["bar"])
        .groupby("bar")
        .agg(
            read_bw_gb=("read_bw_gb", "mean"), read_bw_std=("read_bw_gb", "std")
        )
        .fillna({"read_bw_std": 0})
    )


# --- Plot ---


def _bar_color(key: str):
    """The colour of one bar, for the legend to reuse."""
    bar = next(b for b in BARS if b.key == key)
    return sns.color_palette(bar.ramp, n_colors=9)[bar.shade]


def _setup_style():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})


def plot_pgtable(arch: str, table: pd.DataFrame):
    """One bar per system, Linux first, SPARe's two flush under one tick."""
    _setup_style()

    bar_height = 0.8
    bar_step = bar_height
    group_gap = 0.3

    bars = [b for b in BARS if b.key in table.index]

    ys = []
    y = 0.0
    for i, bar in enumerate(bars):
        if i and bar.label != bars[i - 1].label:
            y += group_gap
        ys.append(y)
        y += bar_step

    # one tick per label, centred on its bars; SPARe owns two
    ticks, ticklabels = [], []
    for label in dict.fromkeys(b.label for b in bars):
        group = [ys[i] for i, b in enumerate(bars) if b.label == label]
        ticks.append(sum(group) / len(group))
        ticklabels.append(label)

    values = [table.loc[b.key, "read_bw_gb"] for b in bars]
    std = [table.loc[b.key, "read_bw_std"] for b in bars]
    colors = [_bar_color(b.key) for b in bars]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    plt.rcParams["hatch.linewidth"] = 1.2
    patches = ax.barh(
        ys,
        values,
        height=bar_height,
        color=colors,
        edgecolor="none",
        linewidth=0,
        xerr=std,
        capsize=1.1,
        error_kw=dict(lw=0.4, capthick=0.5, color="gray", alpha=1.0),
    )
    for patch, bar in zip(patches, bars):
        if bar.hatch:
            patch.set_hatch(bar.hatch)
            patch.set_edgecolor(HATCH_COLOR)
            patch.set_linewidth(0)

    base = table.loc["linux", "read_bw_gb"] if "linux" in table.index else 0
    for i, bar in enumerate(bars):
        if bar.key == "linux" or not base:
            continue
        pct = 100 * (values[i] - base) / base
        ax.text(
            values[i] + std[i],
            ys[i],
            f"  {pct:+.1f}%",
            ha="left",
            va="center",
            fontsize=4.5,
            color="green" if pct > 0 else "red",
        )

    sns.despine(ax=ax)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels, fontsize=6)
    ax.tick_params(axis="x", labelsize=6, length=2, width=1.0)
    ax.tick_params(axis="y", length=2, width=1.0)
    for side in ("bottom", "left"):
        ax.spines[side].set_linewidth(1.0)
    ax.set_xlabel("Read Bandwidth (GB/s)", fontsize=6, labelpad=1)
    ax.set_xlim(0, max(values) * 1.25)
    ax.invert_yaxis()

    legend = [
        Patch(
            facecolor=_bar_color("spare-pt"),
            edgecolor="none",
            linewidth=0,
            label="Non replicated data",
        ),
        Patch(
            facecolor=_bar_color("spare"),
            edgecolor=HATCH_COLOR,
            linewidth=0,
            hatch=HATCH_DATA + HATCH_DATA,
            label="Replicated data",
        ),
    ]

    plt.rcParams["hatch.linewidth"] = 0.8
    ax.legend(
        handles=legend,
        fontsize=4.5,
        frameon=False,
        loc="upper right",
        handlelength=2.8,
        handleheight=1.1,
        labelspacing=0.3,
    )
    fig.tight_layout(pad=0)
    for ext in ("pdf", "svg"):
        path = os.path.join(
            config.PLOT_DIR_FIO,
            f"{config.ARCH_SUBNAMES[arch]}_fio_pgtable.{ext}",
        )
        plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
        print(f"[OK] {path}")
    plt.close(fig)


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

        plot_pgtable(arch, _aggregate(df))
