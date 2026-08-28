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
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from plot_fio import bw_from_fio_output

RESULT_DIR = config.RESULT_DIR

# jsonl file suffix, in plot order
KERNELS = ["spare", "mitosis", "hydra"]
KERNELS = ["mitosis", "hydra", "spare"]
KERNEL_LABELS = {"spare": "SPaRe", "mitosis": "Mitosis", "hydra": "Hydra"}

# two bars per kernel: interleaved, then its replicated run. For SPaRe that
# replicates the whole mapping (page tables + data); for Mitosis/Hydra it's
# the page tables only, hence a different hatch for the second bar
TAGS = ["interleave", "repl"]
HATCH_DATA = "O"
HATCH_PT = "/"
HATCH_COLOR = "0.9"
REPL_HATCH = {"spare": HATCH_DATA, "mitosis": HATCH_PT, "hydra": HATCH_PT}

SIZE = "1G"
BENCHMARK = "pgtable_1G"

# one column of the paper wide, minimal height for 3 groups of 2 bars
FIGSIZE = (3.3, 0.66)

# one ramp per kernel so the three stay apart at a glance
KERNEL_RAMPS = {
    "spare": config.SPARE_COLOR,
    "mitosis": config.LINUX_COLOR,
    "hydra": config.CARREFOUR_COLOR,
}


def _kernel_color(kernel: str, tag: str):
    """One hue per kernel, darker for the replicated bar."""
    ramp = sns.color_palette(KERNEL_RAMPS[kernel], n_colors=9)
    return {"interleave": ramp[3], "repl": ramp[7]}[tag]


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

            # older files carry 768m and 4G run sets too
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
    """Mean and std over the runs of each (kernel, tag), indexed for lookup."""
    return (
        df.groupby(["kernel", "pgt_tag"])
        .agg(
            read_bw_gb=("read_bw_gb", "mean"), read_bw_std=("read_bw_gb", "std")
        )
        .fillna({"read_bw_std": 0})
        .reset_index()
        .set_index(["kernel", "pgt_tag"])
    )


# --- Plot ---


def _setup_style():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})


def plot_pgtable(arch: str, table: pd.DataFrame):
    """Two bars per kernel (interleaved, replicated), grouped with no gap
    within a group and a small gap between kernels."""
    _setup_style()

    bar_height = 0.8
    bar_step = bar_height
    group_gap = 0.3

    bars, ticks, ticklabels = [], [], []
    y = 0
    for kernel in KERNELS:
        start = y
        for tag in TAGS:
            if (kernel, tag) in table.index:
                bars.append((y, kernel, tag))
                y += bar_step
        ticks.append((start + y - bar_step) / 2)
        ticklabels.append(KERNEL_LABELS[kernel])
        y += group_gap

    ys = [b[0] for b in bars]
    values = [table.loc[(k, t), "read_bw_gb"] for _, k, t in bars]
    std = [table.loc[(k, t), "read_bw_std"] for _, k, t in bars]
    colors = [_kernel_color(k, t) for _, k, t in bars]

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
    for patch, (_, kernel, tag) in zip(patches, bars):
        if tag != "interleave":
            patch.set_hatch(REPL_HATCH[kernel])
            patch.set_edgecolor(HATCH_COLOR)
            patch.set_linewidth(0)

    for i, (_, kernel, tag) in enumerate(bars):
        base = table.loc[(kernel, "interleave"), "read_bw_gb"]
        if tag == "interleave" or not base:
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
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(values) * 1.25)
    ax.invert_yaxis()

    legend = [
        Patch(
            facecolor="lightgray",
            edgecolor="none",
            linewidth=0,
            label="Interleaved",
        ),
        Patch(
            facecolor="gray",
            edgecolor=HATCH_COLOR,
            linewidth=0,
            hatch=HATCH_PT + HATCH_PT,
            label="Replicated PT",
        ),
        Patch(
            facecolor="gray",
            edgecolor=HATCH_COLOR,
            linewidth=0,
            hatch=HATCH_DATA,
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

        plot_pgtable(arch, _aggregate(df))
