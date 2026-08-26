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

# jsonl file suffix -> legend label, left to right
KERNELS = [
    ("spare", "SPaRe"),
    ("mitosis", "Mitosis"),
    ("hydra", "Hydra"),
]

# repl-pt is SPaRe only, the other two have no such mode
TAGS = [
    ("interleave", "Interleaved"),
    ("repl-pt", "Replicated PT"),
    ("repl", "Replicated PT + data"),
]

# what each replicated bar replicates
SCOPE_SHORT = {
    ("spare", "repl-pt"): "PT",
    ("spare", "repl"): "data+PT",
    ("mitosis", "repl"): "PT",
    ("hydra", "repl"): "PT",
}

SIZE = "4G"
SIZE_LABEL = "4 GB"
BENCHMARK = "pgtable_4G"

# stats_monitoring column -> strip label, drawn under the bars in this order
METRICS = [("local_pct", "Local\n(%)"), ("upi_out_gb", "UPI\n(GB/s)")]

# wider than tall: it sits in one column of the paper
FIG_WIDTH = 3.6
BAR_WIDTH = 0.075
BAR_GAP = 0.004
YLABEL_SIZE = 6
UPI_LABEL_SIZE = 5.5

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
    for kernel, _ in KERNELS:
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
    keep = ["kernel", "pgt_tag"] + [m for m, _ in METRICS]
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


def _gain_over_interleave(agg: pd.DataFrame) -> pd.DataFrame:
    """Gain of every replicated run over its own kernel's interleave: across
    kernels this would compare page fault paths, not replication."""
    rows = []
    for kernel, group in agg.groupby("kernel"):
        base = group[group["pgt_tag"] == "interleave"]["read_bw_gb"]
        if base.empty or base.iloc[0] == 0:
            continue
        baseline = base.iloc[0]
        for tag, _ in TAGS[1:]:
            repl = group[group["pgt_tag"] == tag]
            if repl.empty:
                continue
            rows.append(
                {
                    "kernel": kernel,
                    "pgt_tag": tag,
                    "gain_pct": 100
                    * (repl["read_bw_gb"].iloc[0] - baseline)
                    / baseline,
                    "gain_std_pct": 100 * repl["read_bw_std"].iloc[0] / baseline,
                }
            )
    return pd.DataFrame(rows)


# --- Plot helpers ---

def _setup_style():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})


def _bar_positions(x, bar_index: int, n_bars: int) -> list:
    group_width = n_bars * BAR_WIDTH + (n_bars - 1) * BAR_GAP
    return [
        pos - group_width / 2 + bar_index * (BAR_WIDTH + BAR_GAP) + BAR_WIDTH / 2
        for pos in x
    ]


def _make_fig(n_strips: int, height: float):
    """Bars on top, one thin strip per hardware metric underneath.

    The counters live on their own axes so the bandwidth axis keeps its scale.
    """
    if not n_strips:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, height))
        return fig, ax, []

    fig, axes = plt.subplots(
        1 + n_strips, 1,
        figsize=(FIG_WIDTH, height + 0.28 * n_strips), sharex=True,
        gridspec_kw={
            "height_ratios": [1] + [0.3] * n_strips, "hspace": 0.1,
        },
    )
    # the axes take the whole canvas, bbox_inches="tight" adds the labels back
    fig.subplots_adjust(left=0.02, right=0.995, top=0.99, bottom=0.02)
    return fig, axes[0], list(axes[1:])


def _plot_strips(strips, positions, values_of, colors, width):
    """The same bars as above, one strip per hardware counter."""
    for strip, (metric, label) in zip(strips, METRICS):
        strip.bar(
            positions, [values_of(metric, i) for i in range(len(positions))],
            width=width, color=colors, edgecolor=colors, linewidth=0.25,
        )
        sns.despine(ax=strip)
        strip.tick_params(axis="y", labelsize=6, length=2)
        strip.tick_params(axis="x", labelsize=6, length=2)
        strip.yaxis.set_major_locator(mtick.MaxNLocator(nbins=2))
        strip.set_ylim(bottom=0)
        strip.set_ylabel(label, fontsize=UPI_LABEL_SIZE)


def _format_ax(ax, x, xlabels: list, ylabel: str, strips: list):
    sns.despine(ax=ax)
    ax.tick_params(axis="y", labelsize=6, length=2)
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.set_ylabel(ylabel, fontsize=YLABEL_SIZE)
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=6))

    # the names go on the bottom row, whichever that is
    bottom = strips[-1] if strips else ax
    if strips:
        ax.tick_params(labelbottom=False)
    bottom.set_xticks(list(x))
    bottom.set_xticklabels(xlabels, fontsize=6)
    bottom.set_xlabel(f"{SIZE_LABEL} working set", fontsize=6)

    fig = ax.get_figure()
    fig.align_ylabels([ax, *strips])


def _save(fig, arch: str, suffix: str):
    path = os.path.join(
        config.PLOT_DIR_FIO, f"{config.ARCH_SUBNAMES[arch]}_{suffix}.pdf"
    )
    plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"[OK] {path}")


# --- Plots ---

def _metric_at(stats: pd.DataFrame, kernel: str, tag: str, metric: str) -> float:
    """One counter of one run set, 0 when unmeasured."""
    if stats.empty or metric not in stats.columns:
        return 0
    row = stats[(stats["kernel"] == kernel) & (stats["pgt_tag"] == tag)]
    return 0 if row.empty else float(row.iloc[0][metric])


def plot_gain(arch: str, gains: pd.DataFrame, stats: pd.DataFrame):
    """One bar per replicated run: what it buys over its own interleaving.

    The strips carry the counters of the replicated run, absolute rather than
    normalised: as a delta the page table only runs collapse onto zero.
    """
    _setup_style()
    gains = gains.set_index(["kernel", "pgt_tag"])
    bars = [
        (kernel, label, tag)
        for kernel, label in KERNELS
        for tag, _ in TAGS[1:]
        if (kernel, tag) in gains.index
    ]
    x = np.arange(len(bars)) * 0.22
    n_strips = 0 if stats.empty else len(METRICS)
    width = 0.11

    fig, ax, strips = _make_fig(n_strips, height=1.0)
    colors = [_kernel_color(kernel, tag) for kernel, _, tag in bars]
    ax.bar(
        x, [gains.loc[(k, tag), "gain_pct"] for k, _, tag in bars],
        yerr=[gains.loc[(k, tag), "gain_std_pct"] for k, _, tag in bars],
        width=width, capsize=0.7, linewidth=0.25,
        error_kw=dict(lw=0.3, capthick=0.3),
        color=colors, edgecolor=colors,
    )

    _plot_strips(
        strips, x,
        lambda metric, i: _metric_at(stats, bars[i][0], bars[i][2], metric),
        colors, width,
    )

    ax.axhline(0, linestyle="--", color="gray", linewidth=0.3, alpha=0.25)
    # the kernel names are the x axis, so no legend is needed
    _format_ax(
        ax, x,
        [f"{label}\n({SCOPE_SHORT[(k, tag)]})" for k, label, tag in bars],
        "Improvement over \nInterleaved (%)", strips,
    )
    _save(fig, arch, "fio_pgtable")


def plot_absolute(arch: str, agg: pd.DataFrame, stats: pd.DataFrame):
    """Raw read bandwidth, interleaved and replicated, for every kernel."""
    _setup_style()
    kernels = [k for k in KERNELS if k[0] in set(agg["kernel"])]
    # three slots per kernel, the groups need room to stay apart
    x = np.arange(len(kernels)) * 0.28
    n_strips = 0 if stats.empty else len(METRICS)

    fig, ax, strips = _make_fig(n_strips, height=1.0)
    base = agg[agg["pgt_tag"] == "interleave"].set_index("kernel")
    bars = []  # (position, kernel, tag) of every bar drawn
    for i, (tag, tlabel) in enumerate(TAGS):
        sub = agg[agg["pgt_tag"] == tag].set_index("kernel")
        # repl-pt is SPaRe only, the other kernels leave that slot empty
        drawn = [
            (pos, kernel)
            for pos, (kernel, _) in zip(
                _bar_positions(x, i, len(TAGS)), kernels
            )
            if kernel in sub.index
        ]
        if not drawn:
            continue
        shade = [_kernel_color(kernel, tag) for _, kernel in drawn]
        ax.bar(
            [pos for pos, _ in drawn],
            [sub.loc[kernel, "read_bw_gb"] for _, kernel in drawn],
            yerr=[sub.loc[kernel, "read_bw_std"] for _, kernel in drawn],
            width=BAR_WIDTH, label=tlabel, capsize=0.7,
            linewidth=0.25, error_kw=dict(lw=0.3, capthick=0.3),
            color=shade, edgecolor=shade,
        )
        bars += [(pos, kernel, tag) for pos, kernel in drawn]

        # in GB/s, not percent: percent rounds Mitosis and Hydra to the same +1%
        if tag == "interleave":
            continue
        for pos, kernel in drawn:
            baseline = base.loc[kernel, "read_bw_gb"]
            value = sub.loc[kernel, "read_bw_gb"]
            if not baseline:
                continue
            ax.annotate(
                f"{value - baseline:+.1f}",
                xy=(pos, value + sub.loc[kernel, "read_bw_std"]),
                xytext=(0, 1.5), textcoords="offset points",
                ha="center", va="bottom", fontsize=4.5, color="dimgray",
            )

    _plot_strips(
        strips, [pos for pos, _, _ in bars],
        lambda metric, i: _metric_at(stats, bars[i][1], bars[i][2], metric),
        [_kernel_color(kernel, tag) for _, kernel, tag in bars], BAR_WIDTH,
    )

    _format_ax(
        ax, x, [label for _, label in kernels], "Read Bandwidth\n(GB/s)", strips,
    )
    # one row above the axes: inside, it covers the bars or their deltas
    ax.legend(
        fontsize=5, ncol=len(TAGS), loc="lower center",
        bbox_to_anchor=(0.5, 1.0), frameon=False,
        handlelength=1.2, columnspacing=1.0, handletextpad=0.4,
    )
    _save(fig, arch, "fio_pgtable_abs")


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

        agg = _aggregate(df)
        plot_gain(arch, _gain_over_interleave(agg), stats)
        plot_absolute(arch, agg, stats)
