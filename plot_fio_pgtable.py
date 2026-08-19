"""What SPaRe's full replication buys over page table only replication.

bench_fio.run_bench_fio_pgt_* writes one jsonl per kernel, each holding an
`interleave` run set and a `repl` one at two working set sizes. SPaRe
replicates the mapping itself (bench.fio carries repl=1), Mitosis and Hydra
replicate the page tables and leave the data interleaved: the strips below the
bars carry the counters that show it, locality and interconnect traffic.

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

# what each one actually replicates, the whole point of the comparison
SCOPE = {"spare": "data + page table", "mitosis": "page table", "hydra": "page table"}
SCOPE_SHORT = {"spare": "data+PT", "mitosis": "PT", "hydra": "PT"}

# the bench runs two working sets; both give the same ranking, so only the
# larger one is plotted: bigger page tables are the best case for the page
# table only kernels, so it is the fair one to show
SIZES = [("768m", "768 MB"), ("4G", "4 GB")]
PLOT_SIZE = "4G"

TAGS = [("interleave", "interleaved"), ("repl", "replicated")]

# stats_monitoring column -> strip label, drawn under the bars in this order
METRICS = [("local_pct", "Local\n(%)"), ("upi_out_gb", "UPI\n(GB/s)")]

# wider than tall: it sits in one column of the paper
FIG_WIDTH = 3.6
BAR_WIDTH = 0.075
# a hair of air between the bars of a group, they read as one block without it
BAR_GAP = 0.004
YLABEL_SIZE = 6
UPI_LABEL_SIZE = 5.5


# one ramp per kernel so the three stay apart at a glance
KERNEL_RAMPS = {
    "spare": config.SPARE_COLOR,
    "mitosis": config.LINUX_COLOR,
    "hydra": config.CARREFOUR_COLOR,
}


def _kernel_color(kernel: str, repl: bool):
    """One hue per kernel, the interleave bar a lighter shade of it."""
    ramp = sns.color_palette(KERNEL_RAMPS[kernel], n_colors=9)
    return ramp[7] if repl else ramp[3]


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

            size = record.get("size", "")
            rows.append(
                {
                    "kernel": kernel,
                    "size": size,
                    "run": int(record.get("run", 1)),
                    # the stats pipeline groups on benchmark / tag, so the
                    # kernel and the size have to ride along in those two
                    "benchmark": f"pgtable_{size}",
                    "tag": f"{kernel}-{record.get('tag', '')}",
                    "pgt_tag": record.get("tag", ""),
                    # all readers, kept for the fio group_by
                    "readratio": 100,
                    "writeratio": 0,
                    # epoch seconds as stamped on the bench machine, converted
                    # to monitor local time by stats_monitoring, not here
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
    """(kernel, size, pgt_tag) -> the hardware counters of that run set.

    Written by `just stats` from pgtable.csv; absent until that has been run,
    in which case the bars simply carry no strips.
    """
    path = os.path.join(RESULT_DIR, arch, "stats", "fio.csv")
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = df[df["benchmark"].astype(str).str.startswith("pgtable_")]
    if df.empty:
        return df

    df = df.copy()
    df["size"] = df["benchmark"].str.removeprefix("pgtable_")
    # tag is "<kernel>-<interleave|repl>", as written into pgtable.csv
    df[["kernel", "pgt_tag"]] = df["tag"].str.rsplit("-", n=1, expand=True)
    keep = ["kernel", "size", "pgt_tag"] + [m for m, _ in METRICS]
    return df[[c for c in keep if c in df.columns]]


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over the runs of each (kernel, size, tag)."""
    return (
        df.groupby(["kernel", "size", "pgt_tag"])
        .agg(
            read_bw_gb=("read_bw_gb", "mean"),
            read_bw_std=("read_bw_gb", "std"),
            nb_runs=("run", "count"),
        )
        .fillna({"read_bw_std": 0})
        .reset_index()
    )


def _gain_over_interleave(agg: pd.DataFrame) -> pd.DataFrame:
    """Replication gain within each kernel: every kernel brings its own
    interleave baseline, so comparing raw bandwidth across kernels would
    compare the kernels' page fault paths, not their replication."""
    rows = []
    for (kernel, size), group in agg.groupby(["kernel", "size"]):
        base = group[group["pgt_tag"] == "interleave"]["read_bw_gb"]
        repl = group[group["pgt_tag"] == "repl"]
        if base.empty or repl.empty or base.iloc[0] == 0:
            continue
        baseline = base.iloc[0]
        rows.append(
            {
                "kernel": kernel,
                "size": size,
                "gain_pct": 100 * (repl["read_bw_gb"].iloc[0] - baseline) / baseline,
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
    # the axes take the whole canvas: bbox_inches="tight" adds the labels back
    # afterwards, so the default margins would only shrink the plotting area
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
    bottom.set_xlabel(f"{dict(SIZES)[PLOT_SIZE]} working set", fontsize=6)

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
    """One counter of one run set at PLOT_SIZE, 0 when unmeasured."""
    if stats.empty or metric not in stats.columns:
        return 0
    row = stats[
        (stats["kernel"] == kernel)
        & (stats["pgt_tag"] == tag)
        & (stats["size"] == PLOT_SIZE)
    ]
    return 0 if row.empty else float(row.iloc[0][metric])


def plot_gain(arch: str, gains: pd.DataFrame, stats: pd.DataFrame):
    """One bar per kernel: what its replication buys over interleaving.

    The strips carry the counters of the *replicated* run, absolute rather
    than normalised: as a delta the two page table only kernels collapse onto
    the zero line and cannot be read at all.
    """
    _setup_style()
    gains = gains[gains["size"] == PLOT_SIZE].set_index("kernel")
    kernels = [k for k in KERNELS if k[0] in gains.index]
    x = np.arange(len(kernels)) * 0.22
    n_strips = 0 if stats.empty else len(METRICS)
    width = 0.11

    fig, ax, strips = _make_fig(n_strips, height=1.0)
    colors = [_kernel_color(k, repl=True) for k, _ in kernels]
    ax.bar(
        x, [gains.loc[k, "gain_pct"] for k, _ in kernels],
        yerr=[gains.loc[k, "gain_std_pct"] for k, _ in kernels],
        width=width, capsize=0.7, linewidth=0.25,
        error_kw=dict(lw=0.3, capthick=0.3),
        color=colors, edgecolor=colors,
    )

    _plot_strips(
        strips, x,
        lambda metric, i: _metric_at(stats, kernels[i][0], "repl", metric),
        colors, width,
    )

    ax.axhline(0, linestyle="--", color="gray", linewidth=0.3, alpha=0.25)
    # the kernel names are the x axis, so no legend is needed
    _format_ax(
        ax, x, [f"{label}\n({SCOPE_SHORT[k]})" for k, label in kernels],
        "Improvement over \nInterleaved (%)", strips,
    )
    _save(fig, arch, "fio_pgtable")


def plot_absolute(arch: str, agg: pd.DataFrame, stats: pd.DataFrame):
    """Raw read bandwidth, interleaved and replicated, for every kernel."""
    _setup_style()
    agg = agg[agg["size"] == PLOT_SIZE]
    kernels = [k for k in KERNELS if k[0] in set(agg["kernel"])]
    x = np.arange(len(kernels)) * 0.21
    n_strips = 0 if stats.empty else len(METRICS)

    fig, ax, strips = _make_fig(n_strips, height=1.0)
    positions, colors, tags = [], [], []
    for i, (tag, tlabel) in enumerate(TAGS):
        pos = _bar_positions(x, i, len(TAGS))
        sub = agg[agg["pgt_tag"] == tag].set_index("kernel")
        shade = [_kernel_color(k, repl=(tag == "repl")) for k, _ in kernels]
        ax.bar(
            pos, [sub.loc[k, "read_bw_gb"] for k, _ in kernels],
            yerr=[sub.loc[k, "read_bw_std"] for k, _ in kernels],
            width=BAR_WIDTH, label=tlabel.capitalize(), capsize=0.7,
            linewidth=0.25, error_kw=dict(lw=0.3, capthick=0.3),
            color=shade, edgecolor=shade,
        )
        positions += pos
        colors += shade
        tags += [tag] * len(kernels)

        # what replication bought, in GB/s over the bar that bought it. Not a
        # percentage: it rounds Mitosis and Hydra to the same +1% and hides
        # that Mitosis gains twice what Hydra does
        if tag != "repl":
            continue
        base = agg[agg["pgt_tag"] == "interleave"].set_index("kernel")
        for p, (kernel, _) in zip(pos, kernels):
            baseline = base.loc[kernel, "read_bw_gb"]
            value = sub.loc[kernel, "read_bw_gb"]
            if not baseline:
                continue
            ax.annotate(
                f"{value - baseline:+.1f}",
                xy=(p, value + sub.loc[kernel, "read_bw_std"]),
                xytext=(0, 1.5), textcoords="offset points",
                ha="center", va="bottom", fontsize=4.5, color="dimgray",
            )

    _plot_strips(
        strips, positions,
        lambda metric, i: _metric_at(
            stats, kernels[i % len(kernels)][0], tags[i], metric
        ),
        colors, BAR_WIDTH,
    )

    _format_ax(
        ax, x, [f"{label}\n({SCOPE_SHORT[k]})" for k, label in kernels],
        "Read Bandwidth\n(GB/s)", strips,
    )
    # two entries only, they fit in the corner
    ax.legend(fontsize=5, ncol=1, framealpha=0.8, edgecolor="none")
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

        agg = _aggregate(df)
        if PLOT_SIZE not in set(agg["size"]):
            continue

        stats = load_stats(arch)
        if stats.empty:
            print(f"[WARN] {arch}: no stats/fio.csv rows, run `just stats`")

        plot_gain(arch, _gain_over_interleave(agg), stats)
        plot_absolute(arch, agg, stats)
