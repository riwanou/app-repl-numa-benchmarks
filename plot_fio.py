import json
import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

import config

# bench_fio.py writes one jsonl per workload and kernel, one fio run per line
JSONL = re.compile(r"readwrite_random_(?P<read>\d+)_(?P<write>\d+)-(repl|default)\.jsonl")

# tag written by bench_fio.run_one -> legend label, palette, shade
SERIES = [
    ("numabalancing", "Linux Vanilla", config.LINUX_COLOR, 3),
    ("interleaved", "Interleave", config.LINUX_COLOR, 5),
    ("repl", "SPARe (No Unreplication)", config.SPARE_COLOR, 3),
    ("unrepl", "SPARe (Unreplication)", config.SPARE_COLOR, 5),
    ("policy", "SPARe (Unrepl. + Dynamic Policy)", config.SPARE_COLOR, 8),
]
SHADES = {config.LINUX_COLOR: 7, config.SPARE_COLOR: 9}

# 80% reads, writes on a part of the file only
WRITTEN = re.compile(r"readwrite_written(?P<pct>\d+)_80_20-(repl|default)\.jsonl")

# one image per panel; in LaTeX one row per machine over the two columns:
# uniform read | write on 2/3, written part read | write on 1/3
WIDTH_UNIFORM = 2.3
WIDTH_WRITTEN = 1.15
# fixed margins in inches, not a tight bbox, so every panel of a row has the
# same axes height and baseline whatever its labels
AXES_HEIGHT = 0.85
# read ticks reach 3 digits, write ticks stay at 2: less room before the label
MARGIN_LEFT = {"read": 0.43, "write": 0.38}
# centre of the y label, from the left edge
YLABEL_X = 0.08
MARGIN_RIGHT = 0.04
MARGIN_TOP = 0.04
MARGIN_TICKS = 0.2
MARGIN_XLABEL = 0.42
# only the bottom row names the x axis
BOTTOM_ROW = "gold"
# 30% already shows the write-heavy side, more room per bar
MIN_READ_RATIO = 30


def bw_from_fio_output(data) -> dict:
    """GB/s of one fio run, summed over its job groups."""
    jobs = data.get("jobs", [])
    return {
        f"{op}_bw_gb": sum(j[op]["bw_bytes"] for j in jobs) / 1000**3
        for op in ("read", "write")
    }


def get_data(fio_dir: str) -> pd.DataFrame:
    rows = []
    for fname in sorted(os.listdir(fio_dir)):
        m = JSONL.fullmatch(fname)
        if not m:
            continue
        with open(os.path.join(fio_dir, fname)) as f:
            for line in f:
                r = json.loads(line)
                rows.append(
                    {
                        "readratio": int(m["read"]),
                        "writeratio": int(m["write"]),
                        "run": r["run"],
                        "benchmark": "readwrite_random",
                        "tag": r["tag"],
                        # epoch seconds on the bench machine, for stats_monitoring
                        "ts_start": r["ts_start"],
                        "ts_end": r["ts_end"],
                        **bw_from_fio_output(r["data"]),
                    }
                )
    return pd.DataFrame(rows)


def get_written_data(fio_dir: str) -> pd.DataFrame:
    rows = []
    for fname in sorted(os.listdir(fio_dir)):
        m = WRITTEN.fullmatch(fname)
        if not m:
            continue
        with open(os.path.join(fio_dir, fname)) as f:
            for line in f:
                r = json.loads(line)
                rows.append(
                    {"pct": int(m["pct"]), "tag": r["tag"], **bw_from_fio_output(r["data"])}
                )
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across the runs of each (tag, ratio)."""
    return (
        df.groupby(["tag", "readratio", "writeratio", "benchmark"])
        .agg(
            read_bw_gb=("read_bw_gb", "mean"),
            write_bw_gb=("write_bw_gb", "mean"),
            read_bw_std=("read_bw_gb", "std"),
            write_bw_std=("write_bw_gb", "std"),
            read_bw_min=("read_bw_gb", "min"),
            write_bw_min=("write_bw_gb", "min"),
            read_bw_max=("read_bw_gb", "max"),
            write_bw_max=("write_bw_gb", "max"),
            nb_runs=("run", "count"),
        )
        .fillna(0)
        .reset_index()
        .sort_values(["readratio", "tag"], ascending=[False, True])
    )


def aggregate_written(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["tag", "pct"])
        .agg(
            read_bw_gb=("read_bw_gb", "mean"),
            write_bw_gb=("write_bw_gb", "mean"),
            read_bw_std=("read_bw_gb", "std"),
            write_bw_std=("write_bw_gb", "std"),
            read_bw_min=("read_bw_gb", "min"),
            write_bw_min=("write_bw_gb", "min"),
            read_bw_max=("read_bw_gb", "max"),
            write_bw_max=("write_bw_gb", "max"),
        )
        .fillna(0)
        .reset_index()
    )


def make_plot_fio():
    os.makedirs(config.PLOT_DIR_FIO, exist_ok=True)
    make_legend()
    for arch, name in config.ARCH_SUBNAMES.items():
        fio_dir = os.path.join(config.RESULT_DIR, arch, "fio")
        if not os.path.isdir(fio_dir):
            continue
        df = get_data(fio_dir)
        if df.empty:
            print(f"No fio data for {arch}, skipping.")
            continue
        df = df.sort_values(["readratio", "tag", "run"], ascending=[False, True, True])
        df.to_csv(os.path.join(fio_dir, "details.csv"), index=False)
        agg = aggregate(df)
        agg.to_csv(os.path.join(fio_dir, "agg.csv"), index=False)
        agg = agg[agg["readratio"] >= MIN_READ_RATIO]
        bottom = name == BOTTOM_ROW
        plot_read_write(
            agg,
            "readratio",
            "Read Ratio (%)" if bottom else None,
            WIDTH_UNIFORM,
            os.path.join(config.PLOT_DIR_FIO, f"{name}_uniform"),
            switch=kill_switch(agg),
        )
        written = get_written_data(fio_dir)
        if not written.empty:
            plot_read_write(
                aggregate_written(written),
                "pct",
                "Written Part (%)" if bottom else None,
                WIDTH_WRITTEN,
                os.path.join(config.PLOT_DIR_FIO, f"{name}_written"),
            )


def kill_switch(agg: pd.DataFrame) -> float | None:
    """x of the switch: between the last ratio where the policy shares (read
    bandwidth closer to unrepl) and the first where it keeps copies."""
    ratios = sorted(agg["readratio"].unique())
    bw = agg.pivot_table(index="readratio", columns="tag", values="read_bw_gb")
    if not {"policy", "repl", "unrepl"} <= set(bw.columns):
        return None
    keeps = [
        abs(bw.loc[r, "policy"] - bw.loc[r, "repl"])
        < abs(bw.loc[r, "policy"] - bw.loc[r, "unrepl"])
        for r in ratios
    ]
    if True not in keeps or keeps.index(True) == 0:
        return None
    return keeps.index(True) - 0.5


def plot_read_write(agg, col, xlabel, width, prefix, switch=None):
    """One image for read bandwidth, one for write, one group of bars per value
    of col; LaTeX puts them side by side."""
    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})
    sns.set_style("ticks")
    sns.set_context("paper")
    for op in ("read", "write"):
        bottom = MARGIN_XLABEL if xlabel else MARGIN_TICKS
        height = MARGIN_TOP + AXES_HEIGHT + bottom
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_axes(
            [
                MARGIN_LEFT[op] / width,
                bottom / height,
                1 - (MARGIN_LEFT[op] + MARGIN_RIGHT) / width,
                AXES_HEIGHT / height,
            ]
        )
        draw_bars(ax, agg, col, op)
        ax.yaxis.set_label_coords((YLABEL_X - MARGIN_LEFT[op]) / (width - MARGIN_LEFT[op] - MARGIN_RIGHT), 0.5)
        ax.yaxis.label.set_verticalalignment("center")
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=6)
        if switch is not None:
            ax.axvline(switch, ls="--", lw=0.5, color="0.35", zorder=1)
        # named once per row, the write panel beside it shares the line
        if switch is not None and op == "read":
            ax.text(
                switch,
                1.0,
                " kill switch",
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=5,
                color="0.35",
            )
        fig.savefig(f"{prefix}_{op}.pdf", dpi=300)
        plt.close(fig)


def draw_bars(ax, agg: pd.DataFrame, col: str, op: str):
    keys = sorted(agg[col].unique())
    x = np.arange(len(keys))
    series = [s for s in SERIES if s[0] in set(agg["tag"])]
    # keep the group of bars centred on the tick whatever the number of series
    width = 0.85 / len(series)

    for i, (tag, label, palette, shade) in enumerate(series):
        sub = agg[agg["tag"] == tag].set_index(col).reindex(keys, fill_value=0)
        color = sns.color_palette(palette, n_colors=SHADES[palette])[shade]
        mean = sub[f"{op}_bw_gb"]
        ax.bar(
            x + (i - (len(series) - 1) / 2) * width,
            mean,
            width,
            # range of the runs: the policy can land either side of its switch
            yerr=[mean - sub[f"{op}_bw_min"], sub[f"{op}_bw_max"] - mean],
            error_kw={"linewidth": 0.3, "capthick": 0.3},
            capsize=0.7,
            label=label,
            color=color,
            edgecolor=color,
            linewidth=0.3,
            zorder=2,
        )

    # every other group on a faint band, to read the groups apart
    for k in x[1::2]:
        ax.axvspan(k - 0.5, k + 0.5, color="#f8f8f8", lw=0, zorder=0)
    ax.set_xlim(-0.5, len(keys) - 0.5)

    # faint dotted rules at each tick, like the pressure plot
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.4, color="0.85", zorder=0)
    sns.despine(ax=ax)
    ax.tick_params(labelsize=6, length=2)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel(rf"$\mathbf{{{op.capitalize()}}}$ (GB/s)", fontsize=7)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=4))


def make_legend():
    """One row over both columns, so the paper places it once."""
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=sns.color_palette(palette, n_colors=SHADES[palette])[shade])
        for _, _, palette, shade in SERIES
    ]
    labels = [label for _, label, _, _ in SERIES]
    fig = plt.figure(figsize=(6.9, 0.2))
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=len(handles),
        fontsize=6.5,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    path = os.path.join(config.PLOT_DIR_FIO, "legend.pdf")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
