import os
import config
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
import numpy as np


RESULT_DIR = config.RESULT_DIR


YLABEL_SIZE = 6.5
XTICK_SIZE = 6.5
XTICK_PAD = 0.5  # points between the method labels and the x axis
VALUE_SIZE = 4.2  # GB/s printed at the end of each bar, stood up
# a white halo, so a value crossing an error bar stays readable
HALO = [pe.withStroke(linewidth=1.2, foreground="white")]

METHODS = [
    "readrandom",
    "multireadrandom",
    "fwdrange",
    "revrange",
    # "overwrite",  # too random, super low bandwidth / op intensity (diff of 5-10 mb), reduce with time (update bench)
    "readwhilewriting",
    "fwdrangewhilewriting",
    "revrangewhilewriting",
]
METHODS_LABELS = [
    "read",
    "mread",
    "fscan",
    "rscan",
    # "overwrite",
    "read-write",
    "fscan-write",
    "rscan-write",
]
VALUE_TAG = "patched-repl"  # the bar the absolute numbers sit on
TAGS_ORDER = [
    # "imbalanced",
    "",
    "interleaved",
    "patched-repl",
]
TAG_LABELS = {
    # "imbalanced": "Imbalanced",
    "": "First Touch",
    "interleaved": "Interleave",
    "patched-repl": "Replication",
}


def _load_data():
    """Mean and std over the rounds of each (tag, test), per machine, in GB/s."""
    frames = []
    for arch in os.listdir(RESULT_DIR):
        csv_path = os.path.join(RESULT_DIR, arch, "rocksdb", "results.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df["mb_sec"] = df["mb_sec"] / 1000  # MB/s -> GB/s
        df["arch"] = arch
        df["test"] = df["test"].str.replace(r"\.[ts]\d+", "", regex=True)
        frames.append(
            df.groupby(["tag", "test", "arch"])["mb_sec"]
            .agg(mb_sec_mean="mean", mb_sec_std="std")
            .reset_index()
        )
    return pd.concat(frames, ignore_index=True)


def _normalize_relative_to_default(df):
    """Mean and std as a percentage over Linux Vanilla (numa balancing)."""
    vanilla = df[df["tag"] == "balancing-" + df["test"]].set_index(
        ["arch", "test"]
    )["mb_sec_mean"]
    base = df.set_index(["arch", "test"]).index.map(vanilla)
    df = df.copy()
    df["mb_sec_mean_pct"] = 100 * (df["mb_sec_mean"] - base) / base
    df["mb_sec_std_pct"] = 100 * df["mb_sec_std"] / base
    return df


def _row_for(arch_data, tag: str, method: str):
    """The row a bar is drawn from, None when this machine skipped it."""
    names = [f"{tag}-{method}"] if tag else [f"default-{method}", method]
    for name in names:
        row = arch_data[arch_data["tag"] == name]
        if len(row) > 0:
            return row.iloc[0]
    return None


BAR_WIDTH = 0.11
GROUP_STEP = 0.63


def _plot_bars(ax, arch_data, show_absolute=False):
    x = np.arange(len(METHODS)) * GROUP_STEP
    group_width = len(TAGS_ORDER) * BAR_WIDTH

    linux = sns.color_palette(config.LINUX_COLOR, n_colors=5)
    spare = sns.color_palette(config.SPARE_COLOR, n_colors=9)
    palettes = {
        "": linux[0],  # the pale orange llama and ann give First Touch
        "interleaved": linux[2],
        "patched-repl": spare[7],
    }

    rows = {
        (tag, method): _row_for(arch_data, tag, method)
        for tag in TAGS_ORDER
        for method in METHODS
    }
    # height of the plotted range, to lift the labels off their bar
    ends = [0.0]
    for row in filter(lambda r: r is not None, rows.values()):
        err = row["mb_sec_std_pct"] if pd.notna(row["mb_sec_std_pct"]) else 0
        ends += [row["mb_sec_mean_pct"] + err, row["mb_sec_mean_pct"] - err]
    span = max(ends) - min(ends)

    for i, tag in enumerate(TAGS_ORDER):
        positions = x - group_width / 2 + (i + 0.5) * BAR_WIDTH
        tag_rows = [rows[(tag, method)] for method in METHODS]

        def col(name):
            return [
                0 if r is None or pd.isna(r[name]) else r[name] for r in tag_rows
            ]

        pcts, abs_values = col("mb_sec_mean_pct"), col("mb_sec_mean")
        values = abs_values if show_absolute else pcts
        stds = col("mb_sec_std") if show_absolute else col("mb_sec_std_pct")

        bars = ax.bar(
            positions,
            values,
            width=BAR_WIDTH,
            label=TAG_LABELS[tag],
            color=palettes[tag],
            edgecolor=palettes[tag],
            yerr=stds,
            capsize=0.6 if show_absolute else 0.7,
            error_kw=dict(lw=0.3, capthick=0.3),
            linewidth=0.25,
        )

        for rect, pct, abs_val, err in zip(bars, pcts, abs_values, stds):
            h = rect.get_height()
            if h == 0:
                continue
            centre = rect.get_x() + rect.get_width() / 2

            if show_absolute:
                ax.text(
                    centre,
                    h + (0.3 if h >= 0 else -0.3),
                    f"{pct:+.0f}%" if pct != 0 else f"{abs_val:.1f} GB/s",
                    ha="center",
                    va="bottom" if h >= 0 else "top",
                    fontsize=2,
                    color="green" if pct > 0 else "red" if pct < 0 else "black",
                )
            elif tag == VALUE_TAG:
                # one throughput per group, off the SPARe bar's upper cap,
                # which a slightly negative bar still pushes above zero
                ax.text(
                    centre,
                    max(h + err, 0) + span * 0.02,
                    f"{abs_val:.1f} GB/s",
                    ha="left",
                    va="bottom",
                    rotation=45,
                    rotation_mode="anchor",
                    fontsize=VALUE_SIZE,
                    zorder=5,
                    path_effects=HALO,
                )


def _fit_texts(ax):
    """Exactly the room the stood-up labels need, no more."""
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()
    ys = [inv.transform(t.get_window_extent().corners())[:, 1] for t in ax.texts]
    if not ys:
        return
    lo, hi = ax.get_ylim()
    pad = 0.02 * (hi - lo)
    ax.set_ylim(
        min(lo, min(y.min() for y in ys) - pad),
        max(hi, max(y.max() for y in ys) + pad),
    )


def make_plot_rocksdb():
    _make_plot_rocksdb_variant(absolute=False)
    _make_plot_rocksdb_variant(absolute=True)


def _make_plot_rocksdb_variant(absolute=False):
    os.makedirs(config.PLOT_DIR_ROCKSDB, exist_ok=True)

    df_all = _load_data()

    df_all_norm = _normalize_relative_to_default(df_all)

    sns.set_style(style="ticks")
    sns.set_context("paper")

    for arch in df_all_norm["arch"].unique():
        arch_data = df_all_norm[df_all_norm["arch"] == arch]

        plt.rcParams.update(
            {"font.family": "serif", "font.serif": "DejaVu Serif"}
        )

        # the percentage figure has no tall bars to hold, so it can
        # sit lower than the absolute one
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(3.3, 1.3 if absolute else 1.25)
        )

        _plot_bars(ax, arch_data, show_absolute=absolute)

        ax.set_axisbelow(True)
        ax.grid(axis="y", ls=":", lw=0.4, color="0.85", zorder=0)
        sns.despine(ax=ax)
        if not absolute:
            ax.axhline(
                0, linestyle="--", color="gray", linewidth=0.3, alpha=0.25
            )

        ax.tick_params(axis="y", labelsize=6, length=2)
        ax.tick_params(axis="x", labelsize=6, length=2, pad=XTICK_PAD)

        ax.set_xticks(np.arange(len(METHODS)) * GROUP_STEP)
        ax.set_xticklabels(METHODS_LABELS, fontsize=XTICK_SIZE, rotation=25)

        if not absolute:
            _fit_texts(ax)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_ylabel(
            "Throughput (GB/s)"
            if absolute
            else "Improvement over \nLinux Vanilla (%)",
            fontsize=YLABEL_SIZE,
        )
        fig.tight_layout(pad=0)
        suffix = "_rocksdb_abs" if absolute else "_rocksdb"
        path = os.path.join(
            config.PLOT_DIR_ROCKSDB, f"{config.ARCH_SUBNAMES[arch]}{suffix}.pdf"
        )
        plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()
