import os
import config
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory
import seaborn as sns
import pandas as pd
import numpy as np


RESULT_DIR = config.RESULT_DIR


def get_std():
    for arch in os.listdir(RESULT_DIR):
        arch_dir = os.path.join(RESULT_DIR, arch, "rocksdb", "outputs")
        if not os.path.isdir(arch_dir):
            continue

        for benchmark in os.listdir(arch_dir):
            benchmark_path = os.path.join(arch_dir, benchmark)
            if os.path.isdir(benchmark_path):
                csv_file = next(
                    (
                        f
                        for f in os.listdir(benchmark_path)
                        if f.endswith(".log.r.csv")
                    ),
                    None,
                )
                if csv_file:
                    csv_path = os.path.join(benchmark_path, csv_file)
                    data = pd.read_csv(csv_path)
                    values = pd.to_numeric(
                        data["interval_qps"], errors="coerce"
                    )

                    mean_val = values.mean()
                    std_val = values.std()

                    print(
                        f"{arch}: {benchmark}: mean = {mean_val:.2f}, std = {std_val:.2f}, std percent = {(std_val / mean_val) * 100:.2f}"
                    )


YLABEL_SIZE = 6.5
XTICK_SIZE = 6.5
XTICK_PAD = 0.5  # points between the method labels and the x axis
VALUE_SIZE = 3  # MB/s printed at the end of each bar, stood up
# a white halo, so a value crossing an error bar stays readable
HALO = [pe.withStroke(linewidth=1.2, foreground="white")]

METHODS = [
    "readrandom",
    "multireadrandom",
    "fwdrange",
    "revrange",
    # "overwrite", # too random, super low bandwidth / op intensity (diff of 5-10 mb), reduce with time (update bench)
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
BASELINE_TAG = ""  # Vanilla, the bar the absolute numbers sit on
TAGS_ORDER = [
    "imbalanced",
    "",
    "interleaved",
    "patched-repl",
]
TAG_LABELS = {
    "imbalanced": "Imbalanced",
    "": "Vanilla",
    "interleaved": "Interleaved",
    "patched-repl": "Replication",
}


# a round more than this far from the median is an outlier: the rounds of a
# tag sit within ~1% of each other, only the odd stall lands this far off
OUTLIER_PCT = 7.5


def _agg_rounds(values: pd.Series) -> pd.Series:
    """Mean and std over the sane rounds, the outliers kept aside."""
    med = values.median()
    keep = (values - med).abs() / med * 100 <= OUTLIER_PCT
    return pd.Series(
        {
            "mb_sec_mean": values[keep].mean(),
            "mb_sec_std": values[keep].std(),
            "outliers": list(values[~keep]),
        }
    )


def _load_data():
    all_data = []
    for arch in os.listdir(RESULT_DIR):
        arch_dir = os.path.join(RESULT_DIR, arch, "rocksdb")
        if not os.path.isdir(arch_dir):
            continue

        csv_path = os.path.join(arch_dir, "results.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        df["arch"] = arch
        df["test"] = (
            df["test"]
            .str.replace(r"\.t\d+", "", regex=True)
            .str.replace(r"\.s\d+", "", regex=True)
        )

        if "nb_runs" in df.columns:
            df = (
                df.groupby(["tag", "test", "arch"])["mb_sec"]
                .apply(_agg_rounds)
                .unstack()
                .reset_index()
            )
        else:
            df["mb_sec_mean"] = df["mb_sec"]
            df["mb_sec_std"] = float("nan")
            df["outliers"] = [[] for _ in range(len(df))]
            df = df[
                ["tag", "test", "arch", "mb_sec_mean", "mb_sec_std", "outliers"]
            ]

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


def _normalize_relative_to_default(group):
    method = group.iloc[0]["test"].rsplit(".", 2)[0]
    default_row = group[group["tag"] == f"balancing-{method}"]
    if default_row.empty:
        return group

    default_mean = default_row["mb_sec_mean"].iloc[0]
    group = group.copy()
    group["mb_sec_mean_pct"] = (
        100 * (group["mb_sec_mean"] - default_mean) / default_mean
    )
    group["mb_sec_std_pct"] = 100 * group["mb_sec_std"] / default_mean
    group["outliers_pct"] = group["outliers"].apply(
        lambda vals: [100 * (v - default_mean) / default_mean for v in vals]
    )

    return group


def _short_value(value: float) -> str:
    """Thousands with one decimal and a small k: "21.4k", "958"."""
    return f"{value / 1000:.1f}k" if value >= 1000 else f"{value:.0f}"


def _row_for(arch_data, tag: str, method: str):
    """The row a bar is drawn from, None when this machine skipped it."""
    if tag:
        row = arch_data[arch_data["tag"] == f"{tag}-{method}"]
    else:
        row = arch_data[arch_data["tag"] == f"default-{method}"]
        if len(row) == 0:
            row = arch_data[arch_data["tag"] == method]
    return row.iloc[0] if len(row) > 0 else None


def _plot_bars(ax, arch_data, show_absolute=False):
    bar_width = 0.11
    bar_gap = 0.0
    n_bars = len(TAGS_ORDER)
    group_width = n_bars * bar_width + (n_bars - 1) * bar_gap
    x = np.arange(len(METHODS)) * 0.63

    linux = sns.color_palette(config.LINUX_COLOR, n_colors=5)
    spare = sns.color_palette(config.SPARE_COLOR, n_colors=9)
    palettes = {
        "imbalanced": linux[0],
        "": linux[1],
        "interleaved": linux[2],
        "patched-repl": spare[7],
    }

    # height of the plotted range, to lift the labels off their bar: only
    # the bars actually drawn, the rest of the rows swing far wider
    drawn = [
        _row_for(arch_data, tag, method)
        for tag in TAGS_ORDER
        for method in METHODS
    ]
    ends = [0.0]
    for row in drawn:
        if row is None:
            continue
        mean = row["mb_sec_mean_pct"]
        err = row["mb_sec_std_pct"] if pd.notna(row["mb_sec_std_pct"]) else 0
        ends += [mean + err, mean - err]
    span = max(ends) - min(ends)

    dropped = []

    for i, tag in enumerate(TAGS_ORDER):
        means = []
        stds = []
        abs_values = []
        abs_stds = []
        outliers = []

        for method in METHODS:
            row = _row_for(arch_data, tag, method)
            if row is not None:
                means.append(row["mb_sec_mean_pct"])
                stds.append(row["mb_sec_std_pct"])
                abs_values.append(row["mb_sec_mean"])
                abs_stds.append(
                    row["mb_sec_std"] if pd.notna(row["mb_sec_std"]) else 0
                )
                outliers.append(
                    row["outliers"] if show_absolute else row["outliers_pct"]
                )
            else:
                means.append(0)
                stds.append(0)
                abs_values.append(0)
                abs_stds.append(0)
                outliers.append([])

        positions = [
            pos - group_width / 2 + i * (bar_width + bar_gap) + bar_width / 2
            for pos in x
        ]

        if show_absolute:
            bar_values = abs_values
            bar_stds = abs_stds
        else:
            bar_values = means
            bar_stds = stds

        if show_absolute:
            error_kw = dict(lw=0.3, capthick=0.3)
            capsize = 0.6
        else:
            error_kw = dict(lw=0.3, capthick=0.3)
            capsize = 0.7

        bars = ax.bar(
            positions,
            bar_values,
            width=bar_width,
            label=TAG_LABELS[tag],
            color=palettes[tag],
            edgecolor=palettes[tag],
            yerr=bar_stds,
            capsize=capsize,
            error_kw=error_kw,
            linewidth=0.25,
        )

        # rounds dropped from the mean, kept visible without letting them
        # stretch the axis: one that falls off the range sits on the edge
        for pos, values in zip(positions, outliers):
            for value in values:
                dropped.append((pos, value, palettes[tag]))

        for rect, pct, abs_val, err in zip(bars, means, abs_values, bar_stds):
            h = rect.get_height()
            if h == 0:
                continue

            # one throughput per group, off the baseline bar: the others
            # are read from it through their percentage
            if not show_absolute and tag == BASELINE_TAG:
                top = (h + err) if h >= 0 else 0
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    top + span * 0.008,
                    _short_value(abs_val),
                    ha="left",
                    va="bottom",
                    rotation=45,
                    rotation_mode="anchor",
                    fontsize=VALUE_SIZE,
                    zorder=5,
                    path_effects=HALO,
                )

            if show_absolute:
                offset = 0.3 if h >= 0 else -0.3
                va = "bottom" if h >= 0 else "top"
                label = f"{pct:+.0f}%" if pct != 0 else f"{abs_val:.0f}"
                color = "green" if pct > 0 else "red" if pct < 0 else "black"

                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    h + offset,
                    label,
                    ha="center",
                    va=va,
                    fontsize=2,
                    color=color,
                )

    return dropped


def _draw_dropped(ax, dropped):
    """The excluded rounds, each at its own value."""
    for pos, value, color in dropped:
        ax.plot(pos, value, marker="x", ms=1.8, mew=0.4, color=color, zorder=4)


def make_plot_rocksdb():
    _make_plot_rocksdb_variant(absolute=False)
    _make_plot_rocksdb_variant(absolute=True)


def _make_plot_rocksdb_variant(absolute=False):
    os.makedirs(config.PLOT_DIR_ROCKSDB, exist_ok=True)

    df_all = _load_data()

    df_all_norm = pd.DataFrame(
        df_all.groupby(["arch", "test"])[df_all.columns.tolist()]
        .apply(_normalize_relative_to_default, include_groups=True)
        .reset_index(drop=True)
    )

    sns.set_style(style="ticks")
    sns.set_context("paper")

    for arch in df_all_norm["arch"].unique():
        arch_data = df_all_norm[df_all_norm["arch"] == arch]

        plt.rcParams.update(
            {"font.family": "serif", "font.serif": "DejaVu Serif"}
        )

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.3, 1.3))

        dropped = _plot_bars(ax, arch_data, show_absolute=absolute)

        ax.set_axisbelow(True)
        ax.grid(axis="y", ls=":", lw=0.4, color="0.85", zorder=0)
        sns.despine(ax=ax)
        if not absolute:
            ax.axhline(
                0, linestyle="--", color="gray", linewidth=0.3, alpha=0.25
            )

        ax.tick_params(axis="y", labelsize=6, length=2)
        ax.tick_params(axis="x", labelsize=6, length=2, pad=XTICK_PAD)

        ax.set_xticks(np.arange(len(METHODS)) * 0.63)
        ax.set_xticklabels(METHODS_LABELS, fontsize=XTICK_SIZE, rotation=25)

        _draw_dropped(ax, dropped)
        if not absolute:
            # exactly the room the stood-up labels need, no more
            fig.canvas.draw()
            inv = ax.transData.inverted()
            ys = [
                inv.transform(t.get_window_extent().corners())[:, 1]
                for t in ax.texts
            ]
            lo, hi = ax.get_ylim()
            if ys:
                pad = 0.02 * (hi - lo)
                ax.set_ylim(
                    min(lo, min(y.min() for y in ys) - pad),
                    max(hi, max(y.max() for y in ys) + pad),
                )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_ylabel(
            "Throughput (MB/s)"
            if absolute
            else "Improvement over \nNUMA Balancing (%)",
            fontsize=YLABEL_SIZE,
        )
        fig.tight_layout(pad=0)
        suffix = "_rocksdb_abs" if absolute else "_rocksdb"
        path = os.path.join(
            config.PLOT_DIR_ROCKSDB, f"{config.ARCH_SUBNAMES[arch]}{suffix}.pdf"
        )
        plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()
