import os
import config
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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


METHODS = [
    "readrandom",
    "multireadrandom",
    "fwdrange",
    "revrange",
    "overwrite",
    "readwhilewriting",
    "fwdrangewhilewriting",
    "revrangewhilewriting",
]
METHODS_LABELS = [
    "read",
    "mread",
    "fscan",
    "rscan",
    "overwrite",
    "read-write",
    "fscan-write",
    "rscan-write",
]
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
                .agg(mb_sec_mean="mean", mb_sec_std="std")
                .reset_index()
            )
        else:
            df["mb_sec_mean"] = df["mb_sec"]
            df["mb_sec_std"] = float("nan")
            df = df[["tag", "test", "arch", "mb_sec_mean", "mb_sec_std"]]

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

    return group


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

    for i, tag in enumerate(TAGS_ORDER):
        means = []
        stds = []
        abs_values = []
        abs_stds = []

        for method in METHODS:
            expected_tag = f"{tag}-{method}" if tag else method
            row = arch_data[arch_data["tag"] == expected_tag]
            if len(row) > 0:
                means.append(row.iloc[0]["mb_sec_mean_pct"])
                stds.append(row.iloc[0]["mb_sec_std_pct"])
                abs_values.append(row.iloc[0]["mb_sec_mean"])
                abs_stds.append(
                    row.iloc[0]["mb_sec_std"]
                    if pd.notna(row.iloc[0]["mb_sec_std"])
                    else 0
                )
            else:
                means.append(0)
                stds.append(0)
                abs_values.append(0)
                abs_stds.append(0)

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
            error_kw = dict(lw=0.2, capthick=0.2)
            capsize = 0.8
        else:
            error_kw = dict(lw=0.4, capthick=0.4)
            capsize = 1.0

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

        for rect, pct, abs_val in zip(bars, means, abs_values):
            h = rect.get_height()
            if h == 0:
                continue

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

        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(3.3, 1.47),
            sharey=True,
        )

        _plot_bars(ax, arch_data, show_absolute=absolute)

        sns.despine(ax=ax)
        if not absolute:
            ax.axhline(
                0, linestyle="--", color="gray", linewidth=0.3, alpha=0.25
            )

        ax.tick_params(axis="y", labelsize=6, length=2)
        ax.tick_params(axis="x", labelsize=6, length=2)

        ax.set_xticks(np.arange(len(METHODS)) * 0.63)
        ax.set_xticklabels(METHODS_LABELS, fontsize=7, rotation=25)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_ylabel(
            "Throughput (MB/s)"
            if absolute
            else "Improvement over \n NUMA Balancing (%)",
            fontsize=7,
        )

        fig.tight_layout(pad=0)
        suffix = "_rocksdb_abs" if absolute else "_rocksdb"
        path = os.path.join(
            config.PLOT_DIR_ROCKSDB, f"{config.ARCH_SUBNAMES[arch]}{suffix}.pdf"
        )
        plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()
