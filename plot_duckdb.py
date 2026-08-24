"""duckdb plots: warm totals per arm, and the per-query breakdown."""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator

import pandas as pd

import config
from duckdb_lib import summary

def stream_label(n):
    return "1 stream" if int(n) == 1 else f"{n} streams"


TAG_LABELS = {
    "firsttouch": "First touch",
    "imbalanced": "Imbalanced",
    "interleaved": "Interleaved",
    "numa-balancing": "NUMA balancing",
    "repl": "Replication",
}


def palette():
    linux = sns.color_palette(config.LINUX_COLOR, n_colors=5)
    spare = sns.color_palette(config.SPARE_COLOR, n_colors=9)
    return {
        "imbalanced": linux[0],
        "firsttouch": linux[1],
        "interleaved": linux[2],
        "numa-balancing": linux[3],
        "repl": spare[7],
    }


def _style():
    sns.set_style(style="ticks")
    sns.set_context("paper")
    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})


def arch_dirs():
    for arch in sorted(os.listdir(config.RESULT_DIR)):
        path = os.path.join(config.RESULT_DIR, arch, "duckdb")
        if os.path.isdir(path):
            yield arch, path


def read(arch_dir, name):
    path = os.path.join(arch_dir, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def _tags(df):
    return [t for t in summary.TAG_ORDER if t in set(df["tag"])]


def _baseline(df):
    tags = _tags(df)
    return summary.BASELINE if summary.BASELINE in tags else tags[0]


def _pct_text(ax, bars, means, base, fontsize, err=0, rotation=0):
    for rect, mean in zip(bars, means):
        if not mean or not base or np.isnan(base):
            continue
        pct = 100 * (mean - base) / base
        color = "green" if pct < 0 else "red" if pct > 0 else "black"
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + err,
            f"{pct:+.0f}%",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
            rotation=rotation,
        )


def plot_totals(totals, arch):
    """Warm total per arm, one panel per bench."""
    warm = totals[totals["phase"] == "warm"]
    benches = [b for b in summary.BENCH_ORDER if b in set(warm["bench"])]
    tags = _tags(warm)
    baseline = _baseline(warm)
    colors = palette()

    fig, axes = plt.subplots(1, len(benches), figsize=(7.0, 1.8))
    axes = np.atleast_1d(axes)

    for ax, bench in zip(axes, benches):
        panel = warm[warm["bench"] == bench]
        groups = sorted(set(panel["streams"]))
        width = 0.8 / len(tags)
        x = np.arange(len(groups))

        for i, tag in enumerate(tags):
            means, stds = [], []
            for group in groups:
                row = panel[
                    (panel["tag"] == tag) & (panel["streams"] == group)
                ]
                means.append(row["mean_s"].iloc[0] if len(row) else 0)
                stds.append(row["std_s"].iloc[0] if len(row) else 0)

            bars = ax.bar(
                x + i * width,
                means,
                width=width,
                label=TAG_LABELS[tag],
                color=colors[tag],
                edgecolor=colors[tag],
                yerr=stds,
                capsize=0.6,
                error_kw=dict(lw=0.3, capthick=0.3, color="gray", alpha=0.5),
                linewidth=0.25,
            )
            if tag != baseline:
                base = [
                    panel[
                        (panel["tag"] == baseline)
                        & (panel["streams"] == group)
                    ]["mean_s"]
                    for group in groups
                ]
                base = [b.iloc[0] if len(b) else np.nan for b in base]
                for rect, mean, b, err in zip(bars, means, base, stds):
                    _pct_text(ax, [rect], [mean], b, 4, err)

        sns.despine(ax=ax)
        ax.set_title(bench, fontsize=7)
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels([stream_label(g) for g in groups], fontsize=6)
        ax.tick_params(axis="both", labelsize=6, length=2)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    axes[0].set_ylabel("Warm run (s)", fontsize=7)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        fontsize=5,
        frameon=False,
        ncol=len(tags),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
    )

    fig.tight_layout(pad=0.2)
    path = os.path.join(
        config.PLOT_DIR_DUCKDB, f"{config.ARCH_SUBNAMES[arch]}_duckdb.pdf"
    )
    plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()


def plot_queries(by_query, arch, bench):
    """One bar per query, warm mean, percentage against the baseline."""
    warm = by_query[
        (by_query["phase"] == "warm") & (by_query["bench"] == bench)
    ]
    if warm.empty:
        return
    groups = sorted(set(warm["streams"]))
    tags = _tags(warm)
    baseline = _baseline(warm)
    colors = palette()
    queries = sorted(set(warm["query"]))

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(max(6.0, 0.2 * len(queries) * len(tags)), 2.4 * len(groups)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, group in zip(axes, groups):
        panel = warm[warm["streams"] == group]
        width = 0.8 / len(tags)
        x = np.arange(len(queries))

        for i, tag in enumerate(tags):
            means = (
                panel[panel["tag"] == tag]
                .set_index("query")["mean_s"]
                .reindex(queries)
                .fillna(0)
            )
            bars = ax.bar(
                x + i * width,
                means,
                width=width,
                label=TAG_LABELS[tag],
                color=colors[tag],
                edgecolor=colors[tag],
                linewidth=0.25,
            )
            if tag != baseline:
                base = (
                    panel[panel["tag"] == baseline]
                    .set_index("query")["mean_s"]
                    .reindex(queries)
                )
                for rect, mean, b in zip(bars, means, base):
                    _pct_text(ax, [rect], [mean], b, 4)

        sns.despine(ax=ax)
        ax.set_ylabel(f"{stream_label(group)} (s)", fontsize=8)
        ax.tick_params(axis="both", labelsize=7, length=2)
        spread = panel["mean_s"].max() / max(panel["mean_s"].min(), 1e-9)
        if spread > 50:
            ax.set_yscale("log")
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    axes[-1].set_xticks(x + 0.4 - width / 2)
    axes[-1].set_xticklabels([f"q{q}" for q in queries], fontsize=7, rotation=90)
    axes[0].set_title(f"{bench} per query", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False, ncol=len(tags))

    fig.tight_layout(pad=0.2)
    path = os.path.join(
        config.PLOT_DIR_DUCKDB,
        f"{config.ARCH_SUBNAMES[arch]}_duckdb_{bench}.pdf",
    )
    plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()


def make_plot_duckdb():
    os.makedirs(config.PLOT_DIR_DUCKDB, exist_ok=True)
    _style()

    for arch, arch_dir in arch_dirs():
        summary.backfill(arch_dir)
        totals = read(arch_dir, summary.TOTALS)
        by_query = read(arch_dir, summary.BY_QUERY)
        if totals.empty or by_query.empty:
            continue
        plot_totals(totals, arch)
        for bench in summary.BENCH_ORDER:
            plot_queries(by_query, arch, bench)


if __name__ == "__main__":
    make_plot_duckdb()
