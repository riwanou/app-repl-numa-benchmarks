import os
import config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator
import numpy as np
import matplotlib.patches as mpatches

RESULT_DIR = config.RESULT_DIR
DATASETS = [
    "glove-100-angular.hdf5",
    # "sift-128-euclidean.hdf5",
    "gist-960-euclidean.hdf5",
]

TAGS_ORDER = [
    "imbalanced-memory",
    "default",
    "interleaved-memory",
    # "numa-balancing",
    "patched-repl",
    # "patched-repl-unrepl",
]
TAG_LABELS = {
    "imbalanced-memory": "Imbalanced",
    "default": "Vanilla",
    "interleaved-memory": "Interleaved",
    # "numa-balancing": "NumaBalancing",
    "patched-repl": "SPaRe",
    # "patched-repl-unrepl": "ReplicationDynamic",
}
linux = sns.color_palette(config.LINUX_COLOR, n_colors=5)
spare = sns.color_palette(config.SPARE_COLOR, n_colors=9)
palettes = {
    "imbalanced-memory": linux[0],
    "default": linux[1],
    "interleaved-memory": linux[2],
    "patched-repl": spare[7],
}

RUNNER_NAMES = ["faiss", "annoy", "usearch"]
N_RUNNERS = len(RUNNER_NAMES)

YLABEL_SIZE = 6.5
UPI_LABEL_SIZE = 6
BAR_WIDTH = 0.095
BAR_GAP = 0.0
X_SPACING = 0.48


def ds_name(dataset: str) -> str:
    return " ".join(dataset.replace(".hdf5", "").split("-")[:2])


DATASET_NAMES = [ds_name(ds) for ds in DATASETS]
N_DATASETS = len(DATASET_NAMES)


def make_plot_ann():
    os.makedirs(config.PLOT_DIR_ANN, exist_ok=True)

    df_main, df_details = get_data(DATASETS)
    df_main_norm = normalize_data(df_main)

    plot_main(df_main_norm)
    plot_main_abs(df_main, df_main_norm)
    plot_details(df_details)


# --- Data loading ---

def _load_tagged_csv(path: str, dataset: str, arch: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        print(f"Warning: CSV {path} not found for {dataset}")
        return None
    df = pd.read_csv(path)
    df = df[df["tag"].isin(TAGS_ORDER + ["numa-balancing"])]
    df["dataset"] = ds_name(dataset)
    df["arch"] = arch
    return df


def load_upi(arch: str) -> dict:
    """(dataset, runner_name, tag) -> upi_out_gb, from the stats pipeline.

    Written by `just stats`; absent until that has been run, in which case the
    bars simply carry no UPI annotation.
    """
    path = os.path.join(RESULT_DIR, arch, "stats", "ann.csv")
    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path)
    if "upi_out_gb" not in df.columns:
        return {}

    return {
        (ds_name(r.dataset), r.runner_name, r.tag): r.upi_out_gb
        for r in df.itertuples()
    }


def get_data(datasets) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and aggregate ANN benchmark data."""
    data_main = []
    data_details = []

    for arch in os.listdir(RESULT_DIR):
        arch_dir = os.path.join(RESULT_DIR, arch, "ann")
        if not os.path.isdir(arch_dir):
            continue

        for dataset in datasets:
            base = dataset.replace(".hdf5", "")
            main_df = _load_tagged_csv(
                os.path.join(arch_dir, f"{base}.csv"), dataset, arch
            )
            details_df = _load_tagged_csv(
                os.path.join(arch_dir, f"{base}-details.csv"), dataset, arch
            )
            if main_df is not None:
                data_main.append(main_df)
            if details_df is not None:
                data_details.append(details_df)

    df_details = pd.concat(data_details, ignore_index=True)
    agg_df = (
        df_details[df_details["run_id"] != 1]
        .groupby(["arch", "runner_name", "dataset", "tag"], as_index=False)
        .agg(mean_qps=("qps", "mean"), std_qps=("qps", "std"))
    )
    return agg_df, df_details


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize QPS relative to the 'numa-balancing' baseline.
    Groups without a numa-balancing row are silently dropped."""

    def _normalize(group):
        rows = group.loc[group["tag"] == "numa-balancing", "mean_qps"]
        if rows.empty:
            return pd.DataFrame()
        baseline = rows.values[0]
        group = group.copy()
        group["mean_qps"] = 100 * (group["mean_qps"] - baseline) / baseline
        group["std_qps"] = 100 * group["std_qps"] / baseline
        return group

    result = (
        df.groupby(["arch", "dataset", "runner_name"])[df.columns.tolist()]
        .apply(_normalize, include_groups=True)
        .reset_index(drop=True)
    )
    if result.empty:
        return result
    return result[result["tag"].isin(TAGS_ORDER)]


# --- Plot helpers ---

def _setup_style():
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})


def _make_subplots(sharey: bool, wspace: float):
    fig, axes = plt.subplots(
        1, N_RUNNERS, figsize=(3.3, 1.2), sharey=sharey,
        gridspec_kw={"wspace": wspace},
    )
    return fig, [axes] if N_RUNNERS == 1 else list(axes)


def _make_subplots_strip(wspace: float):
    """Bars on top, a thin UPI strip underneath sharing the same x.

    Keeps the interconnect numbers off the throughput axis: the bars get the
    full height instead of being squashed by label headroom.
    """
    fig, axes = plt.subplots(
        2, N_RUNNERS, figsize=(3.3, 1.7), sharex="col",
        gridspec_kw={
            "wspace": wspace, "hspace": 0.12, "height_ratios": [1, 0.26],
        },
    )
    top = [axes[0]] if N_RUNNERS == 1 else list(axes[0])
    strip = [axes[1]] if N_RUNNERS == 1 else list(axes[1])
    # share y inside each row so the runners stay comparable
    for ax in top[1:]:
        ax.sharey(top[0])
    for ax in strip[1:]:
        ax.sharey(strip[0])
    return fig, top, strip


def _bar_positions(x, bar_index: int, n_bars: int) -> list:
    group_width = n_bars * BAR_WIDTH + (n_bars - 1) * BAR_GAP
    return [
        pos - group_width / 2 + bar_index * (BAR_WIDTH + BAR_GAP) + BAR_WIDTH / 2
        for pos in x
    ]


def _tag_values(df_runner: pd.DataFrame, tag: str, col: str) -> list:
    """Extract per-dataset values for a given tag, returning 0 for missing rows."""
    values = []
    for ds in DATASET_NAMES:
        row = df_runner[(df_runner["dataset"] == ds) & (df_runner["tag"] == tag)]
        values.append(row.iloc[0][col] if len(row) > 0 else 0)
    return values


def _format_runner_ax(
    ax, x, runner: str, hide_left_spine: bool = False, y_nbins: int = 8,
    xticklabels: bool = True,
):
    sns.despine(ax=ax)
    ax.tick_params(axis="y", labelsize=6, length=2)
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.set_xticks(list(x))
    if xticklabels:
        ax.set_xticklabels(
            [s.replace(" ", "-") for s in DATASET_NAMES], fontsize=7, rotation=25
        )
    else:
        # not set_xticklabels([]): with a shared x that would blank the strip too
        ax.tick_params(labelbottom=False)
    ax.set_title(runner.capitalize(), fontsize=7)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_nbins))
    if hide_left_spine:
        ax.tick_params(left=False, labelleft=False)
        ax.spines["left"].set_visible(False)


def _upi_values(upi: dict, runner: str, tag: str) -> list:
    """Cross-socket traffic each bar is paying for, 0 when unmeasured."""
    values = []
    for ds in DATASET_NAMES:
        value = upi.get((ds, runner, tag))
        values.append(0 if value is None or pd.isna(value) else value)
    return values


def _format_strip_ax(ax, x, hide_left_spine: bool = False):
    sns.despine(ax=ax)
    ax.tick_params(axis="y", labelsize=6, length=2)
    ax.tick_params(axis="x", labelsize=6, length=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [s.replace(" ", "-") for s in DATASET_NAMES], fontsize=7, rotation=25
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=2))
    ax.set_ylim(bottom=0)
    if hide_left_spine:
        ax.tick_params(left=False, labelleft=False)
        ax.spines["left"].set_visible(False)


def _save_figure(fig, path: str):
    plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)


# --- Plot functions ---

def plot_main(df: pd.DataFrame):
    _setup_style()
    x = np.arange(N_DATASETS) * X_SPACING
    n_bars = len(TAGS_ORDER)

    for arch in df["arch"].unique():
        df_arch = df[df["arch"] == arch]

        upi = load_upi(arch)
        # machines with no UPI stats get the plain single-row layout rather
        # than an empty strip
        has_upi = any(v for v in upi.values() if pd.notna(v) and v > 0)
        if has_upi:
            fig, axes, strips = _make_subplots_strip(wspace=0.05)
        else:
            fig, axes = _make_subplots(sharey=True, wspace=0.05)
            strips = [None] * N_RUNNERS

        for idx, runner in enumerate(RUNNER_NAMES):
            ax = axes[idx]
            strip = strips[idx]
            df_runner = df_arch[df_arch["runner_name"] == runner]

            for i, tag in enumerate(TAGS_ORDER):
                means = _tag_values(df_runner, tag, "mean_qps")
                stds = _tag_values(df_runner, tag, "std_qps")
                positions = _bar_positions(x, i, n_bars)
                ax.bar(
                    positions, means, yerr=stds,
                    width=BAR_WIDTH, label=TAG_LABELS[tag], capsize=0.7,
                    linewidth=0.25, error_kw=dict(lw=0.3, capthick=0.3),
                    color=palettes[tag], edgecolor=palettes[tag],
                )
                if strip is not None:
                    strip.bar(
                        positions, _upi_values(upi, runner, tag),
                        width=BAR_WIDTH, linewidth=0.25,
                        color=palettes[tag], edgecolor=palettes[tag],
                    )

            ax.axhline(0, linestyle="--", color="gray", linewidth=0.3, alpha=0.25)
            _format_runner_ax(
                ax, x, runner, hide_left_spine=(idx != 0),
                xticklabels=not has_upi,
            )
            if strip is not None:
                _format_strip_ax(strip, x, hide_left_spine=(idx != 0))

        axes[0].set_ylabel(
            "Improvement over \nNUMA Balancing (%)", fontsize=YLABEL_SIZE
        )
        if has_upi:
            strips[0].set_ylabel("UPI out\n(GB/s)", fontsize=UPI_LABEL_SIZE)
            # same column, but placed clear of each row's tick labels
            fig.align_ylabels([axes[0], strips[0]])

        handles, labels = axes[0].get_legend_handles_labels()
        path = os.path.join(config.PLOT_DIR_ANN, config.ARCH_SUBNAMES[arch])
        _save_figure(fig, f"{path}_ann.pdf")

        fig_legend = plt.figure(figsize=(3.3, 0.5))
        fig_legend.legend(
            handles, labels, fontsize=9, ncol=len(handles),
            edgecolor="white", framealpha=1.0,
        )
        fig_legend.subplots_adjust(left=0, right=1, top=1, bottom=0)
        _save_figure(fig_legend, os.path.join(config.PLOT_DIR_ANN, "legend.pdf"))


def plot_main_abs(df: pd.DataFrame, df_norm: pd.DataFrame):
    """Plot absolute throughput with percentage improvement over NUMA Balancing.
    df holds absolute QPS for all arches; df_norm holds normalized values only
    for arches that have a numa-balancing baseline (used for % annotations)."""
    _setup_style()
    x = np.arange(N_DATASETS) * X_SPACING
    n_bars = len(TAGS_ORDER)

    for arch in df["arch"].unique():
        df_arch = df[df["arch"] == arch]
        df_norm_arch = (
            df_norm[df_norm["arch"] == arch]
            if not df_norm.empty and "arch" in df_norm.columns
            else pd.DataFrame()
        )
        fig, axes = _make_subplots(sharey=False, wspace=0.35)

        for idx, runner in enumerate(RUNNER_NAMES):
            ax = axes[idx]
            df_runner = df_arch[df_arch["runner_name"] == runner]
            df_norm_runner = (
                df_norm_arch[df_norm_arch["runner_name"] == runner]
                if not df_norm_arch.empty
                else pd.DataFrame()
            )

            max_val = df_runner["mean_qps"].max() if len(df_runner) > 0 else 1
            offset = max_val * 0.03

            for i, tag in enumerate(TAGS_ORDER):
                abs_values = _tag_values(df_runner, tag, "mean_qps")
                abs_stds = _tag_values(df_runner, tag, "std_qps")
                pct_values = (
                    _tag_values(df_norm_runner, tag, "mean_qps")
                    if not df_norm_runner.empty
                    else [0] * N_DATASETS
                )

                bars = ax.bar(
                    _bar_positions(x, i, n_bars), abs_values, yerr=abs_stds,
                    width=BAR_WIDTH, label=TAG_LABELS[tag], capsize=0.6,
                    linewidth=0.25, error_kw=dict(lw=0.3, capthick=0.3),
                    color=palettes[tag], edgecolor=palettes[tag],
                )

                for rect, pct in zip(bars, pct_values):
                    if rect.get_height() == 0 or pct == 0:
                        continue
                    color = "green" if pct > 0 else "red"
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + offset,
                        f"{pct:+.0f}%", ha="center", va="bottom", fontsize=2, color=color,
                    )

            _format_runner_ax(ax, x, runner, y_nbins=6)

        axes[0].set_ylabel("Throughput (QPS)", fontsize=7)
        path = os.path.join(config.PLOT_DIR_ANN, config.ARCH_SUBNAMES[arch])
        _save_figure(fig, f"{path}_ann_abs.pdf")


def plot_details(df_details: pd.DataFrame):
    sns.set_style(style="ticks")
    sns.set_context("paper")

    palette = sns.color_palette("Blues", n_colors=len(TAGS_ORDER))
    violin_width = 0.15
    gap = 0.02
    group_width = len(TAGS_ORDER) * violin_width + (len(TAGS_ORDER) - 1) * gap
    offsets = [
        -group_width / 2 + i * (violin_width + gap) + violin_width / 2
        for i in range(len(TAGS_ORDER))
    ]
    x = range(N_DATASETS)

    for arch in df_details["arch"].unique():
        df_arch = df_details[df_details["arch"] == arch]

        fig, axes = plt.subplots(1, N_RUNNERS, figsize=(3 * N_RUNNERS, 3), sharey=False)
        if N_RUNNERS == 1:
            axes = [axes]

        for idx, runner in enumerate(RUNNER_NAMES):
            ax = axes[idx]
            df_runner = df_arch[df_arch["runner_name"] == runner]

            for i, tag in enumerate(TAGS_ORDER):
                df_tag = df_runner[df_runner["tag"] == tag]
                for di, ds in enumerate(DATASET_NAMES):
                    row = df_tag[df_tag["dataset"] == ds]["qps"]
                    if len(row) == 0:
                        continue
                    vp = ax.violinplot(
                        dataset=row, positions=[di + offsets[i]],
                        widths=violin_width, showmeans=False,
                        showextrema=False, showmedians=False,
                    )
                    for b in vp["bodies"]:
                        b.set_facecolor(palette[i])
                        b.set_edgecolor("black")
                        b.set_linewidth(0.3)
                        b.set_alpha(1.0)

            sns.despine(ax=ax)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_xticks(x)
            ax.tick_params(direction="in", axis="x", labelsize=9)
            ax.set_xticklabels(DATASET_NAMES)
            ax.set_title(runner.capitalize(), fontsize=10)
            ax.set_axisbelow(True)
            ax.grid(axis="y", which="major", linestyle="--", linewidth=0.4,
                    color="gray", alpha=0.3)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

        axes[0].set_ylabel(
            "Raw performance\n(mean queries per second, QPS)", fontsize=10
        )
        handles = [
            mpatches.Rectangle(
                (0, 0), 1, 1, facecolor=palette[i], edgecolor="black",
                linewidth=0.3, label=TAG_LABELS[TAGS_ORDER[i]],
            )
            for i in range(len(TAGS_ORDER))
        ]
        legend = fig.legend(
            handles, TAG_LABELS.values(), fontsize=8, title_fontsize=9,
            loc="upper right", bbox_to_anchor=(1, 1),
            edgecolor="white", framealpha=1.0,
        )
        legend.get_frame().set_linewidth(0.4)

        fig.tight_layout()
        path = os.path.join(config.PLOT_DIR_ANN, arch)
        plt.savefig(f"{path}_details.png", bbox_inches="tight", dpi=300)
