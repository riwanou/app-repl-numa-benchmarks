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
    "gist-960-euclidean.hdf5",
]

BASELINE_TAG = "default"  # Vanilla, the bar the QPS numbers sit on
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
    "patched-repl": "SPARe",
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

YLABEL_SIZE = 5.5
TITLE_SIZE = 6.5
TITLE_PAD = 6.5  # points: the stood-up values run up under the title
XTICK_SIZE = 5.5
XTICK_PAD = 0.5  # points between the dataset labels and the x axis
VALUE_SIZE = 3  # QPS printed over each bar, stood up
BAR_WIDTH = 0.095
BAR_GAP = 0.0
X_SPACING = 0.48

# the runner that gets a grey band behind it, like the phase bands in the
# pressure plot, so it reads as its own block
HIGHLIGHT_RUNNER = "annoy"
BAND = "#f8f8f8"
BAND_PAD_X = 0.002  # figure fraction, sits in the gap between two panels
BAND_PAD_TOP = 0.02  # figure fraction, above the panel title
BAND_RADIUS = 3  # corner radius, in points


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
    # drop the warmup
    warmup = (
        df_details.get("warmup", pd.Series(False, index=df_details.index))
        .fillna(False)
        .astype(bool)
    )
    df_details = df_details[~warmup & (df_details["run_id"] != 1)]
    agg_df = df_details.groupby(
        ["arch", "runner_name", "dataset", "tag"], as_index=False
    ).agg(mean_qps=("qps", "mean"), std_qps=("qps", "std"))
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
        group["abs_qps"] = group["mean_qps"]
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
        1,
        N_RUNNERS,
        figsize=(3.3, 1.0),
        sharey=sharey,
        gridspec_kw={"wspace": wspace},
    )
    return fig, [axes] if N_RUNNERS == 1 else list(axes)


def _bar_positions(x, bar_index: int, n_bars: int) -> list:
    group_width = n_bars * BAR_WIDTH + (n_bars - 1) * BAR_GAP
    return [
        pos
        - group_width / 2
        + bar_index * (BAR_WIDTH + BAR_GAP)
        + BAR_WIDTH / 2
        for pos in x
    ]


def _tag_values(df_runner: pd.DataFrame, tag: str, col: str) -> list:
    """Extract per-dataset values for a given tag, returning 0 for missing rows."""
    values = []
    for ds in DATASET_NAMES:
        row = df_runner[
            (df_runner["dataset"] == ds) & (df_runner["tag"] == tag)
        ]
        values.append(row.iloc[0][col] if len(row) > 0 else 0)
    return values


def _format_runner_ax(
    ax,
    x,
    runner: str,
    hide_left_spine: bool = False,
    y_nbins: int = 5,
):
    # faint dotted rules at each tick, like the pressure plot
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.4, color="0.85", zorder=0)
    sns.despine(ax=ax)
    ax.tick_params(axis="y", labelsize=5, length=2)
    ax.tick_params(axis="x", labelsize=6, length=2, pad=XTICK_PAD)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [s.replace(" ", "-") for s in DATASET_NAMES],
        fontsize=XTICK_SIZE,
        rotation=25,
    )
    ax.set_title(runner.capitalize(), fontsize=TITLE_SIZE, pad=TITLE_PAD)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_nbins))
    if hide_left_spine:
        ax.tick_params(left=False, labelleft=False)
        ax.spines["left"].set_visible(False)


def _shade_panel(fig, ax):
    """Grey rounded band behind one runner's panel.

    Follows the axes box, not its tight bbox: it stops on the bottom spine
    rather than running under the x labels, and stays clear of the
    neighbouring panels' labels on either side. Drawn in figure fraction:
    x and y fractions cover different physical distances, so a uniform
    rounding_size would come out as an ellipse; mutation_aspect corrects for
    it from the figure's inches alone, which a "tight" bbox at save time does
    not change.
    """
    fig.canvas.draw()  # settles the layout, so the boxes below are final
    box = ax.get_position()
    # the title lives outside the axes box, so take the top from the tight
    # bbox to bring it inside the band
    title_top = (
        ax.title.get_window_extent().transformed(fig.transFigure.inverted()).y1
    )
    fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
    ax.patch.set_visible(False)  # the white panel would hide the band
    fig.add_artist(
        mpatches.FancyBboxPatch(
            (box.x0 - BAND_PAD_X, box.y0),
            box.width + 2 * BAND_PAD_X,
            title_top + BAND_PAD_TOP - box.y0,
            boxstyle=f"round,pad=0,rounding_size={(BAND_RADIUS / 72) / fig_w}",
            mutation_aspect=fig_w / fig_h,
            transform=fig.transFigure,
            facecolor=BAND,
            edgecolor="none",
            zorder=-1,
        )
    )
    return title_top + BAND_PAD_TOP


def _extend_left_spine(fig, ax, top_fig: float):
    """Run the y axis up to the top of the band, as the pressure QPS plot
    does: the real spine stops at the axes box, which falls short of the
    grey rectangle next to it."""
    box = ax.get_position()
    top = (top_fig - box.y0) / box.height  # figure fraction -> axes fraction
    spine = ax.spines["left"]
    spine.set_visible(False)
    ax.plot(
        [0, 0],
        [0, top],
        transform=ax.transAxes,
        color=spine.get_edgecolor(),
        lw=spine.get_linewidth(),
        clip_on=False,
        zorder=10,
        solid_capstyle="butt",
    )


def _save_figure(fig, path: str):
    plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)


# --- Plot functions ---


def _short_value(value: float) -> str:
    """Thousands with one decimal and a small k: "21.4k", "958"."""
    return f"{value / 1000:.1f}k" if value >= 1000 else f"{value:.0f}"


def _pct_span(df_runner: pd.DataFrame) -> float:
    """Height of one panel's bars, to lift the labels off them."""
    ends = [0.0]
    for tag in TAGS_ORDER:
        means = _tag_values(df_runner, tag, "mean_qps")
        stds = _tag_values(df_runner, tag, "std_qps")
        ends += [m + s for m, s in zip(means, stds)]
        ends += [m - s for m, s in zip(means, stds)]
    return max(ends) - min(ends)


def plot_main(df: pd.DataFrame):
    _setup_style()
    x = np.arange(N_DATASETS) * X_SPACING
    n_bars = len(TAGS_ORDER)

    for arch in df["arch"].unique():
        df_arch = df[df["arch"] == arch]

        fig, axes = _make_subplots(sharey=True, wspace=0.05)

        for idx, runner in enumerate(RUNNER_NAMES):
            ax = axes[idx]
            df_runner = df_arch[df_arch["runner_name"] == runner]

            span = _pct_span(df_runner)

            for i, tag in enumerate(TAGS_ORDER):
                means = _tag_values(df_runner, tag, "mean_qps")
                stds = _tag_values(df_runner, tag, "std_qps")
                abs_values = _tag_values(df_runner, tag, "abs_qps")
                positions = _bar_positions(x, i, n_bars)
                ax.bar(
                    positions,
                    means,
                    yerr=stds,
                    width=BAR_WIDTH,
                    label=TAG_LABELS[tag],
                    capsize=0.7,
                    linewidth=0.25,
                    error_kw=dict(lw=0.3, capthick=0.3),
                    color=palettes[tag],
                    edgecolor=palettes[tag],
                )
                # one QPS per group, off the baseline bar: the others are
                # read from it through their percentage
                if tag != BASELINE_TAG:
                    continue
                for pos, mean, std, value in zip(
                    positions, means, stds, abs_values
                ):
                    if not value:
                        continue
                    top = (mean + std) if mean >= 0 else 0
                    ax.text(
                        pos,
                        top + span * 0.015,
                        _short_value(value),
                        ha="left",
                        va="bottom",
                        rotation=45,
                        rotation_mode="anchor",
                        fontsize=VALUE_SIZE,
                        zorder=3,
                    )

            ax.axhline(
                0, linestyle="--", color="gray", linewidth=0.3, alpha=0.25
            )
            _format_runner_ax(ax, x, runner, hide_left_spine=(idx != 0))

        axes[0].set_ylabel(
            "Improvement over \nNUMA Balancing (%)", fontsize=YLABEL_SIZE
        )

        band_top = _shade_panel(fig, axes[RUNNER_NAMES.index(HIGHLIGHT_RUNNER)])
        _extend_left_spine(fig, axes[0], band_top)

        handles, labels = axes[0].get_legend_handles_labels()
        path = os.path.join(config.PLOT_DIR_ANN, config.ARCH_SUBNAMES[arch])
        _save_figure(fig, f"{path}_ann.pdf")

        fig_legend = plt.figure(figsize=(3.3, 0.5))
        fig_legend.legend(
            handles,
            labels,
            fontsize=9,
            ncol=len(handles),
            edgecolor="white",
            framealpha=1.0,
        )
        fig_legend.subplots_adjust(left=0, right=1, top=1, bottom=0)
        _save_figure(
            fig_legend, os.path.join(config.PLOT_DIR_ANN, "legend.pdf")
        )


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
                    _bar_positions(x, i, n_bars),
                    abs_values,
                    yerr=abs_stds,
                    width=BAR_WIDTH,
                    label=TAG_LABELS[tag],
                    capsize=0.6,
                    linewidth=0.25,
                    error_kw=dict(lw=0.3, capthick=0.3),
                    color=palettes[tag],
                    edgecolor=palettes[tag],
                )

                for rect, pct in zip(bars, pct_values):
                    if rect.get_height() == 0 or pct == 0:
                        continue
                    color = "green" if pct > 0 else "red"
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + offset,
                        f"{pct:+.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=2,
                        color=color,
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

        fig, axes = plt.subplots(
            1, N_RUNNERS, figsize=(3 * N_RUNNERS, 3), sharey=False
        )
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
                        dataset=row,
                        positions=[di + offsets[i]],
                        widths=violin_width,
                        showmeans=False,
                        showextrema=False,
                        showmedians=False,
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
            ax.grid(
                axis="y",
                which="major",
                linestyle="--",
                linewidth=0.4,
                color="gray",
                alpha=0.3,
            )
            ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

        axes[0].set_ylabel(
            "Raw performance\n(mean queries per second, QPS)", fontsize=10
        )
        handles = [
            mpatches.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=palette[i],
                edgecolor="black",
                linewidth=0.3,
                label=TAG_LABELS[TAGS_ORDER[i]],
            )
            for i in range(len(TAGS_ORDER))
        ]
        legend = fig.legend(
            handles,
            TAG_LABELS.values(),
            fontsize=8,
            title_fontsize=9,
            loc="upper right",
            bbox_to_anchor=(1, 1),
            edgecolor="white",
            framealpha=1.0,
        )
        legend.get_frame().set_linewidth(0.4)

        fig.tight_layout()
        path = os.path.join(config.PLOT_DIR_ANN, arch)
        plt.savefig(f"{path}_details.png", bbox_inches="tight", dpi=300)
