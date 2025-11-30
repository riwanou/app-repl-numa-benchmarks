import os
import config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches

RESULT_DIR = config.RESULT_DIR
DATASETS = [
    "glove-100-angular.hdf5",
    # "sift-128-euclidean.hdf5",
    "gist-960-euclidean.hdf5",
]
# DATASETS = ann.lib.DATASETS

TAGS_ORDER = [
    # "imbalanced-memory",
    "interleaved-memory",
    "numa-balancing",
    "patched-repl",
    # "patched-repl-unrepl",
]
TAG_LABELS = {
    # "imbalanced-memory": "Imbalanced",
    "interleaved-memory": "Interleaved",
    "numa-balancing": "NumaBalancing",
    "patched-repl": "Replication",
    # "patched-repl-unrepl": "ReplicationDynamic",
}

RUNNER_NAMES = ["faiss", "annoy", "usearch"]
N_RUNNERS = len(RUNNER_NAMES)


def ds_name(dataset: str) -> str:
    return " ".join(dataset.replace(".hdf5", "").split("-")[:2])


DATASET_NAMES = [ds_name(ds) for ds in DATASETS]
N_DATASETS = len(DATASET_NAMES)


def make_plot_ann():
    os.makedirs(config.PLOT_DIR_ANN, exist_ok=True)

    df_main, df_details = get_data(DATASETS)
    df_main_norm = normalize_data(df_main)

    plot_main(df_main_norm)
    plot_details(df_details)


def get_data(datasets) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Get ANN benchmark data"""
    data_main = []
    data_details = []

    for arch in os.listdir(RESULT_DIR):
        arch_dir = os.path.join(RESULT_DIR, arch, "ann")
        if not os.path.isdir(arch_dir):
            continue

        for dataset in datasets:
            base_name = dataset.replace(".hdf5", "")
            main_csv_path = os.path.join(arch_dir, f"{base_name}.csv")
            details_csv_path = os.path.join(
                arch_dir, f"{base_name}-details.csv"
            )

            if os.path.exists(main_csv_path):
                df = pd.read_csv(main_csv_path)
                df = df[df["tag"].isin(TAGS_ORDER + ["default"])]
                df["dataset"] = ds_name(dataset)
                df["arch"] = arch
                data_main.append(df)
            else:
                print(
                    f"Warning: Main CSV {main_csv_path} not found for {dataset}"
                )

            if os.path.exists(details_csv_path):
                df = pd.read_csv(details_csv_path)
                df = df[df["tag"].isin(TAGS_ORDER + ["default"])]
                df["dataset"] = ds_name(dataset)
                df["arch"] = arch
                data_details.append(df)
            else:
                print(
                    f"Warning: Details CSV {details_csv_path} not found for {dataset}"
                )

    # df_main = pd.concat(data_main, ignore_index=True)
    df_details = pd.concat(data_details, ignore_index=True)

    agg_df = pd.DataFrame(
        df_details[df_details["run_id"] != 1]
        .groupby(["arch", "runner_name", "dataset", "tag"], as_index=False)
        .agg(mean_qps=("qps", "mean"), std_qps=("qps", "std"))
    )

    return agg_df, df_details


def normalize_data(df_main: pd.DataFrame) -> pd.DataFrame:
    """Normalize ANN benchmark data relative to default linux kernel variant"""

    def normalize_relative_to_default(group):
        default_row = group[group["tag"] == "default"]
        default_mean = default_row["mean_qps"].values[0]
        group["mean_qps_copy"] = group["mean_qps"].copy()

        group = group.copy()
        group["mean_qps"] = (
            100 * (group["mean_qps"] - default_mean) / default_mean
        )
        group["std_qps"] = 100 * group["std_qps"] / default_mean

        return group

    df_main_norm = pd.DataFrame(
        df_main.groupby(["arch", "dataset", "runner_name"])[
            df_main.columns.tolist()
        ]
        .apply(normalize_relative_to_default, include_groups=True)
        .reset_index(drop=True)
    )

    return pd.DataFrame(df_main_norm[df_main_norm["tag"].isin(TAGS_ORDER)])


def plot_main(df_main: pd.DataFrame):
    sns.set_style(style="ticks")
    sns.set_context("paper")

    for arch in pd.Series(df_main["arch"]).unique():
        df_arch = pd.DataFrame(df_main[df_main["arch"] == arch])

        fig, axes = plt.subplots(
            nrows=1,
            ncols=N_RUNNERS,
            figsize=(3 * N_RUNNERS, 3),
            sharey=False,
        )
        if N_RUNNERS == 1:
            axes = [axes]

        bar_width = 0.2
        bar_gap = 0.03
        n_bars = len(TAGS_ORDER)
        group_width = n_bars * bar_width + (n_bars - 1) * bar_gap
        x = range(N_DATASETS)
        palette = sns.color_palette("Blues", n_colors=len(TAGS_ORDER))

        for idx in range(N_RUNNERS):
            runner = RUNNER_NAMES[idx]
            ax = axes[idx]
            df_runner = df_arch[df_arch["runner_name"] == runner]

            for i, tag in enumerate(TAGS_ORDER):
                means = []
                stds = []
                for ds in DATASET_NAMES:
                    row = pd.DataFrame(
                        df_runner[
                            (df_runner["dataset"] == ds)
                            & (df_runner["tag"] == tag)
                        ]
                    )
                    if len(row) > 0:
                        means.append(row.iloc[0]["mean_qps"])
                        stds.append(row.iloc[0]["std_qps"])
                    else:
                        means.append(0)
                        stds.append(0)

                positions = [
                    pos
                    - group_width / 2
                    + i * (bar_width + bar_gap)
                    + bar_width / 2
                    for pos in x
                ]

                ax.bar(
                    positions,
                    means,
                    yerr=stds,
                    width=bar_width,
                    label=TAG_LABELS[tag],
                    capsize=2,
                    color=palette[i],
                    edgecolor="black",
                    linewidth=0.3,
                    error_kw=dict(lw=0.6, capthick=0.6),
                )

            sns.despine(ax=ax)
            ax.axhline(
                0, linestyle="--", color="gray", linewidth=0.4, alpha=0.7
            )
            ax.tick_params(axis="y", labelsize=8)
            ax.set_xticks(x)
            ax.tick_params(direction="in", axis="x", labelsize=9)
            ax.set_xticklabels(DATASET_NAMES)
            ax.set_title(f"{runner.capitalize()}", fontsize=10)
            ax.set_axisbelow(True)
            ax.grid(
                axis="y",
                which="major",
                linestyle="--",
                linewidth=0.4,
                color="gray",
                alpha=0.3,
            )

        axes[0].set_ylabel(
            "Performance improvement over Linux (%)\n(mean queries per second, QPS)"
        )
        axes[0].yaxis.label.set_size(10)
        for ax in axes:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            fontsize=8,
            title_fontsize=9,
            loc="upper right",
            bbox_to_anchor=(0.9, 1),
            edgecolor="white",
            framealpha=1.0,
        )
        legend.get_frame().set_linewidth(0.4)

        fig.tight_layout()
        path = os.path.join(config.PLOT_DIR_ANN, arch)
        plt.savefig(f"{path}.png", bbox_inches="tight", dpi=300)


def plot_details(df_details: pd.DataFrame):
    sns.set_style(style="ticks")
    sns.set_context("paper")

    for arch in pd.Series(df_details["arch"]).unique():
        df_arch = pd.DataFrame(df_details[df_details["arch"] == arch])

        fig, axes = plt.subplots(
            nrows=1,
            ncols=N_RUNNERS,
            figsize=(3 * N_RUNNERS, 3),
            sharey=False,
        )
        if N_RUNNERS == 1:
            axes = [axes]

        x = range(N_DATASETS)
        palette = sns.color_palette("Blues", n_colors=len(TAGS_ORDER))

        for idx in range(N_RUNNERS):
            runner = RUNNER_NAMES[idx]
            ax = axes[idx]
            df_runner = df_arch[df_arch["runner_name"] == runner]

            violin_width = 0.15
            gap = 0.02
            group_width = (
                len(TAGS_ORDER) * violin_width + (len(TAGS_ORDER) - 1) * gap
            )
            offsets = [
                -group_width / 2 + i * (violin_width + gap) + violin_width / 2
                for i in range(len(TAGS_ORDER))
            ]

            for i, tag in enumerate(TAGS_ORDER):
                df_tag = df_runner[df_runner["tag"] == tag]
                color = palette[i]

                for di, ds in enumerate(DATASET_NAMES):
                    row = df_tag[df_tag["dataset"] == ds]["qps"]
                    if len(row) == 0:
                        continue

                    pos = di + offsets[i]
                    vp = ax.violinplot(
                        dataset=row,
                        positions=[pos],
                        widths=violin_width,
                        showmeans=False,
                        showextrema=False,
                        showmedians=False,
                    )

                    for b in vp["bodies"]:
                        b.set_facecolor(color)
                        b.set_edgecolor("black")
                        b.set_linewidth(0.3)
                        b.set_alpha(1.0)

            sns.despine(ax=ax)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_xticks(x)
            ax.tick_params(direction="in", axis="x", labelsize=9)
            ax.set_xticklabels(DATASET_NAMES)
            ax.set_title(f"{runner.capitalize()}", fontsize=10)
            ax.set_axisbelow(True)
            ax.grid(
                axis="y",
                which="major",
                linestyle="--",
                linewidth=0.4,
                color="gray",
                alpha=0.3,
            )

        axes[0].set_ylabel("Raw performance\n(mean queries per second, QPS)")
        axes[0].yaxis.label.set_size(10)
        for ax in axes:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=7))

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
