import ann.lib
import os
import config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import MaxNLocator

RESULT_DIR = config.RESULT_DIR
DATASETS = ann.lib.DATASETS


def ds_name(dataset: str) -> str:
    return " ".join(dataset.replace(".hdf5", "").split("-")[:2])


def make_plot_ann():
    os.makedirs(config.PLOT_DIR_ANN, exist_ok=True)

    tags_order = ["imbalanced-memory", "interleaved-memory", "patched-repl"]
    tag_labels = {
        "imbalanced-memory": "Imbalanced",
        "interleaved-memory": "Interleaved",
        "patched-repl": "Replication",
    }
    dataset_names = [ds_name(ds) for ds in DATASETS]

    all_data = []
    for arch in os.listdir(RESULT_DIR):
        arch_dir = os.path.join(RESULT_DIR, arch, "ann")
        if not os.path.isdir(arch_dir):
            continue
        for dataset in DATASETS:
            csv_name = dataset.replace(".hdf5", ".csv")
            csv_path = os.path.join(arch_dir, csv_name)

            df = pd.read_csv(csv_path)
            df = df[df["tag"].isin(tags_order + ["default"])]

            df["dataset"] = ds_name(dataset)
            df["arch"] = arch
            all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)

    def normalize_relative_to_default(group):
        default_row = group[group["tag"] == "default"]
        default_mean = default_row["mean_qps"].values[0]

        group = group.copy()
        group["mean_qps_pct"] = (
            100 * (group["mean_qps"] - default_mean) / default_mean
        )
        group["std_qps_pct"] = 100 * group["std_qps"] / default_mean
        return group

    df_all_norm = pd.DataFrame(
        df_all.groupby(["arch", "dataset", "runner_name"])[
            df_all.columns.tolist()
        ]
        .apply(normalize_relative_to_default, include_groups=True)
        .reset_index(drop=True)
    )

    plot_data = df_all_norm[df_all_norm["tag"].isin(tags_order)]

    sns.set_style(style="ticks")
    sns.set_context("paper")

    for arch in pd.Series(plot_data["arch"]).unique():
        arch_data = pd.DataFrame(plot_data[plot_data["arch"] == arch])

        runner_names = ["faiss", "annoy", "usearch"]
        n_runners = len(runner_names)

        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_runners,
            figsize=(3 * n_runners, 3),
            sharey=True,
        )
        if n_runners == 1:
            axes = [axes]

        n_datasets = len(dataset_names)
        bar_width = 0.2
        bar_gap = 0.03
        n_bars = len(tags_order)
        group_width = n_bars * bar_width + (n_bars - 1) * bar_gap
        x = range(n_datasets)
        palette = sns.color_palette("Blues", n_colors=len(tags_order))

        for idx in range(n_runners):
            runner = runner_names[idx]
            ax = axes[idx]
            runner_data = arch_data[arch_data["runner_name"] == runner]

            for i, tag in enumerate(tags_order):
                means = []
                stds = []
                for ds in dataset_names:
                    row = pd.DataFrame(
                        runner_data[
                            (runner_data["dataset"] == ds)
                            & (runner_data["tag"] == tag)
                        ]
                    )
                    if len(row) > 0:
                        means.append(row.iloc[0]["mean_qps_pct"])
                        stds.append(row.iloc[0]["std_qps_pct"])
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
                bars = ax.bar(
                    positions,
                    means,
                    yerr=stds,
                    width=bar_width,
                    label=tag_labels[tag],
                    capsize=2,
                    color=palette[i],
                    edgecolor="black",
                    linewidth=0.3,
                    error_kw=dict(lw=0.6, capthick=0.6),
                )
                for j, bar in enumerate(bars):
                    ds = dataset_names[j]
                    abs_row = pd.DataFrame(
                        runner_data[
                            (runner_data["dataset"] == ds)
                            & (runner_data["tag"] == tag)
                        ]
                    )
                    abs_qps = abs_row.iloc[0]["mean_qps"]
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (1 if bar.get_height() >= 0 else -1),
                        f"{int(abs_qps)}",
                        ha="center",
                        va="bottom" if bar.get_height() >= 0 else "top",
                        fontsize=5,
                        rotation=0,
                    )

            ax.axhline(
                0, linestyle="--", color="gray", linewidth=0.4, alpha=0.7
            )
            sns.despine(ax=ax)

            ax.tick_params(axis="y", labelsize=8)
            ax.set_xticks(x)
            ax.tick_params(direction="in", axis="x", labelsize=9)
            ax.set_xticklabels(dataset_names)
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
            bbox_to_anchor=(1, 1),
            edgecolor="white",
            framealpha=1.0,
        )
        legend.get_frame().set_linewidth(0.4)

        fig.tight_layout()
        path = os.path.join(config.PLOT_DIR_ANN, arch)
        plt.savefig(f"{path}.png", bbox_inches="tight", dpi=300)
