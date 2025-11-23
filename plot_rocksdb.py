import os
import config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


RESULT_DIR = config.RESULT_DIR


def make_plot_rocksdb():
    os.makedirs(config.PLOT_DIR_ROCKSDB, exist_ok=True)

    methods = [
        "readrandom",
        "multireadrandom",
        "fwdrange",
        "revrange",
        "readwhilewriting",
        "overwrite",
        # "fwdrangewhilewriting",
        # "revrangewhilewriting",
    ]
    tags_order = [
        # "imbalanced",
        "interleaved",
        "balancing",
        "patched-repl",
        "patched-repl-unrepl",
    ]
    tag_labels = {
        # "imbalanced": "Imbalanced",
        "interleaved": "Interleaved",
        "balancing": "NumaBalancing",
        "patched-repl": "Replication",
        "patched-repl-unrepl": "ReplicationDynamic",
    }

    all_data = []
    arch = "IntelR_XeonR_Gold_6130_CPU_@_2.10GHz_X86_64"
    for arch in os.listdir(RESULT_DIR):
        arch_dir = os.path.join(RESULT_DIR, arch, "rocksdb")
        if not os.path.isdir(arch_dir):
            continue

        csv_path = os.path.join(arch_dir, "results.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df["arch"] = arch
        all_data.append(df)

    df_all = pd.concat(all_data, ignore_index=True)

    def normalize_relative_to_default(group):
        method = group.iloc[0]["test"].rsplit(".", 2)[0]
        default_row = group[group["tag"] == f"{method}"]
        if default_row.empty:
            return group

        default_mean = default_row["mb_sec"].iloc[0]
        group = group.copy()
        group["mb_sec_pct"] = (
            100 * (group["mb_sec"] - default_mean) / default_mean
        )

        return group

    df_all_norm = pd.DataFrame(
        df_all.groupby(["arch", "test"])[df_all.columns.tolist()]
        .apply(normalize_relative_to_default, include_groups=True)
        .reset_index(drop=True)
    )

    sns.set_style(style="ticks")
    sns.set_context("paper")

    for arch in df_all_norm["arch"].unique():
        arch_data = pd.DataFrame(df_all_norm[df_all_norm["arch"] == arch])

        n_methods = len(methods)
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(12, 3),
            sharey=False,
        )

        bar_width = 0.2
        bar_gap = 0.03
        n_bars = len(tags_order)
        group_width = n_bars * bar_width + (n_bars - 1) * bar_gap
        x = range(n_methods)
        palette = sns.color_palette("Blues", n_colors=len(tags_order))

        for i, tag in enumerate(tags_order):
            means = []
            for method in methods:
                expected_tag = f"{tag}-{method}"
                row = pd.DataFrame(
                    arch_data[(arch_data["tag"] == expected_tag)]
                )
                if len(row) > 0:
                    means.append(row.iloc[0]["mb_sec_pct"])
                else:
                    means.append(0)

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
                width=bar_width,
                label=tag_labels[tag],
                capsize=2,
                color=palette[i],
                edgecolor="black",
                linewidth=0.3,
            )

        ax.axhline(0, linestyle="--", color="gray", linewidth=0.4, alpha=0.7)
        sns.despine(ax=ax)

        ax.tick_params(axis="y", labelsize=8)
        ax.set_xticks(x)
        ax.tick_params(direction="in", axis="x", labelsize=9)
        ax.set_xticklabels(methods)

        ax.set_axisbelow(True)
        ax.grid(
            axis="y",
            which="major",
            linestyle="--",
            linewidth=0.4,
            color="gray",
            alpha=0.3,
        )

        ax.set_ylabel(
            "Performance improvement over Linux (%)\n(measured in MB/s)"
        )
        ax.yaxis.label.set_size(10)

        handles, labels = ax.get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            fontsize=8,
            title_fontsize=9,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            edgecolor="white",
            framealpha=1.0,
        )
        legend.get_frame().set_linewidth(0.4)

        fig.tight_layout()
        path = os.path.join(config.PLOT_DIR_ROCKSDB, f"{arch}.png")
        plt.savefig(path, dpi=300)
