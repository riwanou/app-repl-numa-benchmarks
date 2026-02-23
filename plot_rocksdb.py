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


def make_plot_rocksdb():
    # get_std()
    # return
    os.makedirs(config.PLOT_DIR_ROCKSDB, exist_ok=True)

    methods = [
        "readrandom",
        "multireadrandom",
        "fwdrange",
        "revrange",
        "overwrite",
        "readwhilewriting",
        "fwdrangewhilewriting",
        "revrangewhilewriting",
    ]
    methods_labels = [
        "read",
        "mread",
        "fscan",
        "rscan",
        "overwrite",
        "read-write",
        "fscan-write",
        "rscan-write",
    ]
    tags_order = [
        "imbalanced",
        "",
        "interleaved",
        # "balancing",
        "patched-repl",
        # "patched-repl-unrepl",
    ]
    tag_labels = {
        "imbalanced": "Imbalanced",
        "": "Vanilla",
        "interleaved": "Interleaved",
        # "balancing": "NumaBalancing",
        "patched-repl": "Replication",
        # "patched-repl-unrepl": "ReplicationDynamic",
    }
    linux = sns.color_palette(config.LINUX_COLOR, n_colors=5)
    spare = sns.color_palette(config.SPARE_COLOR, n_colors=9)
    palettes = {
        "imbalanced": linux[0],
        "": linux[1],
        "interleaved": linux[2],
        "patched-repl": spare[7],
    }

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
            .str.replace(r"\.t\d+", "", regex=True)  # remove .t64
            .str.replace(r"\.s\d+", "", regex=True)  # remove .s1
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

    df_all = pd.concat(all_data, ignore_index=True)

    def normalize_relative_to_default(group):
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

    df_all_norm = pd.DataFrame(
        df_all.groupby(["arch", "test"])[df_all.columns.tolist()]
        .apply(normalize_relative_to_default, include_groups=True)
        .reset_index(drop=True)
    )

    sns.set_style(style="ticks")
    sns.set_context("paper")

    for arch in df_all_norm["arch"].unique():
        arch_data = pd.DataFrame(df_all_norm[df_all_norm["arch"] == arch])

        plt.rcParams.update(
            {"font.family": "serif", "font.serif": "DejaVu Serif"}
        )

        n_methods = len(methods)
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(3.3, 1.47),
            sharey=True,
        )

        bar_width = 0.11
        bar_gap = 0.0
        n_bars = len(tags_order)
        group_width = n_bars * bar_width + (n_bars - 1) * bar_gap
        x = np.arange(n_methods) * 0.63

        for i, tag in enumerate(tags_order):
            means = []
            stds = []
            for method in methods:
                expected_tag = f"{tag}-{method}"
                if tag == "":
                    expected_tag = f"{method}"
                row = pd.DataFrame(
                    arch_data[(arch_data["tag"] == expected_tag)]
                )
                if len(row) > 0:
                    means.append(row.iloc[0]["mb_sec_mean_pct"])
                    stds.append(row.iloc[0]["mb_sec_std_pct"])
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
                width=bar_width,
                label=tag_labels[tag],
                color=palettes[tag],
                edgecolor=palettes[tag],
                yerr=stds,
                capsize=0.6,
                error_kw={"linewidth": 0.4, "capthick": 0.4},
                linewidth=0.25,
            )

            # ax.bar(
            #     positions,
            #     means,
            #     width=bar_width,
            #     color="none",
            #     hatch=hatches[i],
            #     edgecolor="darkblue",
            #     linewidth=0,
            #     alpha=0.55,
            # )

        sns.despine(ax=ax)
        ax.axhline(0, linestyle="--", color="gray", linewidth=0.3, alpha=0.25)

        ax.tick_params(axis="y", labelsize=6, length=2)
        ax.tick_params(axis="x", labelsize=6, length=2)

        ax.set_xticks(x)
        ax.set_xticklabels(methods_labels, fontsize=7, rotation=25)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_ylabel("Improvement over \n NUMA Balancing (%)", fontsize=7)

        # handles, labels = ax.get_legend_handles_labels()
        # legend = fig.legend(
        #     handles,
        #     labels,
        #     fontsize=4,
        #     title_fontsize=9,
        #     loc="upper right",
        #     bbox_to_anchor=(1.0, 1.0),
        #     edgecolor="white",
        #     framealpha=1.0,
        # )
        # legend.get_frame().set_linewidth(0.4)

        fig.tight_layout(pad=0)
        path = os.path.join(
            config.PLOT_DIR_ROCKSDB, f"{config.ARCH_SUBNAMES[arch]}_rocksdb.pdf"
        )
        plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
