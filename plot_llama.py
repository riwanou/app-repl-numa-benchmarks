import os
import config
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
import numpy as np


RESULT_DIR = config.RESULT_DIR


def make_plot_llama():
    os.makedirs("plots/llama", exist_ok=True)

    methods = ["gen_128", "gen_256", "gen_512"]
    methods_labels = ["gen_128", "gen_256", "gen_512"]

    tags_order = ["baseline", "distribute", "repl"]
    tag_labels = {
        "baseline": "Baseline",
        "distribute": "Interleaved",
        "repl": "Replication",
    }

    linux = sns.color_palette(config.LINUX_COLOR, n_colors=5)
    spare = sns.color_palette(config.SPARE_COLOR, n_colors=9)
    palettes = {
        "baseline": linux[1],
        "distribute": linux[0],
        "repl": spare[7],
    }

    all_data = []
    for arch in os.listdir(RESULT_DIR):
        arch_dir = os.path.join(RESULT_DIR, arch, "llama")
        if not os.path.isdir(arch_dir):
            continue

        for tag in ["baseline", "distribute", "repl"]:
            csv_path = os.path.join(arch_dir, f"{tag}.csv")
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                n_gen = row["n_gen"]

                if n_gen == 128:
                    method = "gen_128"
                elif n_gen == 256:
                    method = "gen_256"
                elif n_gen == 512:
                    method = "gen_512"
                else:
                    continue

                all_data.append({
                    "arch": arch,
                    "tag": tag,
                    "test": method,
                    "avg_ts": row["avg_ts"],
                    "stddev_ts": row["stddev_ts"],
                })

    df_all = pd.DataFrame(all_data)

    df_all = df_all.groupby(["arch", "tag", "test"]).agg({
        "avg_ts": "mean",
        "stddev_ts": "mean",
    }).reset_index()

    def normalize_relative_to_baseline(group):
        baseline_row = group[group["tag"] == "baseline"]
        if baseline_row.empty:
            return group

        baseline_mean = baseline_row["avg_ts"].iloc[0]
        group = group.copy()
        group["avg_ts_pct"] = (
            100 * (group["avg_ts"] - baseline_mean) / baseline_mean
        )
        return group

    df_all_norm = pd.DataFrame(
        df_all.groupby(["arch", "test"])[df_all.columns.tolist()]
        .apply(normalize_relative_to_baseline, include_groups=True)
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

        bar_width = 0.25
        x = np.arange(n_methods)

        for i, tag in enumerate(tags_order):
            means = []
            stds = []
            pcts = []

            for method in methods:
                row = arch_data[
                    (arch_data["tag"] == tag) & (arch_data["test"] == method)
                ]
                if len(row) > 0:
                    means.append(row.iloc[0]["avg_ts"])
                    stds.append(row.iloc[0]["stddev_ts"])
                    pcts.append(row.iloc[0]["avg_ts_pct"])
                else:
                    means.append(0)
                    stds.append(0)
                    pcts.append(0)

            positions = [pos + i * bar_width for pos in x]

            bars = ax.bar(
                positions,
                means,
                width=bar_width,
                label=tag_labels[tag],
                color=palettes[tag],
                edgecolor=palettes[tag],
                yerr=stds,
                capsize=0.8,
                error_kw=dict(lw=0.2, capthick=0.2, color="gray", alpha=0.5),
                linewidth=0.25,
            )

            for rect, pct, tag in zip(bars, pcts, [tags_order[i]] * len(bars)):
                h = rect.get_height()
                if h == 0:
                    continue

                if tag == "baseline":
                    continue

                offset = 0.3 if h >= 0 else -0.3
                va = "bottom" if h >= 0 else "top"
                color = "green" if pct > 0 else "red" if pct < 0 else "black"

                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    h + offset,
                    f"{pct:+.0f}%",
                    ha="center",
                    va=va,
                    fontsize=3,
                    color=color,
                )

        sns.despine(ax=ax)

        ax.tick_params(axis="y", labelsize=6, length=2)
        ax.tick_params(axis="x", labelsize=6, length=2)

        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(methods_labels, fontsize=7)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_ylabel("Throughput (tokens/s)", fontsize=7)

        fig.tight_layout(pad=0)
        path = os.path.join(
            "plots/llama", f"{config.ARCH_SUBNAMES[arch]}_llama.pdf"
        )
        plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()
