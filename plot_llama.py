import os
import config
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd


RESULT_DIR = config.RESULT_DIR

# prompt processing then token generation, llama-bench's two default tests
TESTS = ["pp512", "tg128"]
TEST_LABELS = {"pp512": "Prompt (pp512)", "tg128": "Generation (tg128)"}

# left group: no --numa distribute. right group: with it.
LEFT = ["baseline", "repl"]
RIGHT = ["distribute", "interleaved", "repl-distribute"]
TAGS = LEFT + RIGHT

TAG_LABELS = {
    "baseline": "Baseline",
    "repl": "Replication",
    "distribute": "Distribute",
    "interleaved": "Interleaved",
    "repl-distribute": "Replication",
}

# a gap between the two groups, so the split reads without a divider
POSITIONS = {tag: i for i, tag in enumerate(LEFT)}
POSITIONS.update({tag: len(LEFT) + 0.6 + i for i, tag in enumerate(RIGHT)})


def load():
    rows = []
    for arch in os.listdir(RESULT_DIR):
        arch_dir = os.path.join(RESULT_DIR, arch, "llama")
        if not os.path.isdir(arch_dir):
            continue

        for tag in TAGS:
            csv_path = os.path.join(arch_dir, f"{tag}.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            if "test" not in df.columns:
                continue  # a csv from the old llama-bench --output csv
            for _, row in df.iterrows():
                rows.append({
                    "arch": arch,
                    "tag": tag,
                    "test": row["test"],
                    "avg_ts": row["avg_ts"],
                    "stddev_ts": row["stddev_ts"],
                })
    return pd.DataFrame(rows)


def make_plot_llama():
    os.makedirs(config.PLOT_DIR_LLAMA, exist_ok=True)

    linux = sns.color_palette(config.LINUX_COLOR, n_colors=5)
    spare = sns.color_palette(config.SPARE_COLOR, n_colors=9)
    palettes = {
        "baseline": linux[0],
        "distribute": linux[1],
        "interleaved": linux[2],
        "repl": spare[7],
        "repl-distribute": spare[7],
    }

    df_all = load()
    if df_all.empty:
        return

    sns.set_style(style="ticks")
    sns.set_context("paper")
    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})

    for arch in df_all["arch"].unique():
        arch_data = df_all[df_all["arch"] == arch]

        fig, axes = plt.subplots(
            nrows=1, ncols=len(TESTS), figsize=(3.3, 1.47)
        )

        for ax, test in zip(axes, TESTS):
            test_data = arch_data[arch_data["test"] == test]

            base = test_data[test_data["tag"] == "baseline"]["avg_ts"]
            base = base.iloc[0] if len(base) else None

            for tag in TAGS:
                row = test_data[test_data["tag"] == tag]
                if row.empty:
                    continue
                mean = row.iloc[0]["avg_ts"]
                std = row.iloc[0]["stddev_ts"]

                ax.bar(
                    POSITIONS[tag],
                    mean,
                    width=0.8,
                    color=palettes[tag],
                    edgecolor=palettes[tag],
                    yerr=std,
                    capsize=0.6,
                    error_kw=dict(
                        lw=0.3, capthick=0.3, color="gray", alpha=0.5
                    ),
                    linewidth=0.25,
                )

                if base is None or tag == "baseline":
                    continue

                pct = 100 * (mean - base) / base
                ax.text(
                    POSITIONS[tag],
                    mean + std + 0.02 * mean,
                    f"{pct:+.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=3,
                    color="green" if pct > 0 else "red",
                )

            sns.despine(ax=ax)
            ax.tick_params(axis="y", labelsize=6, length=2)
            ax.tick_params(axis="x", labelsize=6, length=0)

            ax.set_xticks([POSITIONS[t] for t in TAGS])
            ax.set_xticklabels(
                [TAG_LABELS[t] for t in TAGS],
                fontsize=4,
                rotation=45,
                ha="right",
            )
            ax.set_xlabel(TEST_LABELS[test], fontsize=6)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

        axes[0].set_ylabel("Throughput (tokens/s)", fontsize=7)

        fig.tight_layout(pad=0, w_pad=0.6)
        path = os.path.join(
            config.PLOT_DIR_LLAMA, f"{config.ARCH_SUBNAMES[arch]}_llama.pdf"
        )
        plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()
