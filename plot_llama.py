import os
import config
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd


RESULT_DIR = config.RESULT_DIR

# prompt processing then token generation, llama-bench's two default tests
TESTS = [("pp512", "Prompt (pp512)"), ("tg128", "Generation (tg128)")]

# bench_llama writes one csv per kernel variant
CSVS = ["llama", "llama-repl"]

# the reference of the percentages
BASELINE = "baseline"

# bottom to top, baseline first
TAGS = ["baseline", "distribute", "interleaved", "repl", "repl-distribute"]
TAG_LABELS = {
    "baseline": "Baseline",
    "distribute": "Distribute",
    "interleaved": "Interleaved",
    "repl": "Replication",
    "repl-distribute": "Replication + Distribute",
}


def load():
    rows = []
    for arch in os.listdir(RESULT_DIR):
        arch_dir = os.path.join(RESULT_DIR, arch, "llama")
        if not os.path.isdir(arch_dir):
            continue

        for name in CSVS:
            csv_path = os.path.join(arch_dir, f"{name}.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                rows.append({
                    "arch": arch,
                    "tag": row["tag"],
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
        "repl": spare[5],
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

        # one cluster per test, stacked: the two differ by 10x in tokens/s and
        # would not share an axis
        fig, axes = plt.subplots(
            nrows=len(TESTS), ncols=1, figsize=(3.3, 2.0), sharey=True
        )

        for ax, (test, test_label) in zip(axes, TESTS):
            test_data = arch_data[arch_data["test"] == test].set_index("tag")
            tags = [t for t in TAGS if t in test_data.index]
            if not tags:
                continue

            base = test_data.loc[BASELINE, "avg_ts"] if BASELINE in tags else None
            means = [test_data.loc[t, "avg_ts"] for t in tags]

            ax.barh(
                range(len(tags)), means,
                height=0.7,
                color=[palettes[t] for t in tags],
                edgecolor=[palettes[t] for t in tags],
                xerr=[test_data.loc[t, "stddev_ts"] for t in tags],
                capsize=0.6,
                error_kw=dict(lw=0.3, capthick=0.3, color="gray", alpha=0.5),
                linewidth=0.25,
            )

            for i, tag in enumerate(tags):
                if base is None or tag == BASELINE:
                    continue
                pct = 100 * (means[i] - base) / base
                ax.text(
                    means[i] + test_data.loc[tag, "stddev_ts"],
                    i,
                    f"  {pct:+.1f}%",
                    ha="left", va="center", fontsize=4,
                    color="green" if pct > 0 else "red",
                )

            sns.despine(ax=ax)
            ax.set_yticks(range(len(tags)))
            ax.set_yticklabels([TAG_LABELS[t] for t in tags], fontsize=4)
            ax.tick_params(axis="x", labelsize=6, length=2)
            ax.tick_params(axis="y", length=0)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            # room for the percentage at the right of the longest bar
            ax.set_xlim(0, max(means) * 1.25)
            ax.set_xlabel(f"{test_label} throughput (tokens/s)", fontsize=6)

        fig.tight_layout(pad=0, h_pad=0.8)
        path = os.path.join(
            config.PLOT_DIR_LLAMA, f"{config.ARCH_SUBNAMES[arch]}_llama.pdf"
        )
        plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=300)
        plt.close()
