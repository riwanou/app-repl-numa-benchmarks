import os
import config
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from plot_ann import _setup_style, _save_figure

# ann on usearch gist: one node against the whole machine, per placement
COMPARE = "locality-usearch-gist-local-vs-firsttouch-vs-balancing-vs-interleaved-vs-imbalanced-vs-repl.csv"
ROWS = ["local-1node", "imbalanced-allnodes", "firsttouch-allnodes", "balancing-allnodes", "interleaved-allnodes", "repl-allnodes"]
LABELS = ["1 node", "Imbalanced", "First Touch", "NumaBal.", "Interleave", "SPARe"]
ARCHS = [("Silver", "Silver x2"), ("Gold", "Gold x4")]
linux = sns.color_palette(config.LINUX_COLOR, n_colors=6)
spare = sns.color_palette(config.SPARE_COLOR, n_colors=9)
COLORS = [linux[4], linux[3], linux[0], linux[1], linux[2], spare[7]]


def make_plot_locality():
    os.makedirs(os.path.join(config.PLOT_DIR, "locality"), exist_ok=True)
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.15), gridspec_kw={"wspace": 0.25})
    for ax, (key, title) in zip(axes, ARCHS):
        arch = next(a for a in os.listdir(config.RESULT_DIR) if key in a)
        df = pd.read_csv(os.path.join(config.RESULT_DIR, arch, "stats", "compare", COMPARE), index_col=0)
        qps = df.loc["qps", ROWS].astype(float)
        loc = df.loc["local_pct", ROWS].astype(float)
        x = range(len(ROWS))
        ax.bar(x, qps / qps["local-1node"], color=COLORS, width=0.7)
        for i, r in enumerate(ROWS):
            ax.text(i, qps[r] / qps["local-1node"] + 0.05, f"{loc[r]:.0f}%", ha="center", va="bottom", fontsize=4.5)
        ax.set_title(title, fontsize=6.5, pad=8)
        ax.set_xticks(list(x), LABELS, fontsize=4.5, rotation=35, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="y", labelsize=5.5)
        ax.set_ylim(0, (qps / qps["local-1node"]).max() * 1.25)
        ax.axhline(1, color="grey", lw=0.4, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("QPS vs 1 node", fontsize=5.5)
    _save_figure(fig, os.path.join(config.PLOT_DIR, "locality", "locality.pdf"))


if __name__ == "__main__":
    make_plot_locality()
