import os
import config
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
import re

RESULT_DIR = config.RESULT_DIR
pattern = re.compile(
    r"(?P<benchmark>\w+)"  # pgtable, mem, alloc, etc.
    r"_(?P<method>\w+)"  # mmap, madvise, main, etc.
    r"_repl_(?P<replication>[01])"  # replication enabled (0 or 1)
    r"_(?P<threads>\d+)\.csv"  # number of threads
)


def make_plot_microbench():
    os.makedirs(config.PLOT_DIR_MICROBENCH, exist_ok=True)
    for arch in os.listdir(RESULT_DIR):
        if arch != "IntelR_XeonR_Gold_6130_CPU_@_2.10GHz_X86_64":
            continue

        microbench_dir = os.path.join(RESULT_DIR, arch, "microbench")
        if not os.path.isdir(microbench_dir):
            continue
        make_plot_microbench_arch(arch)


def make_plot_microbench_arch(arch: str):
    combined_df = get_data(arch)

    elapsed_y_label = [
        "SPaRe",
        "Carrefour",
    ]

    df_pg = combined_df[combined_df["benchmark"] == "pgtable"]
    tag_labels_pg = {
        "pgtable_norepl_default": "NoRepl",
        "pgtable_repl_repl": "Repl",
        "pgtable_norepl_repl": "NoReplAfter",
    }
    tags_pg = [
        "pgtable_norepl_default",
        "pgtable_repl_repl",
        "pgtable_norepl_repl",
    ]
    plot_microbench(
        arch,
        "pgtable",
        df_pg,
        "elapsed_ms",
        tag_labels_pg,
        tags_pg,
        ylabels=elapsed_y_label,
    )

    df_alloc = combined_df[combined_df["benchmark"] == "alloc"]
    tag_labels_alloc = {
        "alloc_default": "Default",
        "alloc_repl_repl": "Replication",
    }
    tags_alloc = [
        "alloc_default",
        "alloc_repl_repl",
    ]
    plot_microbench(
        arch,
        "alloc",
        df_alloc,
        "elapsed_ms",
        tag_labels_alloc,
        tags_alloc,
        ylabels=elapsed_y_label,
    )

    df_mem = combined_df[combined_df["benchmark"] == "mem"]
    df_mem["unique_tag"] = df_mem["tag"] + "_" + df_mem["method_tag"]
    agg_mem = (
        df_mem.groupby(["method", "threads", "unique_tag"])["mem_used"]
        .max()
        .reset_index()
        .sort_values(["method", "threads"])
    )
    tag_labels = {
        "mem_repl_main": "Base Memory",
        "mem_repl_repl_main": "Replication Memory",
        "mem_main_repl_repl": "Main + Replication Memory",
    }
    agg_mem["tag_label"] = agg_mem["unique_tag"].map(tag_labels)
    print(agg_mem)


def get_data(arch: str) -> pd.DataFrame:
    data = []

    dirs = [
        os.path.join(RESULT_DIR, arch, "microbench"),
        os.path.join(RESULT_DIR, arch, "microbench", "carrefour"),
    ]

    for d in dirs:
        for fname in os.listdir(d):
            if not fname.endswith(".csv"):
                continue

            match = pattern.match(fname)
            if not match:
                print(f"Skipping unknown file format: {fname}")
                continue

            benchmark = match["benchmark"]
            nthreads = match["threads"]
            method = match["method"]
            is_repl = bool(int(match["replication"]))
            is_main = False
            if "_main" in benchmark:
                benchmark = benchmark.replace("_main", "")
                is_main = True

            path = os.path.join(d, fname)
            df = pd.read_csv(path)
            df["benchmark"] = benchmark
            df["method"] = method
            df["threads"] = nthreads
            df["method_tag"] = "repl" if is_repl else "default"
            if is_main:
                df["method_tag"] = "repl_main"

            data.append(df)

    combined_df = pd.concat(data, ignore_index=True)
    return combined_df


def plot_microbench(
    arch,
    title,
    df_param,
    value_col,
    tag_labels,
    tags,
    ylabels,
    showText=True,
    y_fontsize=10,
    tick_fontsize=8,
):
    df = df_param.copy()
    df["threads"] = df["threads"].astype(int)
    df["unique_tag"] = df["tag"] + "_" + df["method_tag"]

    threads = sorted(df["threads"].unique())
    n_threads = len(threads)
    methods = ["mmap", "madvise"]

    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})

    fig, axes = plt.subplots(
        nrows=len(methods),
        ncols=n_threads,
        figsize=(3.2, 1.2),
        sharey="row",
        gridspec_kw={"wspace": 0.05, "hspace": 0.85},
    )

    palette = sns.color_palette("Blues", n_colors=3)

    sns.set_style("ticks")
    sns.set_context("paper")

    if len(methods) == 1:
        axes = [axes]
    if n_threads == 1:
        axes = [[ax] for ax in axes]

    for midx, method in enumerate(methods):
        df_m = df[df["method"] == method]

        for tidx, t in enumerate(threads):
            ax = axes[midx][tidx]
            df_t = df_m[df_m["threads"] == t]

            if df_t.empty:
                ax.axis("off")
                continue

            agg = df_t.groupby("unique_tag")[value_col].agg(["mean", "std"])
            mean_col = "mean_norm"
            std_col = "std_norm"

            default_val = agg.loc[tags[0], "mean"]
            agg["mean_norm"] = agg["mean"] / default_val
            std_raw = df_t.groupby("unique_tag")["elapsed_ms"].std()
            agg["std_norm"] = std_raw / default_val

            agg = agg.reindex(tags, fill_value=0)
            ntags = 0

            xpos = np.arange(len(tags))
            bar_width = 0.22

            for i, tag in enumerate(tags):
                height = agg.loc[tag, mean_col]
                yerr_val = agg.loc[tag, std_col]
                ntags += 1

                bar = ax.bar(
                    xpos[i],
                    height,
                    width=1.0,
                    # yerr=yerr_val,
                    label=tag_labels.get(tag, tag),
                    capsize=1.0,
                    color=palette[i],
                    edgecolor=palette[i],
                    linewidth=0.25,
                    # error_kw=dict(lw=0.6, capthick=0.6),
                    zorder=2,
                )

                if showText:
                    ax.text(
                        bar[0].get_x() + bar[0].get_width() / 2,
                        height + 0.02,
                        f"x{height:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        color="black",
                        zorder=3,
                    )

            sns.despine(ax=ax, left=tidx != 0)
            if tidx != 0:
                ax.tick_params(
                    labelleft=False,
                    left=False,
                )

            ax.set_xticks([])
            ax.set_xlabel("")
            # ax.grid(
            #     axis="y",
            #     which="major",
            #     linestyle="--",
            #     linewidth=0.4,
            #     color="gray",
            #     alpha=0.3,
            #     zorder=1,
            # )

            xpos = 0.5 if ntags == 2 else 1
            ax.set_xticks([xpos])

            ax.tick_params(axis="y", labelsize=6, length=2)
            ax.tick_params(axis="x", labelsize=6, length=2)

            ax.set_xticklabels([str(t)], fontsize=7)

            ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=4))
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

    for midx, ylabel in enumerate(ylabels):
        axes[midx][0].set_ylabel(ylabel, fontsize=7)
        axes[midx][0].yaxis.set_label_coords(-0.50, 0.5)  # align vertically

    # handles, labels = axes[0][0].get_legend_handles_labels()
    # legend = fig.legend(
    #     handles,
    #     labels,
    #     fontsize=8,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.0),
    #     edgecolor="white",
    #     framealpha=1.0,
    #     ncol=len(handles),
    #     frameon=False,
    # )
    # legend.get_frame().set_linewidth(0.4)

    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    path = os.path.join(config.PLOT_DIR_MICROBENCH, arch)
    plt.savefig(f"{path}_{title}.svg", bbox_inches="tight", dpi=300)
    plt.savefig(f"{path}_{title}.png", bbox_inches="tight", dpi=300)
