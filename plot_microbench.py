import os
import config
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
import re

palettes = [
    sns.color_palette(config.CARREFOUR_COLOR, n_colors=9)[3],
    sns.color_palette(config.SPARE_COLOR, n_colors=9)[7],
]

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

    df_pg = combined_df[combined_df["benchmark"] == "pgtable"]
    plot_microbench_sync(
        arch,
        df_pg,
    )

    df_alloc = combined_df[combined_df["benchmark"] == "alloc"]
    plot_microbench_alloc(
        arch,
        df_alloc,
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


def plot_microbench_sync(
    arch,
    df_param,
    showText=True,
):
    df = df_param.copy()
    df["threads"] = df["threads"].astype(int)
    df["unique_tag"] = df["tag"] + "_" + df["method_tag"]

    threads = sorted(df["threads"].unique())
    n_threads = len(threads)
    methods = ["madvise", "mmap"]

    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})
    sns.set_style("ticks")
    sns.set_context("paper")
    tags = [
        "pgtable_norepl_repl",
        "pgtable_repl_repl",
    ]

    for midx, tag in enumerate(tags):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_threads,
            figsize=(3.2, 1.1),
            sharey=True,
            gridspec_kw={"wspace": 0.05},
        )

        for tidx, t in enumerate(threads):
            ax = axes[tidx]
            df_t0 = df[df["threads"] == t]

            if df_t0.empty:
                ax.axis("off")
                continue

            for i, method in enumerate(methods):
                df_t = df_t0[df_t0["method"] == method]

                agg = df_t.groupby("unique_tag")["elapsed_ms"].agg(
                    ["mean", "std"]
                )
                mean_col = "mean_norm"
                std_col = "std_norm"

                default_val = agg.loc["pgtable_norepl_default", "mean"]
                agg["mean_norm"] = agg["mean"] / default_val
                agg["std_norm"] = agg["std"] / default_val

                agg = agg.reindex(tags, fill_value=0)
                ntags = 0

                xpos = np.arange(len(tags))
                height = agg.loc[tag, mean_col]
                yerr_val = agg.loc[tag, std_col]
                ntags += 1

                bar = ax.bar(
                    xpos[i],
                    height,
                    width=1.0,
                    label=method,
                    yerr=yerr_val,
                    capsize=1.0,
                    color=palettes[i],
                    edgecolor=palettes[i],
                    error_kw=dict(lw=0.5, capthick=0.5),
                    linewidth=0.25,
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

            sns.despine(ax=ax, left=True)
            if tidx != 0:
                ax.tick_params(
                    labelleft=False,
                    left=False,
                )

            ax.set_xticks([0.5])
            ax.tick_params(axis="y", labelsize=6, length=2)
            ax.set_xticklabels(
                [f"{str(t)} node{'s' if t > 1 else ''}"], fontsize=6
            )

            ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
            ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
            ax.set_ylim(1.0, 22)

        axes[0].yaxis.set_visible(False)

        if tag == "pgtable_norepl_repl":
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                ["Carrefour", "SPaRe"],
                bbox_to_anchor=(0.5, 1.0),
                loc="upper center",
                fontsize=8,
                ncol=len(handles),
                edgecolor="none",
            )

        path = os.path.join(
            config.PLOT_DIR_MICROBENCH, config.ARCH_SUBNAMES[arch]
        )
        plt.savefig(
            f"{path}_{tag}.pdf", bbox_inches="tight", pad_inches=0, dpi=300
        )
        # plt.savefig(
        #     f"{path}_{tag}.png", bbox_inches="tight", pad_inches=0, dpi=300
        # )
        plt.close(fig)

    # axes[0][0].set_ylabel("not repl", fontsize=7)
    # axes[0][0].yaxis.set_label_coords(-0.50, 0.5)  # align vertically
    # axes[1][0].set_ylabel("repl", fontsize=7)
    # axes[1][0].yaxis.set_label_coords(-0.50, 0.5)  # align vertically
    # axes[0][0].yaxis.set_visible(False)
    # axes[1][0].yaxis.set_visible(False)

    # # handles, labels = axes[0][0].get_legend_handles_labels()
    # # legend = fig.legend(
    # #     handles,
    # #     labels,
    # #     fontsize=8,
    # #     loc="upper center",
    # #     bbox_to_anchor=(0.5, 1.0),
    # #     edgecolor="white",
    # #     framealpha=1.0,
    # #     ncol=len(handles),
    # #     frameon=False,
    # # )
    # # legend.get_frame().set_linewidth(0.4)

    # plt.subplots_adjust(top=0.9)
    # path = os.path.join(config.PLOT_DIR_MICROBENCH, arch)
    # plt.savefig(
    #     f"{path}_pgtable.svg", bbox_inches="tight", pad_inches=0, dpi=300
    # )
    # plt.savefig(
    #     f"{path}_pgtable.png", bbox_inches="tight", pad_inches=0, dpi=300
    # )


# def plot_microbench_sync(
#     arch,
#     df_param,
#     showText=True,
# ):
#     ylabels = [
#         "SPaRe",
#         "Carrefour",
#     ]

#     tags = [
#         "pgtable_norepl_default",
#         "pgtable_repl_repl",
#         "pgtable_norepl_repl",
#     ]

#     tag_labels = {
#         "pgtable_norepl_default": "NoRepl",
#         "pgtable_repl_repl": "Repl",
#         "pgtable_norepl_repl": "NoReplAfter",
#     }

#     df = df_param.copy()
#     df["threads"] = df["threads"].astype(int)
#     df["unique_tag"] = df["tag"] + "_" + df["method_tag"]

#     threads = sorted(df["threads"].unique())
#     n_threads = len(threads)
#     methods = ["madvise", "mmap"]

#     plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})

#     fig, axes = plt.subplots(
#         ncols=n_threads,
#         figsize=(3.2, 1.2),
#         sharey=True,
#         gridspec_kw={"wspace": 0.05},
#     )

#     palettes = [
#         sns.color_palette("RdPu", n_colors=1),
#         sns.color_palette("Blues", n_colors=1),
#     ]
#     hatches = ["xxxxxxx", "......"]

#     sns.set_style("ticks")
#     sns.set_context("paper")
#     plt.rcParams["hatch.linewidth"] = 0.4

#     if len(methods) == 1:
#         axes = [axes]

#     for tidx, t in enumerate(threads):
#         ax = axes[tidx]
#         df_t0 = df[df["threads"] == t]

#         if df_t0.empty:
#             ax.axis("off")
#             continue

#         for i, method in enumerate(methods):
#             df_t = df_t0[df_t0["method"] == method]

#             agg = df_t.groupby("unique_tag")["elapsed_ms"].agg(["mean", "std"])
#             mean_col = "mean_norm"

#             default_val = agg.loc[tags[0], "mean"]
#             agg["mean_norm"] = agg["mean"] / default_val
#             std_raw = df_t.groupby("unique_tag")["elapsed_ms"].std()
#             agg["std_norm"] = std_raw / default_val

#             agg = agg.reindex(tags, fill_value=0)
#             ntags = 0
#             bar_width = 0.2
#             gap = 0.05

#             group_center = [-(bar_width + gap / 2), (bar_width + gap / 2)]

#             for ti, tag in enumerate(
#                 ["pgtable_repl_repl", "pgtable_norepl_repl"]
#             ):
#                 height = agg.loc[tag, mean_col]
#                 ntags += 1

#                 group_offset = (ti - 0.5) * bar_width
#                 x = group_center[i] + group_offset

#                 if ti == 0:
#                     bar = ax.bar(
#                         x,
#                         height,
#                         width=bar_width,
#                         label=tag_labels.get(tag, tag),
#                         capsize=1.0,
#                         linewidth=0.25,
#                         color=palettes[i],
#                         edgecolor=palettes[i],
#                         zorder=1.0,
#                     )
#                 else:
#                     bar = ax.bar(
#                         x,
#                         height,
#                         width=bar_width,
#                         label=tag_labels.get(tag, tag),
#                         capsize=1.0,
#                         linewidth=0.25,
#                         facecolor="none",
#                         hatch=hatches[0],
#                         color=palettes[i],
#                         edgecolor=palettes[i],
#                         zorder=1.0,
#                     )

#                 if showText:
#                     ax.text(
#                         bar[0].get_x() + bar[0].get_width() / 2,
#                         height + 0.02,
#                         f"x{height:.1f}",
#                         ha="center",
#                         va="bottom",
#                         fontsize=4,
#                         fontweight="bold",
#                         color="black",
#                         zorder=3,
#                     )

#         sns.despine(ax=ax, left=tidx != 0)
#         if tidx != 0:
#             ax.tick_params(
#                 labelleft=False,
#                 left=False,
#             )

#         ax.set_xticks([0])

#         ax.tick_params(axis="y", labelsize=6, length=2)
#         ax.tick_params(axis="x", labelsize=6, length=2)
#         ax.set_xticklabels([f"{str(t)} node{'s' if t > 1 else ''}"], fontsize=7)

#         ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=4))
#         ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

#     # for midx, ylabel in enumerate(ylabels):
#     #     axes[midx][0].set_ylabel(ylabel, fontsize=7)
#     #     axes[midx][0].yaxis.set_label_coords(-0.50, 0.5)  # align vertically

#     # handles, labels = axes[0][0].get_legend_handles_labels()
#     # legend = fig.legend(
#     #     handles,
#     #     labels,
#     #     fontsize=8,
#     #     loc="upper center",
#     #     bbox_to_anchor=(0.5, 1.0),
#     #     edgecolor="white",
#     #     framealpha=1.0,
#     #     ncol=len(handles),
#     #     frameon=False,
#     # )
#     # legend.get_frame().set_linewidth(0.4)

#     fig.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     path = os.path.join(config.PLOT_DIR_MICROBENCH, arch)
#     plt.savefig(f"{path}_pgtable.svg", bbox_inches="tight", dpi=300)
#     plt.savefig(f"{path}_pgtable.png", bbox_inches="tight", dpi=300)


def plot_microbench_alloc(
    arch,
    df_param,
    showText=True,
):
    tags = [
        "alloc_default",
        "alloc_repl_repl",
    ]

    tag_labels = {
        "alloc_default": "Default",
        "alloc_repl_repl": "Replication",
    }

    df = df_param.copy()
    df["threads"] = df["threads"].astype(int)
    df["unique_tag"] = df["tag"] + "_" + df["method_tag"]

    threads = sorted(df["threads"].unique())
    n_threads = len(threads)
    methods = ["madvise", "mmap"]

    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_threads,
        figsize=(3.2, 1.1),
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )

    sns.set_style("ticks")
    sns.set_context("paper")

    if len(methods) == 1:
        axes = [axes]

    for tidx, t in enumerate(threads):
        ax = axes[tidx]
        df_t0 = df[df["threads"] == t]

        if df_t0.empty:
            ax.axis("off")
            continue

        for i, method in enumerate(methods):
            df_t = df_t0[df_t0["method"] == method]

            agg = df_t.groupby("unique_tag")["elapsed_ms"].agg(["mean", "std"])
            mean_col = "mean_norm"
            std_col = "std_norm"

            default_val = agg.loc[tags[0], "mean"]
            agg["mean_norm"] = agg["mean"] / default_val
            std_raw = df_t.groupby("unique_tag")["elapsed_ms"].std()
            agg["std_norm"] = std_raw / default_val

            agg = agg.reindex(tags, fill_value=0)
            ntags = 0

            bar_width = 0.3
            gap = 0.0

            group_center = [-(gap / 2), (gap / 2)]

            tag = "alloc_repl_repl"
            height = agg.loc[tag, mean_col]
            yerr_val = agg.loc[tag, std_col]
            ntags += 1

            group_offset = (i - 0.5) * bar_width
            x = group_center[i] + group_offset

            bar = ax.bar(
                x,
                height,
                width=bar_width,
                yerr=yerr_val,
                label=tag_labels.get(tag, tag),
                capsize=1.0,
                color=palettes[i],
                edgecolor=palettes[i],
                linewidth=0.25,
                error_kw=dict(lw=0.6, capthick=0.6),
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

        sns.despine(ax=ax, left=True)
        if tidx != 0:
            ax.tick_params(
                labelleft=False,
                left=False,
            )

        ax.set_xticks([0])

        ax.tick_params(axis="y", labelsize=6, length=2)
        ax.set_xticklabels([f"{str(t)} node{'s' if t > 1 else ''}"], fontsize=6)

        ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

    # axes[0].set_ylabel("repl", fontsize=7)
    axes[0].yaxis.set_visible(False)
    axes[0].yaxis.set_label_coords(-0.50, 0.5)  # align vertically
    axes[0].set_ylim(1.0, 22)

    # for midx, ylabel in enumerate(ylabels):
    #     axes[midx][0].set_ylabel(ylabel, fontsize=7)
    #     axes[midx][0].yaxis.set_label_coords(-0.50, 0.5)  # align vertically

    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    path = os.path.join(config.PLOT_DIR_MICROBENCH, config.ARCH_SUBNAMES[arch])
    # plt.savefig(f"{path}_alloc.svg", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.savefig(f"{path}_alloc.pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    # plt.savefig(f"{path}_alloc.png", bbox_inches="tight", pad_inches=0, dpi=300)

    handles, labels = axes[0].get_legend_handles_labels()
    fig_legend = plt.figure(figsize=(3.3, 0.5))
    legend = fig_legend.legend(
        handles,
        ["Carrefour", "SPaRe"],
        fontsize=8,
        edgecolor="white",
        framealpha=1.0,
        ncol=len(handles),
    )
    fig_legend.subplots_adjust(left=0, right=1, top=1, bottom=0)
    path = os.path.join(config.PLOT_DIR_MICROBENCH, "legend")
    # plt.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig_legend)
