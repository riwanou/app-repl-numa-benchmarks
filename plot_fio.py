from multiprocessing.context import BaseContext
import os
import config
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import re
import seaborn as sns
import pandas as pd
import numpy as np
import json

RESULT_DIR = config.RESULT_DIR
pattern = re.compile(
    r"(?P<benchmark>\w+)"  # readwrite, read
    r"_(?P<distrib>\w+)"  # random, zipf
    r"_(?P<readratio>\d+)"  # number of reader jobs
    r"_(?P<writeratio>\d+)"  # number of writer jobs
    r"(?P<repl>_repl)?"  # matches "_repl" if present
    r"(?P<unrepl>_unrepl)?"  # matches "_unrepl" if present
    r"(?P<default>_default)?"  # matches "_default" if present
    r"\.json$"  # exactly one .json at the end
)


def make_plot_fio():
    os.makedirs(config.PLOT_DIR_FIO, exist_ok=True)

    # for arch in os.listdir(config.RESULT_DIR):
    #     arch_dir = os.path.join(RESULT_DIR, arch)
    #     if not os.path.isdir(arch_dir):
    #         continue
    #     make_plot_fio_arch(arch)

    make_plot_fio_arch("IntelR_XeonR_Silver_4216_CPU_@_2.10GHz_X86_64")
    make_plot_fio_arch("IntelR_XeonR_Gold_6130_CPU_@_2.10GHz_X86_64")
    # make_plot_fio_arch("INTELR_XEONR_PLATINUM_8568Y+_X86_64")


def make_plot_fio_arch(arch):
    combined_df = get_data(arch)
    combined_df = combined_df.sort_values(
        by=["tag", "read_bw_gb"], ascending=False
    ).reset_index(drop=True)
    # print(combined_df)

    plot_fio(
        arch,
        "read",
        combined_df,
        value_col="read_bw_gb",
        std_col="read_bw_std",
        ylabel="$\mathbf{Read}$ Bandwidth (GB/s)",
        is_write=False,
    )
    plot_fio(
        arch,
        "write",
        combined_df,
        value_col="write_bw_gb",
        std_col="write_bw_std",
        ylabel="$\mathbf{Write}$ Bandwidth (GB/s)",
        is_write=True,
    )


def get_data(arch: str) -> pd.DataFrame:
    data = []
    dir = os.path.join(RESULT_DIR, arch, "fio")

    for fname in os.listdir(dir):
        if not fname.endswith(".json"):
            continue

        match = pattern.match(fname)
        if not match:
            print(f"Skipping unknown file format: {fname}")
            continue

        benchmark = match["benchmark"]
        distrib = match["distrib"]
        readratio = match["readratio"]
        writeratio = match["writeratio"]
        is_repl = match["repl"] is not None
        is_unrepl = match["unrepl"] is not None
        is_default = match["default"] is not None

        if distrib != "random":
            continue

        path = os.path.join(dir, fname)
        if os.path.getsize(path) == 0:
            continue
        try:
            with open(path, "r") as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        read_bw_gb = 0
        write_bw_gb = 0

        for job in json_data.get("jobs", []):
            read_stats = job.get("read", {})
            write_stats = job.get("write", {})

            read_bw_gb = read_stats.get("bw_bytes", 0) / (1000**3)
            write_bw_gb = write_stats.get("bw_bytes", 0) / (1000**3)
            # read_bw_gb = read_stats.get("bw_mean", 0) / (1000**2)
            # write_bw_gb = write_stats.get("bw_mean", 0) / (1000**2)

            rbm = read_stats.get("bw_mean", 0) / (1000**2)
            rbd = read_stats.get("bw_dev", 0) / (1000**2)
            wbm = write_stats.get("bw_mean", 0) / (1000**2)
            wbd = write_stats.get("bw_dev", 0) / (1000**2)

            read_bw_std_pct = (rbd / rbm) * 100 if rbm > 0 else 0
            write_bw_std_pct = (wbd / wbm) * 100 if wbm > 0 else 0

        tag = ""
        if is_repl:
            tag = "repl"
        if is_unrepl:
            tag = "unrepl"
        if is_default:
            tag = "default"

        df = pd.DataFrame(
            [
                {
                    "readratio": readratio,
                    "writeratio": writeratio,
                    "read_bw_gb": read_bw_gb,
                    "write_bw_gb": write_bw_gb,
                    "read_bw_std": rbd,
                    "write_bw_std": wbd,
                    "read_bw_std_pct": read_bw_std_pct,
                    "write_bw_std_pct": write_bw_std_pct,
                    "benchmark": f"{benchmark}_{distrib}",
                    "tag": tag,
                }
            ]
        )
        data.append(df)

    combined_df = pd.concat(data, ignore_index=True)

    print(
        f"Avg std percent — Read:  {combined_df['read_bw_std_pct'].mean():.3f}%  Write: {combined_df['write_bw_std_pct'].mean():.3f}%"
    )

    return combined_df


def plot_fio(arch, title, df_param, value_col, std_col, ylabel, is_write=False):
    df = df_param.copy()
    df["readratio"] = df["readratio"].astype(int)

    read_ratios = sorted(df["readratio"].unique())
    x = np.arange(len(read_ratios))
    width = 0.21

    df_repl = df[df["tag"] == "repl"].set_index("readratio")
    df_unrepl = df[df["tag"] == "unrepl"].set_index("readratio")
    df_default = df[df["tag"] == "default"].set_index("readratio")
    df_normal = df[df["tag"] == ""].set_index("readratio")

    read_bw_default = [
        df_default.loc[r, value_col] if r in df_default.index else 0
        for r in read_ratios
    ]
    read_std_default = [
        df_default.loc[r, std_col] if r in df_default.index else 0
        for r in read_ratios
    ]
    read_bw_normal = [
        df_normal.loc[r, value_col] if r in df_normal.index else 0
        for r in read_ratios
    ]
    read_std_normal = [
        df_normal.loc[r, std_col] if r in df_normal.index else 0
        for r in read_ratios
    ]
    read_bw_repl = [
        df_repl.loc[r, value_col] if r in df_repl.index else 0
        for r in read_ratios
    ]
    read_std_repl = [
        df_repl.loc[r, std_col] if r in df_repl.index else 0
        for r in read_ratios
    ]
    read_bw_unrepl = [
        df_unrepl.loc[r, value_col] if r in df_unrepl.index else 0
        for r in read_ratios
    ]
    read_std_unrepl = [
        df_unrepl.loc[r, std_col] if r in df_unrepl.index else 0
        for r in read_ratios
    ]

    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})

    sns.set_style("ticks")
    sns.set_context("paper")
    fig, ax = plt.subplots(
        figsize=(3.31, 1.65),
    )

    linux = sns.color_palette(config.LINUX_COLOR, n_colors=7)
    palette = sns.color_palette(config.SPARE_COLOR, n_colors=9)

    ax.bar(
        x - 1.5 * width,
        read_bw_default,
        width,
        # yerr=read_std_default,
        # error_kw=dict(lw=1, capthick=1),
        # capsize=0.3,
        label="Vanilla",
        color=linux[1],
        edgecolor=linux[1],
        linewidth=0.3,
        zorder=2,
    )
    ax.bar(
        x - 0.5 * width,
        read_bw_normal,
        width,
        # yerr=read_std_normal,
        # error_kw=dict(lw=1, capthick=1),
        # capsize=0.3,
        label="NumaBalancing",
        color=linux[3],
        edgecolor=linux[3],
        linewidth=0.3,
        zorder=2,
    )
    ax.bar(
        x + 0.5 * width,
        read_bw_repl,
        width,
        # yerr=read_std_repl,
        # error_kw=dict(lw=1, capthick=1),
        # capsize=0.3,
        label="SPaRe (No Unreplication)",
        color=palette[5],
        edgecolor=palette[5],
        linewidth=0.3,
        zorder=2,
    )
    ax.bar(
        x + 1.5 * width,
        read_bw_unrepl,
        width,
        # yerr=read_std_unrepl,
        # error_kw=dict(lw=1, capthick=1),
        # capsize=0.3,
        label="SPaRe",
        color=palette[7],
        edgecolor=palette[7],
        linewidth=0.3,
        zorder=2,
    )

    # ax.grid(
    #     axis="y",
    #     which="major",
    #     linestyle="--",
    #     linewidth=0.4,
    #     color="gray",
    #     alpha=0.3,
    #     zorder=1,
    # )

    sns.despine(ax=ax)

    ax.tick_params(axis="y", labelsize=6, length=2)
    ax.tick_params(axis="x", labelsize=6, length=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}%" for r in read_ratios])
    # ax.set_xlabel("Read Ratio (%)")
    ax.set_ylabel(ylabel, fontsize=7)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=6))
    ax.set_xlabel("Read Ratio (%)", fontsize=6)

    # legend inside figure
    ax.legend(
        loc="upper left" if not is_write else "upper right",
        fontsize=5,
        ncol=1,
        framealpha=0.8,
        edgecolor="none",
    )

    fig.tight_layout()
    path = os.path.join(config.PLOT_DIR_FIO, config.ARCH_SUBNAMES[arch])
    plt.savefig(
        f"{path}_{title}.pdf", bbox_inches="tight", pad_inches=0, dpi=300
    )

    handles, labels = ax.get_legend_handles_labels()
    fig_legend = plt.figure(figsize=(3.3, 0.5))
    legend = fig_legend.legend(
        handles,
        labels,
        fontsize=8,
        ncol=len(handles),
        edgecolor="white",
        framealpha=1.0,
    )

    fig_legend.subplots_adjust(left=0, right=1, top=1, bottom=0)
    path = os.path.join(config.PLOT_DIR_FIO, "legend")
    # plt.savefig(f"{path}.pdf", bbox_inches="tight", pad_inches=0, dpi=300)
    # plt.savefig(f"{path}.png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig_legend)
