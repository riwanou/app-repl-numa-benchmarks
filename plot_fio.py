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
    make_plot_fio_arch("INTELR_XEONR_PLATINUM_8568Y+_X86_64")


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
        ylabel="Read Bandwidth (GB/s)",
    )
    plot_fio(
        arch,
        "write",
        combined_df,
        value_col="write_bw_gb",
        ylabel="Write Bandwidth (GB/s)",
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

        if distrib != "zipf":
            continue

        path = os.path.join(dir, fname)
        if os.path.getsize(path) == 0:
            continue
        with open(path, "r") as f:
            json_data = json.load(f)

        read_bw_gb = 0
        write_bw_gb = 0

        for job in json_data.get("jobs", []):
            read_bw_gb = job.get("read", {}).get("bw_bytes", 0) / 1000_000_000
            write_bw_gb = job.get("write", {}).get("bw_bytes", 0) / 1000_000_000

        tag = ""
        if is_repl:
            tag = "repl"
        if is_unrepl:
            tag = "unrepl"

        df = pd.DataFrame(
            [
                {
                    "readratio": readratio,
                    "writeratio": writeratio,
                    "read_bw_gb": read_bw_gb,
                    "write_bw_gb": write_bw_gb,
                    "benchmark": f"{benchmark}_{distrib}",
                    "tag": tag,
                }
            ]
        )
        data.append(df)

    combined_df = pd.concat(data, ignore_index=True)
    return combined_df


def plot_fio(arch, title, df_param, value_col, ylabel):
    df = df_param.copy()
    df["readratio"] = df["readratio"].astype(int)

    read_ratios = sorted(df["readratio"].unique())
    x = np.arange(len(read_ratios))
    width = 0.25

    df_repl = df[df["tag"] == "repl"].set_index("readratio")
    df_unrepl = df[df["tag"] == "unrepl"].set_index("readratio")
    df_normal = df[df["tag"] == ""].set_index("readratio")

    read_bw_normal = [
        df_normal.loc[r, value_col] if r in df_normal.index else 0
        for r in read_ratios
    ]
    read_bw_repl = [
        df_repl.loc[r, value_col] if r in df_repl.index else 0
        for r in read_ratios
    ]
    read_bw_unrepl = [
        df_unrepl.loc[r, value_col] if r in df_unrepl.index else 0
        for r in read_ratios
    ]

    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})

    sns.set_style("ticks")
    sns.set_context("paper")
    fig, ax = plt.subplots(
        figsize=(3.3, 1.2),
    )

    palette = sns.color_palette("Blues", n_colors=3)

    ax.bar(
        x - 1.5 * width,
        read_bw_normal,
        width,
        label="Default",
        capsize=3,
        color=palette[0],
        edgecolor=palette[0],
        linewidth=0.3,
        zorder=2,
    )
    ax.bar(
        x - 0.5 * width,
        read_bw_repl,
        width,
        label="Replication",
        capsize=3,
        color=palette[1],
        edgecolor=palette[1],
        linewidth=0.3,
        zorder=2,
    )
    ax.bar(
        x + 0.5 * width,
        read_bw_unrepl,
        width,
        label="UnReplication",
        capsize=3,
        color=palette[2],
        edgecolor=palette[2],
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

    # handles, labels = ax.get_legend_handles_labels()
    # legend = fig.legend(
    #     handles,
    #     labels,
    #     fontsize=8,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.0),
    #     ncol=2,
    #     edgecolor="white",
    #     framealpha=1.0,
    # )
    # legend.get_frame().set_linewidth(0.4)

    fig.tight_layout()
    path = os.path.join(config.PLOT_DIR_FIO, arch)
    plt.savefig(
        f"{path}_{title}.svg", bbox_inches="tight", pad_inches=0, dpi=300
    )
