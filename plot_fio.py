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
    r"(?:_run(?P<run>\d+))?"  # match _run<nb_run>
    r"\.json$"  # exactly one .json at the end
)

# bench_fio.py writes one jsonl per (workload, variant): the "-default" file
# holds the unreplicated runs, the "-repl" file the replicated ones. The tag of
# each individual run lives inside the record, not in the filename.
jsonl_pattern = re.compile(
    r"(?P<benchmark>[a-z]+)"  # readwrite, read
    r"_(?P<distrib>[a-z]+)"  # random, zipf
    r"_(?P<readratio>\d+)"  # read ratio (%)
    r"_(?P<writeratio>\d+)"  # write ratio (%)
    r"-(?P<variant>repl|default)"  # replicated or not
    r"\.jsonl$"
)

# tag (as written by bench_fio.run_one) -> legend label and (palette, index),
# ordered left to right. "linux" is the Oranges ramp, "spare" the Blues one.
SERIES = [
    ("numabalancing", "NUMA Balancing", ("linux", 3)),
    ("interleaved", "Interleaved", ("linux", 5)),
    ("repl", "SPARe (No Unreplication)", ("spare", 3)),
    ("unrepl-bound", "SPARe", ("spare", 5)),
    ("unrepl-interleaved", "SPARe (Interleaved)", ("spare", 8)),
]

# tags used by the older per-run .json result files
LEGACY_TAGS = {"": "numabalancing", "unrepl": "unrepl-bound"}

# jobs start staggered on every run, not just the first one
DROP_FIRST_RUN = False

# fio divides the whole group's io_bytes by *one* job's 30s runtime. On yeti4
# the jobs start staggered (elapsed up to 97s for a 51s run), so those bytes
# are really spread over a much longer window and the bandwidth comes out
# inflated, up to 2.3x at write-heavy ratios. Divide by the wall-clock span of
# the measured phase instead. tall4 is unaffected (<1%).
USE_WALL_BW = True


def make_plot_fio():
    os.makedirs(config.PLOT_DIR_FIO, exist_ok=True)

    for arch in os.listdir(RESULT_DIR):
        if arch not in config.ARCH_SUBNAMES:
            continue
        if not os.path.isdir(os.path.join(RESULT_DIR, arch, "fio")):
            continue
        make_plot_fio_arch(arch)


def make_plot_fio_arch(arch):
    combined_df = get_data(arch)
    if combined_df.empty:
        print(f"No random fio data for {arch}, skipping.")
        return

    agg_df = aggregate(combined_df)

    combined_df["readratio"] = combined_df["readratio"].astype(int)
    combined_df = combined_df.sort_values(
        by=["readratio", "tag", "run"], ascending=[False, True, True]
    ).reset_index(drop=True)

    # the csvs keep every run; only the bars drop the first one
    result_dir = os.path.join(RESULT_DIR, arch, "fio")
    combined_df.to_csv(os.path.join(result_dir, "details.csv"), index=False)
    agg_df.to_csv(os.path.join(result_dir, "agg.csv"), index=False)

    plot_df = agg_df
    # old results are a single run
    if DROP_FIRST_RUN and (combined_df["run"] > 1).any():
        plot_df = aggregate(combined_df[combined_df["run"] > 1])

    read_col = "read_bw_wall" if USE_WALL_BW else "read_bw"
    write_col = "write_bw_wall" if USE_WALL_BW else "write_bw"

    plot_fio(
        arch,
        "read",
        plot_df,
        value_col=f"{read_col}_gb",
        std_col=f"{read_col}_std",
        ylabel="$\mathbf{Read}$ Bandwidth (GB/s)",
        is_write=False,
    )
    plot_fio(
        arch,
        "write",
        plot_df,
        value_col=f"{write_col}_gb",
        std_col=f"{write_col}_std",
        ylabel="$\mathbf{Write}$ Bandwidth (GB/s)",
        is_write=True,
    )


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across the runs of each (tag, ratio, benchmark)."""
    agg_df = (
        df.groupby(["tag", "readratio", "writeratio", "benchmark"])
        .agg(
            read_bw_gb=("read_bw_gb", "mean"),
            write_bw_gb=("write_bw_gb", "mean"),
            read_bw_std=("read_bw_gb", "std"),  # std across runs
            write_bw_std=("write_bw_gb", "std"),  # std across runs
            read_bw_wall_gb=("read_bw_wall_gb", "mean"),
            write_bw_wall_gb=("write_bw_wall_gb", "mean"),
            read_bw_wall_std=("read_bw_wall_gb", "std"),
            write_bw_wall_std=("write_bw_wall_gb", "std"),
            wall_s=("wall_s", "mean"),
            nb_runs=("run", "count"),
        )
        .reset_index()
    )
    # std is NaN for a single run; matplotlib wants a number for yerr
    std_cols = [
        "read_bw_std", "write_bw_std", "read_bw_wall_std", "write_bw_wall_std",
    ]
    agg_df[std_cols] = agg_df[std_cols].fillna(0)
    agg_df["readratio"] = agg_df["readratio"].astype(int)
    return agg_df.sort_values(
        by=["readratio", "tag"], ascending=[False, True]
    ).reset_index(drop=True)


def wall_seconds(json_data) -> float:
    """Wall-clock span of the measured phase, in seconds: job_start is stamped
    when the first job leaves its ramp, timestamp_ms when fio writes the
    output. Wider than fio's runtime whenever the jobs are staggered."""
    jobs = json_data.get("jobs", [])
    if not jobs or "timestamp_ms" not in json_data:
        return 0.0
    return (json_data["timestamp_ms"] - jobs[0].get("job_start", 0)) / 1000


def bw_from_fio_output(json_data) -> dict:
    """Sum the bandwidth of every job group in one fio json output."""
    read_bytes = write_bytes = 0
    read_bw = write_bw = 0.0
    read_bw_mean = read_bw_dev = 0.0
    write_bw_mean = write_bw_dev = 0.0

    for job in json_data.get("jobs", []):
        read_stats = job.get("read", {})
        write_stats = job.get("write", {})

        read_bw += read_stats.get("bw_bytes", 0) / (1000**3)
        write_bw += write_stats.get("bw_bytes", 0) / (1000**3)

        read_bytes += read_stats.get("io_bytes", 0)
        write_bytes += write_stats.get("io_bytes", 0)

        read_bw_mean += read_stats.get("bw_mean", 0) / (1000**2)
        read_bw_dev += read_stats.get("bw_dev", 0) / (1000**2)
        write_bw_mean += write_stats.get("bw_mean", 0) / (1000**2)
        write_bw_dev += write_stats.get("bw_dev", 0) / (1000**2)

    wall = wall_seconds(json_data)

    return {
        "read_bw_gb": read_bw,
        "write_bw_gb": write_bw,
        # same bytes, divided by the real window every job ran in
        "read_bw_wall_gb": read_bytes / wall / (1000**3) if wall > 0 else 0,
        "write_bw_wall_gb": write_bytes / wall / (1000**3) if wall > 0 else 0,
        "wall_s": wall,
        "read_bw_std": read_bw_dev,
        "write_bw_std": write_bw_dev,
        "read_bw_std_pct": (read_bw_dev / read_bw_mean) * 100
        if read_bw_mean > 0
        else 0,
        "write_bw_std_pct": (write_bw_dev / write_bw_mean) * 100
        if write_bw_mean > 0
        else 0,
    }


def get_data_jsonl(dir: str) -> list:
    """Read the *-repl.jsonl / *-default.jsonl files written by bench_fio."""
    rows = []

    for fname in sorted(os.listdir(dir)):
        # pgtable_*.jsonl belongs to the microbench plots, not here
        if not fname.endswith(".jsonl") or not fname.startswith("readwrite_"):
            continue

        match = jsonl_pattern.match(fname)
        if not match:
            print(f"Skipping unknown jsonl file format: {fname}")
            continue
        if match["distrib"] != "random":
            continue

        path = os.path.join(dir, fname)
        with open(path) as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping {path}:{lineno}: {e}")
                    continue

                rows.append(
                    {
                        "readratio": match["readratio"],
                        "writeratio": match["writeratio"],
                        "run": int(record.get("run", 1)),
                        "benchmark": f"{match['benchmark']}_{match['distrib']}",
                        "tag": record.get("tag", ""),
                        # epoch seconds, as stamped on the bench machine.
                        # stats_monitoring turns them into the monitor's local
                        # time; converting here would use the *plotting*
                        # machine's timezone instead.
                        "ts_start": record.get("ts_start"),
                        "ts_end": record.get("ts_end"),
                        **bw_from_fio_output(record.get("data", {})),
                    }
                )

    return rows


def get_data(arch: str) -> pd.DataFrame:
    dir = os.path.join(RESULT_DIR, arch, "fio")

    rows = get_data_jsonl(dir)
    if rows:
        combined_df = pd.DataFrame(rows)
        print(
            f"Avg std percent — Read:  {combined_df['read_bw_std_pct'].mean():.3f}%"
            f"  Write: {combined_df['write_bw_std_pct'].mean():.3f}%"
        )
        return combined_df

    return get_data_legacy_json(dir)


def get_data_legacy_json(dir: str) -> pd.DataFrame:
    data = []

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
        run = match["run"] if match["run"] else "1"

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

        tag = ""
        if is_repl:
            tag = "repl"
        if is_unrepl:
            tag = "unrepl"
        if is_default:
            tag = "default"

        data.append(
            {
                "readratio": readratio,
                "writeratio": writeratio,
                "run": int(run),
                "benchmark": f"{benchmark}_{distrib}",
                "tag": LEGACY_TAGS.get(tag, tag),
                **bw_from_fio_output(json_data),
            }
        )

    if not data:
        return pd.DataFrame()

    combined_df = pd.DataFrame(data)

    print(
        f"Avg std percent — Read:  {combined_df['read_bw_std_pct'].mean():.3f}%  Write: {combined_df['write_bw_std_pct'].mean():.3f}%"
    )

    return combined_df


def plot_fio(arch, title, df_param, value_col, std_col, ylabel, is_write=False):
    df = df_param.copy()
    df["readratio"] = df["readratio"].astype(int)

    read_ratios = sorted(df["readratio"].unique())
    x = np.arange(len(read_ratios))

    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})

    sns.set_style("ticks")
    sns.set_context("paper")
    fig, ax = plt.subplots(
        figsize=(3.31, 1.65),
    )

    palettes = {
        "linux": sns.color_palette(config.LINUX_COLOR, n_colors=7),
        "spare": sns.color_palette(config.SPARE_COLOR, n_colors=9),
    }

    error_kw = {"linewidth": 0.3, "capthick": 0.3}
    capsize = 0.7

    series = []
    for tag, label, (palette_name, shade) in SERIES:
        sub = df[df["tag"] == tag].set_index("readratio")
        values = [
            sub.loc[r, value_col] if r in sub.index else 0 for r in read_ratios
        ]
        if all(v == 0 for v in values):
            continue
        stds = [
            sub.loc[r, std_col] if r in sub.index else 0 for r in read_ratios
        ]
        series.append((values, stds, label, palettes[palette_name][shade]))

    # keep the group of bars centred on the tick whatever the number of series
    width = 0.85 / max(len(series), 1)

    for i, (values, stds, label, color) in enumerate(series):
        pos = x + (i - (len(series) - 1) / 2) * width
        ax.bar(
            pos, values, width, yerr=stds,
            error_kw=error_kw, capsize=capsize, label=label,
            color=color, edgecolor=color, linewidth=0.3, zorder=2,
        )

    # faint dotted rules at each tick, like the pressure plot
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.4, color="0.85", zorder=0)

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
