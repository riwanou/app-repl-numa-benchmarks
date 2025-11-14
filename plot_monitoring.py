import config
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULT_DIR = config.RESULT_DIR

TIME_WINDOWS = {
    "ann": [],
    "ann-repl": [],
    "rocksdb-repl": [
        ("2025-09-04T13:45:12.607589", "2025-09-04T13:47:14.062101"),
        ("2025-09-04T13:51:18.489159", "2025-09-04T13:53:20.251285"),
    ],
    "tmp": [
        ("2025-09-04T13:45:12.489159", "2025-09-04T13:53:20.251285"),
    ],
}


def get_variant_time_windows(variant: str) -> list[tuple[str, str]]:
    return TIME_WINDOWS.get(variant, [])


def filter_by_time_windows(
    df: pd.DataFrame, windows: list[tuple[str, str]], time_col: str
) -> pd.DataFrame:
    filtered = pd.DataFrame()
    for start, end in windows:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        filtered = pd.concat(
            [
                filtered,
                df[(df[time_col] >= start_dt) & (df[time_col] <= end_dt)],
            ]
        )
    return pd.DataFrame(filtered)


def make_plot_monitoring():
    os.makedirs(config.PLOT_DIR_MONITORING, exist_ok=True)
    plot("ann")
    plot("rocksdb")
    plot("ann-repl")
    plot("rocksdb-repl")
    plot("tmp")


def plot(variant: str):
    df_pcm, df_pcm_memory, df_mem = get_data(variant)
    plot_pcm(df_pcm, variant)
    plot_pcm_memory(df_pcm_memory, variant)
    plot_mem(df_mem, variant)


def get_data(variant: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_pcm = []
    data_pcm_memory = []
    data_mem = []

    for arch in os.listdir(RESULT_DIR):
        arch_dir = os.path.join(RESULT_DIR, arch, "monitor")
        if not os.path.isdir(arch_dir):
            continue

        pcm_csv_path = os.path.join(arch_dir, f"pcm_{variant}.csv")
        if os.path.exists(pcm_csv_path):
            df = pd.read_csv(pcm_csv_path, header=[0, 1])
            df["arch"] = arch
            data_pcm.append(df)

        pcm_memory_csv_path = os.path.join(
            arch_dir, f"pcm_memory_{variant}.csv"
        )
        if os.path.exists(pcm_memory_csv_path):
            df = pd.read_csv(pcm_memory_csv_path, header=[0, 1])
            df["arch"] = arch
            data_pcm_memory.append(df)

        mem_csv_path = os.path.join(arch_dir, f"mem_{variant}.csv")
        if os.path.exists(mem_csv_path):
            df = pd.read_csv(mem_csv_path)
            df["arch"] = arch
            data_mem.append(df)

    df_pcm = (
        pd.concat(data_pcm, ignore_index=True) if data_pcm else pd.DataFrame()
    )
    df_pcm_memory = (
        pd.concat(data_pcm_memory, ignore_index=True)
        if data_pcm_memory
        else pd.DataFrame()
    )
    df_mem = (
        pd.concat(data_mem, ignore_index=True) if data_mem else pd.DataFrame()
    )

    return df_pcm, df_pcm_memory, df_mem


def plot_pcm(df, variant: str):
    if df.empty:
        return

    sns.set_style(style="ticks")
    sns.set_context("paper")
    palette = sns.color_palette("Blues", n_colors=3)

    for arch in pd.Series(df["arch"]).unique():
        df_arch = pd.DataFrame(df[df["arch"] == arch])
        df_arch["time_dt"] = pd.to_datetime(
            df_arch[("System", "Date")].astype(str)
            + " "
            + df_arch[("System", "Time")].astype(str)
        )
        time_windows = get_variant_time_windows(variant)
        if time_windows:
            df_arch = filter_by_time_windows(df_arch, time_windows, "time_dt")

        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(3, 6, height_ratios=[1, 1, 1])

        # memory locality bandwidth

        ax0 = fig.add_subplot(gs[0, 0:3])
        ax0.stackplot(
            df_arch["time_dt"],
            [
                df_arch[("Socket 0", "LMB")] / 1024,
                df_arch[("Socket 0", "RMB")] / 1024,
            ],
            labels=["Local", "Remote"],
            colors=palette,
            edgecolor="none",
        )
        ax0.set_title("Memory Locality Node 0")
        ax0.set_ylabel("Memory Bandwidth (GB)")
        ax0.legend(edgecolor="white", framealpha=1.0, loc="upper right")
        ax0.tick_params(axis="x")

        ax = fig.add_subplot(gs[0, 3:6], sharey=ax0)
        ax.stackplot(
            df_arch["time_dt"],
            [
                df_arch[("Socket 1", "LMB")] / 1024,
                df_arch[("Socket 1", "RMB")] / 1024,
            ],
            labels=["Local", "Remote"],
            colors=palette,
            edgecolor="none",
        )
        ax.set_title("Memory Locality Node 1")
        ax.set_ylabel("Memory Bandwidth (GB)")
        ax.legend(edgecolor="white", framealpha=1.0, loc="upper right")
        ax.tick_params(axis="x")

        # LLC latency plot

        ax = fig.add_subplot(gs[1, 0:2])
        ax.plot(
            df_arch["time_dt"],
            df_arch[("System", "LLCRDMISSLAT (ns)")],
            label="LLC read miss latency",
            color=palette[0],
        )
        ax.set_title("System LLC Read Miss Latency")
        ax.set_ylabel("Miss Latency (ns)")
        ax.legend(edgecolor="white", framealpha=1.0, loc="upper right")
        ax.tick_params(axis="x")

        # UPI traffic memory ratio

        ax = fig.add_subplot(gs[1, 2:4])
        ax.plot(
            df_arch["time_dt"],
            df_arch[("System", "UPItoMC")],
            label="UPI Traffic Ratio",
            color=palette[0],
        )
        ax.set_title("System UPI Traffic to Memory Traffic")
        ax.set_ylabel("UPI Traffic Ratio")
        ax.legend(edgecolor="white", framealpha=1.0, loc="upper right")
        ax.tick_params(axis="x")

        # UPI traffic memory bandwidth

        ax = fig.add_subplot(gs[1, 4:6])
        ax.plot(
            df_arch["time_dt"],
            df_arch[("System", "TotalUPIin")] / (1024),
            label="UPI In",
            color=palette[0],
        )
        ax.plot(
            df_arch["time_dt"],
            df_arch[("System", "TotalUPIout")] / (1024),
            label="UPI Out",
            color=palette[1],
        )
        ax.set_title("System UPI Traffic")
        ax.set_ylabel("Traffic (GB)")
        ax.legend(edgecolor="white", framealpha=1.0, loc="upper right")
        ax.tick_params(axis="x")

        # Cache Miss

        ax = fig.add_subplot(gs[2, 0:3])
        ax.plot(
            df_arch["time_dt"],
            df_arch[("System", "L3MISS")],
            label="L3 Miss",
            color=palette[0],
        )
        ax.plot(
            df_arch["time_dt"],
            df_arch[("System", "L2MISS")],
            label="L2 Miss",
            color=palette[1],
        )
        ax.set_title("Cache Misses")
        ax.set_ylabel("Cache line misses in millions")
        ax.legend(edgecolor="white", framealpha=1.0, loc="upper right")
        ax.tick_params(axis="x")

        # Cache Miss per Instruction

        ax = fig.add_subplot(gs[2, 3:6])
        ax.plot(
            df_arch["time_dt"],
            df_arch[("System", "L3MPI")],
            label="L3 Miss Per Instruction",
            color=palette[0],
        )
        ax.plot(
            df_arch["time_dt"],
            df_arch[("System", "L2MPI")],
            label="L2 Miss Per Instruction",
            color=palette[1],
        )
        ax.set_title("Cache Misses Per Instruction")
        ax.set_ylabel("Average cache misses per INST")
        ax.legend(edgecolor="white", framealpha=1.0, loc="upper right")
        ax.tick_params(axis="x")

        plt.tight_layout()
        path = os.path.join(config.PLOT_DIR_MONITORING, arch)
        plt.savefig(f"{path}_pcm_{variant}.png", bbox_inches="tight", dpi=300)


def plot_pcm_memory(df, variant: str):
    if df.empty:
        return

    sns.set_style(style="ticks")
    sns.set_context("paper")
    palette = sns.color_palette("Blues", n_colors=6)

    for arch in pd.Series(df["arch"]).unique():
        df_arch = pd.DataFrame(df[df["arch"] == arch])
        df_arch["time_dt"] = pd.to_datetime(
            df_arch[("Unnamed: 0_level_0", "Date")].astype(str)
            + " "
            + df_arch[("Unnamed: 1_level_0", "Time")].astype(str)
        )
        time_windows = get_variant_time_windows(variant)
        if time_windows:
            df_arch = filter_by_time_windows(df_arch, time_windows, "time_dt")

        read_channels_sock0 = [
            col
            for col in df_arch.columns
            if col[0] == "SKT0"
            and col[1].startswith("Ch")
            and col[1].endswith("Read")
            and "PMM" not in col[1]
        ]
        write_channels_sock0 = [
            col
            for col in df_arch.columns
            if col[0] == "SKT0"
            and col[1].startswith("Ch")
            and col[1].endswith("Write")
            and "PMM" not in col[1]
        ]

        read_channels_sock1 = [
            col
            for col in df_arch.columns
            if col[0] == "SKT1"
            and col[1].startswith("Ch")
            and col[1].endswith("Read")
            and "PMM" not in col[1]
        ]
        write_channels_sock1 = [
            col
            for col in df_arch.columns
            if col[0] == "SKT1"
            and col[1].startswith("Ch")
            and col[1].endswith("Write")
            and "PMM" not in col[1]
        ]

        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(4, 4, height_ratios=[1, 1, 1, 1])

        # system bandwidth

        ax0 = fig.add_subplot(gs[0, 0:4])
        ax0.plot(
            df_arch["time_dt"],
            df_arch[("System", "Memory")] / (1024),
            label="Memory",
            color=palette[3],
        )
        ax0.set_title("System Memory Bandwidth")
        ax0.set_ylabel("Memory Bandwidth (GB)")
        ax0.legend(edgecolor="white", framealpha=1.0, loc="upper right")
        ax0.tick_params(axis="x")

        ax = fig.add_subplot(gs[1, 0:2], sharey=ax0)
        ax.plot(
            df_arch["time_dt"],
            df_arch[("System", "Read")] / (1024),
            label="Mem Read",
            color=palette[3],
        )
        ax.set_title("System Read Bandwidth")
        ax.set_ylabel("Read Bandwidth (GB)")
        ax.legend(edgecolor="white", framealpha=1.0, loc="upper right")
        ax.tick_params(axis="x")

        ax = fig.add_subplot(gs[1, 2:4], sharey=ax0)
        ax.plot(
            df_arch["time_dt"],
            df_arch[("System", "Write")] / (1024),
            label="Mem Write",
            color=palette[3],
        )
        ax.set_title("System Write Bandwidth")
        ax.set_ylabel("Write Bandwidth (GB)")
        ax.legend(edgecolor="white", framealpha=1.0, loc="upper right")
        ax.tick_params(axis="x")

        # read bandwidth channels

        ax1 = fig.add_subplot(gs[2, 0:2])
        ax1.stackplot(
            df_arch["time_dt"],
            [df_arch[col] / 1024 for col in read_channels_sock0],
            labels=[col[1] for col in read_channels_sock0],
            colors=palette,
            edgecolor="none",
        )
        ax1.set_title("System Read Bandwidth Node 0 (Channel)")
        ax1.set_ylabel("Read Bandwidth (GB)")
        ax1.legend(edgecolor="white", framealpha=1.0, loc="upper right")
        ax1.tick_params(axis="x")

        ax1 = fig.add_subplot(gs[2, 2:4])
        ax1.stackplot(
            df_arch["time_dt"],
            [df_arch[col] / 1024 for col in read_channels_sock1],
            labels=[col[1] for col in read_channels_sock1],
            colors=palette,
            edgecolor="none",
        )
        ax1.set_title("System Read Bandwidth Node 1 (Channel)")
        ax1.set_ylabel("Read Bandwidth (GB)")
        ax1.legend(edgecolor="white", framealpha=1.0, loc="upper right")
        ax1.tick_params(axis="x")

        # write bandwidth channels

        ax = fig.add_subplot(gs[3, 0:2], sharey=ax1)
        ax.stackplot(
            df_arch["time_dt"],
            [df_arch[col] / 1024 for col in write_channels_sock0],
            labels=[col[1] for col in write_channels_sock0],
            colors=palette,
            edgecolor="none",
        )
        ax.set_title("System Write Bandwidth Node 0 (Channel)")
        ax.set_ylabel("Write Bandwidth (GB)")
        ax.legend(edgecolor="white", framealpha=1.0, loc="upper right")
        ax.tick_params(axis="x")

        ax = fig.add_subplot(gs[3, 2:4], sharey=ax1)
        ax.stackplot(
            df_arch["time_dt"],
            [df_arch[col] / 1024 for col in write_channels_sock1],
            labels=[col[1] for col in write_channels_sock1],
            colors=palette,
            edgecolor="none",
        )
        ax.set_title("System Write Bandwidth Node 1 (Channel)")
        ax.set_ylabel("Write Bandwidth (GB)")
        ax.legend(edgecolor="white", framealpha=1.0, loc="upper right")
        ax.tick_params(axis="x")

        plt.tight_layout()
        path = os.path.join(config.PLOT_DIR_MONITORING, arch)
        plt.savefig(
            f"{path}_pcm_memory_{variant}.png", bbox_inches="tight", dpi=300
        )


def plot_mem(df: pd.DataFrame, variant: str):
    if df.empty:
        return

    sns.set_style(style="ticks")
    sns.set_context("paper")
    palette = sns.color_palette("Blues", n_colors=4)

    for arch in pd.Series(df["arch"]).unique():
        df_arch = pd.DataFrame(df[df["arch"] == arch])
        df_arch["time_dt"] = pd.to_datetime(df_arch["time"])
        time_windows = get_variant_time_windows(variant)
        if time_windows:
            df_arch = filter_by_time_windows(df_arch, time_windows, "time_dt")

        nodes = ["Node0", "Node1"]
        fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 1])

        # row 0, system
        ax = fig.add_subplot(gs[0, 0:4])
        ax.stackplot(
            df_arch["time_dt"],
            [
                df_arch["mapped"] / (1024 * 1024),
                df_arch["anon"] / (1024 * 1024),
                df_arch["pageTable"] / (1024 * 1024),
            ],
            labels=["Mapped Files", "Anonymous Memory", "Page Tables"],
            colors=palette,
            edgecolor="none",
        )
        ax.set_title("System Memory Stats")
        ax.set_ylabel("Memory (GB)")
        ax.legend(edgecolor="white", framealpha=1.0, loc="upper right")
        ax.tick_params(axis="x")
        ax.set_xlabel("Time")

        # row 1, per node memory usage
        for i, node in enumerate(nodes):
            ax = fig.add_subplot(gs[1, (i * 2) : ((i + 1) * 2)])

            ax.stackplot(
                df_arch["time_dt"],
                [
                    df_arch[f"{node}_mapped"] / (1024 * 1024),
                    df_arch[f"{node}_anon"] / (1024 * 1024),
                    df_arch[f"{node}_pageTable"] / (1024 * 1024),
                ],
                labels=["Mapped Files", "Anonymous Memory", "Page Table"],
                colors=palette,
                edgecolor="none",
            )

            ax.set_title(f"{node} Memory Stats")
            ax.set_ylabel("Memory (GB)")
            ax.legend(edgecolor="white", framealpha=1.0, loc="upper right")
            ax.tick_params(axis="x")

        # row 2, per node page table
        for i, node in enumerate(nodes):
            ax = fig.add_subplot(gs[2, (i * 2) : ((i + 1) * 2)])

            ax.plot(
                df_arch["time_dt"],
                df_arch[f"{node}_pageTable"] / (1024),
                label="Page Table",
                color=palette[2],
            )

            ax.set_title(f"{node} Memory Stats (Page Table)")
            ax.set_ylabel("Memory (MB)")
            ax.legend(edgecolor="white", framealpha=1.0, loc="upper right")
            ax.tick_params(axis="x")

        plt.tight_layout()
        path = os.path.join(config.PLOT_DIR_MONITORING, arch)
        plt.savefig(f"{path}_mem_{variant}.png", bbox_inches="tight", dpi=300)
