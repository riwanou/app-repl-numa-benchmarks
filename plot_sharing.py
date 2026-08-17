"""DRAM bandwidth over time, with the bench phases shaded.

Shows whether a number is a steady state or a slice of a curve, which a phase
mean cannot say. One plot per arch that has the capture.

    uv run run.py plot-sharing
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

import config

MB_TO_GB = 1 / 1024


def load(arch: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(config.RESULT_DIR, arch, "monitor", f"pcm_memory_{label}.csv"),
        header=[0, 1],
    )
    time = pd.to_datetime(
        df[("Unnamed: 0_level_0", "Date")].astype(str)
        + " "
        + df[("Unnamed: 1_level_0", "Time")].astype(str),
        errors="coerce",
    )
    return pd.DataFrame(
        {
            "time": time,
            "read": pd.to_numeric(df[("System", "Read")], errors="coerce") * MB_TO_GB,
            "write": pd.to_numeric(df[("System", "Write")], errors="coerce") * MB_TO_GB,
        }
    ).dropna()


def coherence(arch: str, label: str) -> pd.DataFrame:
    """The directory counters, summed over sockets, in millions per second."""
    path = os.path.join(
        config.RESULT_DIR, arch, "monitor", f"perf_coherence_{label}.csv"
    )
    if not os.path.exists(path):
        return pd.DataFrame()

    lines = open(path).read().splitlines()
    anchor = pd.to_datetime(lines[0].split("# start ")[1].strip())
    rows = []
    for line in lines[2:]:
        f = line.split(",")
        if len(f) < 6 or line.startswith("#"):
            continue
        try:
            rows.append((round(float(f[0])), f[5].strip(), int(f[3])))
        except ValueError:
            continue
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["offset", "event", "value"])
    wide = df.pivot_table(index="offset", columns="event", values="value",
                          aggfunc="sum") / 1e6
    wide["time"] = anchor + pd.to_timedelta(wide.index.to_series(), unit="s")
    return wide


def phases(arch: str, label: str) -> pd.DataFrame:
    """The bench's own phase windows, when it wrote a results.csv."""
    path = os.path.join(config.RESULT_DIR, arch, label, "results.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in ("start_time", "end_time"):
        df[col] = pd.to_datetime(df[col])
    return df


def plot(arch: str, sub: str, label: str):
    df = load(arch, label)
    start = df["time"].iloc[0]
    secs = (df["time"] - start).dt.total_seconds()

    coh = coherence(arch, label)
    fig, (ax, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 9), sharex=True, height_ratios=[2, 1, 1]
    )
    ax.plot(secs, df["read"], label="read", lw=1)
    ax.plot(secs, df["write"], label="write", lw=1)

    if not coh.empty:
        csecs = (coh["time"] - start).dt.total_seconds()
        for event, name in (
            ("UNC_M2M_DIRECTORY_UPDATE.ANY", "directory updates"),
            ("UNC_CHA_DIR_LOOKUP.SNP", "snoops"),
        ):
            if event in coh:
                ax2.plot(csecs, coh[event], label=name, lw=1)
        ax2.set_ylabel("M/s")
        ax2.legend()

        for state in ("I", "S", "A"):
            event = f"UNC_M2M_DIRECTORY_LOOKUP.STATE_{state}"
            if event in coh:
                ax3.plot(csecs, coh[event], label=f"state {state}", lw=1)
        ax3.set_ylabel("lookups M/s")
        ax3.legend()

    for _, row in phases(arch, label).iterrows():
        a = (row["start_time"] - start).total_seconds()
        b = (row["end_time"] - start).total_seconds()
        name = f"{row.get('policy', '')} {row['phase']}".strip()
        for axis in (ax, ax2, ax3):
            axis.axvspan(a, b, color="grey", alpha=0.12)
        ax.text(a, ax.get_ylim()[1], name, fontsize=6, rotation=90, va="top")

    ax3.set_xlabel("seconds")
    ax.set_ylabel("GB/s")
    ax.set_title(f"{label} on {sub}")
    ax.legend()

    os.makedirs(config.PLOT_DIR_SHARING, exist_ok=True)
    out = os.path.join(config.PLOT_DIR_SHARING, f"{sub}_{label}.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def make_plot_sharing(label: str = "sharing"):
    for arch, sub in config.ARCH_SUBNAMES.items():
        capture = os.path.join(
            config.RESULT_DIR, arch, "monitor", f"pcm_memory_{label}.csv"
        )
        if os.path.exists(capture):
            plot(arch, sub, label)


if __name__ == "__main__":
    make_plot_sharing(sys.argv[1] if len(sys.argv) > 1 else "sharing")
