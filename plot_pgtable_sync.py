"""Mean us per munmap, with the slowdown over vanilla.

uv run run.py plot-pgtable-sync
"""

import glob
import os
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

import config

BASE = "pgtable_sync_baseline"
LABELS = {
    "shared": "page-table shared",
    "diverged": "page-table diverged",
}
COLORS = {
    "shared": sns.color_palette(config.SPARE_COLOR, n_colors=9)[3],
    "diverged": sns.color_palette(config.SPARE_COLOR, n_colors=9)[7],
}
MS_TO_US = 1000

# chunk size sweep: (subfolder under pgtable_sync/, label), left to right
CHUNKS = [
    (None, "1 page\n4KB"),
    ("16kb_chunks", "4 pages\n16KB"),
    ("32kb_chunks", "8 pages\n32KB"),
]
CHUNK_BAR_WIDTH = 0.28
# (page table state, tag, x offset from the group centre)
CHUNK_STATES = [
    ("vanilla", BASE, -CHUNK_BAR_WIDTH),
    ("shared", "pgtable_sync_norepl", 0.0),
    ("diverged", "pgtable_sync_repl", CHUNK_BAR_WIDTH),
]
CHUNK_LABELS = {
    "vanilla": "Vanilla",
    "shared": "SPARe, page-table shared",
    "diverged": "SPARe, page-table diverged",
}
CHUNK_COLORS = {
    "vanilla": sns.color_palette(config.LINUX_COLOR, n_colors=9)[4],
    "shared": COLORS["shared"],
    "diverged": COLORS["diverged"],
}


def load(arch: str, subfolder: str | None = None) -> pd.DataFrame:
    """One row per tag, over whatever csvs sit in that chunk size's folder."""
    folder = os.path.join(config.RESULT_DIR, arch, "microbench", "pgtable_sync")
    if subfolder:
        folder = os.path.join(folder, subfolder)
    paths = glob.glob(os.path.join(folder, "*.csv"))
    if not paths:
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return df.groupby("tag")["elapsed_ms"].agg(["mean", "std"]) * MS_TO_US


BAR_WIDTH = 0.22
BAR_STEP = 0.25  # small separation between bars of the same group
GROUP_GAP = 0.15  # Vanilla stands apart from the SPARe group
NODE_TAG_RE = re.compile(r"^pgtable_sync_repl_(\d+)n$")


def diverged_bars(df: pd.DataFrame) -> list[tuple[str, str]]:
    """(tick label, tag) for SPARe's diverged region, all nodes last."""
    sweep = []
    for tag in df.index:
        m = NODE_TAG_RE.match(tag)
        if m:
            sweep.append((int(m.group(1)), str(tag)))
    bars = [
        (f"{n}-nd" + ("s" if n > 1 else ""), tag) for n, tag in sorted(sweep)
    ]
    if "pgtable_sync_repl" in df.index:
        bars.append(("all", "pgtable_sync_repl"))
    return bars


def bars_for(df: pd.DataFrame) -> list[tuple[str, str, str, float]]:
    """(state, tag, tick label, x) for one arch, left to right."""
    bars = [("shared", BASE, "shared", 0.0)]
    spare_x = BAR_STEP + GROUP_GAP
    if "pgtable_sync_norepl" in df.index:
        bars.append(("shared", "pgtable_sync_norepl", "shared", spare_x))
    for i, (label, tag) in enumerate(diverged_bars(df)):
        bars.append(("diverged", tag, label, spare_x + BAR_STEP * (i + 1)))
    return bars


def draw(ax, df: pd.DataFrame, bars, ylabel: str = "µs", tick_fs=6):
    """Shared region for Vanilla then SPARe, then SPARe's diverged region
    once per node count. Ticks name the bar, the row under them the system."""
    base = df.loc[BASE, "mean"] if BASE in df.index else None
    tallest = max(df.loc[t, "mean"] + df.loc[t, "std"] for _, t, _, _ in bars)

    for state, tag, _, x in bars:
        height = df.loc[tag, "mean"]
        ax.bar(
            x,
            height,
            width=BAR_WIDTH,
            yerr=df.loc[tag, "std"],
            capsize=1.5,
            color=COLORS[state],
            edgecolor=COLORS[state],
            error_kw=dict(lw=0.4, capthick=0.4),
            zorder=2,
        )
        # the factor only says something on the SPARe bars
        if base and tag != BASE:
            ax.text(
                x,
                height + df.loc[tag, "std"] + tallest * 0.04,
                f"x{height / base:.2f}",
                ha="center",
                va="bottom",
                fontsize=6,
                zorder=3,
            )

    spare_xs = [x for _, _, _, x in bars[1:]]
    groups = [("Vanilla", 0.0)]
    if spare_xs:
        groups.append(("SPARe", sum(spare_xs) / len(spare_xs)))
    for label, x in groups:
        ax.text(
            x,
            1.02,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6,
        )

    # faint dotted rules at each tick, like the pressure plot
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.4, color="0.85", zorder=0)
    sns.despine(ax=ax)
    ax.set_xticks([x for _, _, _, x in bars])
    ax.set_xticklabels([label for _, _, label, _ in bars], fontsize=tick_fs)
    ax.tick_params(axis="x", length=2, pad=1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7, labelpad=1)
    ax.tick_params(axis="y", labelsize=6, length=2, pad=1)
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=3))
    ax.set_ylim(0, tallest * 1.25)
    ax.set_xlim(-BAR_STEP, bars[-1][3] + BAR_STEP)


def plot(arch: str, sub: str, df: pd.DataFrame, ylabel: str = "µs"):
    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})
    sns.set_style("ticks")
    sns.set_context("paper")

    bars = bars_for(df)
    fig, ax = plt.subplots(figsize=(0.34 * len(bars) + 0.75, 0.75))
    draw(ax, df, bars, ylabel)

    os.makedirs(config.PLOT_DIR_PGTABLE_SYNC, exist_ok=True)
    out = os.path.join(config.PLOT_DIR_PGTABLE_SYNC, f"{sub}.pdf")
    plt.savefig(out, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"[OK] {out}")
    print(df)


# silver left, gold right, in one compact figure
BOTH_ARCHS = [("silver", "Silver x2"), ("gold", "Gold x4")]


def plot_both(frames: dict[str, pd.DataFrame]):
    """Both machines side by side, each with its own y axis."""
    panels = [(t, frames[s]) for s, t in BOTH_ARCHS if s in frames]
    if len(panels) < 2:
        return

    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})
    sns.set_style("ticks")
    sns.set_context("paper")

    layouts = [bars_for(df) for _, df in panels]
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(0.36 * sum(len(b) for b in layouts) + 0.7, 0.75),
        gridspec_kw=dict(
            width_ratios=[len(b) for b in layouts], wspace=0.15
        ),
    )
    for ax, (title, df), bars in zip(axes, panels, layouts):
        draw(ax, df, bars, ylabel="µs" if ax is axes[0] else "", tick_fs=5.5)
        ax.text(
            sum(ax.get_xlim()) / 2,
            -0.28,
            title,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
        )

    os.makedirs(config.PLOT_DIR_PGTABLE_SYNC, exist_ok=True)
    out = os.path.join(config.PLOT_DIR_PGTABLE_SYNC, "silver_gold.pdf")
    plt.savefig(out, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"[OK] {out}")


def legend_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=COLORS[state],
            edgecolor=COLORS[state],
            label=LABELS[state],
        )
        for state in ("shared", "diverged")
    ]


def make_legend():
    """Its own file, so a paper can place it once over several archs."""
    fig = plt.figure(figsize=(3.3, 0.2))
    fig.legend(
        handles=legend_handles(), loc="center", ncol=2, fontsize=7, frameon=False
    )

    os.makedirs(config.PLOT_DIR_PGTABLE_SYNC, exist_ok=True)
    out = os.path.join(config.PLOT_DIR_PGTABLE_SYNC, "legend.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"[OK] {out}")


def plot_chunks(arch: str, sub: str):
    """Mean us per munmap across chunk sizes, log scale, with the slowdown
    over that chunk size's own vanilla bar."""
    plt.rcParams.update({"font.family": "serif", "font.serif": "DejaVu Serif"})
    sns.set_style("ticks")
    sns.set_context("paper")

    fig, ax = plt.subplots(figsize=(3.2, 1.3))
    any_data = False

    for gi, (subfolder, label) in enumerate(CHUNKS):
        df = load(arch, subfolder)
        if df.empty:
            continue
        any_data = True
        base = df.loc[BASE, "mean"] if BASE in df.index else None

        for state, tag, dx in CHUNK_STATES:
            if tag not in df.index:
                continue
            height = df.loc[tag, "mean"]
            ax.bar(
                gi + dx,
                height,
                width=CHUNK_BAR_WIDTH,
                yerr=df.loc[tag, "std"],
                capsize=1.5,
                color=CHUNK_COLORS[state],
                edgecolor=CHUNK_COLORS[state],
                error_kw=dict(lw=0.4, capthick=0.4),
                zorder=2,
            )
            if base and tag != BASE:
                ax.text(
                    gi + dx,
                    height * 1.12,
                    f"x{height / base:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.5,
                    zorder=3,
                )

    if not any_data:
        plt.close(fig)
        return

    ax.set_yscale("log")
    # faint dotted rules at each tick, like the pressure plot
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.4, color="0.85", zorder=0)
    sns.despine(ax=ax)
    ax.set_xticks(range(len(CHUNKS)))
    ax.set_xticklabels([label for _, label in CHUNKS], fontsize=6.5)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.set_ylabel("µs (log scale)", fontsize=7, labelpad=1)
    ax.tick_params(axis="y", labelsize=6, length=2, pad=1)
    ax.yaxis.set_major_formatter(mtick.ScalarFormatter())

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CHUNK_COLORS[s])
        for s, _, _ in CHUNK_STATES
    ]
    fig.legend(
        handles,
        [CHUNK_LABELS[s] for s, _, _ in CHUNK_STATES],
        bbox_to_anchor=(0.5, 1.12),
        loc="upper center",
        fontsize=6.5,
        ncol=3,
        edgecolor="none",
        columnspacing=1.0,
    )

    os.makedirs(config.PLOT_DIR_PGTABLE_SYNC, exist_ok=True)
    out = os.path.join(config.PLOT_DIR_PGTABLE_SYNC, f"{sub}_chunks.pdf")
    plt.savefig(out, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"[OK] {out}")


def make_plot_pgtable_sync():
    make_legend()
    frames = {}
    for arch, sub in config.ARCH_SUBNAMES.items():
        df = load(arch)
        if not df.empty:
            frames[sub] = df
            # side by side with silver, so only silver carries the unit
            plot(arch, sub, df, ylabel="" if sub == "gold" else "µs")
        plot_chunks(arch, sub)
    plot_both(frames)


if __name__ == "__main__":
    make_plot_pgtable_sync()
