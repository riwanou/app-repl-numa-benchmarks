"""QPS over time under the memory.high staircase (the paper figure), and a
per-phase summary of what each limit cost.

    uv run run.py plot-pressure
"""

import os

import matplotlib

# headless, and pinned rather than left to the default: the rounded phase
# band precomputes its position in display pixels, and macOS's native backend
# applies a Retina pixel-ratio scale Agg does not
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import config

STOCK = ["firsttouch", "interleaved", "numa-balancing"]
REPL = ["repl-bound", "repl-firsttouch", "repl-interleaved", "repl-dynamic"]

VARIANT_LABELS = {
    "firsttouch": "First Touch",
    "interleaved": "Interleave",
    "numa-balancing": "Linux Vanilla",
    "repl-bound": "SPARe (Bind)",
    "repl-firsttouch": "SPARe (First Touch)",
    "repl-interleaved": "SPARe (Interleave)",
    "repl-dynamic": "SPARe (Dynamic)",
}

# the paper figure leaves SPARe's first touch out: single node is the
# placement dynamic starts from, so it shows what adapting buys
PAPER = STOCK + ["repl-bound", "repl-interleaved", "repl-dynamic"]

# an interleaved variant is drawn as its twin's color, dashed
INTERLEAVED_TWIN = {
    "interleaved": "firsttouch",
    "repl-interleaved": "repl-firsttouch",
}
COLORS = {
    "firsttouch": "#FBBC04",
    "numa-balancing": "#7a4fa3",
    "repl-dynamic": "#EA4335",
}

# summary reports steady state: skip this much of each phase, the rest is
# the reclaim transient rather than what the limit costs
STEADY_FROM = 1 / 2

# the M limits we run, spelled the way their phase labels spell them
LIMIT_NAMES = {"4500M": "4.5G", "3500M": "3.5G", "2560M": "2.5G"}
# no memory.high at all: the phase label names what the machine actually has
MACHINE_RAM = {"silver": "384G", "gold": "768G"}

BAND = "#f8f8f8"  # phase shading, alternating
BAND_TOP = 1.1  # in axes fraction, to cover the label
BAND_RADIUS = 6  # points
LINEWIDTH = 2.1
SPINE_WIDTH = 2.0


def short(arch: str) -> str:
    return config.ARCH_SUBNAMES.get(arch, arch)


def base(arch: str, variant: str) -> str:
    return os.path.join(config.RESULT_DIR, arch, "pressure", f"ann-pressure-{variant}")


def variant_label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant)


def variant_color(variant: str):
    """Patched in blues, stock in oranges, like plot_ann; vanilla, balancing
    and dynamic get a hue each."""
    variant = INTERLEAVED_TWIN.get(variant, variant)
    if variant in COLORS:
        return COLORS[variant]
    if variant in REPL:
        return sns.color_palette(config.SPARE_COLOR, n_colors=3 + len(REPL))[
            3 + REPL.index(variant)
        ]
    return sns.color_palette(config.LINUX_COLOR, n_colors=3 + len(STOCK))[
        3 + STOCK.index(variant)
    ]


def variant_style(variant: str) -> str:
    return "--" if variant in INTERLEAVED_TWIN else "-"


def limit_name(limit: str) -> str:
    return LIMIT_NAMES.get(limit, limit)


def phases(arch: str, variant: str) -> pd.DataFrame:
    df = pd.read_csv(
        f"{base(arch, variant)}-phases.csv", parse_dates=["start_time", "end_time"]
    )
    t0 = df.start_time.iloc[0]
    df["start_s"] = (df.start_time - t0).dt.total_seconds()
    df["end_s"] = (df.end_time - t0).dt.total_seconds()
    return df


def runs(arch: str, variant: str) -> pd.DataFrame:
    return pd.read_csv(
        f"{base(arch, variant)}-ann.csv", parse_dates=["start_time"]
    ).dropna(subset=["phase"])


def plot_qps(arch: str, variants: list[str]):
    """The paper figure: QPS over time, every variant on one axis."""
    variants = [v for v in variants if v in PAPER]
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=150)
    x_max = 0
    for variant in variants:
        df = runs(arch, variant)
        t0 = phases(arch, variant).start_time.iloc[0]
        df = df.assign(elapsed=(df.start_time - t0).dt.total_seconds())
        df = df.sort_values("elapsed")
        x_max = max(x_max, df.elapsed.max())
        ax.plot(
            df.elapsed,
            df.qps.rolling(5, center=True).median(),
            lw=LINEWIDTH,
            color=variant_color(variant),
            ls=variant_style(variant),
        )

    ax.set_xlabel("time (s)", fontsize=20)
    ax.set_ylabel("QPS", fontsize=20, labelpad=1)
    ax.set_xlim(left=0, right=x_max + 1)
    # headroom, so the phase labels do not sit on top of the curves
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.12)
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.tick_params(labelsize=19, length=4, width=SPINE_WIDTH)
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.9, color="0.8", zorder=0)
    # margins fixed by hand rather than bbox_inches="tight": the rounded band
    # is placed from the figure geometry, which "tight" changes at save time
    fig.subplots_adjust(left=0.095, right=0.99, top=0.9, bottom=0.16)
    shade_phases(ax, phases(arch, variants[0]), MACHINE_RAM.get(short(arch)))

    # the y-axis already carries the shared origin: hide x's own 0
    for tick in ax.xaxis.get_major_ticks():
        if tick.get_loc() == 0:
            tick.tick1line.set_visible(False)
            tick.label1.set_visible(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
    # left spine redrawn up to the top of the phase band; the projecting cap
    # squares off the corner with the bottom spine
    ax.plot(
        [0, 0],
        [0, BAND_TOP],
        transform=ax.transAxes,
        color="black",
        lw=SPINE_WIDTH,
        clip_on=False,
        zorder=10,
        solid_capstyle="projecting",
    )

    out = os.path.join(config.PLOT_DIR_PRESSURE, f"{short(arch)}_qps.pdf")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def shade_phases(ax, ph: pd.DataFrame, max_label: str | None):
    """Every other phase shaded with a rounded band, its limit on top.
    The band lives in data seconds by axes fraction, so the corner radius is
    corrected by the axes aspect, from figure geometry alone so a PDF comes
    out like a PNG."""
    ax.figure.canvas.draw()  # settles autoscale, so xlim below is final
    pos = ax.get_position()
    xlim = ax.get_xlim()
    scale_x = pos.width * ax.figure.get_figwidth() / (xlim[1] - xlim[0])
    aspect = scale_x / (pos.height * ax.figure.get_figheight())
    rounding = (BAND_RADIUS / 72) / scale_x
    blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, row in ph.iterrows():
        if i % 2:
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (row.start_s, 0),
                    row.end_s - row.start_s,
                    BAND_TOP,
                    boxstyle=f"round,pad=0,rounding_size={rounding}",
                    mutation_aspect=aspect,
                    transform=blended,
                    facecolor=BAND,
                    edgecolor="none",
                    clip_on=False,
                    zorder=0,
                )
            )
        text = max_label if row.limit == "max" and max_label else limit_name(row.limit)
        ax.text(
            (row.start_s + row.end_s) / 2,
            1.02,
            text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=18,
            fontweight="bold",
        )


def plot_qps_legend(variants: list[str]):
    """Its own file, so the two arch figures share one legend: baselines on
    the first row, SPARe on the second (the legend fills column by column)."""
    variants = [v for v in variants if v in PAPER]
    rows = [[v for v in variants if v in STOCK], [v for v in variants if v in REPL]]
    order = [v for pair in zip(*rows) for v in pair]
    fig = plt.figure(figsize=(9, 0.9))
    fig.legend(
        [
            Line2D([], [], lw=LINEWIDTH, color=variant_color(v), ls=variant_style(v))
            for v in order
        ],
        [variant_label(v) for v in order],
        loc="center",
        fontsize=17,
        ncol=len(rows[0]),
        columnspacing=1.5,
        handlelength=1.6,
        frameon=False,
    )
    out = os.path.join(config.PLOT_DIR_PRESSURE, "qps_legend.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def read_locality(arch: str, variant: str, t0, t1) -> pd.Series:
    """Share of memory accesses served by the local socket, from pcm."""
    label = "ann-pressure-repl" if variant in REPL else "ann-pressure"
    path = os.path.join(config.RESULT_DIR, arch, "monitor", f"pcm_{label}.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path, header=[0, 1])
    date = df[[c for c in df.columns if c[1] == "Date"][0]].astype(str)
    time = df[[c for c in df.columns if c[1] == "Time"][0]].astype(str)
    t = pd.to_datetime(date + " " + time, errors="coerce")
    inside = (t >= t0) & (t <= t1)
    return pd.to_numeric(df.loc[inside, ("System", "LOCAL")], errors="coerce")


def plot_summary(arch: str, variants: list[str]):
    """Per phase, steady state: median QPS, memory used and local accesses,
    against the limit it ran under."""
    fig, (ax, ax_mem, ax_loc) = plt.subplots(
        3, 1, figsize=(9, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 1.2]}
    )
    for variant in variants:
        ph = phases(arch, variant)
        order, x = list(ph.phase), range(len(ph))
        df = runs(arch, variant)
        cg = pd.read_csv(f"{base(arch, variant)}-cgroup.csv")
        style = {"marker": "o", "color": variant_color(variant)}

        qps, loc, mem = [], [], []
        for _, r in ph.iterrows():
            begins = r.start_time + (r.end_time - r.start_time) * STEADY_FROM
            g = df[(df.start_time >= begins) & (df.phase == r.phase)]
            qps.append(g.qps.median())
            v = read_locality(arch, variant, begins, r.end_time)
            loc.append(v.median() if not v.empty else float("nan"))
            c = cg[cg.phase == r.phase]
            mem.append(c.current_mb.iloc[-1] / 1024 if not c.empty else float("nan"))
        ax.plot(x, qps, label=variant_label(variant), **style)
        ax_loc.plot(x, loc, **style)
        ax_mem.plot(x, mem, **style)

    limits = list(ph.limit)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [p if p == limit_name(lim) else f"{p}\n{lim}" for p, lim in zip(order, limits)]
    )
    ax.set_ylabel("median QPS (steady state)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax_mem.set_ylabel("memory used (GB)")
    ax_mem.set_ylim(bottom=0)
    ax_loc.set_ylabel("local memory\naccesses (%)")
    ax_loc.set_ylim(0, 105)
    ax_loc.set_xlabel("memory.high")
    for a in (ax, ax_mem, ax_loc):
        a.grid(axis="y", alpha=0.3)

    out = os.path.join(config.PLOT_DIR_PRESSURE_DETAILS, f"{short(arch)}_summary.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def make_plot_pressure():
    os.makedirs(config.PLOT_DIR_PRESSURE_DETAILS, exist_ok=True)
    for arch in sorted(os.listdir(config.RESULT_DIR)):
        available = [v for v in STOCK + REPL if os.path.exists(f"{base(arch, v)}-ann.csv")]
        if not available:
            continue
        plot_qps(arch, available)
        plot_qps_legend(available)
        plot_summary(arch, available)
