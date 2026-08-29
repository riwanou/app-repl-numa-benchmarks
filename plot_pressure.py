"""QPS and bandwidth over time under the memory.high staircase, with the
counters that explain them underneath, all cut on the same phase windows.

    uv run run.py plot-pressure
"""

import os
import re

import matplotlib

# headless, and pinned rather than left to the default: the rounded phase
# band in plot_qps precomputes its position in display pixels, and macOS's
# native backend applies a Retina pixel-ratio scale Agg does not, throwing
# that precomputed position off
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
import seaborn as sns
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

import config

# plots are made from synced results, so walk every arch rather than the
# machine we happen to run on (config.PLATFORM only matches the bench host)
RESULT_DIR = config.RESULT_DIR

# the monitor CSVs hold every variant of a run, phases.csv cuts them apart
STOCK = ["firsttouch", "interleaved", "numa-balancing"]
REPL = ["repl-bound", "repl-firsttouch", "repl-interleaved", "repl-dynamic"]

# legend names, the variant tag itself when absent
VARIANT_LABELS = {
    "firsttouch": "Vanilla",
    "interleaved": "Interleaved",
    "numa-balancing": "NUMA Balancing",
    "repl-bound": "SPARe",
    "repl-firsttouch": "SPARe (Vanilla)",
    "repl-interleaved": "SPARe (Interleaved)",
    "repl-dynamic": "SPARe (Dynamic)",
}

BAND = "#f8f8f8"  # phase shading, alternating

# counters are read every 0.5s, so ~2s of samples: enough to average out the
# scan bursts without hiding a phase transition
EVENT_WINDOW = 4

# summary reports steady state: skip this much of each phase, the rest is
# the reclaim transient rather than what the limit costs
STEADY_FROM = 2 / 3

QPS_FIGSIZE = (12, 4.2)
QPS_FONTSIZE = 17  # legend
QPS_PHASE_FONTSIZE = 20  # phase labels
QPS_TICK_FONTSIZE = 19  # axis numbers, x and y alike
QPS_AXIS_LABEL_FONTSIZE = 20  # "time (s)" / "QPS"
QPS_TICK_WIDTH = 2.0
QPS_SPINE_WIDTH = QPS_TICK_WIDTH
QPS_BAND_TOP = 1.1  # phase shading top, in axes fraction, to cover the label
QPS_BAND_RADIUS = 6  # phase shading corner radius, in points


def plot_qps(arch: str, variants: list[str], full_gb: float = 0):
    """QPS over time, every variant on one axis. Kept separate from
    plot_metric: this is the one that goes in the paper, so its style is
    tuned by hand rather than shared with the debug plots.

    repl-firsttouch is left out, it sits on top of repl-bound.
    """
    variants = [v for v in variants if v != "repl-firsttouch"]

    fig, ax = plt.subplots(figsize=QPS_FIGSIZE, dpi=150)
    band = None
    x_max = 0
    handles, labels = [], []
    for variant in variants:
        ph = phases(arch, variant)
        band = band if band is not None else ph
        t0 = ph.start_time.iloc[0]
        df = pd.read_csv(
            f"{base(arch, variant)}-ann.csv", parse_dates=["start_time"]
        ).dropna(subset=["phase"])
        df["elapsed"] = (df.start_time - t0).dt.total_seconds()
        df = df.sort_values("elapsed")
        x, y = df.elapsed, df.qps.rolling(5, center=True).median()
        x_max = max(x_max, x.max())
        color = variant_color(variant)
        if variant == "repl-dynamic":
            # a soft glow behind the dashed line so it stands out from the
            # solid curves it overlaps; carried into the legend too, so its
            # swatch matches what's on the plot
            (glow,) = ax.plot(x, y, lw=4.5, color=color, alpha=0.2, zorder=1.5)
            (dash,) = ax.plot(x, y, lw=1.4, ls="--", color=color, zorder=1.6)
            handles.append((glow, dash))
        else:
            (line,) = ax.plot(x, y, lw=1.4, color=color)
            handles.append(line)
        labels.append(variant_label(variant))

    ax.set_xlabel("time (s)", fontsize=QPS_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("QPS", fontsize=QPS_AXIS_LABEL_FONTSIZE, labelpad=1)
    ax.set_xlim(left=0, right=x_max + 1)
    # headroom, so the phase labels do not sit on top of the curves
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.12)
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.tick_params(
        axis="both", labelsize=QPS_TICK_FONTSIZE, length=4, width=QPS_TICK_WIDTH
    )
    # faint dotted rules at each QPS tick, to read a level off the curves
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", lw=0.9, color="0.8", zorder=0)
    plot_qps_legend(variants)
    # margins fixed by hand rather than bbox_inches="tight": the rounded
    # band below is placed in pixels computed before saving, and a "tight"
    # bbox resizes the canvas at save time, invalidating them
    fig.subplots_adjust(left=0.095, right=0.99, top=0.9, bottom=0.16)
    shade_phases(
        ax,
        band,
        label=True,
        full_gb=full_gb,
        divider=False,
        band_top=QPS_BAND_TOP,
        radius=QPS_BAND_RADIUS,
        label_fontsize=QPS_PHASE_FONTSIZE,
        label_one_line=True,
        max_label=MACHINE_RAM.get(short(arch)),
    )

    # the y-axis already carries the shared origin, so x's own 0 is
    # redundant; hide just that tick's mark and label rather than dropping
    # it from the locator, which would let autoscale re-expand the view
    for tick in ax.xaxis.get_major_ticks():
        if tick.get_loc() == 0:
            tick.tick1line.set_visible(False)
            tick.label1.set_visible(False)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(QPS_SPINE_WIDTH)
    # the real left spine stops at the axes box; redraw it up to the top of
    # the phase band so it doesn't fall short of the gray rectangle. It stops
    # on the bottom spine, with a projecting cap: that extends it by exactly
    # half a linewidth, squaring off the corner instead of overshooting it
    ax.spines["left"].set_visible(False)
    ax.plot(
        [0, 0],
        [0, QPS_BAND_TOP],
        transform=ax.transAxes,
        color="black",
        lw=QPS_SPINE_WIDTH,
        clip_on=False,
        zorder=10,
        solid_capstyle="projecting",
    )

    os.makedirs(config.PLOT_DIR_PRESSURE, exist_ok=True)
    os.makedirs(config.PLOT_DIR_PRESSURE_DETAILS, exist_ok=True)
    # only the pdf is the paper figure, the svg is a working copy
    for ext, out_dir in (
        ("pdf", config.PLOT_DIR_PRESSURE),
        ("svg", config.PLOT_DIR_PRESSURE_DETAILS),
    ):
        out = os.path.join(out_dir, f"{short(arch)}_qps.{ext}")
        fig.savefig(out, dpi=150)
        print(f"[OK] {out}")
    plt.close(fig)


def plot_qps_legend(variants: list[str]):
    """Its own file, so the two arch figures can share one legend."""
    fig = plt.figure(figsize=(9, 0.9))

    def entry(variant):
        color = variant_color(variant)
        if variant == "repl-dynamic":
            glow = Line2D([], [], lw=4.5, color=color, alpha=0.2)
            dash = Line2D([], [], lw=1.4, ls="--", color=color)
            return (glow, dash), variant_label(variant)
        return Line2D([], [], lw=1.4, color=color), variant_label(variant)

    # baselines on the first row, SPARe on the second: the legend fills
    # column by column, so the two rows are zipped together here
    rows = [[v for v in variants if v in STOCK], [v for v in variants if v in REPL]]
    handles, labels = [], []
    for pair in zip(*rows):
        for variant in pair:
            handle, label = entry(variant)
            handles.append(handle)
            labels.append(label)

    fig.legend(
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="center",
        fontsize=QPS_FONTSIZE,
        ncol=len(rows[0]),
        columnspacing=1.5,
        handlelength=1.6,
        frameon=False,
    )
    os.makedirs(config.PLOT_DIR_PRESSURE, exist_ok=True)
    out = os.path.join(config.PLOT_DIR_PRESSURE, "qps_legend.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02, dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def base(arch: str, variant: str) -> str:
    return os.path.join(RESULT_DIR, arch, "pressure", f"ann-pressure-{variant}")


def monitor_dir(arch: str) -> str:
    return os.path.join(RESULT_DIR, arch, "monitor")


def monitor_label(variant: str) -> str:
    return "ann-pressure-repl" if variant in REPL else "ann-pressure"


def phases(arch: str, variant: str) -> pd.DataFrame:
    df = pd.read_csv(
        f"{base(arch, variant)}-phases.csv",
        parse_dates=["start_time", "end_time"],
    )
    t0 = df.start_time.iloc[0]
    df["start_s"] = (df.start_time - t0).dt.total_seconds()
    df["end_s"] = (df.end_time - t0).dt.total_seconds()
    return df


def rate(df: pd.DataFrame, col: str) -> pd.Series:
    """Cumulative counter -> per second, on the sample grid."""
    return pd.to_numeric(df[col], errors="coerce").diff() / df.elapsed.diff()


def smooth(values: pd.Series, window: int = EVENT_WINDOW) -> pd.Series:
    """Rolling mean of a rate. numa balancing works in bursts: at the 0.5s
    sample grid a series swings between 0 and 10k every other point, and on a
    log axis that paints the whole panel solid. A centered mean keeps the
    area under the curve and shows the burst rate instead of the sampling."""
    return values.rolling(window, center=True, min_periods=1).mean()


def read_bandwidth(arch: str, variant: str, t0, t1) -> pd.DataFrame:
    """pcm-memory system read/write, cut to this variant's window."""
    path = os.path.join(
        monitor_dir(arch), f"pcm_memory_{monitor_label(variant)}.csv"
    )
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, header=[0, 1])
    date = df[[c for c in df.columns if c[1] == "Date"][0]].astype(str)
    time = df[[c for c in df.columns if c[1] == "Time"][0]].astype(str)
    df["t"] = pd.to_datetime(date + " " + time, errors="coerce")
    df = df[(df.t >= t0) & (df.t <= t1)].copy()
    df["elapsed"] = (df.t - t0).dt.total_seconds()
    df["read_gb"] = (
        pd.to_numeric(df[("System", "Read")], errors="coerce") / 1024
    )
    df["write_gb"] = (
        pd.to_numeric(df[("System", "Write")], errors="coerce") / 1024
    )
    return df


def read_locality(arch: str, variant: str, t0, t1) -> pd.Series:
    """Share of memory accesses served by the local socket, from pcm. The
    hardware view, and unlike pg_stats it exists for the stock variants too."""
    path = os.path.join(monitor_dir(arch), f"pcm_{monitor_label(variant)}.csv")
    if not os.path.exists(path):
        return pd.Series(dtype=float)
    df = pd.read_csv(path, header=[0, 1])
    date = df[[c for c in df.columns if c[1] == "Date"][0]].astype(str)
    time = df[[c for c in df.columns if c[1] == "Time"][0]].astype(str)
    t = pd.to_datetime(date + " " + time, errors="coerce")
    inside = (t >= t0) & (t <= t1)
    return pd.to_numeric(df.loc[inside, ("System", "LOCAL")], errors="coerce")


def read_sched_events(arch: str, variant: str, t0, t1) -> pd.DataFrame:
    """perf stat tracepoints. Its timestamps are relative to its own start,
    so the first line of the file anchors them."""
    path = os.path.join(monitor_dir(arch), f"perf_{monitor_label(variant)}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        head = f.readline()
    if not head.startswith("# start "):
        return pd.DataFrame()
    anchor = pd.to_datetime(head.removeprefix("# start ").strip())
    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        usecols=[0, 1, 3],
        names=["offset", "count", "event"],
    )
    df["t"] = anchor + pd.to_timedelta(
        pd.to_numeric(df.offset, errors="coerce"), unit="s"
    )
    df = df[(df.t >= t0) & (df.t <= t1)].copy()
    df["elapsed"] = (df.t - t0).dt.total_seconds()
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df["event"] = df.event.str.removeprefix("sched:sched_").str.removesuffix(
        "_numa"
    )
    return df


PG_NODE = re.compile(
    r"^node (\d+)(\(main\))?\s*: locality=([\d.]+) ptes=([\d.]+)K"
)


def read_coverage(arch: str, variant: str) -> pd.DataFrame:
    """Replication coverage per phase, from the pg_stats blocks in the log.

    Per node, pg_stats reports what fraction of the pages its page table
    points at are local. Averaged over nodes that is the share of accesses
    served by a local copy: 100% is fully replicated, and 1/nodes is what a
    single copy gives, which is where the stock variants sit by construction.
    """
    path = f"{base(arch, variant)}.log"
    if not os.path.exists(path):
        return pd.DataFrame()
    rows, phase, nodes = [], None, []

    def flush():
        if phase and nodes:
            n = len(nodes)
            ptes = sum(x[1] for x in nodes) / n
            rows.append(
                {
                    "phase": phase,
                    "nodes": n,
                    "coverage": sum(x[0] for x in nodes) / n * 100,
                    "gb_max": ptes * n * 4096 / 1024**3,
                }
            )

    for line in open(path):
        if line.startswith("== pg_stats @"):
            flush()
            phase = line.split("@")[1].replace("end", "").strip()
            nodes = []
        elif phase and (m := PG_NODE.match(line)):
            nodes.append((float(m.group(3)), float(m.group(4)) * 1000))
    flush()
    df = pd.DataFrame(rows)
    if not df.empty:
        # coverage is memory weighted: a single copy already reads 1/N local,
        # so back out the share of pages that actually carry a duplicate.
        # coverage = 1/N + (1 - 1/N) * duplicated
        floor = 100 / df.nodes
        # not "duplicated": that name collides with DataFrame.duplicated
        df["dup_pct"] = ((df.coverage - floor) / (100 - floor) * 100).clip(
            lower=0
        )
        df["gb_replicated"] = df.gb_max * df.coverage / 100  # whole mapping
        df["gb_duplicated"] = df.gb_max * (1 - 1 / df.nodes) * df.dup_pct / 100
    return df


# the M limits we run, spelled the way their phase labels spell them
LIMIT_NAMES = {"4500M": "4.5G", "3500M": "3.5G", "2560M": "2.5G"}

# no memory.high at all: the phase label names what the machine actually has
MACHINE_RAM = {"silver": "384G", "gold": "768G"}


def _display_limit(limit: str) -> str:
    """memory.high values in M read as GB, e.g. "2560M" -> "2.5G"."""
    return LIMIT_NAMES.get(limit, limit)


def fits_pct(limit: str, full_gb: float) -> str:
    """How much of a fully replicated footprint this limit still allows."""
    if limit == "max" or not full_gb:
        return "100%"
    gb = float(limit[:-1]) / (1024 if limit.endswith("M") else 1)
    return f"{min(100, gb / full_gb * 100):.0f}%"


def shade_phases(
    ax,
    ph,
    label: bool = False,
    full_gb: float = 0,
    divider: bool = True,
    band_top: float = 1.0,
    radius: float = 0,
    label_fontsize: float = 8,
    label_one_line: bool = False,
    max_label: str | None = None,
):
    # x in data seconds, y in axes fraction: those two units cover very
    # different physical distances, so a uniform rounding_size on a patch
    # in that mixed space draws an ellipse, not a round corner. mutation_aspect
    # corrects for it, computed from figure/subplot geometry alone (inches
    # and the data xlim) rather than rendered pixels, so it comes out the
    # same in a vector PDF/SVG as in a raster PNG.
    if radius:
        ax.figure.canvas.draw()  # settles autoscale, so xlim below is final
        pos = ax.get_position()
        axes_w_in = pos.width * ax.figure.get_figwidth()
        axes_h_in = pos.height * ax.figure.get_figheight()
        xlim = ax.get_xlim()
        scale_x = axes_w_in / (xlim[1] - xlim[0])  # inches per data-second
        mutation_aspect = scale_x / axes_h_in
        rounding_size = (radius / 72) / scale_x  # radius in points -> seconds
    blended = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, row in ph.iterrows():
        if i % 2:
            if radius:
                ax.add_patch(
                    mpatches.FancyBboxPatch(
                        (row.start_s, 0),
                        row.end_s - row.start_s,
                        band_top,
                        boxstyle=f"round,pad=0,rounding_size={rounding_size}",
                        mutation_aspect=mutation_aspect,
                        transform=blended,
                        facecolor=BAND,
                        edgecolor="none",
                        clip_on=False,
                        zorder=0,
                    )
                )
            else:
                ax.axvspan(
                    row.start_s,
                    row.end_s,
                    ymax=band_top,
                    color=BAND,
                    zorder=0,
                    clip_on=False,
                )
        if divider:
            ax.axvline(row.start_s, color="0.6", lw=0.6, zorder=1)
        if label:
            text = _display_limit(row.limit)
            if row.limit == "max" and max_label:
                text = max_label
            pct = fits_pct(row.limit, full_gb) if full_gb else ""
            if pct:
                text += f" ({pct})" if label_one_line else f"\n{pct}"
            ax.text(
                (row.start_s + row.end_s) / 2,
                1.02,
                text,
                transform=ax.get_xaxis_transform(),
                ha="center",
                fontsize=label_fontsize,
            )


def full_footprint(arch: str) -> float:
    """The fully replicated footprint: every copy, plus what the process
    needs anyway. Measured off the repl variants, which all agree."""
    fulls = []
    for variant in REPL:
        cov = read_coverage(arch, variant)
        if cov.empty:
            continue
        cg = pd.read_csv(f"{base(arch, variant)}-cgroup.csv")
        first = cg[cg.phase == cov.phase.iloc[0]]
        if not first.empty:
            r = cov.iloc[0]
            fulls.append(
                first.current_mb.iloc[-1] / 1024 - r.gb_replicated + r.gb_max
            )
    return sum(fulls) / len(fulls) if fulls else 0


def plot_variant(arch: str, variant: str, full_gb: float = 0):
    ph = phases(arch, variant)
    t0, t1 = ph.start_time.iloc[0], ph.end_time.iloc[-1]

    runs = pd.read_csv(
        f"{base(arch, variant)}-ann.csv", parse_dates=["start_time"]
    )
    runs = runs.dropna(subset=["phase"])
    runs["elapsed"] = (runs.start_time - t0).dt.total_seconds()

    cg = pd.read_csv(f"{base(arch, variant)}-cgroup.csv")
    bw = read_bandwidth(arch, variant, t0, t1)
    ev = read_sched_events(arch, variant, t0, t1)

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    fig.suptitle(f"pressure staircase: {variant} ({short(arch)})", y=0.995)

    # 1. QPS, the primary result
    ax = axes[0]
    ax.plot(runs.elapsed, runs.qps, lw=1, color="#1f77b4")
    ax.set_ylabel("QPS")
    ax.set_ylim(bottom=0)
    shade_phases(ax, ph, label=True, full_gb=full_gb)

    # 2. memory read bandwidth, the other primary
    ax = axes[1]
    if not bw.empty:
        ax.plot(bw.elapsed, bw.read_gb, lw=1, color="#2ca02c")
    ax.set_ylabel("read GB/s")
    ax.set_ylim(bottom=0)
    shade_phases(ax, ph)

    # 3. what the cgroup is holding, and what the limit let it hold
    ax = axes[2]
    gb = 1024**3
    ax.plot(cg.elapsed, cg.current_mb / 1024, lw=1, label="current", color="k")
    ax.plot(
        cg.elapsed,
        pd.to_numeric(cg.anon, errors="coerce") / gb,
        lw=1,
        label="anon (replicas)",
        color="#9467bd",
    )
    ax.plot(
        cg.elapsed,
        pd.to_numeric(cg.file, errors="coerce") / gb,
        lw=1,
        label="file (index)",
        color="#8c564b",
    )
    # how much of that is copies, from pg_stats. One dump per phase, so it is
    # a step, and the label is the share of the index it stands for
    cov = read_coverage(arch, variant)
    if not cov.empty:
        by_phase = cov.set_index("phase")
        for _, r in ph.iterrows():
            if r.phase not in by_phase.index:
                continue
            c = by_phase.loc[r.phase]
            ax.hlines(
                c.gb_duplicated,
                r.start_s,
                r.end_s,
                color="#1f77b4",
                lw=2,
            )
            ax.text(
                (r.start_s + r.end_s) / 2,
                c.gb_duplicated + 0.25,
                f"{c.dup_pct:.0f}% dup",
                ha="center",
                fontsize=8,
                color="#1f77b4",
            )
        ax.plot(
            [], [], color="#1f77b4", lw=2, label="duplicates (subset of anon)"
        )

    ax.set_ylabel("GB")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    shade_phases(ax, ph)

    # 4. the explanatory counters: why the curves above moved. Log scale, the
    # rates span three decades, and only series that fired, to keep it legible
    ax = axes[3]
    # reclaimed and reclaimed_replicas are different pages: the first is a
    # main copy going out, the second a duplicate, so pin their colors rather
    # than letting the cycler pair them by accident.
    # reclaimed_replicas_from_main is disjoint from reclaimed_replicas, not a
    # subset: replicas actually lost is the two summed
    series = {
        "replicas allocated/s": (
            rate(cg, "repl_repl_allocated")
            if "repl_repl_allocated" in cg.columns
            else None,
            "#2ca02c",
        ),
        "reclaimed (main)/s": (
            rate(cg, "repl_reclaimed")
            if "repl_reclaimed" in cg.columns
            else None,
            "#d62728",
        ),
        "replicas reclaimed/s": (
            rate(cg, "repl_reclaimed_replicas")
            if "repl_reclaimed_replicas" in cg.columns
            else None,
            "#9467bd",
        ),
        # the headroom floor doing its job: a replica refused and pointed at
        # main instead, so it never became memory reclaim had to take back
        "replicas skipped (pressure)/s": (
            rate(cg, "repl_skipped_pressure")
            if "repl_skipped_pressure" in cg.columns
            else None,
            "#e7ba52",
        ),
    }
    if variant == "numa-balancing":
        series = {
            "numa hint faults/s": (rate(cg, "vm_numa_hint_faults"), "#1f77b4"),
            "pages migrated/s": (rate(cg, "vm_numa_pages_migrated"), "#ff7f0e"),
            **series,
            "replicas lost with their main/s": (
                rate(cg, "repl_reclaimed_replicas_from_main")
                if "repl_reclaimed_replicas_from_main" in cg.columns
                else None,
                "#8c564b",
            ),
        }
    peak = 0
    for label, (values, color) in series.items():
        if values is not None and values.fillna(0).sum() > 0:
            y = smooth(values)
            peak = max(peak, y.max())
            ax.plot(cg.elapsed, y, lw=1, label=label, color=color)
    # the sched tracepoints are a different family, so keep them off the
    # counter colors: dotted alone reads as the same series at this density
    if variant == "numa-balancing":
        ev_colors = {"move": "#17becf", "stick": "#bcbd22", "swap": "#e377c2"}
        for name, g in ev.groupby("event") if not ev.empty else []:
            if g["count"].sum() > 0:
                y = smooth(g["count"] / 0.5)
                peak = max(peak, y.max())
                ax.plot(
                    g.elapsed,
                    y,
                    lw=1,
                    ls=":",
                    color=ev_colors.get(name, "0.5"),
                    label=f"{name}/s",
                )
    ax.set_yscale("log")
    ax.set_ylabel(f"events/s ({EVENT_WINDOW // 2}s mean)")  # 0.5s samples
    ax.set_xlabel("time (s)")
    if peak > 0:
        # floor the axis a couple of decades under the peak, and leave a band
        # on top for the legend to sit in rather than over the curves
        ax.set_ylim(max(peak / 1e4, 0.5), peak * 30)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", fontsize=7, ncol=3, framealpha=0.9)
    shade_phases(ax, ph)

    # locality is a ratio of two small counters, so smooth it over ~5s before
    # dividing rather than per sample
    hint = pd.to_numeric(cg["vm_numa_hint_faults"], errors="coerce").diff()
    if hint.fillna(0).sum() > 0:
        local = pd.to_numeric(
            cg["vm_numa_hint_faults_local"], errors="coerce"
        ).diff()
        win = 10
        pct = (local.rolling(win).sum() / hint.rolling(win).sum() * 100).where(
            hint.rolling(win).sum() > 20
        )
        twin = ax.twinx()
        twin.plot(cg.elapsed, pct, lw=1.2, color="0.35", ls="--")
        twin.set_ylabel("% hint faults local (dashed)")
        twin.set_ylim(0, 100)

    os.makedirs(config.PLOT_DIR_PRESSURE_DETAILS, exist_ok=True)
    out = os.path.join(
        config.PLOT_DIR_PRESSURE_DETAILS, f"{short(arch)}_{variant}.png"
    )
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def variant_label(variant: str) -> str:
    return VARIANT_LABELS.get(variant, variant)


def variant_color(variant: str):
    """Patched in blues, stock in oranges, same convention as plot_ann.
    dynamic gets its own hue, a fourth blue is too close to the others."""
    if variant == "repl-dynamic":
        return "#0e6e51"  # darker teal, leaning green
    stock = sns.color_palette(config.LINUX_COLOR, n_colors=3 + len(STOCK))
    patched = sns.color_palette(config.SPARE_COLOR, n_colors=3 + len(REPL))
    if variant in REPL:
        return patched[3 + REPL.index(variant)]
    return stock[3 + STOCK.index(variant)]


def plot_metric(
    arch: str, variants: list[str], metric: str, full_gb: float = 0
):
    """All variants on one axis, one metric. QPS and bandwidth get a plot
    each: on shared axes the twelve traces hide each other."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    band = None

    for variant in variants:
        ph = phases(arch, variant)
        band = band if band is not None else ph
        t0, t1 = ph.start_time.iloc[0], ph.end_time.iloc[-1]
        color = variant_color(variant)

        if metric == "bandwidth":
            df = read_bandwidth(arch, variant, t0, t1)
            if df.empty:
                continue
            x, y = df.elapsed, df.read_gb.rolling(10, center=True).median()
        elif metric == "memory":
            df = pd.read_csv(f"{base(arch, variant)}-cgroup.csv")
            x, y = (
                df.elapsed,
                df.current_mb / 1024,
            )  # already a step, no smoothing
        else:
            df = pd.read_csv(
                f"{base(arch, variant)}-ann.csv", parse_dates=["start_time"]
            ).dropna(subset=["phase"])
            df["elapsed"] = (df.start_time - t0).dt.total_seconds()
            df = df.sort_values("elapsed")
            x, y = df.elapsed, df.qps.rolling(5, center=True).median()
        ax.plot(x, y, lw=1.4, color=color, label=variant_label(variant))

    ax.set_xlabel("time (s)")
    ax.set_ylabel(
        {
            "bandwidth": "memory read bandwidth (GB/s)",
            "memory": "cgroup memory.current (GB)",
        }.get(metric, "QPS")
    )
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    shade_phases(ax, band, label=True, full_gb=full_gb)

    os.makedirs(config.PLOT_DIR_PRESSURE_DETAILS, exist_ok=True)
    out = os.path.join(
        config.PLOT_DIR_PRESSURE_DETAILS, f"{short(arch)}_{metric}.png"
    )
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def plot_summary(arch: str, variants: list[str]):
    """QPS against the limit it ran under, over how much of the index the
    limit still let us replicate."""
    fig, (ax, ax_mem, ax_loc, ax_cov) = plt.subplots(
        4,
        1,
        figsize=(9, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1.2, 1.2, 1.2]},
    )
    order = limits = None
    nodes = 0
    fulls = []
    for variant in variants:
        path = f"{base(arch, variant)}-ann.csv"
        if not os.path.exists(path):
            continue
        ph = phases(arch, variant)
        order = list(ph.phase)
        limits = list(ph.limit)
        runs = pd.read_csv(path, parse_dates=["start_time"]).dropna(
            subset=["phase"]
        )
        # steady state only: the reclaim transient at the top of a phase is
        # not what the limit costs, so drop the first STEADY_FROM of it
        med = {}
        for _, r in ph.iterrows():
            begins = r.start_time + (r.end_time - r.start_time) * STEADY_FROM
            g = runs[(runs.start_time >= begins) & (runs.phase == r.phase)]
            med[r.phase] = g.qps.median()
        ax.plot(
            range(len(order)),
            [med.get(p, float("nan")) for p in order],
            marker="o",
            color=variant_color(variant),
            label=variant_label(variant),
        )

        loc = []
        for _, r in ph.iterrows():
            begins = r.start_time + (r.end_time - r.start_time) * STEADY_FROM
            v = read_locality(arch, variant, begins, r.end_time)
            loc.append(v.median() if not v.empty else float("nan"))
        ax_loc.plot(
            range(len(order)), loc, marker="o", color=variant_color(variant)
        )

        cg_all = pd.read_csv(f"{base(arch, variant)}-cgroup.csv")
        ax_mem.plot(
            range(len(order)),
            [
                cg_all[cg_all.phase == p].current_mb.iloc[-1] / 1024
                if not cg_all[cg_all.phase == p].empty
                else float("nan")
                for p in order
            ],
            marker="o",
            color=variant_color(variant),
        )

        cov = read_coverage(arch, variant)
        if not cov.empty:
            nodes = int(cov.nodes.iloc[0])
            # The ceiling a limit allows: everything that is not a copy is
            # reclaimable dead weight and goes first (base falls from ~4G to
            # ~0.4G as soon as pressure arrives), so what the limit really
            # buys is (limit - irreducible base) worth of copies.
            cg = pd.read_csv(f"{base(arch, variant)}-cgroup.csv")
            per_phase = cov.set_index("phase")
            bases = []
            for p in order:
                c = cg[cg.phase == p]
                if not c.empty and p in per_phase.index:
                    bases.append(
                        c.current_mb.iloc[-1] / 1024
                        - per_phase.loc[p].gb_replicated
                    )
            if bases:
                # (all copies, base at max) -> their sum is the fully
                # replicated footprint the limits are measured against
                fulls.append((cov.gb_max.median(), bases[0]))
            by_phase = per_phase.dup_pct
            ax_cov.plot(
                range(len(order)),
                [by_phase.get(p, float("nan")) for p in order],
                marker="o",
                color=variant_color(variant),
            )

    # a limit only buys what it is worth against the fully replicated
    # footprint, so label each one with the share of it that still fits
    full_gb = sum(f[0] + f[1] for f in fulls) / len(fulls) if fulls else 0
    if order:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            [
                p if p == _display_limit(limit) else f"{p}\n{limit}"
                for p, limit in zip(order, limits)
            ]
        )
        # the budget gets its own axis on top, so the only percentage on the
        # left is the one the curves are plotted against
        if full_gb:
            top = ax.secondary_xaxis("top")
            top.set_xticks(range(len(order)))
            top.set_xticklabels([fits_pct(x, full_gb) for x in limits])
            top.set_xlabel(f"% of the {full_gb:.1f}GB full replication needs")
    ax.set_ylabel("median QPS (steady state)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    # the percentage is a share of the index, so pair it with the GB it
    # stands for: 80% is easier to read as "6.0 of 7.5GB of copies"
    if fulls and nodes:
        dup_gb = sum(f[0] for f in fulls) / len(fulls) * (1 - 1 / nodes)
        ax_gb = ax_cov.secondary_yaxis(
            "right",
            functions=(
                lambda pct: pct / 100 * dup_gb,
                lambda gb: gb / dup_gb * 100,
            ),
        )
        ax_gb.set_ylabel(f"GB of duplicates (of {dup_gb:.1f}GB)")
    ax_loc.set_ylabel("local memory\naccesses (%)")
    ax_loc.set_ylim(0, 105)
    ax_loc.grid(axis="y", alpha=0.3)
    ax_mem.set_ylabel("memory used (GB)")
    ax_mem.set_ylim(bottom=0)
    ax_mem.grid(axis="y", alpha=0.3)
    # not "index replicated": at 25% the whole index is still there, just in
    # one copy. This is how much of it each node reaches locally.
    ax_cov.set_ylabel("pages actually\nduplicated (%)")
    ax_cov.set_xlabel("memory.high")
    ax_cov.set_ylim(0, 105)
    ax_cov.grid(axis="y", alpha=0.3)

    os.makedirs(config.PLOT_DIR_PRESSURE_DETAILS, exist_ok=True)
    out = os.path.join(
        config.PLOT_DIR_PRESSURE_DETAILS, f"{short(arch)}_summary.png"
    )
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def read_node_mem(arch: str, variant: str) -> pd.DataFrame:
    """Per node anon / mapped, from collect_mem. Machine wide rather than
    cgroup scoped, so it only reads cleanly on an otherwise idle host."""
    path = os.path.join(monitor_dir(arch), f"mem_{monitor_label(variant)}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["t"] = pd.to_datetime(df.time)
    return df


def short(arch: str) -> str:
    return config.ARCH_SUBNAMES.get(arch, arch)


def make_plot_pressure():
    found = False
    for arch in sorted(os.listdir(RESULT_DIR)):
        available = [
            v
            for v in STOCK + REPL
            if os.path.exists(f"{base(arch, v)}-ann.csv")
        ]
        if not available:
            continue
        found = True
        full_gb = full_footprint(arch)
        for variant in available:
            plot_variant(arch, variant, full_gb)
        plot_qps(arch, available, full_gb)
        plot_metric(arch, available, "bandwidth", full_gb)
        plot_metric(arch, available, "memory", full_gb)
        plot_summary(arch, available)
    if not found:
        print(f"[WARN] no pressure results under {RESULT_DIR}/*/pressure")
