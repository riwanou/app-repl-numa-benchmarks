"""DRAM bandwidth over time, with the bench phases shaded.

Shows whether a number is a steady state or a slice of a curve, which a phase
mean cannot say. One plot per arch that has the capture.

    uv run run.py plot-sharing
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

import config

MB_TO_GB = 1 / 1024

# for the title, keyed by the short arch name
MACHINES = {
    "gold": "Intel Xeon Gold 6130",
    "silver": "Intel Xeon Silver 4216",
    "plat": "Intel Xeon Platinum 8568Y+",
    "gold5320": "Intel Xeon Gold 5320",
}

# orange is the directory churn everywhere it appears: the write bandwidth and
# the updates that cause it. Checked for colourblind separation and contrast
# as a set.
BLUE, ORANGE, VIOLET, AQUA = "#2a78d6", "#eb6834", "#4a3aa7", "#1baf7a"
INK = "#52514e"
BAND = "#f4f3f0"

# I and A run on top of each other in the shared phase, so they get the pair
# that is furthest apart: aqua vs orange, not two blues. A only costs a snoop
# when the request comes from a socket other than the one it recorded, so the
# label says what A holds, not what it forces: remote and disjoint sit at ~100%
# A with zero snoops.
STATE_COLORS = {"I": AQUA, "A": ORANGE, "S": VIOLET}
STATE_LABELS = {
    "I": "I  no remote copy",
    "A": "A  maybe a remote copy",
    "S": "S  shared, both clean",
}

# what each phase does, in place of the bench's own shorthand names
PHASE_LABELS = {
    "local": "readers near",
    "remote": "readers far",
    "disjoint": "split, own lines",
    "shared": "split, same lines",
}
POLICY_LABELS = {
    "membind": "data 1 node · access 2 nodes",
    "interleaved": "data 2 nodes · access 2 nodes",
}


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


def core(arch: str, label: str) -> pd.DataFrame:
    """The core side capture, for the demand L3 hit ratio."""
    path = os.path.join(config.RESULT_DIR, arch, "monitor", f"pcm_{label}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    # this capture puts Date and Time under System, not in their own group
    df = pd.read_csv(path, header=[0, 1])
    time = pd.to_datetime(
        df[("System", "Date")].astype(str) + " " + df[("System", "Time")].astype(str),
        errors="coerce",
    )
    return pd.DataFrame(
        {
            "time": time,
            # a ratio in the capture, a percentage on the axis
            "l3hit": pd.to_numeric(df[("System", "L3HIT")], errors="coerce") * 100,
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


PAD_FRAC = 0.06   # the gap left between two phases, as a share of one phase


def phase_span(windows: list[tuple[float, float]]) -> tuple[float, float]:
    """What one phase measures, to the nearest 10 s, and the gap to leave after
    it. Taken from the run rather than hardcoded: the bench's SECS moves."""
    span = round(max(b - a for a, b in windows) / 10) * 10
    return span, span * PAD_FRAC


def compress(secs: pd.Series, windows: list[tuple[float, float]],
             span: float, pad: float) -> pd.Series:
    """Real seconds to axis seconds, with the gap between two phases taken
    out. That gap is the next phase's 8 GB memset plus the idle sync, which is
    neither measured nor plotted, and leaving it in pushed every phase along by
    the gaps before it, so no tick landed on a phase boundary. Laid end to end
    a phase starts at a multiple of the span and the round ticks fall where
    they belong. Samples outside a window come back NaN, which is what breaks
    the line between phases."""
    out = pd.Series(float("nan"), index=secs.index, dtype=float)
    for i, (a, b) in enumerate(windows):
        inside = (secs >= a) & (secs <= b)
        out[inside] = i * (span + pad) + (secs[inside] - a)
    return out


def phase_box(i: int, a: float, b: float, span: float,
              pad: float) -> tuple[float, float]:
    """Where phase i sits on the axis."""
    return i * (span + pad), i * (span + pad) + (b - a)


def steady(v: float) -> str:
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}" if v >= 1 else f"{v:.2f}"


def annotate(axis, t, entries, windows, boxes):
    """Print where each line settles, at the right of its phase, a few points
    above that line. The offset is in points off the value itself, not a slice
    of the panel: a fraction of the height throws a label sitting at 61 on a
    panel scaled to 1100 far up into empty space. Two labels are only pushed
    apart when they would actually touch, and stepped over any other line they
    would land on. The tail of the window, not the mean: most phases open with
    a transient that no steady number should include."""
    per_point = axis.figure.dpi / 72

    def points(value):
        return axis.transData.transform((0, value))[1] / per_point

    for (a, b), (lo, hi) in zip(windows, boxes):
        tail = (t >= b - (b - a) * 0.4) & (t <= b)
        marks = []
        for values, color in entries:
            value = values[tail].median()
            if not pd.isna(value):
                marks.append([points(value), value, steady(value), color])

        # the other lines are obstacles too: a label for 0.20 sitting just
        # above its own line still lands on the read line plateauing at 18
        marks.sort()
        lines = [mark[0] for mark in marks]
        placed = None
        for mark in marks:
            target = mark[0] + 5
            if placed is not None:
                target = max(target, placed + 9)
            for line in lines:            # ascending, so one pass clears all
                if target - 3 <= line <= target + 8:
                    target = line + 6
            placed = target
            mark[0] = target - points(mark[1])

        for offset, value, text, color in marks:
            axis.annotate(text, (hi - (hi - lo) * 0.03, value),
                          textcoords="offset points", xytext=(0, offset),
                          ha="right", va="bottom", fontsize=7, color=color,
                          annotation_clip=False)


def draw(axis, x, y, label, color, style="-"):
    """One line per phase: NaN x in the gaps, so nothing joins them up."""
    axis.plot(x, y, label=label, lw=1.1, ls=style, color=color)


# Every gap below is an inch, placed by hand. Equal panel heights are the
# point: a legend offset in axes fractions gives a different gap on every row
# when the rows differ in height, which is what made the spacing look random.
FIG_W, FIG_H = 6.2, 9.6
LEFT, RIGHT = 0.107, 0.985
LEGEND = 0.17     # the line of keys above a panel
LEGEND_GAP = 0.05  # legend to the panel it labels
PANEL_GAP = 0.11  # panel to the next legend
PANELS = 4        # stacked measures per block
AXIS = 0.40       # tick labels plus "seconds" under a block
HEADER = 0.30     # the policy line and the rule under it
PHASES = 0.24     # the row of phase names
BLOCK_GAP = 0.12  # between the two blocks
MARGIN = 0.10


def plot(arch: str, sub: str, label: str):
    df = load(arch, label)
    start = df["time"].iloc[0]
    secs = (df["time"] - start).dt.total_seconds()
    coh = coherence(arch, label)
    csecs = (coh["time"] - start).dt.total_seconds() if not coh.empty else None

    spans = [
        (
            (row["start_time"] - start).total_seconds(),
            (row["end_time"] - start).total_seconds(),
            row.get("policy", ""),
            row["phase"],
            float(row["read_gb_s"]),
        )
        for _, row in phases(arch, label).iterrows()
    ]
    policies = list(dict.fromkeys(p for _, _, p, _, _ in spans))

    # panels take whatever the fixed gaps leave, so the page is always full
    # and never overruns
    per_block = (HEADER + PANELS * (LEGEND + LEGEND_GAP)
                 + (PANELS - 1) * PANEL_GAP + AXIS)
    panel = (
        FIG_H - 2 * MARGIN - BLOCK_GAP * (len(policies) - 1)
        - (per_block + PHASES) * len(policies)
    ) / (PANELS * len(policies))

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    inch = 1 / FIG_H          # one inch as a figure fraction
    cursor = 1 - MARGIN * inch

    for i, policy in enumerate(policies):
        group = [(a, b, ph, gbs) for a, b, p, ph, gbs in spans
                 if p == policy]
        windows = [(a, b) for a, b, _, _ in group]
        base = group[0][0]
        windows = [(a - base, b - base) for a, b in windows]

        cursor -= HEADER * inch
        fig.text((LEFT + RIGHT) / 2, cursor + 0.075 * inch,
                 POLICY_LABELS.get(policy, policy), ha="center", va="bottom",
                 fontsize=10.5, color="#0b0b0b")
        fig.add_artist(Line2D([LEFT, RIGHT], [cursor, cursor],
                              color="#cfcecb", lw=1))

        cursor -= PHASES * inch

        panels = []
        for _ in range(PANELS):
            legend_y = cursor - LEGEND * inch
            cursor = legend_y - LEGEND_GAP * inch - panel * inch
            panels.append((fig.add_axes([LEFT, cursor, RIGHT - LEFT,
                                         panel * inch]), legend_y))
            cursor -= PANEL_GAP * inch

        (ax, _), (ax2, _), (ax3, _), (ax4, _) = panels
        cursor += PANEL_GAP * inch - AXIS * inch
        fig.text((LEFT + RIGHT) / 2, cursor + 0.04 * inch, "seconds",
                 ha="center", va="bottom", fontsize=8, color=INK)
        cursor -= BLOCK_GAP * inch

        span, pad = phase_span(windows)
        x = compress(secs - base, windows, span, pad)

        # what the threads actually read, flat across each phase because the
        # bench reports one number per phase. On the same axis as the DRAM
        # line on purpose: the gap between them is the reuse between threads.
        app = pd.Series(float("nan"), index=secs.index, dtype=float)
        for (a, b), (_, _, _, gbs) in zip(windows, group):
            app[((secs - base) >= a) & ((secs - base) <= b)] = gbs

        draw(ax, x, app, "app reads", VIOLET)
        draw(ax, x, df["read"], "DRAM reads", BLUE)
        draw(ax, x, df["write"], "DRAM writes", ORANGE)
        entries = {ax: [(app, VIOLET), (df["read"], BLUE), (df["write"], ORANGE)],
                   ax2: [], ax3: [], ax4: []}

        cores = core(arch, label)
        if not cores.empty:
            ksecs = (cores["time"] - start).dt.total_seconds()
            draw(ax4, compress(ksecs - base, windows, span, pad),
                 cores["l3hit"], "demand L3 hits", AQUA)
            entries[ax4].append((cores["l3hit"], AQUA))

        if not coh.empty:
            cx = compress(csecs - base, windows, span, pad)

            # snoops sit exactly on the updates whenever there is churn, so
            # dash them: the overlap is the point, a second solid line hides one
            for event, name, style in (
                ("UNC_M2M_DIRECTORY_UPDATE.ANY", "directory updates", "-"),
                ("UNC_CHA_DIR_LOOKUP.SNP", "snoops", "--"),
            ):
                if event in coh:
                    color = ORANGE if style == "-" else BLUE
                    draw(ax2, cx, coh[event], name, color, style)
                    entries[ax2].append((coh[event], color))

            # what state each lookup found, as a rate. A share of the total
            # reads as an abstraction; a rate says how much is going on.
            events = {s: f"UNC_M2M_DIRECTORY_LOOKUP.STATE_{s}" for s in "ISA"}
            for state in ("I", "A", "S"):
                if events[state] in coh:
                    draw(ax3, cx, coh[events[state]], STATE_LABELS[state],
                         STATE_COLORS[state])
                    entries[ax3].append((coh[events[state]],
                                         STATE_COLORS[state]))

        # a band per phase, the gaps left white: that is what separates one
        # phase from the next
        boxes = [phase_box(k, a, b, span, pad)
                 for k, (a, b) in enumerate(windows)]
        for lo, hi in boxes:
            for axis in (ax, ax2, ax3, ax4):
                axis.axvspan(lo, hi, color=BAND, lw=0, zorder=0)

        span_ax = ax.get_xaxis_transform()
        for (a, b), (_, _, phase, _) in zip(boxes, group):
            ax.text((a + b) / 2, 1 + (LEGEND + LEGEND_GAP + 0.06) / panel,
                    PHASE_LABELS.get(phase, phase), transform=span_ax,
                    ha="center", va="bottom", fontsize=8.5, color=INK)

        for axis, name in ((ax, "reads and writes\nGB/s"),
                           (ax2, "directory\nupdates M/s"),
                           (ax3, "directory\nlookups M/s"),
                           (ax4, "demand L3\nhits %")):
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#cfcecb")
            axis.tick_params(colors=INK, labelsize=7.5, length=2.5,
                             color="#cfcecb")
            axis.set_axisbelow(True)
            axis.margins(x=0.015, y=0.10)
            axis.set_ylabel(name, color=INK, fontsize=8, labelpad=2)
            axis.set_xlim(-pad / 2, boxes[-1][1] + pad / 2)
            # a tick at each phase boundary and each midpoint; with the gaps
            # off the axis they land on round numbers
            ticks = [(0.0, 0.0)] + [
                (k * (span + pad) + h, k * span + h)
                for k in range(len(boxes)) for h in (span / 2, span)
            ]
            axis.set_xticks([t for t, _ in ticks])
            axis.set_xticklabels([f"{v:g}" for _, v in ticks])
            # each block scales to its own data: a scale shared with the
            # other let one block's start up transient flatten this one, plus
            # a little headroom for the steady values printed on the lines
            axis.set_ylim(0, axis.get_ylim()[1] * 1.12)
            if axis is not ax4:
                axis.tick_params(labelbottom=False)
            # the panels are all one height, so this offset is the same gap
            # everywhere on the figure
            axis.legend(loc="lower center",
                        bbox_to_anchor=(0.5, 1 + LEGEND_GAP / panel),
                        ncol=len(axis.get_legend_handles_labels()[0]),
                        fontsize=8, frameon=False, labelcolor=INK,
                        handlelength=1.3, handletextpad=0.5,
                        columnspacing=1.8, borderpad=0)

        # after the limits are final: the labels are placed against them
        annotate(ax, secs - base, entries[ax], windows, boxes)
        if not coh.empty:
            for counters in (ax2, ax3):
                annotate(counters, csecs - base, entries[counters], windows,
                         boxes)
        if not cores.empty:
            annotate(ax4, ksecs - base, entries[ax4], windows, boxes)

    os.makedirs(config.PLOT_DIR_SHARING, exist_ok=True)
    out = os.path.join(config.PLOT_DIR_SHARING, f"{sub}_{label}.pdf")
    fig.savefig(out)
    print(f"[OK] {out}")
    plt.close(fig)


def make_plot_sharing(label: str = "sharing"):
    for arch, sub in config.ARCH_SUBNAMES.items():
        capture = os.path.join(
            config.RESULT_DIR, arch, "monitor", f"pcm_memory_{label}.csv"
        )
        if os.path.exists(capture):
            plot(arch, sub, label)


if __name__ == "__main__":
    make_plot_sharing(sys.argv[1] if len(sys.argv) > 1 else "sharing")
