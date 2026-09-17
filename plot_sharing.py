"""DRAM bandwidth over time, with the bench phases shaded.

Shows whether a number is a steady state or a slice of a curve, which a phase
mean cannot say. One plot per arch that has the capture.

    uv run run.py plot-sharing
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
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
    "S": "S  shared clean copy",
}

# what each phase does, in place of the bench's own shorthand names
PHASE_LABELS = {
    "local": "all near",
    "remote": "all far",
    "disjoint": "own lines",
    "shared": "same lines",
    "pingpong": "same lines, ping pong",
}
POLICY_LABELS = {"membind": "data on 1 node", "interleaved": "data on 2 nodes"}


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


PAD_FRAC = 0.06   # the gap left between two phases, as a share of one phase


def phase_boxes(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Where each phase sits on the axis, laid end to end with a gap between.
    Phases are not all the same length: shared runs far longer than the rest."""
    pad = max(b - a for a, b in windows) * PAD_FRAC
    boxes, x = [], 0.0
    for a, b in windows:
        boxes.append((x, x + (b - a)))
        x += (b - a) + pad
    return boxes


def compress(secs: pd.Series, windows: list[tuple[float, float]],
             boxes: list[tuple[float, float]]) -> pd.Series:
    """Real seconds to axis seconds, with the gap between two phases taken
    out. That gap is the next phase's 8 GB memset plus the idle sync, which is
    neither measured nor plotted, and leaving it in pushed every phase along by
    the gaps before it, so no tick landed on a phase boundary. Laid end to end
    the phases sit end to end and a tick lands on every boundary. Samples
    outside a window come back NaN, which is what breaks the line between
    phases."""
    out = pd.Series(float("nan"), index=secs.index, dtype=float)
    for (a, b), (lo, _) in zip(windows, boxes):
        inside = (secs >= a) & (secs <= b)
        out[inside] = lo + (secs[inside] - a)
    return out


def steady(v: float) -> str:
    return f"{v:.0f}" if v >= 10 else f"{v:.1f}" if v >= 1 else f"{v:.2f}"


def annotate(axis, t, entries, windows, boxes):
    """Print where each line settles, at the right of its phase.

    The offset is in points off the value, so a label stays with its own line
    whatever the panel is scaled to. Labels stack above their lines and step
    over any other line they would land on; if the stack would reach the legend
    the top ones go under their lines instead. Values come from the tail of the
    phase, not the mean, since most phases open with a transient."""
    per_point = axis.figure.dpi / 72

    def points(value):
        return axis.transData.transform((0, value))[1] / per_point

    ceiling = points(axis.get_ylim()[1])
    floor = points(axis.get_ylim()[0])

    for (a, b), (lo, hi) in zip(windows, boxes):
        tail = (t >= b - (b - a) * 0.4) & (t <= b)
        marks = []
        for values, color in entries:
            value = values[tail].median()
            if not pd.isna(value):
                marks.append((points(value), value, steady(value), color))
        marks.sort()
        lines = [mark[0] for mark in marks]

        def stack(items):
            """Where each of these sits once it is clear of its own line, of
            the line above it, and of the label under it."""
            out, placed = [], None
            for pos, _, _, _ in items:
                target = pos + 2
                if placed is not None:
                    target = max(target, placed + 9)
                for line in lines:   # ascending, so one pass clears them all
                    if target - 3 <= line <= target + 8:
                        target = line + 3
                placed = target
                out.append(target)
            return out

        # Try putting the top k labels under their lines, smallest k first.
        # Keep the first arrangement that fits the panel and still reads in
        # value order, top to bottom. Flipping one label on its own breaks that
        # order: it lands under the labels of smaller values.
        placement, flipped = stack(marks), 0
        for k in range(len(marks) + 1):
            above, below = marks[: len(marks) - k], marks[len(marks) - k :]
            ys, under = stack(above), None
            for pos, _, _, _ in reversed(below):
                under = pos - 2 if under is None else min(pos - 2, under - 9)
                for line in reversed(lines):  # descending, one pass clears all
                    if under - 8 <= line <= under + 2:
                        under = line - 4
                ys.append(under)
            ys[len(above):] = reversed(ys[len(above):])
            if (max(ys) + 6 <= ceiling and min(ys) >= floor
                    and all(a < b for a, b in zip(ys, ys[1:]))):
                placement, flipped = ys, k
                break

        x = hi - (hi - lo) * 0.03
        for j, ((pos, value, text, color), y) in enumerate(zip(marks, placement)):
            axis.annotate(text, (x, value), textcoords="offset points",
                          xytext=(0, y - pos), ha="right",
                          va="top" if j >= len(marks) - flipped else "bottom",
                          fontsize=7, color=color, annotation_clip=False)

def draw(axis, x, y, label, color, style="-"):
    """One line per phase: NaN x in the gaps, so nothing joins them up."""
    axis.plot(x, y, label=label, lw=1.1, ls=style, color=color)


# Every gap below is an inch, placed by hand. Equal panel heights are the
# point: a legend offset in axes fractions gives a different gap on every row
# when the rows differ in height, which is what made the spacing look random.
FIG_W = 6.2
PANEL = 0.92      # every panel, same height; the figure follows from it
LEFT, RIGHT = 0.107, 0.985
LEGEND = 0.17     # the line of keys above a panel
LEGEND_GAP = 0.05  # legend to the panel it labels
PANEL_GAP = 0.11  # panel to the next legend
PANELS = 3        # stacked measures per block
AXIS = 0.40       # tick labels plus "seconds" under a block
HEADER = 0.30     # the policy line and the rule under it
PHASES = 0.26     # the row of phase names
COL_GAP = 0.14    # between the two columns
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
        )
        for _, row in phases(arch, label).iterrows()
    ]

    columns = []
    for policy in dict.fromkeys(p for _, _, p, _ in spans):
        group = [(a, b, ph) for a, b, p, ph in spans if p == policy]
        base = group[0][0]
        windows = [(a - base, b - base) for a, b, _ in group]
        boxes = phase_boxes(windows)
        columns.append((policy, group, base, windows, boxes))

    fig_h = (2 * MARGIN + HEADER + PHASES + AXIS
             + PANELS * (LEGEND + LEGEND_GAP + PANEL)
             + (PANELS - 1) * PANEL_GAP)
    fig = plt.figure(figsize=(FIG_W, fig_h))
    inch, wide = 1 / fig_h, 1 / FIG_W

    # a column per policy, as wide as the time it covers, so both columns run
    # at the same seconds per inch and can be read against each other
    covered = [col[4][-1][1] for col in columns]
    room = RIGHT - LEFT - COL_GAP * wide * (len(columns) - 1)
    widths = [room * c / sum(covered) for c in covered]
    lefts, edge = [], LEFT
    for width in widths:
        lefts.append(edge)
        edge += width + COL_GAP * wide

    # the rows are shared: one legend each, over the top of both columns
    rows, y = [], 1 - (MARGIN + HEADER + PHASES) * inch
    for _ in range(PANELS):
        y -= (LEGEND + LEGEND_GAP + PANEL) * inch
        rows.append(y)
        y -= PANEL_GAP * inch

    def up(inches):
        """An offset above a panel, in that panel's own fraction."""
        return 1 + inches / PANEL

    drawn = []
    for i, (policy, group, base, windows, boxes) in enumerate(columns):
        ax, ax2, ax3 = (
            fig.add_axes([lefts[i], row, widths[i], PANEL * inch])
            for row in rows
        )

        x = compress(secs - base, windows, boxes)
        draw(ax, x, df["read"], "reads", BLUE)
        draw(ax, x, df["write"], "writes", ORANGE)
        entries = {ax: [(df["read"], BLUE), (df["write"], ORANGE)],
                   ax2: [], ax3: []}

        if not coh.empty:
            cx = compress(csecs - base, windows, boxes)
            # the writes panel 1 measures, in lines instead of bytes. The
            # counter misses every update that rides a writeback, so the two
            # are the same traffic counted two ways.
            lines_s = np.interp(csecs, secs, df["write"] * 1e3 / 64)
            writeback = pd.Series(lines_s, index=csecs.index)
            draw(ax2, cx, writeback, "writes / 64 B", BLUE, "--")
            entries[ax2].append((writeback, BLUE))

            event = "UNC_M2M_DIRECTORY_UPDATE.ANY"
            if event in coh:
                draw(ax2, cx, coh[event], "directory updates", ORANGE)
                entries[ax2].append((coh[event], ORANGE))

            events = {s: f"UNC_M2M_DIRECTORY_LOOKUP.STATE_{s}" for s in "ISA"}
            for state in ("I", "A", "S"):
                if events[state] in coh:
                    draw(ax3, cx, coh[events[state]], STATE_LABELS[state],
                         STATE_COLORS[state])
                    entries[ax3].append((coh[events[state]],
                                         STATE_COLORS[state]))

        for lo, hi in boxes:
            for axis in (ax, ax2, ax3):
                axis.axvspan(lo, hi, color=BAND, lw=0, zorder=0)

        span_ax = ax.get_xaxis_transform()
        for (a, b), (_, _, phase) in zip(boxes, group):
            ax.text((a + b) / 2, up(LEGEND_GAP + LEGEND + 0.07),
                    PHASE_LABELS.get(phase, phase), transform=span_ax,
                    ha="center", va="bottom", fontsize=8, color=INK)

        rule = up(LEGEND_GAP + LEGEND + PHASES)
        ax.plot([boxes[0][0], boxes[-1][1]], [rule, rule], transform=span_ax,
                color="#cfcecb", lw=1, clip_on=False)
        ax.text(0.5, rule + 0.075 / PANEL, POLICY_LABELS.get(policy, policy),
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=10.5, color="#0b0b0b")

        for axis, name in ((ax, "DRAM GB/s"),
                           (ax2, "directory\nwrites M/s"),
                           (ax3, "directory\nlookups M/s")):
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#cfcecb")
            axis.tick_params(colors=INK, labelsize=7.5, length=2.5,
                             color="#cfcecb")
            axis.set_axisbelow(True)
            axis.margins(x=0.015, y=0.10)
            edge = max(b - a for a, b in windows) * PAD_FRAC / 2
            axis.set_xlim(-edge, boxes[-1][1] + edge)
            # a tick at each phase boundary and midpoint, labelled with the
            # measured seconds so far
            ticks, cum = [(0.0, 0.0)], 0.0
            for (a, b), (lo, hi) in zip(windows, boxes):
                ticks.append(((lo + hi) / 2, cum + (b - a) / 2))
                ticks.append((hi, cum + (b - a)))
                cum += b - a
            axis.set_xticks([t for t, _ in ticks])
            axis.set_xticklabels([f"{round(v / 10) * 10:g}" for _, v in ticks])
            if axis is not ax3:
                axis.tick_params(labelbottom=False)
            if i:
                axis.tick_params(labelleft=False)
            else:
                axis.set_ylabel(name, color=INK, fontsize=8, labelpad=2)
                # one legend per row, centred over both columns
                axis.legend(
                    loc="lower center",
                    bbox_to_anchor=(((LEFT + RIGHT) / 2 - lefts[0]) / widths[0],
                                    up(LEGEND_GAP)),
                    ncol=len(axis.get_legend_handles_labels()[0]),
                    fontsize=8, frameon=False, labelcolor=INK,
                    handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
                    # both default to font sized padding, which is most of the
                    # space under a legend
                    borderpad=0, borderaxespad=0)

        drawn.append(((ax, ax2, ax3), entries, base, windows, boxes))
        fig.text(lefts[i] + widths[i] / 2, rows[-1] - 0.32 * inch, "seconds",
                 ha="center", va="bottom", fontsize=8, color=INK)

    # one scale per row, so the columns read against each other and only the
    # left one needs its numbers
    for row in zip(*(axes for axes, _, _, _, _ in drawn)):
        # headroom so a value can be printed over its line rather than under
        top = max(axis.get_ylim()[1] for axis in row) * 1.25
        for axis in row:
            axis.set_ylim(0, top)

    # after the scales are final: the labels are placed against them
    for axes, entries, base, windows, boxes in drawn:
        annotate(axes[0], secs - base, entries[axes[0]], windows, boxes)
        if not coh.empty:
            for counters in axes[1:]:
                annotate(counters, csecs - base, entries[counters], windows,
                         boxes)

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
