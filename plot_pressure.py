"""QPS and bandwidth over time under the memory.high staircase, with the
counters that explain them underneath, all cut on the same phase windows.

    uv run run.py plot-pressure
"""

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config

# plots are made from synced results, so walk every arch rather than the
# machine we happen to run on (config.PLATFORM only matches the bench host)
RESULT_DIR = config.RESULT_DIR

# the monitor CSVs hold every variant of a run, phases.csv cuts them apart
STOCK = ["firsttouch", "interleaved", "numa-balancing"]
REPL = ["repl-bound", "repl-firsttouch", "repl-interleaved"]

BAND = "#e8e8e8"  # phase shading, alternating

# summary reports steady state: skip this much of each phase, the rest is
# the reclaim transient rather than what the limit costs
STEADY_FROM = 2 / 3


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
    df["read_gb"] = pd.to_numeric(df[("System", "Read")], errors="coerce") / 1024
    df["write_gb"] = (
        pd.to_numeric(df[("System", "Write")], errors="coerce") / 1024
    )
    return df


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


PG_NODE = re.compile(r"^node (\d+)(\(main\))?\s*: locality=([\d.]+) ptes=([\d.]+)K")


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
        df["gb_replicated"] = df.gb_max * df.coverage / 100
    return df


def fits_pct(limit: str, full_gb: float) -> str:
    """How much of a fully replicated footprint this limit still allows."""
    if limit == "max" or not full_gb:
        return "100%"
    return f"{min(100, float(limit.rstrip('G')) / full_gb * 100):.0f}%"


def shade_phases(ax, ph, label: bool = False, full_gb: float = 0):
    for i, row in ph.iterrows():
        if i % 2:
            ax.axvspan(row.start_s, row.end_s, color=BAND, zorder=0)
        ax.axvline(row.start_s, color="0.6", lw=0.6, zorder=1)
        if label:
            text = row.limit
            if full_gb:
                text += f"\n{fits_pct(row.limit, full_gb)}"
            ax.text(
                (row.start_s + row.end_s) / 2,
                1.02,
                text,
                transform=ax.get_xaxis_transform(),
                ha="center",
                fontsize=8,
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

    # 2. memory bandwidth, the other primary
    ax = axes[1]
    if not bw.empty:
        ax.plot(bw.elapsed, bw.read_gb, lw=1, label="read", color="#2ca02c")
        ax.plot(bw.elapsed, bw.write_gb, lw=1, label="write", color="#d62728")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("GB/s")
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
                c.gb_replicated,
                r.start_s,
                r.end_s,
                color="#1f77b4",
                lw=2,
            )
            ax.text(
                (r.start_s + r.end_s) / 2,
                c.gb_replicated + 0.25,
                f"{c.coverage:.0f}%",
                ha="center",
                fontsize=8,
                color="#1f77b4",
            )
        ax.plot([], [], color="#1f77b4", lw=2, label="replicated (of index)")

    ax.set_ylabel("GB")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    shade_phases(ax, ph)

    # 4. the explanatory counters: why the curves above moved. Log scale, the
    # rates span three decades, and only series that fired, to keep it legible
    ax = axes[3]
    series = {
        "numa hint faults/s": rate(cg, "vm_numa_hint_faults"),
        "pages migrated/s": rate(cg, "vm_numa_pages_migrated"),
        "replicas allocated/s": rate(cg, "repl_repl_allocated")
        if "repl_repl_allocated" in cg.columns
        else None,
        "replicas reclaimed/s": rate(cg, "repl_reclaimed_replicas")
        if "repl_reclaimed_replicas" in cg.columns
        else None,
    }
    for label, values in series.items():
        if values is not None and values.fillna(0).sum() > 0:
            ax.plot(cg.elapsed, values, lw=1, label=label)
    for name, g in (ev.groupby("event") if not ev.empty else []):
        if g["count"].sum() > 0:
            ax.plot(g.elapsed, g["count"] / 0.5, lw=1, ls=":", label=f"{name}/s")
    ax.set_yscale("log")
    ax.set_ylabel("events/s")
    ax.set_xlabel("time (s)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper left", fontsize=7, ncol=2)
    shade_phases(ax, ph)

    # locality is a ratio of two small counters, so smooth it over ~5s before
    # dividing rather than per sample
    hint = pd.to_numeric(cg["vm_numa_hint_faults"], errors="coerce").diff()
    if hint.fillna(0).sum() > 0:
        local = pd.to_numeric(
            cg["vm_numa_hint_faults_local"], errors="coerce"
        ).diff()
        win = 10
        pct = (
            local.rolling(win).sum() / hint.rolling(win).sum() * 100
        ).where(hint.rolling(win).sum() > 20)
        twin = ax.twinx()
        twin.plot(cg.elapsed, pct, lw=1.2, color="0.35", ls="--")
        twin.set_ylabel("% hint faults local (dashed)")
        twin.set_ylim(0, 100)

    os.makedirs(config.PLOT_DIR_PRESSURE, exist_ok=True)
    out = os.path.join(
        config.PLOT_DIR_PRESSURE, f"{short(arch)}_{variant}.png"
    )
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def variant_color(variant: str):
    """Patched in blues, stock in oranges, same convention as plot_ann."""
    stock = sns.color_palette(config.LINUX_COLOR, n_colors=6)
    patched = sns.color_palette(config.SPARE_COLOR, n_colors=6)
    if variant in REPL:
        return patched[3 + REPL.index(variant)]
    return stock[3 + STOCK.index(variant)]


def plot_metric(arch: str, variants: list[str], metric: str, full_gb: float = 0):
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
            x, y = df.elapsed, df.current_mb / 1024  # already a step, no smoothing
        else:
            df = pd.read_csv(
                f"{base(arch, variant)}-ann.csv", parse_dates=["start_time"]
            ).dropna(subset=["phase"])
            df["elapsed"] = (df.start_time - t0).dt.total_seconds()
            df = df.sort_values("elapsed")
            x, y = df.elapsed, df.qps.rolling(5, center=True).median()
        ax.plot(x, y, lw=1.4, color=color, label=variant)

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

    os.makedirs(config.PLOT_DIR_PRESSURE, exist_ok=True)
    out = os.path.join(config.PLOT_DIR_PRESSURE, f"{short(arch)}_{metric}.png")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def plot_summary(arch: str, variants: list[str]):
    """QPS against the limit it ran under, over how much of the index the
    limit still let us replicate."""
    fig, (ax, ax_mem, ax_cov) = plt.subplots(
        3,
        1,
        figsize=(9, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1.2, 1.2]},
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
            label=variant,
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
            by_phase = per_phase.coverage
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
            [limit if p == limit else f"{p}\n{limit}" for p, limit in zip(order, limits)]
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

    # one copy on N nodes serves 1/N of accesses locally: that is where the
    # stock variants sit, and where replication ends up once squeezed
    if nodes:
        ax_cov.axhline(100 / nodes, color="0.4", ls="--", lw=1)
        ax_cov.text(
            0.01,
            100 / nodes + 3,
            f"1 copy = {100 / nodes:.0f}%",
            fontsize=8,
            color="0.3",
        )

    # the percentage is a share of the index, so pair it with the GB it
    # stands for: 80% is easier to read as "6.0 of 7.5GB of copies"
    if fulls:
        copies_gb = sum(f[0] for f in fulls) / len(fulls)
        ax_gb = ax_cov.secondary_yaxis(
            "right",
            functions=(
                lambda pct: pct / 100 * copies_gb,
                lambda gb: gb / copies_gb * 100,
            ),
        )
        ax_gb.set_ylabel(f"GB replicated (of {copies_gb:.1f}GB)")
    ax_mem.set_ylabel("memory used (GB)")
    ax_mem.set_ylim(bottom=0)
    ax_mem.grid(axis="y", alpha=0.3)
    # not "index replicated": at 25% the whole index is still there, just in
    # one copy. This is how much of it each node reaches locally.
    ax_cov.set_ylabel("pages local per node (%)")
    ax_cov.set_xlabel("memory.high")
    ax_cov.set_ylim(0, 105)
    ax_cov.grid(axis="y", alpha=0.3)

    os.makedirs(config.PLOT_DIR_PRESSURE, exist_ok=True)
    out = os.path.join(config.PLOT_DIR_PRESSURE, f"{short(arch)}_summary.png")
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
        plot_metric(arch, available, "qps", full_gb)
        plot_metric(arch, available, "bandwidth", full_gb)
        plot_metric(arch, available, "memory", full_gb)
        plot_summary(arch, available)
    if not found:
        print(f"[WARN] no pressure results under {RESULT_DIR}/*/pressure")
