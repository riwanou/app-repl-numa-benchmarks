"""Mean QPS per pressure level, one row per variant and phase.

The pressure plot draws QPS over time; this is the same runs read as a
table, so a phase can be quoted as a number rather than eyeballed off a
curve. Written to `results/<arch>/stats/pressure.csv`, next to the other
per bench stat files.

    uv run run.py stats-pressure
"""

import os

import pandas as pd

import config
import plot_pressure as pp


def variant_stats(arch: str, variant: str) -> pd.DataFrame:
    """One row per phase: mean QPS over the phase, and over its steady state."""
    runs = pd.read_csv(
        f"{pp.base(arch, variant)}-ann.csv", parse_dates=["start_time"]
    ).dropna(subset=["phase"])

    rows = []
    for _, r in pp.phases(arch, variant).iterrows():
        g = runs[runs.phase == r.phase]
        # the reclaim transient at the top of a phase is not what the limit
        # costs, so the steady state columns drop the first STEADY_FROM of it
        begins = r.start_time + (r.end_time - r.start_time) * pp.STEADY_FROM
        steady = g[g.start_time >= begins]
        rows.append(
            {
                "arch": pp.short(arch),
                "variant": variant,
                "label": pp.variant_label(variant),
                "phase": r.phase,
                "limit": r.limit,
                "qps_mean_steady": steady.qps.mean(),
                "qps_std": steady.qps.std(),
                "qps_median": steady.qps.median(),
                "runs": len(steady),
                "qps_mean_full_phase": g.qps.mean(),
                "runs_full_phase": len(g),
            }
        )
    return pd.DataFrame(rows)


def make_stats_pressure():
    found = False
    for arch in sorted(os.listdir(config.RESULT_DIR)):
        available = [
            v
            for v in pp.STOCK + pp.REPL
            if os.path.exists(f"{pp.base(arch, v)}-ann.csv")
        ]
        if not available:
            continue
        found = True

        df = pd.concat(
            [variant_stats(arch, v) for v in available], ignore_index=True
        )
        output = os.path.join(
            config.RESULT_DIR, arch, "stats", "pressure.csv"
        )
        os.makedirs(os.path.dirname(output), exist_ok=True)
        df.to_csv(output, index=False)
        print(f"[OK] {len(df)} rows -> {output}")

        # the same table the way it is read: a level per row, a variant per
        # column, in the order the phases ran. Keyed on the variant tag, not
        # the label: repl-bound and repl-firsttouch share one label
        wide = df.pivot_table(
            index="phase", columns="variant", values="qps_mean_steady", sort=False
        )[available]
        print(wide.round(1).to_string(), "\n")

    if not found:
        print(f"[WARN] no pressure results under {config.RESULT_DIR}/*/pressure")


if __name__ == "__main__":
    make_stats_pressure()
