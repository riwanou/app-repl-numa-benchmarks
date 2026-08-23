"""Aggregate one duckdb run into the shared summary csvs.

The runners call this when they finish a batch of queries, so the aggregates
are written as the bench goes and never need a separate pass. Pass 1 is the
cold run and is kept as its own row, never averaged with the warm ones.
"""
import csv
import os
from statistics import mean, stdev

RAW_PREFIXES = ("tpch_", "clickbench_")

TOTALS = "summary_totals.csv"
BY_QUERY = "summary_by_query.csv"

TOTAL_FIELDS = ["bench", "compression", "tag", "phase", "mean_s", "std_s", "n"]
QUERY_FIELDS = TOTAL_FIELDS[:4] + ["query"] + TOTAL_FIELDS[4:]

BENCH_ORDER = ["tpch-sf10", "tpch-sf30", "clickbench"]
COMPRESSION_ORDER = ["compressed", "uncompressed"]
TAG_ORDER = [
    "firsttouch",
    "imbalanced",
    "interleaved",
    "numa-balancing",
    "repl",
]
# every percentage is read against this arm
BASELINE = "numa-balancing"


def compression_of(variant):
    """'-raw' is the uncompressed build of the same database."""
    return "uncompressed" if str(variant).endswith("-raw") else "compressed"


def _agg(values):
    return {
        "mean_s": round(mean(values), 4),
        "std_s": round(stdev(values), 4) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _upsert(path, fields, rows, keys):
    """Replace the rows of this run, keep every other run's."""
    old = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            old = list(csv.DictReader(f))

    fresh = {tuple(str(row[key]) for key in keys) for row in rows}
    kept = [r for r in old if tuple(r[key] for key in keys) not in fresh]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept + rows)


def backfill(result_dir):
    """Rebuild the summaries from the raw csvs.

    The runners write their aggregate as they go; this covers runs made before
    they did. Upserting the same rows again is harmless, so it just runs.
    """
    if not os.path.isdir(result_dir):
        return
    for name in sorted(os.listdir(result_dir)):
        if not name.endswith(".csv") or not name.startswith(RAW_PREFIXES):
            continue
        with open(os.path.join(result_dir, name), newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        results = [
            {
                "pass": int(r["pass"]),
                "query": int(r["query"]),
                "elapsed_s": float(r["elapsed_s"]),
            }
            for r in rows
        ]
        if "sf" in rows[0]:
            variant = rows[0]["sf"]
            bench = f"tpch-sf{variant.removesuffix('-raw')}"
        else:
            variant = rows[0]["variant"]
            bench = "clickbench"
        write_summary(
            results, bench, compression_of(variant), rows[0]["tag"], result_dir
        )


def write_summary(results, bench, compression, tag, result_dir):
    """Totals and per query means for one run, merged into the summary csvs."""
    passes = sorted({r["pass"] for r in results})
    warm = [p for p in passes if p != 1]
    queries = sorted({r["query"] for r in results})
    key = {"bench": bench, "compression": compression, "tag": tag}

    def elapsed(pass_nr, query=None):
        return [
            r["elapsed_s"]
            for r in results
            if r["pass"] == pass_nr and (query is None or r["query"] == query)
        ]

    totals = []
    per_pass = {p: sum(elapsed(p)) for p in passes}
    if 1 in passes:
        totals.append(key | {"phase": "cold"} | _agg([per_pass[1]]))
    if warm:
        totals.append(
            key | {"phase": "warm"} | _agg([per_pass[p] for p in warm])
        )

    by_query = []
    for query in queries:
        if 1 in passes:
            by_query.append(
                key | {"phase": "cold", "query": query} | _agg(elapsed(1, query))
            )
        if warm:
            by_query.append(
                key
                | {"phase": "warm", "query": query}
                | _agg([v for p in warm for v in elapsed(p, query)])
            )

    os.makedirs(result_dir, exist_ok=True)
    _upsert(
        os.path.join(result_dir, TOTALS),
        TOTAL_FIELDS,
        totals,
        ["bench", "compression", "tag", "phase"],
    )
    _upsert(
        os.path.join(result_dir, BY_QUERY),
        QUERY_FIELDS,
        by_query,
        ["bench", "compression", "tag", "phase", "query"],
    )
