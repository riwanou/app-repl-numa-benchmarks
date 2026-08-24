"""Aggregate a duckdb run into the summary csvs. Pass 1 is the cold run."""
import csv
import os
from statistics import mean, stdev

RAW_PREFIXES = ("tpch_", "clickbench_")

TOTALS = "summary_totals.csv"
BY_QUERY = "summary_by_query.csv"

TOTAL_FIELDS = ["bench", "streams", "tag", "phase", "mean_s", "std_s", "n"]
QUERY_FIELDS = TOTAL_FIELDS[:4] + ["query"] + TOTAL_FIELDS[4:]

BENCH_ORDER = ["tpch-sf10", "tpch-sf30", "clickbench"]
TAG_ORDER = [
    "firsttouch",
    "imbalanced",
    "interleaved",
    "numa-balancing",
    "repl",
]
BASELINE = "numa-balancing"


def _agg(values):
    return {
        "mean_s": round(mean(values), 4),
        "std_s": round(stdev(values), 4) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def query_time(rows):
    """One client's time for a query, meaned over the streams."""
    return mean([r["elapsed_s"] for r in rows])


def pass_total(rows):
    """One client's total query time, meaned over the streams."""
    per_stream = {}
    for r in rows:
        per_stream.setdefault(r.get("stream", 0), []).append(r["elapsed_s"])
    return mean([sum(v) for v in per_stream.values()])


def _upsert(path, fields, rows, keys):
    """Replace this run's rows, keep the others."""
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


def write_summary(results, bench, streams, tag, result_dir):
    """Merge one run's totals and per query means into the summary csvs."""
    passes = sorted({r["pass"] for r in results})
    warm = [p for p in passes if p != 1]
    queries = sorted({r["query"] for r in results})
    key = {"bench": bench, "streams": streams, "tag": tag}

    def rows_of(pass_nr, query=None):
        return [
            r
            for r in results
            if r["pass"] == pass_nr and (query is None or r["query"] == query)
        ]

    totals = []
    per_pass = {p: pass_total(rows_of(p)) for p in passes}
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
                key
                | {"phase": "cold", "query": query}
                | _agg([query_time(rows_of(1, query))])
            )
        if warm:
            by_query.append(
                key
                | {"phase": "warm", "query": query}
                | _agg([query_time(rows_of(p, query)) for p in warm])
            )

    os.makedirs(result_dir, exist_ok=True)
    _upsert(
        os.path.join(result_dir, TOTALS),
        TOTAL_FIELDS,
        totals,
        ["bench", "streams", "tag", "phase"],
    )
    _upsert(
        os.path.join(result_dir, BY_QUERY),
        QUERY_FIELDS,
        by_query,
        ["bench", "streams", "tag", "phase", "query"],
    )


def backfill(result_dir):
    """Rebuild the summaries from the raw csvs, skipping the retired -raw ones."""
    if not os.path.isdir(result_dir):
        return
    for name in sorted(os.listdir(result_dir)):
        if not name.endswith(".csv") or not name.startswith(RAW_PREFIXES):
            continue
        with open(os.path.join(result_dir, name), newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        head = rows[0]
        if "-raw" in head.get("sf", "") or head.get("variant") == "-raw":
            continue
        bench = f"tpch-sf{head['sf']}" if "sf" in head else "clickbench"

        results = [
            {
                "pass": int(r["pass"]),
                "query": int(r["query"]),
                "elapsed_s": float(r["elapsed_s"]),
                "stream": int(r.get("stream", 0)),
            }
            for r in rows
        ]
        write_summary(
            results, bench, int(head.get("streams", 1)), head["tag"], result_dir
        )
