"""Run the ClickBench queries over an mmap-attached database and write a csv."""
import argparse
import csv
import os
import time

import duckdb

import config
from duckdb_lib import summary

DB_DIR = config.DUCKDB_DB_DIR
RESULT_DIR = config.RESULT_DIR_DUCKDB
QUERIES = os.path.join(config.DUCKDB_CB_DIR, "queries.sql")
PASSES = 20
THREADS = config.NUM_THREADS


def db_path(variant):
    return f"{DB_DIR}/hits{variant}.db"


def run_clickbench(variant, tag, passes=PASSES, threads=THREADS):
    """Run the ClickBench queries `passes` times, write the csv, return the rows."""
    db = db_path(variant)
    if not os.path.exists(db):
        raise FileNotFoundError(f"{db} not found, run `just build-duckdb` first")

    queries = [q.strip() for q in open(QUERIES) if q.strip()]

    con = duckdb.connect()
    con.execute(f"ATTACH '{db}' AS t (READ_ONLY, IO_MODE MMAP)")
    con.execute("USE t")
    con.execute(f"SET threads={threads}")

    results = []
    for p in range(1, passes + 1):
        for nr, sql in enumerate(queries, start=1):
            start = config.get_time()
            t0 = time.perf_counter()
            n = len(con.execute(sql).fetchall())
            elapsed = round(time.perf_counter() - t0, 4)
            end = config.get_time()
            results.append({"variant": variant or "base", "tag": tag, "pass": p,
                            "warmup": p == 1, "query": nr, "elapsed_s": elapsed, "rows": n,
                            "start_time": start, "end_time": end})
            print(f"{variant or 'base'} {tag} pass {p} Q{nr:<2} {elapsed:7.3f}s", flush=True)
    con.close()

    summary.write_summary(
        results,
        "clickbench",
        summary.compression_of(variant),
        tag,
        RESULT_DIR,
    )

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(f"{RESULT_DIR}/clickbench_{variant or 'base'}_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["variant", "tag", "pass", "warmup", "query", "elapsed_s", "rows",
                           "start_time", "end_time"])
        w.writeheader()
        w.writerows(results)

    warmup = sum(r["elapsed_s"] for r in results if r["warmup"])
    steady = [sum(r["elapsed_s"] for r in results if r["pass"] == p)
              for p in range(2, passes + 1)]
    print(f"{variant or 'base'} {tag}: warmup (pass 1) {warmup:.2f}s")
    print(f"{variant or 'base'} {tag}: steady (pass 2-{passes}) {[round(s, 2) for s in steady]}"
          f" mean {sum(steady) / len(steady):.2f}s")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="", help="'' for compressed, '-raw' for uncompressed")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run_clickbench(a.variant, a.tag)
