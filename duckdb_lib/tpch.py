"""Run the duckdb TPC-H queries over an mmap-attached database and write a csv."""
import argparse
import csv
import os
import time

import duckdb

import config
from duckdb_lib import summary

DB_DIR = config.DUCKDB_DB_DIR
RESULT_DIR = config.RESULT_DIR_DUCKDB
PASSES = 20
THREADS = config.NUM_THREADS


def db_path(sf):
    return f"{DB_DIR}/tpch-sf{sf}.db"


def run_tpch(sf, tag, passes=PASSES, threads=THREADS):
    """Run the 22 TPC-H queries `passes` times, write the csv, return the rows."""
    db = db_path(sf)
    if not os.path.exists(db):
        raise FileNotFoundError(f"{db} not found, run `just build-duckdb` first")

    con = duckdb.connect()
    con.execute("LOAD tpch")
    con.execute(f"ATTACH '{db}' AS t (READ_ONLY, IO_MODE MMAP)")
    con.execute("USE t")
    con.execute(f"SET threads={threads}")

    queries = con.execute(
        "SELECT query_nr, query FROM tpch_queries() ORDER BY query_nr"
    ).fetchall()

    results = []
    for p in range(1, passes + 1):
        for nr, sql in queries:
            start = config.get_time()
            t0 = time.perf_counter()
            n = len(con.execute(sql).fetchall())
            elapsed = round(time.perf_counter() - t0, 4)
            end = config.get_time()
            results.append({"sf": sf, "tag": tag, "pass": p, "warmup": p == 1,
                            "query": nr, "elapsed_s": elapsed, "rows": n,
                            "start_time": start, "end_time": end})
            print(f"sf{sf} {tag} pass {p} Q{nr:<2} {elapsed:7.3f}s", flush=True)
    con.close()

    summary.write_summary(
        results,
        f"tpch-sf{str(sf).removesuffix('-raw')}",
        summary.compression_of(sf),
        tag,
        RESULT_DIR,
    )

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(f"{RESULT_DIR}/tpch_sf{sf}_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["sf", "tag", "pass", "warmup", "query", "elapsed_s", "rows",
                           "start_time", "end_time"])
        w.writeheader()
        w.writerows(results)

    # pass 1 pays the cold read and the replica build, so it is reported apart
    warmup = sum(r["elapsed_s"] for r in results if r["warmup"])
    steady = [sum(r["elapsed_s"] for r in results if r["pass"] == p)
              for p in range(2, passes + 1)]
    print(f"sf{sf} {tag}: warmup (pass 1) {warmup:.2f}s")
    print(f"sf{sf} {tag}: steady (pass 2-{passes}) {[round(s, 2) for s in steady]}"
          f" mean {sum(steady) / len(steady):.2f}s")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run_tpch(a.sf, a.tag)
