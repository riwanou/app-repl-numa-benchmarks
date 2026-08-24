"""Run the duckdb TPC-H queries over an mmap-attached database and write a csv."""
import argparse
import csv
import os
import threading
import time

import duckdb

import config
from duckdb_lib import summary

DB_DIR = config.DUCKDB_DB_DIR
RESULT_DIR = config.RESULT_DIR_DUCKDB
PASSES = 10
THREADS = config.NUM_THREADS

FIELDS = ["sf", "streams", "stream", "tag", "pass", "warmup", "query",
          "elapsed_s", "rows", "start_time", "end_time"]


def db_path(sf):
    return f"{DB_DIR}/tpch-sf{sf}.db"


def pass_totals(results, passes):
    """Per-client total of each pass."""
    return [
        summary.pass_total([r for r in results if r["pass"] == p])
        for p in range(1, passes + 1)
    ]


def run_stream(cur, queries, stream, p, sf, tag, streams, out, lock, stop):
    """One client, query order rotated so streams differ."""
    order = queries[stream:] + queries[:stream]
    for nr, sql in order:
        if stop.is_set():
            return
        start = config.get_time()
        t0 = time.perf_counter()
        n = len(cur.execute(sql).fetchall())
        elapsed = round(time.perf_counter() - t0, 4)
        end = config.get_time()
        with lock:
            out.append({"sf": sf, "streams": streams, "stream": stream,
                        "tag": tag, "pass": p, "warmup": p == 1, "query": nr,
                        "elapsed_s": elapsed, "rows": n,
                        "start_time": start, "end_time": end})
        print(f"sf{sf} {tag} s{stream} pass {p} Q{nr:<2} {elapsed:7.3f}s", flush=True)


def run_tpch(sf, tag, passes=PASSES, threads=THREADS, streams=1):
    """Run the 22 TPC-H queries `passes` times over `streams` clients."""
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

    cursors = []
    for _ in range(streams):
        cur = con.cursor()
        cur.execute("USE t")  # a cursor starts in the default catalog
        cursors.append(cur)

    results = []
    lock = threading.Lock()
    stop = threading.Event()
    try:
        for p in range(1, passes + 1):
            workers = [
                threading.Thread(
                    target=run_stream,
                    args=(cursors[s], queries, s, p, sf, tag, streams,
                          results, lock, stop),
                    daemon=True,
                )
                for s in range(streams)
            ]
            for w in workers:
                w.start()
            for w in workers:
                w.join()
    except KeyboardInterrupt:
        stop.set()
        for cur in cursors:
            cur.interrupt()
        raise
    con.close()

    summary.write_summary(results, f"tpch-sf{sf}", streams, tag, RESULT_DIR)

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(f"{RESULT_DIR}/tpch_sf{sf}_s{streams}_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(results)

    totals = pass_totals(results, passes)
    print(f"sf{sf} {tag} s{streams}: cold (pass 1) {totals[0]:.2f}s")
    print(f"sf{sf} {tag} s{streams}: warm mean {sum(totals[1:]) / len(totals[1:]):.2f}s")

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--passes", type=int, default=PASSES)
    a = ap.parse_args()
    run_tpch(a.sf, a.tag, passes=a.passes, streams=a.streams)
