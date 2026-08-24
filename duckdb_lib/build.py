"""Fetch and build the duckdb databases. Idempotent: a re-run only fills gaps."""
import os
import subprocess
import sys

import duckdb

import config

DB_DIR = config.DUCKDB_DB_DIR
CB_DIR = config.DUCKDB_CB_DIR

TPCH_SF = [10, 30]
TPCH_URL = "https://blobs.duckdb.org/data/tpch-sf{sf}.db"
HITS_URL = "https://datasets.clickhouse.com/hits_compatible/hits.parquet"
CB_RAW = "https://raw.githubusercontent.com/ClickHouse/ClickBench/main/duckdb"
CB_FILES = ["create.sql", "queries.sql", "load", "query"]

# the parquet stores the time columns as epoch ints
CB_INSERT = """
INSERT INTO hits
SELECT * REPLACE (
    make_date(EventDate) AS EventDate,
    epoch_ms(EventTime * 1000) AS EventTime,
    epoch_ms(ClientEventTime * 1000) AS ClientEventTime,
    epoch_ms(LocalEventTime * 1000) AS LocalEventTime)
FROM read_parquet('{src}', binary_as_string=True);
"""


def log(msg):
    print(f"[build] {msg}", flush=True)


def download(url, dest):
    if os.path.exists(dest):
        log(f"have {os.path.basename(dest)}")
        return
    log(f"downloading {url}")
    # -c so an interrupted build resumes
    subprocess.run(["wget", "-c", "-q", "-O", dest, url], check=True)


def build_tpch():
    for sf in TPCH_SF:
        download(TPCH_URL.format(sf=sf), os.path.join(DB_DIR, f"tpch-sf{sf}.db"))


def build_clickbench():
    os.makedirs(CB_DIR, exist_ok=True)
    for f in CB_FILES:
        dest = os.path.join(CB_DIR, f)
        if not os.path.exists(dest):
            log(f"fetching clickbench/{f}")
            subprocess.run(["curl", "-sSf", "-o", dest, f"{CB_RAW}/{f}"], check=True)

    parquet = os.path.join(CB_DIR, "hits.parquet")
    hits = os.path.join(DB_DIR, "hits.db")

    if not os.path.exists(hits):
        download(HITS_URL, parquet)
        log("building hits.db with the official ClickBench DDL")
        tmp = hits + ".part"
        if os.path.exists(tmp):
            os.remove(tmp)
        con = duckdb.connect(tmp)
        con.execute("SET storage_compatibility_version='latest'")
        con.execute(open(os.path.join(CB_DIR, "create.sql")).read())
        con.execute(CB_INSERT.format(src=parquet))
        con.close()
        os.rename(tmp, hits)
    else:
        log("have hits.db")

    if os.path.exists(parquet):
        log("removing hits.parquet (no longer needed)")
        os.remove(parquet)


def main():
    os.makedirs(DB_DIR, exist_ok=True)
    duckdb.connect().execute("INSTALL tpch")

    build_tpch()
    build_clickbench()

    log("done. databases:")
    for f in sorted(os.listdir(DB_DIR)):
        if f.endswith(".db"):
            log(f"  {f:24} {os.path.getsize(os.path.join(DB_DIR, f)) / 1e9:8.1f} GB")


if __name__ == "__main__":
    sys.exit(main())
