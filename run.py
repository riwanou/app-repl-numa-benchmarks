import argparse
import bench_ann
import bench_rocksdb


parser = argparse.ArgumentParser()
parser.add_argument(
    "bench",
    choices=["ann", "ann-repl", "rocksdb", "rocksdb-repl"],
    help="Application benchmark to run",
)
args = parser.parse_args()

if args.bench == "ann":
    bench_ann.run_bench_ann()
elif args.bench == "ann-repl":
    bench_ann.run_bench_ann_repl()
elif args.bench == "rocksdb":
    bench_rocksdb.run_bench_rocksdb()
elif args.bench == "rocksdb-repl":
    bench_rocksdb.run_bench_rocksdb_repl()
