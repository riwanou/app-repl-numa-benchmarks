import argparse
import bench_ann
import bench_rocksdb
import plot_ann


parser = argparse.ArgumentParser()
parser.add_argument(
    "run",
    choices=[
        "ann",
        "ann-repl",
        "rocksdb",
        "rocksdb-repl",
        "plot-ann",
    ],
    help="Variant to run",
)
args = parser.parse_args()

if args.run == "ann":
    bench_ann.run_bench_ann()
elif args.run == "ann-repl":
    bench_ann.run_bench_ann_repl()
elif args.run == "rocksdb":
    bench_rocksdb.run_bench_rocksdb()
elif args.run == "rocksdb-repl":
    bench_rocksdb.run_bench_rocksdb_repl()
elif args.run == "plot-ann":
    plot_ann.make_plot_ann()
