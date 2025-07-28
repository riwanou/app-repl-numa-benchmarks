import argparse
import bench_ann
import bench_rocksdb
import monitoring
import plot_ann
import plot_rocksdb
import plot_monitoring

parser = argparse.ArgumentParser()
parser.add_argument(
    "run",
    choices=[
        "ann",
        "ann-repl",
        "rocksdb",
        "rocksdb-repl",
        "plot-ann",
        "plot-rocksdb",
        "plot-monitoring",
    ],
    help="Variant to run",
)
args = parser.parse_args()


def bench_and_monitor(bench_fn, label):
    monitor = monitoring.Monitoring(label)
    monitor.start()
    try:
        bench_fn()
    except Exception as e:
        print(f"Error: {e}")
    else:
        monitor.mv_output_files()
    finally:
        monitor.stop()


if args.run == "ann":
    bench_and_monitor(bench_ann.run_bench_ann, "ann")
elif args.run == "ann-repl":
    bench_and_monitor(bench_ann.run_bench_ann_repl, "ann-repl")
elif args.run == "rocksdb":
    bench_and_monitor(bench_rocksdb.run_bench_rocksdb, "rocksdb")
elif args.run == "rocksdb-repl":
    bench_and_monitor(bench_rocksdb.run_bench_rocksdb_repl, "rocksdb-repl")
elif args.run == "plot-ann":
    plot_ann.make_plot_ann()
elif args.run == "plot-rocksdb":
    plot_rocksdb.make_plot_rocksdb()
elif args.run == "plot-monitoring":
    plot_monitoring.make_plot_monitoring()
