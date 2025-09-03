import argparse
import bench_ann
import bench_rocksdb
import bench_micro
import monitoring

parser = argparse.ArgumentParser()
parser.add_argument(
    "run",
    choices=[
        "ann",
        "ann-repl",
        "rocksdb",
        "rocksdb-repl",
        "bench-pgtable-own",
        "bench-pgtable-carrefour",
        "bench-alloc-own",
        "bench-alloc-carrefour",
        "bench-mem-own",
        "bench-mem-carrefour",
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
elif args.run == "bench-pgtable-own":
    bench_micro.run_bench_pgtable("mmap")
elif args.run == "bench-pgtable-carrefour":
    bench_micro.run_bench_pgtable("madvise")
elif args.run == "bench-alloc-own":
    bench_micro.run_bench_alloc("mmap")
elif args.run == "bench-alloc-carrefour":
    bench_micro.run_bench_alloc("madvise")
elif args.run == "bench-mem-own":
    bench_micro.run_bench_mem("mmap")
elif args.run == "bench-mem-carrefour":
    bench_micro.run_bench_mem("madvise")
elif args.run == "plot-ann":
    import plot_ann

    plot_ann.make_plot_ann()
elif args.run == "plot-rocksdb":
    import plot_rocksdb

    plot_rocksdb.make_plot_rocksdb()
elif args.run == "plot-monitoring":
    import plot_monitoring

    plot_monitoring.make_plot_monitoring()
