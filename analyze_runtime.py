import os
import pandas as pd
import json
import config


def get_bench_time(arch_dir, bench, ref_year=None):
    results = {
        "normal": {"start": None, "end": None, "duration": 0},
        "repl": {"start": None, "end": None, "duration": 0},
    }

    if bench == "ann":
        ann_dir = os.path.join(arch_dir, "ann")
        if not os.path.isdir(ann_dir):
            return results

        for f in os.listdir(ann_dir):
            if not f.endswith(".csv"):
                continue
            try:
                df = pd.read_csv(os.path.join(ann_dir, f))
                if "start_time" not in df.columns:
                    continue

                for idx, row in df.iterrows():
                    start = pd.to_datetime(row["start_time"])
                    if ref_year and start.year != ref_year:
                        continue
                    end = pd.to_datetime(row["end_time"])
                    tag = row.get("tag", "")

                    is_repl = "repl" in str(tag)

                    variant = "repl" if is_repl else "normal"

                    if (
                        results[variant]["start"] is None
                        or start < results[variant]["start"]
                    ):
                        results[variant]["start"] = start
                    if (
                        results[variant]["end"] is None
                        or end > results[variant]["end"]
                    ):
                        results[variant]["end"] = end
                    results[variant]["duration"] += (
                        end - start
                    ).total_seconds()
            except:
                pass

    elif bench == "rocksdb":
        csv_path = os.path.join(arch_dir, "rocksdb", "results.csv")
        if not os.path.exists(csv_path):
            return results

        try:
            df = pd.read_csv(csv_path)
            if "start_time" not in df.columns:
                return results

            for idx, row in df.iterrows():
                start = pd.to_datetime(row["start_time"])
                if ref_year and start.year != ref_year:
                    continue
                end = pd.to_datetime(row["end_time"])
                tag = row.get("tag", "")

                is_repl = "patched-repl" in str(tag)

                variant = "repl" if is_repl else "normal"

                if (
                    results[variant]["start"] is None
                    or start < results[variant]["start"]
                ):
                    results[variant]["start"] = start
                if (
                    results[variant]["end"] is None
                    or end > results[variant]["end"]
                ):
                    results[variant]["end"] = end
                results[variant]["duration"] += (end - start).total_seconds()
        except:
            pass

    elif bench == "llama":
        llama_dir = os.path.join(arch_dir, "llama")
        if not os.path.isdir(llama_dir):
            return results

        baseline_csv = os.path.join(llama_dir, "baseline.csv")
        if os.path.exists(baseline_csv):
            try:
                df = pd.read_csv(baseline_csv)
                if "test_time" in df.columns:
                    times = pd.to_datetime(df["test_time"])
                    times = (
                        times[times.dt.year == ref_year] if ref_year else times
                    )
                    if len(times) > 0:
                        results["normal"]["start"] = times.min()
            except:
                pass

        distribute_csv = os.path.join(llama_dir, "distribute.csv")
        if os.path.exists(distribute_csv):
            try:
                df = pd.read_csv(distribute_csv)
                if "test_time" in df.columns:
                    times = pd.to_datetime(df["test_time"])
                    times = (
                        times[times.dt.year == ref_year] if ref_year else times
                    )
                    if len(times) > 0:
                        results["normal"]["end"] = times.max()
            except:
                pass

        if results["normal"]["start"] and results["normal"]["end"]:
            results["normal"]["duration"] = (
                results["normal"]["end"] - results["normal"]["start"]
            ).total_seconds()

        repl_csv = os.path.join(llama_dir, "repl.csv")
        if os.path.exists(repl_csv):
            try:
                df = pd.read_csv(repl_csv)
                if "test_time" in df.columns:
                    times = pd.to_datetime(df["test_time"])
                    times = (
                        times[times.dt.year == ref_year] if ref_year else times
                    )
                    if len(times) > 0:
                        results["repl"]["start"] = times.min()
                        results["repl"]["end"] = times.max()
                        results["repl"]["duration"] = (
                            results["repl"]["end"] - results["repl"]["start"]
                        ).total_seconds()
            except:
                pass

    elif bench == "fio":
        fio_dir = os.path.join(arch_dir, "fio")
        if not os.path.isdir(fio_dir):
            return results

        for f in os.listdir(fio_dir):
            if not f.endswith(".json"):
                continue
            if "random" not in f:
                continue
            if "_run" not in f:
                continue

            is_repl = "repl" in f or "unrepl" in f
            variant = "repl" if is_repl else "normal"

            try:
                with open(os.path.join(fio_dir, f)) as fp:
                    data = json.load(fp)
                    if "timestamp_ms" in data:
                        start = pd.to_datetime(data["timestamp_ms"], unit="ms")
                        if ref_year and start.year != ref_year:
                            continue
                        runtime_us = data["jobs"][0].get("job_runtime", 0)
                        duration = runtime_us / 1e6
                        end = start + pd.Timedelta(seconds=duration)

                        if (
                            results[variant]["start"] is None
                            or start < results[variant]["start"]
                        ):
                            results[variant]["start"] = start
                        if (
                            results[variant]["end"] is None
                            or end > results[variant]["end"]
                        ):
                            results[variant]["end"] = end
            except:
                pass

        # Calculate duration from start to end
        if results["normal"]["start"] and results["normal"]["end"]:
            results["normal"]["duration"] = (
                results["normal"]["end"] - results["normal"]["start"]
            ).total_seconds()
        if results["repl"]["start"] and results["repl"]["end"]:
            results["repl"]["duration"] = (
                results["repl"]["end"] - results["repl"]["start"]
            ).total_seconds()

    return results


def get_ref_year(arch_dir):
    ann_dir = os.path.join(arch_dir, "ann")
    if os.path.isdir(ann_dir):
        for f in os.listdir(ann_dir):
            if f.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(ann_dir, f))
                    if "start_time" in df.columns:
                        years = pd.to_datetime(df["start_time"]).dt.year
                        if len(years) > 0:
                            return int(years.max())
                except:
                    pass
    return 2025


def format_duration(seconds):
    if (
        seconds is None
        or (isinstance(seconds, float) and pd.isna(seconds))
        or seconds == 0
    ):
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def format_ts(ts):
    if ts is None or pd.isna(ts):
        return "N/A"
    return ts.strftime("%Y-%m-%d %H:%M")


def main():
    for arch in sorted(os.listdir(config.RESULT_DIR)):
        if arch == "unknown-cpu_ARM64":
            continue
        arch_dir = os.path.join(config.RESULT_DIR, arch)
        if not os.path.isdir(arch_dir):
            continue

        arch_short = config.ARCH_SUBNAMES.get(
            arch, arch.split("_")[2] if len(arch.split("_")) > 2 else arch
        )
        ref_year = get_ref_year(arch_dir)

        print(f"\n{'=' * 60}")
        print(f"Architecture: {arch_short} (ref year: {ref_year})")
        print(f"{'=' * 60}")

        for variant in ["normal", "repl"]:
            print(f"\n--- {variant.upper()} ---")

            total_duration = 0

            for bench in ["ann", "rocksdb", "llama", "fio"]:
                timing = get_bench_time(arch_dir, bench, ref_year)

                if timing[variant]["start"] is not None:
                    print(
                        f"  {bench:8}: {format_ts(timing[variant]['start'])} - {format_ts(timing[variant]['end'])} ({format_duration(timing[variant]['duration'])})"
                    )
                    total_duration += timing[variant]["duration"]
                else:
                    print(f"  {bench:8}: N/A")

            print(f"  {'Total':8}: {format_duration(total_duration)}")

        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
