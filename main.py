import argparse
import requests
import os
import h5py
import numpy as np
import csv
import multiprocessing
import cpuinfo
import re
import mod_faiss
import mod_annoy
import mod_usearch

NUM_RUNS = 10
TAG = "default"
NUM_THREADS = multiprocessing.cpu_count()
CPU_INFO = cpuinfo.get_cpu_info()
PLATFORM = (
    re.sub(
        r"\s+",
        "_",
        re.sub(r"[()]", "", CPU_INFO.get("brand_raw", "unknown-cpu")).strip(),
    )
    + "_"
    + CPU_INFO.get("arch", "unknown-arch")
)
DATA_DIR = "data"
INDEX_DIR = "indices"
RESULT_DIR = os.path.join("results", PLATFORM)

CONFIG = {
    "glove-100-angular.hdf5": {
        "faiss": {"nlist": 100, "nprobe": 20},
        "annoy": {"trees": 100, "search_k": 200_000},
        "usearch": {"e_search": 4000},
    },
    "sift-128-euclidean.hdf5": {
        "faiss": {"nlist": 100, "nprobe": 10},
        "annoy": {"trees": 100, "search_k": 40_000},
        "usearch": {"e_search": 256},
    },
    "gist-960-euclidean.hdf5": {
        "faiss": {"nlist": 100, "nprobe": 10},
        "annoy": {"trees": 100, "search_k": 500_000},
        "usearch": {"e_search": 3000},
    },
}

DATASETS = list(CONFIG.keys())

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=int, default=NUM_RUNS, help="Create the indices"
)
parser.add_argument(
    "--threads", type=int, default=NUM_THREADS, help="Number of threads"
)
parser.add_argument("--tag", default=TAG, help="CSV filename tag")
parser.add_argument(
    "--datasets",
    nargs="*",
    default=DATASETS,
    help="List of datasets to use (default: all datasets)",
)
parser.add_argument(
    "--recreate-index", action="store_true", help="Re create the indices"
)
parser.add_argument("--bench", action="store_true", help="Bench the ann search")
parser.add_argument(
    "--faiss", action="store_true", help="Evaluate faiss benchmark"
)
parser.add_argument(
    "--annoy", action="store_true", help="Evaluate annoy benchmark"
)
parser.add_argument(
    "--usearch", action="store_true", help="Evaluate usearch benchmark"
)
args = parser.parse_args()


def download_data(dataset: str, path: str):
    if os.path.exists(path):
        return

    url = f"http://ann-benchmarks.com/{dataset}"
    print(f"Downloading {dataset} from {url} ...")

    response = requests.get(url)
    response.raise_for_status()
    with open(path, "wb") as f:
        f.write(response.content)

    print(f"Downloaded {dataset} to {path}")


def create_faiss(dataset: str, dataset_config):
    index_path = os.path.join(INDEX_DIR, f"{dataset}.ivf")
    config = dataset_config.get("faiss", {})
    runner = mod_faiss.Faiss()
    return runner, index_path, config, "faiss"


def create_annoy(dataset: str, dataset_config):
    index_path = os.path.join(INDEX_DIR, f"{dataset}.ann")
    config = dataset_config.get("annoy", {})
    runner = mod_annoy.Annoy()
    return runner, index_path, config, "annoy"


def create_usearch(dataset: str, dataset_config):
    index_path = os.path.join(INDEX_DIR, f"{dataset}.usearch")
    config = dataset_config.get("usearch", {})
    runner = mod_usearch.Usearch()
    return runner, index_path, config, "usearch"


def runner_create_index(
    create_f, dataset: str, dataset_config, train: h5py.Dataset
):
    runner, index_path, config, _ = create_f(dataset, dataset_config)
    if not args.recreate_index and os.path.exists(index_path):
        return
    runner.create_index(train, index_path, config)
    pass


def save_bench(
    dataset: str,
    tag: str,
    runner_name: str,
    nb_runs: int,
    mean_recall,
    mean_time,
    std_time,
    mean_qps,
    std_qps,
):
    path = os.path.join(RESULT_DIR, f"{dataset}.csv")
    header = [
        "runner_name",
        "nb_runs",
        "tag",
        "mean_recall",
        "mean_time",
        "std_time",
        "mean_qps",
        "std_qps",
    ]

    if os.path.isfile(path):
        with open(path, mode="r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
            data_rows = rows[1:] if len(rows) > 1 else []
            data_rows = [
                row
                for row in data_rows
                if not (
                    row[0] == runner_name
                    and int(row[1]) == nb_runs
                    and row[2] == tag
                )
            ]
    else:
        data_rows = []

    new_row = list(
        map(
            str,
            [
                runner_name,
                nb_runs,
                tag,
                mean_recall,
                mean_time,
                std_time,
                mean_qps,
                std_qps,
            ],
        )
    )

    data_rows.append(new_row)
    data_rows.sort(key=lambda r: (r[0], int(r[1]), r[2]))

    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)


def runner_bench(
    create_f,
    dataset: str,
    dataset_config,
    train: h5py.Dataset,
    test: h5py.Dataset,
    neighbors: h5py.Dataset,
):
    runner, index_path, config, runner_name = create_f(dataset, dataset_config)
    runner.load_index(train, index_path, args.threads, config)

    k = neighbors.shape[1]
    total = neighbors.shape[0] * k
    n = test.shape[0]

    recalls = []
    total_times = []
    qpss = []

    for run in range(args.run):
        print(f"Run {run + 1}/{args.run} ({args.tag})")

        pred_vecs, total_time = runner.query_batch(test, k)
        hits = 0

        for i, pred_indices in enumerate(pred_vecs):
            pred_keys = pred_indices
            true_keys = neighbors[i][:k].tolist()
            hits += len(set(pred_keys) & set(true_keys))

        recall = hits / total
        qps = n / total_time

        recalls.append(recall)
        total_times.append(total_time)
        qpss.append(qps)

    mean_recall = np.mean(recalls)
    mean_time = np.mean(total_times)
    std_time = np.std(total_times)
    mean_qps = np.mean(qpss)
    std_qps = np.std(qpss)

    save_bench(
        dataset,
        args.tag,
        runner_name,
        args.run,
        mean_recall,
        mean_time,
        std_time,
        mean_qps,
        std_qps,
    )

    print(f"Mean Recall@{k}: {mean_recall:.4f}")
    print(f"Mean Total search time: {mean_time:.6f} ± {std_time:.6f} seconds")
    print(f"Mean Queries per second (QPS): {mean_qps:.2f} ± {std_qps:.2f}\n")


def run():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    for dataset in args.datasets:
        print(f"-- Dataset {dataset} --")
        path = os.path.join(DATA_DIR, dataset)
        download_data(dataset, path)

        with h5py.File(path, "r") as f:
            train = f["train"]
            test = f["test"]
            neighbors = f["neighbors"]
            if not isinstance(train, h5py.Dataset):
                raise TypeError(f"'train' is not a dataset but {type(train)}")
            if not isinstance(test, h5py.Dataset):
                raise TypeError(f"'test' is not a dataset but {type(test)}")
            if not isinstance(neighbors, h5py.Dataset):
                raise TypeError(
                    f"'neighbors' is not a dataset but {type(neighbors)}"
                )

            dataset_base, _ = os.path.splitext(dataset)
            dataset_config = CONFIG.get(dataset, {})
            train = train[:]
            test = test[:]
            neighbors = neighbors[:]

            if args.faiss:
                runner_create_index(
                    create_faiss, dataset_base, dataset_config, train
                )
            if args.annoy:
                runner_create_index(
                    create_annoy, dataset_base, dataset_config, train
                )
            if args.usearch:
                runner_create_index(
                    create_usearch, dataset_base, dataset_config, train
                )

            if args.bench:
                if args.faiss:
                    print("== Benching Faiss ==")
                    runner_bench(
                        create_faiss,
                        dataset_base,
                        dataset_config,
                        train,
                        test,
                        neighbors,
                    )
                if args.annoy:
                    print("== Benching Annoy ==")
                    runner_bench(
                        create_annoy,
                        dataset_base,
                        dataset_config,
                        train,
                        test,
                        neighbors,
                    )
                if args.usearch:
                    print("== Benching Usearch ==")
                    runner_bench(
                        create_usearch,
                        dataset_base,
                        dataset_config,
                        train,
                        test,
                        neighbors,
                    )


run()
