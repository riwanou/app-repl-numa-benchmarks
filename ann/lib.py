import requests
import os
import h5py
import numpy as np
import csv
import time
from . import mod_faiss
from . import mod_annoy
from . import mod_usearch

MIN_NB_RUNS = 3
MIN_STD_TIME = 0.1
MAX_BENCH_TIME = 60

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


def create_faiss(index_dir: str, dataset: str, dataset_config):
    index_path = os.path.join(index_dir, f"{dataset}.ivf")
    config = dataset_config.get("faiss", {})
    runner = mod_faiss.Faiss()
    return runner, index_path, config, "faiss"


def create_annoy(index_dir: str, dataset: str, dataset_config):
    index_path = os.path.join(index_dir, f"{dataset}.ann")
    config = dataset_config.get("annoy", {})
    runner = mod_annoy.Annoy()
    return runner, index_path, config, "annoy"


def create_usearch(index_dir: str, dataset: str, dataset_config):
    index_path = os.path.join(index_dir, f"{dataset}.usearch")
    config = dataset_config.get("usearch", {})
    runner = mod_usearch.Usearch()
    return runner, index_path, config, "usearch"


def runner_create_index(
    create_f,
    index_dir: str,
    dataset: str,
    dataset_config,
    train: h5py.Dataset,
    recreate_index: bool,
):
    runner, index_path, config, _ = create_f(index_dir, dataset, dataset_config)
    if not recreate_index and os.path.exists(index_path):
        return
    runner.create_index(train, index_path, config)
    pass


def save_bench(
    result_dir: str,
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
    path = os.path.join(result_dir, f"{dataset}.csv")
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
                if not (row[0] == runner_name and row[2] == tag)
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
    index_dir: str,
    result_dir: str,
    dataset: str,
    dataset_config,
    train: h5py.Dataset,
    test: h5py.Dataset,
    neighbors: h5py.Dataset,
    tag: str,
    threads: int,
):
    runner, index_path, config, runner_name = create_f(
        index_dir, dataset, dataset_config
    )
    runner.load_index(train, index_path, threads, config)

    k = neighbors.shape[1]
    total = neighbors.shape[0] * k
    n = test.shape[0]

    recalls = []
    total_times = []
    qpss = []

    start_time = time.time()
    nb_runs = 0

    while True:
        nb_runs += 1

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

        mean_time = np.mean(total_times)
        std_time = np.std(total_times)
        elapsed_time = time.time() - start_time

        print(
            f"Run {nb_runs} (min {MIN_NB_RUNS}) [{tag}] elapsed {elapsed_time:.2f}s (max {MAX_BENCH_TIME}s) +- {std_time:.4f}s (min {MIN_STD_TIME:.4f})"
        )
        if std_time <= MIN_STD_TIME and nb_runs >= MIN_NB_RUNS:
            break
        if elapsed_time >= MAX_BENCH_TIME:
            break

    mean_recall = np.mean(recalls)
    mean_qps = np.mean(qpss)
    std_qps = np.std(qpss)

    save_bench(
        result_dir,
        dataset,
        tag,
        runner_name,
        nb_runs,
        mean_recall,
        mean_time,
        std_time,
        mean_qps,
        std_qps,
    )

    print(
        f"[{tag}] Recall@{k}: {mean_recall:.4f}  Time: {mean_time:.4f} ± {std_time:.4f}s  QPS: {mean_qps:.2f} ± {std_qps:.2f}"
    )


def run(
    data_dir: str,
    index_dir: str,
    result_dir: str,
    datasets,
    faiss: bool,
    annoy: bool,
    usearch: bool,
    bench: bool,
    recreate_index: bool,
    tag: str,
    threads: int,
):
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    for dataset in datasets:
        print(f"-- Dataset {dataset} --")
        path = os.path.join(data_dir, dataset)
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

            if faiss:
                runner_create_index(
                    create_faiss,
                    index_dir,
                    dataset_base,
                    dataset_config,
                    train,
                    recreate_index,
                )
            if annoy:
                runner_create_index(
                    create_annoy,
                    index_dir,
                    dataset_base,
                    dataset_config,
                    train,
                    recreate_index,
                )
            if usearch:
                runner_create_index(
                    create_usearch,
                    index_dir,
                    dataset_base,
                    dataset_config,
                    train,
                    recreate_index,
                )

            if bench:
                if faiss:
                    print("== Benching Faiss ==")
                    runner_bench(
                        create_faiss,
                        index_dir,
                        result_dir,
                        dataset_base,
                        dataset_config,
                        train,
                        test,
                        neighbors,
                        tag,
                        threads,
                    )
                if annoy:
                    print("== Benching Annoy ==")
                    runner_bench(
                        create_annoy,
                        index_dir,
                        result_dir,
                        dataset_base,
                        dataset_config,
                        train,
                        test,
                        neighbors,
                        tag,
                        threads,
                    )
                if usearch:
                    print("== Benching Usearch ==")
                    runner_bench(
                        create_usearch,
                        index_dir,
                        result_dir,
                        dataset_base,
                        dataset_config,
                        train,
                        test,
                        neighbors,
                        tag,
                        threads,
                    )
