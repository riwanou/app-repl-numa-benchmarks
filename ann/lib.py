import requests
import os
import h5py
import numpy as np
import csv
import time
from config import get_time, sh
from . import mod_faiss
from . import mod_annoy
from . import mod_usearch

# per variant warmup, shared measured window
WARMUP_TIME = 30
MEASURE_TIME = 60

CONFIG = {
    "glove-100-angular.hdf5": {
        "faiss": {"nlist": 100, "nprobe": 39},
        "annoy": {"trees": 100, "search_k": 250_000},
        "usearch": {"e_search": 5000},
    },
    "gist-960-euclidean.hdf5": {
        "faiss": {"nlist": 100, "nprobe": 18},
        "annoy": {"trees": 100, "search_k": 500_000},
        "usearch": {"e_search": 2500},
    },
}

DATASETS = list(CONFIG.keys())


def sync_drop_caches():
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")


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
    runner.create_index(train[:], index_path, config)
    pass


def save_bench(
    result_dir: str,
    dataset: str,
    tag: str,
    runner_name: str,
    nb_runs: int,
    start_time: str,
    end_time: str,
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
        "start_time",
        "end_time",
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
                start_time,
                end_time,
            ],
        )
    )

    data_rows.append(new_row)
    data_rows.sort(key=lambda r: (r[0], int(r[1]), r[2]))

    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)


def save_bench_details(
    result_dir: str,
    dataset: str,
    tag: str,
    runner_name: str,
    recalls,
    total_times,
    qpss,
    run_start_times,
    run_end_times,
    warmups,
):
    path = os.path.join(result_dir, f"{dataset}-details.csv")
    header = [
        "runner_name",
        "tag",
        "run_id",
        "recall",
        "total_time",
        "qps",
        "start_time",
        "end_time",
        "warmup",
    ]

    if os.path.isfile(path):
        with open(path, mode="r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
            data_rows = rows[1:] if len(rows) > 1 else []
            data_rows = [
                row
                for row in data_rows
                if not (row[0] == runner_name and row[1] == tag)
            ]
    else:
        data_rows = []

    for i, (recall, total_time, qps, run_start, run_end, warmup) in enumerate(
        zip(
            recalls,
            total_times,
            qpss,
            run_start_times,
            run_end_times,
            warmups,
        ),
        1,
    ):
        data_rows.append(
            list(
                map(
                    str,
                    [
                        runner_name,
                        tag,
                        i,
                        recall,
                        total_time,
                        qps,
                        run_start,
                        run_end,
                        warmup,
                    ],
                )
            )
        )

    data_rows.sort(key=lambda r: (r[0], r[1], int(r[2])))

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
    running_time: int,
    warmup_time: int,
    measure_time: int,
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
    run_start_times = []
    run_end_times = []
    warmups = []

    begin = time.time()
    start_time = get_time()
    measure_begin = None

    nb_runs = 0
    while True:
        # warmup on the clock it started on
        warmup = time.time() - begin < warmup_time
        if not warmup and measure_begin is None:
            measure_begin = time.time()
        run_start_time = get_time()
        pred_vecs, total_time = runner.query_batch(test, k)
        run_end_time = get_time()
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
        run_start_times.append(run_start_time)
        run_end_times.append(run_end_time)
        warmups.append(warmup)

        elapsed_time = time.time() - begin
        nb_runs += 1

        measured = 0 if measure_begin is None else time.time() - measure_begin
        print(
            f"Run {nb_runs} [{tag}]{' warmup' if warmup else ''} run "
            f"{total_time:.2f}s QPS {qps:.2f} elapsed {elapsed_time:.2f}s "
            f"measured {measured:.2f}s"
        )

        # the pressure bench owns its duration
        if running_time:
            if elapsed_time >= running_time:
                break
        elif measured >= measure_time:
            break

    end_time = get_time()

    # steady state only
    kept = [i for i, warmup in enumerate(warmups) if not warmup]
    if not kept:
        print(f"[WARN] [{tag}] every run fell in the warmup, summarising all")
        kept = list(range(nb_runs))

    mean_recall = np.mean([recalls[i] for i in kept])
    mean_time = np.mean([total_times[i] for i in kept])
    std_time = np.std([total_times[i] for i in kept])
    mean_qps = np.mean([qpss[i] for i in kept])
    std_qps = np.std([qpss[i] for i in kept])

    save_bench(
        result_dir,
        dataset,
        tag,
        runner_name,
        len(kept),
        start_time,
        end_time,
        mean_recall,
        mean_time,
        std_time,
        mean_qps,
        std_qps,
    )

    save_bench_details(
        result_dir,
        dataset,
        tag,
        runner_name,
        recalls,
        total_times,
        qpss,
        run_start_times,
        run_end_times,
        warmups,
    )

    print(
        f"[{tag}] {len(kept)}/{nb_runs} runs kept  Recall@{k}: {mean_recall:.4f}  "
        f"Time: {mean_time:.4f} ± {std_time:.4f}s  QPS: {mean_qps:.2f} ± {std_qps:.2f}"
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
    running_time: int,
    warmup_time: int,
    measure_time: int,
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
            # train stays lazy: the bench only reads its shape, and
            # materialising it costs 3.8G of anon that nothing ever touches
            # again. runner_create_index pulls it in when it has to build.
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
                sync_drop_caches()

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
                        running_time,
                        warmup_time,
                        measure_time,
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
                        running_time,
                        warmup_time,
                        measure_time,
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
                        running_time,
                        warmup_time,
                        measure_time,
                    )
