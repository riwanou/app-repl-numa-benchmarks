import h5py
import multiprocessing.pool
import time
import numpy as np
from annoy import AnnoyIndex


class Annoy:
    def _annoy_index(self, dims: int, path: str):
        if "angular" in path:
            index = AnnoyIndex(dims, "angular")
        elif "euclidean" in path:
            index = AnnoyIndex(dims, "euclidean")
        else:
            raise ValueError("Unsupported format")
        return index

    def create_index(self, train: h5py.Dataset, index_path: str, config):
        _, dims = train.shape
        trees = config["trees"]

        print(f"Creating index {index_path}, dims={dims}, trees={trees}")

        index = self._annoy_index(dims, index_path)
        for i, vec in enumerate(train):
            index.add_item(i, vec.tolist())
        index.build(trees)
        index.save(index_path)

        print(f"Index created {index_path}")

    def load_index(self, train: h5py.Dataset, index_path: str, config):
        _, dims = train.shape
        search_k = config["search_k"]
        threads = config["threads"]

        index = self._annoy_index(dims, index_path)
        index.load(index_path)

        self._index = index
        self._pool = multiprocessing.pool.ThreadPool(threads)
        self._search_k = search_k

        print(
            f"Index loaded {index_path}, dims={dims}, search_k={search_k}, threads={threads}"
        )

    def query_batch(self, test: h5py.Dataset, k: int):
        def query_f(i_query):
            _, query = i_query
            pred = self._index.get_nns_by_vector(
                query, k, search_k=self._search_k
            )
            return pred

        start_time = time.perf_counter()
        pred = self._pool.map(query_f, enumerate(test))
        end_time = time.perf_counter()
        total_time = end_time - start_time
        return pred, total_time
