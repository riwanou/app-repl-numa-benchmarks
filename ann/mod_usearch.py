import h5py
import numpy as np
import time
from usearch.index import Index


class Usearch:
    def _usearch_index(self, dims: int, path: str, e_search: int | None = None):
        if "angular" in path:
            index = Index(
                ndim=dims, dtype="bf16", metric="cos", expansion_search=e_search
            )
        elif "euclidean" in path:
            index = Index(
                ndim=dims,
                dtype="bf16",
                metric="l2sq",
                expansion_search=e_search,
            )
        else:
            raise ValueError("Unsupported format")
        return index

    def create_index(self, train: h5py.Dataset, index_path: str, _):
        nvecs, dims = train.shape

        print(f"Creating index {index_path}, dims={dims}")

        index = self._usearch_index(dims, index_path)
        keys = np.arange(nvecs)
        index.add(keys, train)
        index.save(index_path)

        print(f"Index created {index_path}")

    def load_index(self, train: h5py.Dataset, index_path: str, _: int, config):
        _, dims = train.shape
        e_search = config["e_search"]

        index = self._usearch_index(dims, index_path, e_search=e_search)
        index.view(index_path)

        self._index = index

        print(f"Index loaded {index_path}, dims={dims}, e_search={e_search}")

    def query_batch(self, test: h5py.Dataset, k: int):
        start_time = time.perf_counter()
        matches = self._index.search(test, k, threads=0)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        return matches.keys, total_time
