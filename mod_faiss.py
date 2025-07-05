import h5py
import numpy as np
import sklearn
import time
import faiss


class Faiss:
    def create_index(self, train: h5py.Dataset, index_path: str, config):
        _, dims = train.shape
        nlist = config["nlist"]

        print(f"Creating index {index_path}, dims={dims}, nlist={nlist}")

        if "angular" in index_path:
            train = sklearn.preprocessing.normalize(train, axis=1, norm="l2")

        if train.dtype != np.float32:
            train = train.astype(np.float32)  # type: ignore

        quantizer = faiss.IndexFlatL2(dims)
        index = faiss.IndexIVFFlat(quantizer, dims, nlist, faiss.METRIC_L2)
        index.train(train)  # type: ignore
        index.add(train)  # type: ignore
        faiss.write_index(index, index_path)

        print(f"Index created {index_path}")

    def load_index(self, train: h5py.Dataset, index_path: str, config):
        _, dims = train.shape
        nprobe = config["nprobe"]

        index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP)
        index.nprobe = nprobe

        self._index = index

        print(f"Index loaded {index_path}, dims={dims}, nprobe={nprobe}")

    def query_batch(self, test: h5py.Dataset, k: int):
        start_time = time.perf_counter()
        _, I = self._index.search(test, k)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        return I.tolist(), total_time
