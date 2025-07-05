# ANN simple benchmarks

Datasets are from the ANN benchmark project (https://github.com/erikbern/ann-benchmarks).
Comparison of faiss IVF, usearch and annoy using mmaped data index.

The goal is to evaluate our NUMA kernel optimization which replicates the file backed mmap areas, thus maximizing memory access locality on the NUMA nodes.

We measure throughput using batch query as Query Per Second (QPS) for all of them (simulated on annoy using a threadpool).
