# NUMA replication Benchmarks

Benchmark applications to see the impact of our Linux kernel's memory replication optimization in NUMA systems.
Our NUMA kernel optimization replicates the file backed mmap areas, thus maximizing memory access locality on the NUMA nodes.

## ANN simple benchmarks

Datasets are from the ANN benchmark project (https://github.com/erikbern/ann-benchmarks).
Comparison of faiss IVF, usearch and annoy using mmaped data index.
We measure throughput using batch query as Query Per Second (QPS) for all of them (simulated on annoy using a threadpool).

## Rocks db

We use db_bench on mostly read benchmarks with the `mmap_read` option to use file backed memory.
