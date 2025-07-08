# use all the cores, no numa policy
echo 3 > /proc/sys/vm/drop_caches
uv run main.py --faiss --annoy --usearch --bench --run 10

# limited number of cores, check numa effect
# full local
echo 3 > /proc/sys/vm/drop_caches
numactl --membind=0 --cpunodebind=0 uv run main.py --faiss --annoy --usearch --bench --run 10 --tag local
 
# full remote
echo 3 > /proc/sys/vm/drop_caches
numactl --membind=0 --cpunodebind=1 uv run main.py --faiss --annoy --usearch --bench --run 10 --tag distant

# cpu interleaved on 2 nodes, number of cores from one node 
CPUS=$(uv run one_node_cpus_interleaved.py)
echo 3 > /proc/sys/vm/drop_caches
numactl --physcpubind=$CPUS uv run main.py --faiss --annoy --usearch --bench --run 10 --tag balanced
