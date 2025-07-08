# Use all the cores
echo 3 > /proc/sys/vm/drop_caches
uv run main.py --faiss --annoy --usearch --bench --run 10

# Use only cores in one node
# full local
echo 3 > /proc/sys/vm/drop_caches
numactl --membind=0 --cpunodebind=0 uv run main.py --faiss --annoy --usearch --bench --run 10 --tag local
 
# full remote
echo 3 > /proc/sys/vm/drop_caches
numactl --membind=0 --cpunodebind=1 uv run main.py --faiss --annoy --usearch --bench --run 10 --tag distant
