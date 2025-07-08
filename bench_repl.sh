# baseline patched, no repl
echo 3 > /proc/sys/vm/drop_caches
uv run main.py --faiss --annoy --usearch --bench --run 10 --tag patched

# baseline patched, repl
# register index extension for replication 
echo 1 > /sys/kernel/debug/repl_pt/clear_registered
echo .ivf > /sys/kernel/debug/repl_pt/registered
echo .ann > /sys/kernel/debug/repl_pt/registered
echo .usearch > /sys/kernel/debug/repl_pt/registered
# run
echo 3 > /proc/sys/vm/drop_caches
echo 1 > /sys/kernel/debug/repl_pt/policy
uv run main.py --faiss --annoy --usearch --bench --run 10 --tag patched-repl
echo 0 > /sys/kernel/debug/repl_pt/policy

# limited number of cores, check numa effect
# full local, no repl
CPUS=$(uv run one_node_cpus_interleaved.py)
echo 3 > /proc/sys/vm/drop_caches
numactl --physcpubind=$CPUS uv run main.py --faiss --annoy --usearch --bench --run 10 --tag patched-balanced

# full local, repl
echo 3 > /proc/sys/vm/drop_caches
echo 1 > /sys/kernel/debug/repl_pt/policy
numactl --physcpubind=$CPUS uv run main.py --faiss --annoy --usearch --bench --run 10 --tag patched-repl-balanced
echo 0 > /sys/kernel/debug/repl_pt/policy
