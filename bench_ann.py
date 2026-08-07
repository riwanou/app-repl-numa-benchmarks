import subprocess
from config import sh


def get_interleaved_cpus_one_node() -> str:
    """
    Get half the CPUs from one NUMA node, interleaved.
    """

    def get_cpus(node):
        lines = subprocess.check_output(["numactl", "--hardware"], text=True)
        for line in lines.splitlines():
            if f"node {node} cpus:" in line:
                return [int(cpu) for cpu in line.split(":")[1].split()]
        return []

    cpus0, cpus1 = get_cpus(0), get_cpus(1)
    half = min(len(cpus0), len(cpus1)) // 2
    selected = cpus0[:half] + cpus1[:half]
    return ",".join(map(str, selected))


def run_bench(tag: str) -> str:
    return f"uv run run_ann.py --faiss --annoy --usearch --bench --tag {tag}"


def run_bench_ann():
    # disable numa balancing
    sh("echo 0 > /proc/sys/kernel/numa_balancing")

    # all cores
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"{run_bench('default')}")

    # worst case (mem in 1 node)
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"numactl --membind={0} {run_bench('imbalanced-memory')}")

    # best case (interleaved)
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"numactl --interleave=all {run_bench('interleaved-memory')}")

    # a case (numa balancing)
    sh("echo 1 > /proc/sys/kernel/numa_balancing")
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"{run_bench('numa-balancing')}")
    sh("echo 0 > /proc/sys/kernel/numa_balancing")


def run_bench_ann_repl():
    sh("echo 0 > /sys/kernel/debug/repl_pt/main_placement")

    # baseline patched, all cores, repl
    sh("echo 1 > /sys/kernel/debug/repl_pt/clear_registered")
    sh("echo .ivf > /sys/kernel/debug/repl_pt/registered")
    sh("echo .ann > /sys/kernel/debug/repl_pt/registered")
    sh("echo .usearch > /sys/kernel/debug/repl_pt/registered")

    # run
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"""(
      echo 1 > /sys/kernel/debug/repl_pt/policy &&
      {run_bench("patched-repl")};
      echo 0 > /sys/kernel/debug/repl_pt/policy
    )""")


def run_bench_ann_pressure():
    sh("echo 1 > /proc/sys/kernel/numa_balancing")
    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"{run_bench('pressure-numa-balancing')}")
    sh("echo 0 > /proc/sys/kernel/numa_balancing")


def run_bench_ann_pressure_repl():
    sh("echo 0 > /sys/kernel/debug/repl_pt/main_placement")

    sh("echo 1 > /sys/kernel/debug/repl_pt/clear_registered")
    sh("echo .ivf > /sys/kernel/debug/repl_pt/registered")
    sh("echo .ann > /sys/kernel/debug/repl_pt/registered")
    sh("echo .usearch > /sys/kernel/debug/repl_pt/registered")

    sh("echo 3 > /proc/sys/vm/drop_caches")
    sh(f"""(
      echo 1 > /sys/kernel/debug/repl_pt/policy &&
      {run_bench("pressure-patched-repl")};
      echo 0 > /sys/kernel/debug/repl_pt/policy
    )""")
