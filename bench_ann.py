import subprocess
from dataclasses import dataclass

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


PRESSURE_DATASET = "gist-960-euclidean.hdf5"


def run_bench(tag: str) -> str:
    return f"uv run run_ann.py --faiss --annoy --usearch --bench --tag {tag}"


def run_bench_ann():
    # disable numa balancing
    sh("echo 0 > /proc/sys/kernel/numa_balancing")

    # all cores
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    sh(f"{run_bench('default')}")

    # worst case (mem in 1 node)
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    sh(f"numactl --membind={0} {run_bench('imbalanced-memory')}")

    # best case (interleaved)
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    sh(f"numactl --interleave=all {run_bench('interleaved-memory')}")

    # a case (numa balancing)
    sh("echo 1 > /proc/sys/kernel/numa_balancing")
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
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
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    sh(f"""(
      echo 1 > /sys/kernel/debug/repl_pt/policy &&
      {run_bench("patched-repl")};
      echo 0 > /sys/kernel/debug/repl_pt/policy
    )""")


# The pressure bench is the odd one out: it does not run its own command, it
# hands it to pressure.py, which runs it inside a squeezed cgroup.

# main_placement: where the main copy of a replicated table goes
MAIN_BOUND = 0
MAIN_FIRSTTOUCH = 1
MAIN_INTERLEAVED = 2


@dataclass(frozen=True)
class PressureVariant:
    tag: str
    numactl: str = ""
    numa_balancing: bool = False
    main_placement: int | None = None  # None on the stock kernel
    # Seconds to let the variant reach steady state before the plan starts, so
    # the first phase measures a converged system and not a ramp. Index load
    # (~15s) happens inside it. 60s is what replication needs to converge;
    # static placements need only the load, and AutoNUMA needs far more (see
    # numa-balancing below).
    settle: int = 60


PRESSURE_VARIANTS = [
    # AutoNUMA does not even start hurting until ~40s after launch: 1s scan
    # delay, then 256M of address space per pass with a 1s floor on the
    # period, so a 3.84G index needs a dozen-odd passes before coverage bites.
    # QPS then decays over another ~30s to a plateau ~40% below the ramp. A
    # 60s settle would put the knee inside the first phase.
    PressureVariant("numa-balancing", numa_balancing=True, settle=100),
    PressureVariant(
        "interleaved", numactl="numactl --interleave=all", settle=30
    ),
]

PRESSURE_VARIANTS_REPL = [
    PressureVariant("repl-bound", main_placement=MAIN_BOUND),
    PressureVariant("repl-firsttouch", main_placement=MAIN_FIRSTTOUCH),
    PressureVariant("repl-interleaved", main_placement=MAIN_INTERLEAVED),
]


def run_bench_pressure(variant: PressureVariant, running_time: int) -> str:
    """Set the knobs, return the command for pressure.py to run. Every knob is
    set explicitly so a variant cannot inherit the previous one's state."""
    sh(f"echo {int(variant.numa_balancing)} > /proc/sys/kernel/numa_balancing")

    cmd = (
        f"uv run run_ann.py --usearch --bench --tag pressure-{variant.tag}"
        f" --datasets {PRESSURE_DATASET} --running-time {running_time}"
    )
    if variant.numactl:
        cmd = f"{variant.numactl} {cmd}"

    if variant.main_placement is not None:
        sh(f"echo {variant.main_placement} >"
           " /sys/kernel/debug/repl_pt/main_placement")
        sh("echo 1 > /sys/kernel/debug/repl_pt/clear_registered")
        sh("echo .ivf > /sys/kernel/debug/repl_pt/registered")
        sh("echo .ann > /sys/kernel/debug/repl_pt/registered")
        sh("echo .usearch > /sys/kernel/debug/repl_pt/registered")
        cmd = f"""(
          echo 1 > /sys/kernel/debug/repl_pt/policy &&
          {cmd};
          echo 0 > /sys/kernel/debug/repl_pt/policy
        )"""

    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    return cmd


def run_bench_ann_pressure():
    sh("echo 1 > /proc/sys/kernel/numa_balancing")
    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    sh(f"{run_bench_pressure('pressure-numa-balancing')}")
    sh("echo 0 > /proc/sys/kernel/numa_balancing")


def run_bench_ann_pressure_repl():
    sh("echo 0 > /sys/kernel/debug/repl_pt/main_placement")

    sh("echo 1 > /sys/kernel/debug/repl_pt/clear_registered")
    sh("echo .ivf > /sys/kernel/debug/repl_pt/registered")
    sh("echo .ann > /sys/kernel/debug/repl_pt/registered")
    sh("echo .usearch > /sys/kernel/debug/repl_pt/registered")

    sh("sync; echo 3 > /proc/sys/vm/drop_caches")
    sh(f"""(
      echo 1 > /sys/kernel/debug/repl_pt/policy &&
      {run_bench_pressure("pressure-patched-repl")};
      echo 0 > /sys/kernel/debug/repl_pt/policy
    )""")
