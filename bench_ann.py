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

# AutoNUMA plateaus at ~135s, the others by run 2
WARMUP = 30
WARMUP_BALANCING = 150


def run_bench(tag: str, warmup: int = WARMUP) -> str:
    return (
        "uv run run_ann.py --faiss --annoy --usearch --bench"
        f" --tag {tag} --warmup-time {warmup}"
    )


def build_ann():
    """Download every dataset and build the index each runner needs. Existing
    indices are kept, pass --recreate-index to run_ann.py to rebuild them."""
    sh("uv run run_ann.py --faiss --annoy --usearch")


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
    sh(f"{run_bench('numa-balancing', WARMUP_BALANCING)}")
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
MAIN_DYNAMIC = 3  # bound while there is room, interleaved once there is not


@dataclass(frozen=True)
class PressureVariant:
    tag: str
    numactl: str = ""
    numa_balancing: bool = False
    main_placement: int | None = None  # None on the stock kernel
    # some warmup time
    settle: int = 30


PRESSURE_VARIANTS = [
    # no numactl, no balancing: the kernel default, which is first touch
    PressureVariant("firsttouch"),
    PressureVariant("interleaved", numactl="numactl --interleave=all"),
    PressureVariant("numa-balancing", numa_balancing=True),
]

PRESSURE_VARIANTS_REPL = [
    PressureVariant("repl-bound", main_placement=MAIN_BOUND),
    PressureVariant("repl-firsttouch", main_placement=MAIN_FIRSTTOUCH),
    PressureVariant("repl-interleaved", main_placement=MAIN_INTERLEAVED),
    PressureVariant("repl-dynamic", main_placement=MAIN_DYNAMIC),
]


def run_bench_pressure(variant: PressureVariant, running_time: int) -> str:
    """Set the knobs, return the command for pressure.py to run. Every knob is
    set explicitly so a variant cannot inherit the previous one's state."""
    sh(f"echo {int(variant.numa_balancing)} > /proc/sys/kernel/numa_balancing")

    # no warmup, the phases split these runs at plot time
    cmd = (
        f"uv run run_ann.py --usearch --bench --tag pressure-{variant.tag}"
        f" --datasets {PRESSURE_DATASET} --running-time {running_time}"
        f" --warmup-time 0"
    )
    if variant.numactl:
        cmd = f"{variant.numactl} {cmd}"

    if variant.main_placement is not None:
        sh(
            f"echo {variant.main_placement} >"
            " /sys/kernel/debug/repl_pt/main_placement"
        )
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
