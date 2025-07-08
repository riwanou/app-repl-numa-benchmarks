import subprocess


def get_cpus(node):
    for line in subprocess.check_output(
        ["numactl", "--hardware"], text=True
    ).splitlines():
        if f"node {node} cpus:" in line:
            return list(map(int, line.split(":")[1].split()))
    return []


cpus0 = get_cpus(0)
cpus1 = get_cpus(1)
half = len(cpus0) // 2
selected = cpus0[:half] + cpus1[:half]
print(",".join(map(str, selected)))
