import time
import csv
import os
import argparse
from config import get_time


def parse_meminfo():
    meminfo = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split(":")
            if len(parts) != 2:
                continue
            key = parts[0].strip()
            value = int(parts[1].strip().split()[0])
            meminfo[key] = value

    mapped = {
        "total": meminfo.get("MemTotal", 0),
        "free": meminfo.get("MemFree", 0),
        "available": meminfo.get("MemAvailable", 0),
        "dirty": meminfo.get("Dirty", 0),
        "anon": meminfo.get("AnonPages", 0),
        "mapped": meminfo.get("Mapped", 0),
        "pageTable": meminfo.get("PageTables", 0),
        "vmallocTotal": meminfo.get("VmallocTotal", 0),
        "vmallocUsed": meminfo.get("VmallocUsed", 0),
    }

    return mapped


def parse_numa_meminfo():
    node_data = {}
    node_base = "/sys/devices/system/node"

    if not os.path.exists(node_base):
        return node_data

    for node in sorted(os.listdir(node_base)):
        if not node.startswith("node"):
            continue

        node_id = node.replace("node", "")
        meminfo_path = os.path.join(node_base, node, "meminfo")

        meminfo = {}
        if os.path.exists(meminfo_path):
            try:
                with open(meminfo_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            key = parts[2].rstrip(":")
                            value = int(parts[3])
                            meminfo[key] = value

                node_data[node_id] = {
                    "total": meminfo.get("MemTotal", 0),
                    "free": meminfo.get("MemFree", 0),
                    "used": meminfo.get("MemUsed", 0),
                    "dirty": meminfo.get("Dirty", 0),
                    "anon": meminfo.get("AnonPages", 0),
                    "mapped": meminfo.get("Mapped", 0),
                    "pageTable": meminfo.get("PageTables", 0),
                }
            except Exception:
                continue

    return node_data


def parse_own_proc_meminfo():
    meminfo = {}
    with open("/proc/self/status") as f:
        for line in f:
            if any(
                line.startswith(field)
                for field in ["VmRSS:", "RssAnon:", "RssFile:", "VmPTE:"]
            ):
                key, val = line.split(":", 1)
                meminfo[key.strip()] = int(val.strip().split()[0])

    return meminfo


def collect_mem(mem_csv, interval):
    numa_nodes = parse_numa_meminfo()

    base_fields = [
        "time",
        "total",
        "free",
        "available",
        "dirty",
        "anon",
        "mapped",
        "pageTable",
        "vmallocTotal",
        "vmallocUsed",
        "own_VmRSS",
        "own_RssAnon",
        "own_RssFile",
        "own_VmPTE",
    ]

    node_fields = []
    for node_id in sorted(numa_nodes.keys()):
        node_fields.extend(
            [
                f"Node{node_id}_total",
                f"Node{node_id}_free",
                f"Node{node_id}_used",
                f"Node{node_id}_dirty",
                f"Node{node_id}_anon",
                f"Node{node_id}_mapped",
                f"Node{node_id}_pageTable",
            ]
        )

    fieldnames = base_fields + node_fields

    with open(mem_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            timestamp = get_time()
            meminfo = parse_meminfo()
            numa_stats = parse_numa_meminfo()
            own_mem = parse_own_proc_meminfo()

            row = {
                "time": timestamp,
                "total": meminfo["total"],
                "free": meminfo["free"],
                "available": meminfo["available"],
                "dirty": meminfo["dirty"],
                "anon": meminfo["anon"],
                "mapped": meminfo["mapped"],
                "pageTable": meminfo["pageTable"],
                "vmallocTotal": meminfo["vmallocTotal"],
                "vmallocUsed": meminfo["vmallocUsed"],
                "own_VmRSS": own_mem.get("VmRSS", 0),
                "own_RssAnon": own_mem.get("RssAnon", 0),
                "own_RssFile": own_mem.get("RssFile", 0),
                "own_VmPTE": own_mem.get("VmPTE", 0),
            }

            for node_id, stats in numa_stats.items():
                row[f"Node{node_id}_total"] = stats["total"]
                row[f"Node{node_id}_free"] = stats["free"]
                row[f"Node{node_id}_used"] = stats["used"]
                row[f"Node{node_id}_dirty"] = stats["dirty"]
                row[f"Node{node_id}_anon"] = stats["anon"]
                row[f"Node{node_id}_mapped"] = stats["mapped"]
                row[f"Node{node_id}_pageTable"] = stats["pageTable"]

            writer.writerow(row)
            csvfile.flush()
            time.sleep(interval)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--interval", type=int, default=1, help="Sampling interval in seconds"
)
parser.add_argument(
    "-csv", "--csv_path", type=str, required=True, help="Output CSV file path"
)
args = parser.parse_args()


collect_mem(args.csv_path, args.interval)
