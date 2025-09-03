import asyncio
import time

from qemu.qmp import QMPClient
from subprocess import call

MAX_NB_TRY = 10


async def main():
    qmp = QMPClient("replication-vm")
    nbtry = 0

    while nbtry < MAX_NB_TRY:
        try:
            await qmp.connect("/tmp/qmp-sock")
            break
        except Exception as e:
            print(
                f"Failed to connect to QEMU (try {nbtry} / {MAX_NB_TRY}): {e}"
            )
            nbtry += 1
            time.sleep(1)

    res = await qmp.execute("query-status")
    print(f"VM status: {res['status']}")

    vcpus = await qmp.execute("query-cpus-fast")
    for cpu in vcpus:
        tid = cpu["thread-id"]
        vcpuid = cpu["cpu-index"]
        call(["taskset", "-pc", str(vcpuid), str(tid)])

    await qmp.disconnect()


asyncio.run(main())
