import re
import shutil
import subprocess


class MooreThreadsGPUStat:
    """Parse GPU info from mthreads-gmi for a single Moore Threads MTT GPU."""

    def __init__(self, gpu_id: int, name: str, mem_used_mb: int, mem_total_mb: int):
        self.gpu_id = gpu_id
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb

    def to_dict(self) -> dict:
        return {
            "gpu_id": self.gpu_id,
            "name": self.name,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
        }

    def __repr__(self) -> str:
        return (
            f"MooreThreadsGPUStat(gpu_id={self.gpu_id}, name={self.name!r}, "
            f"mem_used={self.mem_used_mb}MB, mem_total={self.mem_total_mb}MB)"
        )


class MooreThreadsGPUStatCollection:
    """Collects stats for all Moore Threads GPUs via mthreads-gmi.

    Parses the device rows from ``mthreads-gmi``.  Each device occupies
    three consecutive lines; the first line carries the key fields:

        0    MTT S80        |00000000:01:00.0    |98%   1339MiB(16384MiB)

    The memory column format is ``UsedMiB(TotalMiB)``.
    """

    def __init__(self, gpus: list[MooreThreadsGPUStat]):
        self.gpus = gpus

    def to_dict(self) -> dict:
        return {"gpus": [gpu.to_dict() for gpu in self.gpus]}

    def __len__(self) -> int:
        return len(self.gpus)

    def __iter__(self):
        return iter(self.gpus)

    def __repr__(self) -> str:
        return f"MooreThreadsGPUStatCollection({self.gpus!r})"

    @classmethod
    def new_query(cls) -> "MooreThreadsGPUStatCollection":
        """
        Query mthreads-gmi and return a MooreThreadsGPUStatCollection.
        Returns an empty collection if mthreads-gmi is not installed or fails.
        """
        if shutil.which("mthreads-gmi") is None:
            return cls([])

        try:
            result = subprocess.run(
                ["mthreads-gmi"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return cls([])

        if result.returncode != 0 or not result.stdout.strip():
            return cls([])

        return cls(_parse_mthreads_gmi(result.stdout))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Device row pattern:
#   "0    MTT S80        |00000000:01:00.0    |98%   1339MiB(16384MiB)"
# We anchor on a leading digit (GPU ID), capture the device name before the
# first pipe, then capture the memory field at the end.
_DEVICE_ROW = re.compile(
    r"^(\d+)\s+"          # GPU ID
    r"([\w\s]+?)"         # device name (lazy, stops before extra spaces/pipe)
    r"\s*\|[^|]+"         # PCIe column (ignored)
    r"\|[^|]*?"           # util/temp column (ignored)
    r"(\d+)MiB\((\d+)MiB\)"  # UsedMiB(TotalMiB)
)


def _parse_mthreads_gmi(output: str) -> list[MooreThreadsGPUStat]:
    gpus: list[MooreThreadsGPUStat] = []
    for line in output.splitlines():
        line = line.strip()
        m = _DEVICE_ROW.match(line)
        if not m:
            continue
        try:
            gpus.append(MooreThreadsGPUStat(
                gpu_id=int(m.group(1)),
                name=m.group(2).strip(),
                mem_used_mb=int(m.group(3)),
                mem_total_mb=int(m.group(4)),
            ))
        except ValueError:
            continue
    return gpus


if __name__ == "__main__":
    import json
    collection = MooreThreadsGPUStatCollection.new_query()
    print(json.dumps(collection.to_dict(), indent=2))
