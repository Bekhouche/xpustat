import json
import re
import shutil
import subprocess
from typing import Optional


class IntelGPUStat:
    """Parse GPU info from xpu-smi for a single Intel Arc / Data Center GPU."""

    def __init__(self, device_id: int, name: str, mem_used_mb: int, mem_total_mb: int):
        self.device_id = device_id
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
        }

    def __repr__(self) -> str:
        return (
            f"IntelGPUStat(device_id={self.device_id}, name={self.name!r}, "
            f"mem_used={self.mem_used_mb}MB, mem_total={self.mem_total_mb}MB)"
        )


class IntelGPUStatCollection:
    """Collects stats for all Intel GPUs via xpu-smi.

    Steps:
      1. ``xpu-smi discovery -j``  → JSON device list (device_id, device_name)
      2. ``xpu-smi dump -d <id> -m 18 -n 1``  → CSV, metric 18 = GPU Memory Used (MiB)
      3. ``xpu-smi discovery -d <id>``  → text, parse "Memory Physical Size: X MiB"
    """

    def __init__(self, gpus: list[IntelGPUStat]):
        self.gpus = gpus

    def to_dict(self) -> dict:
        return {"gpus": [gpu.to_dict() for gpu in self.gpus]}

    def __len__(self) -> int:
        return len(self.gpus)

    def __iter__(self):
        return iter(self.gpus)

    def __repr__(self) -> str:
        return f"IntelGPUStatCollection({self.gpus!r})"

    @staticmethod
    def _run(*args: str, timeout: int = 10) -> Optional[str]:
        try:
            result = subprocess.run(
                ["xpu-smi"] + list(args),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None
            return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    @classmethod
    def new_query(cls) -> "IntelGPUStatCollection":
        """
        Query xpu-smi and return an IntelGPUStatCollection.
        Returns an empty collection if xpu-smi is not installed or fails.
        """
        if shutil.which("xpu-smi") is None:
            return cls([])

        discovery_out = cls._run("discovery", "-j")
        if discovery_out is None:
            return cls([])

        try:
            discovery_data = json.loads(discovery_out)
        except json.JSONDecodeError:
            return cls([])

        device_list = discovery_data.get("device_list", [])
        gpus: list[IntelGPUStat] = []

        for entry in device_list:
            device_id = entry.get("device_id")
            name = entry.get("device_name", "Unknown Intel GPU")
            if device_id is None:
                continue

            mem_used_mb = _query_mem_used(cls, device_id)
            mem_total_mb = _query_mem_total(cls, device_id)

            gpus.append(IntelGPUStat(
                device_id=device_id,
                name=name,
                mem_used_mb=mem_used_mb,
                mem_total_mb=mem_total_mb,
            ))

        return cls(gpus)


# Matches: "GPU Memory Used (MiB),  24" (last value on a CSV dump line)
_MEM_USED_CSV = re.compile(r",\s*([\d.]+)\s*$")
# Matches: "Memory Physical Size: 32768.00 MiB"
_MEM_TOTAL_TEXT = re.compile(r"Memory Physical Size\s*:\s*([\d.]+)\s*MiB", re.IGNORECASE)


def _query_mem_used(cls: type[IntelGPUStatCollection], device_id: int) -> int:
    """Return current GPU memory used in MiB via xpu-smi dump metric 18."""
    out = cls._run("dump", "-d", str(device_id), "-m", "18", "-n", "1")
    if out is None:
        return 0
    for line in out.splitlines():
        if line.startswith("Timestamp") or not line.strip():
            continue
        m = _MEM_USED_CSV.search(line)
        if m:
            try:
                return int(float(m.group(1)))
            except ValueError:
                pass
    return 0


def _query_mem_total(cls: type[IntelGPUStatCollection], device_id: int) -> int:
    """Return total GPU memory in MiB by parsing xpu-smi discovery text output."""
    out = cls._run("discovery", "-d", str(device_id))
    if out is None:
        return 0
    m = _MEM_TOTAL_TEXT.search(out)
    if m:
        try:
            return int(float(m.group(1)))
        except ValueError:
            pass
    return 0


if __name__ == "__main__":
    collection = IntelGPUStatCollection.new_query()
    print(json.dumps(collection.to_dict(), indent=2))
