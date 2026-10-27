"""
tpu.py — Google Cloud TPU chips via ``tpu-info``.

``tpu-info --format json dashboard`` prints::

    {
      "timestamp": "...",
      "tpu_info_version": "0.5.1",
      "libtpu_version": "0.0.18",
      "topology": "...",
      "devices": [
        {
          "device_id": 0,
          "device_path": "/dev/vfio/0",
          "memory_used_bytes": 18538430464,
          "memory_total_bytes": 33554432000,
          "duty_cycle_pct": 100.0,
          "hbm_bandwidth_util_pct": 12.5,
          "process": {"pid": 799657, "name": "python3"}   # or null
        }
      ]
    }

``tpu-info --version`` adds the accelerator type (``v5e``, ``v6e`` …), which
becomes part of the device name. Metrics need a running TPU workload
(JAX / PyTorch-XLA); with none, memory and duty cycle read 0.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from ._device import DeviceMixin
from ._process import ProcessInfo, enrich_full

_MIB = 1024 * 1024
_ACCEL_TYPE = re.compile(r"accelerator type:\s*([\w-]+)", re.IGNORECASE)


class TPUChipStat(DeviceMixin):
    """Stats for a single Cloud TPU chip."""

    vendor = "tpu"
    _index_attr = "device_id"

    def __init__(
        self,
        device_id: int,
        name: str,
        mem_used_mb: int,
        mem_total_mb: int,
        util_pct: Optional[int] = None,
        hbm_bw_util_pct: Optional[float] = None,
        device_path: Optional[str] = None,
        processes: Optional[List[ProcessInfo]] = None,
    ) -> None:
        self.device_id = device_id
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb
        #: TensorCore duty cycle (%), the closest thing to GPU utilization.
        self.util_pct = util_pct
        self.hbm_bw_util_pct = hbm_bw_util_pct
        self.device_path = device_path
        self.processes: List[ProcessInfo] = processes or []

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "device_path": self.device_path,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
            "mem_free_mb": self.mem_free_mb,
            "util_pct": self.util_pct,
            "hbm_bw_util_pct": self.hbm_bw_util_pct,
            "processes": [p.to_dict() for p in self.processes],
        }

    def __repr__(self) -> str:
        return (
            f"TPUChipStat(index={self.device_id}, name={self.name!r}, "
            f"mem_used={self.mem_used_mb}MB, mem_total={self.mem_total_mb}MB)"
        )


class TPUChipStatCollection:
    """Collects stats for all Cloud TPU chips via ``tpu-info``."""

    def __init__(self, chips: List[TPUChipStat]) -> None:
        self.chips = chips

    def to_dict(self) -> dict:
        return {"chips": [c.to_dict() for c in self.chips]}

    def __len__(self) -> int:
        return len(self.chips)

    def __iter__(self):
        return iter(self.chips)

    def __getitem__(self, index):
        return self.chips[index]

    def __repr__(self) -> str:
        return f"TPUChipStatCollection({self.chips!r})"

    @staticmethod
    def _run(args: List[str], timeout: int = 20) -> Optional[str]:
        try:
            r = subprocess.run(["tpu-info"] + args, capture_output=True, text=True, timeout=timeout)
            return r.stdout if r.returncode == 0 and r.stdout.strip() else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    @classmethod
    def new_query(cls) -> "TPUChipStatCollection":
        """
        Query ``tpu-info`` and return a TPUChipStatCollection.
        Returns an empty collection if ``tpu-info`` is not installed or fails.
        """
        if shutil.which("tpu-info") is None:
            return cls([])

        # The two invocations are independent; run them concurrently.
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_dash = pool.submit(cls._run, ["--format", "json", "dashboard"])
            f_ver = pool.submit(cls._run, ["--version"], 10)
        dash = f_dash.result()
        if dash is None:
            return cls([])
        return cls(_parse_dashboard(dash, _accelerator_type(f_ver.result())))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _accelerator_type(version_output: Optional[str]) -> Optional[str]:
    if not version_output:
        return None
    m = _ACCEL_TYPE.search(version_output)
    return m.group(1) if m else None


def _parse_dashboard(output: str, accel_type: Optional[str] = None) -> List[TPUChipStat]:
    # tpu-info may print warnings before the JSON object; find the first brace.
    start = output.find("{")
    if start < 0:
        return []
    try:
        data = json.loads(output[start:])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []

    name = f"TPU {accel_type}" if accel_type else "Cloud TPU"
    chips: List[TPUChipStat] = []
    for entry in data.get("devices") or []:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("device_id", len(chips)))
        except (TypeError, ValueError):
            continue

        used = entry.get("memory_used_bytes") or 0
        total = entry.get("memory_total_bytes") or 0
        duty = entry.get("duty_cycle_pct")
        bw = entry.get("hbm_bandwidth_util_pct")

        processes: List[ProcessInfo] = []
        proc = entry.get("process")
        if isinstance(proc, dict) and proc.get("pid"):
            try:
                pid = int(proc["pid"])
                processes.append(enrich_full(pid, str(proc.get("name") or "?"), gpu_mem_mb=None))
            except (TypeError, ValueError):
                pass

        chips.append(TPUChipStat(
            device_id=idx,
            name=name,
            mem_used_mb=int(used) // _MIB if isinstance(used, (int, float)) else 0,
            mem_total_mb=int(total) // _MIB if isinstance(total, (int, float)) else 0,
            util_pct=int(round(duty)) if isinstance(duty, (int, float)) else None,
            hbm_bw_util_pct=float(bw) if isinstance(bw, (int, float)) else None,
            device_path=str(entry["device_path"]) if entry.get("device_path") else None,
            processes=processes,
        ))
    return chips


if __name__ == "__main__":
    collection = TPUChipStatCollection.new_query()
    print(json.dumps(collection.to_dict(), indent=2))
