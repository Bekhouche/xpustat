"""
neuron.py — AWS Neuron devices (Inferentia / Trainium) via ``neuron-ls``.

``neuron-ls --json-output`` prints one object per Neuron device::

    [
      {
        "neuron_device": 0,
        "bdf": "cc:00.0",
        "nc_count": 4,
        "memory_size": 103079215104,
        "neuroncore_ids": [0, 1, 2, 3],
        "neuron_processes": [
          {"pid": 113985, "command": "neuron-bench exec ...", "neuron_runtime_version": "2.0.0"}
        ],
        ...
      }
    ]

``neuron-ls`` reports inventory and the processes attached to each device,
but not live memory usage; that only comes from the streaming
``neuron-monitor`` daemon. ``mem_used_mb`` is therefore ``0`` and
``util_pct`` is ``None`` for this backend.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import List, Optional

from ._device import DeviceMixin
from ._process import ProcessInfo, enrich_full

_MIB = 1024 * 1024


class NeuronDeviceStat(DeviceMixin):
    """Stats for a single AWS Neuron device."""

    vendor = "neuron"
    _index_attr = "neuron_device"

    def __init__(
        self,
        neuron_device: int,
        name: str,
        mem_used_mb: int,
        mem_total_mb: int,
        nc_count: Optional[int] = None,
        bdf: Optional[str] = None,
        processes: Optional[List[ProcessInfo]] = None,
        neuroncore_ids: Optional[List[int]] = None,
    ) -> None:
        self.neuron_device = neuron_device
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb
        #: Number of NeuronCores on the device.
        self.nc_count = nc_count
        self.bdf = bdf
        self.processes: List[ProcessInfo] = processes or []
        #: Global NeuronCore ids on this device; NEURON_RT_VISIBLE_CORES lists these.
        self.neuroncore_ids: List[int] = list(neuroncore_ids or [])

    @property
    def util_pct(self) -> Optional[int]:
        return None

    @property
    def _visible_ids(self) -> List[int]:
        return self.neuroncore_ids

    def to_dict(self) -> dict:
        return {
            "neuron_device": self.neuron_device,
            "name": self.name,
            "nc_count": self.nc_count,
            "neuroncore_ids": self.neuroncore_ids,
            "bdf": self.bdf,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
            "mem_free_mb": self.mem_free_mb,
            "util_pct": self.util_pct,
            "processes": [p.to_dict() for p in self.processes],
        }

    def __repr__(self) -> str:
        return (
            f"NeuronDeviceStat(index={self.neuron_device}, name={self.name!r}, "
            f"cores={self.nc_count}, mem_total={self.mem_total_mb}MB)"
        )


class NeuronDeviceStatCollection:
    """Collects stats for all AWS Neuron devices via ``neuron-ls``."""

    def __init__(self, devices: List[NeuronDeviceStat]) -> None:
        self.devices = devices

    def to_dict(self) -> dict:
        return {"devices": [d.to_dict() for d in self.devices]}

    def __len__(self) -> int:
        return len(self.devices)

    def __iter__(self):
        return iter(self.devices)

    def __getitem__(self, index):
        return self.devices[index]

    def __repr__(self) -> str:
        return f"NeuronDeviceStatCollection({self.devices!r})"

    @classmethod
    def new_query(cls) -> "NeuronDeviceStatCollection":
        """
        Query ``neuron-ls`` and return a NeuronDeviceStatCollection.
        Returns an empty collection if ``neuron-ls`` is not installed or fails.
        """
        binary = shutil.which("neuron-ls") or shutil.which("/opt/aws/neuron/bin/neuron-ls")
        if binary is None:
            return cls([])

        try:
            result = subprocess.run(
                [binary, "--json-output"],
                capture_output=True, text=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return cls([])

        if result.returncode != 0 or not result.stdout.strip():
            return cls([])

        return cls(_parse_neuron_ls(result.stdout))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_neuron_ls(output: str) -> List[NeuronDeviceStat]:
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    devices: List[NeuronDeviceStat] = []
    for entry in data:
        if not isinstance(entry, dict) or "neuron_device" not in entry:
            continue
        try:
            idx = int(entry["neuron_device"])
        except (TypeError, ValueError):
            continue

        nc = entry.get("nc_count")
        nc_count = int(nc) if isinstance(nc, int) else None
        core_ids = [int(c) for c in (entry.get("neuroncore_ids") or []) if isinstance(c, int)]
        mem = entry.get("memory_size")
        mem_total_mb = int(mem) // _MIB if isinstance(mem, int) else 0

        processes: List[ProcessInfo] = []
        for p in entry.get("neuron_processes") or []:
            if not isinstance(p, dict) or p.get("pid") is None:
                continue
            try:
                pid = int(p["pid"])
            except (TypeError, ValueError):
                continue
            cmd = str(p.get("command") or "")
            fallback = os.path.basename(cmd.split()[0]) if cmd.strip() else "?"
            processes.append(enrich_full(pid, fallback, gpu_mem_mb=None))

        cores = f" ({nc_count} NeuronCores)" if nc_count else ""
        devices.append(NeuronDeviceStat(
            neuron_device=idx,
            name=f"AWS Neuron{cores}",
            mem_used_mb=0,
            mem_total_mb=mem_total_mb,
            nc_count=nc_count,
            bdf=str(entry["bdf"]) if entry.get("bdf") else None,
            processes=processes,
            neuroncore_ids=core_ids,
        ))
    return devices


if __name__ == "__main__":
    collection = NeuronDeviceStatCollection.new_query()
    print(json.dumps(collection.to_dict(), indent=2))
