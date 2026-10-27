"""
rebellions.py — Rebellions ATOM / REBEL NPUs via ``rbln-stat --json``.

``rbln-stat --json`` (``rbln-smi`` on older SDKs) prints::

    {
      "KMD_version": "3.0.0",
      "devices": [
        {"npu": 0, "name": "RBLN-CA22", "uuid": "...", "device": "rbln0", "status": "normal",
         "fw_ver": "3.0.0", "pci": {"bus_id": "0000:03:00.0", "numa_node": "0", ...},
         "temperature": "31C", "card_power": "17521800uW", "pstate": "P14",
         "memory": {"used": "0", "total": "16877879296"}, "util": "0.0", "location": 5}
      ],
      "contexts": [
        {"ctx_id": "...", "npu": 0, "process": "worker", "pid": "42", "memalloc": "1MiB", "util_info": "12.5"}
      ]
    }

Numbers arrive as strings with units (``"31C"``, ``"17521800uW"``, bytes as
decimal strings), and ``pid`` or ``npu`` may be strings or integers
depending on the SDK version; the parser accepts both. On ATOM Max boards
``rbln-stat`` lists one entry per die, so a card appears several times
with the same ``card_power``.

The field names follow the ``rbln-stat`` JSON as consumed by the all-smi
project; the backend has not been run on hardware.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Any, List, Optional

from ._device import DeviceMixin
from ._process import ProcessInfo, enrich_full

_MIB = 1024 * 1024
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


class RebellionsNPUStat(DeviceMixin):
    """Stats for a single Rebellions NPU (one die on multi-die boards)."""

    vendor = "rebellions"
    _index_attr = "npu_index"

    def __init__(
        self,
        npu_index: int,
        name: str,
        mem_used_mb: int,
        mem_total_mb: int,
        util_pct: Optional[int] = None,
        temp_c: Optional[int] = None,
        power_w: Optional[float] = None,
        pstate: Optional[int] = None,
        status: Optional[str] = None,
        uuid: Optional[str] = None,
        device_node: Optional[str] = None,
        bus_id: Optional[str] = None,
        fw_version: Optional[str] = None,
        location: Optional[int] = None,
        processes: Optional[List[ProcessInfo]] = None,
    ) -> None:
        self.npu_index = npu_index
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb
        self.util_pct = util_pct
        self.temp_c = temp_c
        self.power_w = power_w
        self.pstate = pstate
        #: ``"normal"`` when healthy.
        self.status = status
        self.uuid = uuid
        self.device_node = device_node
        self.bus_id = bus_id
        self.fw_version = fw_version
        self.location = location
        self.processes: List[ProcessInfo] = processes or []

    @property
    def healthy(self) -> Optional[bool]:
        if self.status is None:
            return None
        return self.status.lower() == "normal"

    def to_dict(self) -> dict:
        return {
            "npu_index": self.npu_index,
            "name": self.name,
            "uuid": self.uuid,
            "device_node": self.device_node,
            "bus_id": self.bus_id,
            "fw_version": self.fw_version,
            "location": self.location,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
            "mem_free_mb": self.mem_free_mb,
            "util_pct": self.util_pct,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "pstate": self.pstate,
            "status": self.status,
            "healthy": self.healthy,
            "processes": [p.to_dict() for p in self.processes],
        }

    def __repr__(self) -> str:
        return (
            f"RebellionsNPUStat(index={self.npu_index}, name={self.name!r}, "
            f"mem_used={self.mem_used_mb}MB, mem_total={self.mem_total_mb}MB)"
        )


class RebellionsNPUStatCollection:
    """Collects stats for all Rebellions NPUs via ``rbln-stat``."""

    def __init__(self, npus: List[RebellionsNPUStat]) -> None:
        self.npus = npus

    def to_dict(self) -> dict:
        return {"npus": [n.to_dict() for n in self.npus]}

    def __len__(self) -> int:
        return len(self.npus)

    def __iter__(self):
        return iter(self.npus)

    def __getitem__(self, index):
        return self.npus[index]

    def __repr__(self) -> str:
        return f"RebellionsNPUStatCollection({self.npus!r})"

    @classmethod
    def new_query(cls) -> "RebellionsNPUStatCollection":
        binary = shutil.which("rbln-stat") or shutil.which("rbln-smi")
        if binary is None:
            return cls([])
        try:
            result = subprocess.run([binary, "--json"], capture_output=True, text=True, timeout=10)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return cls([])
        if result.returncode != 0 or not result.stdout.strip():
            return cls([])
        return cls(_parse_rbln_stat(result.stdout))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _num(value: Any) -> Optional[float]:
    """``"31C"`` -> 31.0, ``"17521800uW"`` -> 17521800.0, ``5`` -> 5.0."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = _NUM.search(value)
        if m:
            return float(m.group(0))
    return None


def _int(value: Any) -> Optional[int]:
    v = _num(value)
    return int(v) if v is not None else None


def _bytes(value: Any) -> Optional[int]:
    """Bytes from ``"16877879296"`` or ``"1MiB"`` / ``"512KiB"`` / ``"2GiB"``."""
    v = _num(value)
    if v is None:
        return None
    text = str(value).lower()
    for suffix, mult in (("gib", 1024 ** 3), ("gb", 1024 ** 3), ("mib", _MIB), ("mb", _MIB), ("kib", 1024), ("kb", 1024)):
        if text.endswith(suffix):
            return int(v * mult)
    return int(v)


def _parse_rbln_stat(output: str) -> List[RebellionsNPUStat]:
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []

    npus: List[RebellionsNPUStat] = []
    for entry in data.get("devices") or []:
        if not isinstance(entry, dict):
            continue
        idx = _int(entry.get("npu"))
        if idx is None:
            continue
        mem = entry.get("memory") if isinstance(entry.get("memory"), dict) else {}
        pci = entry.get("pci") if isinstance(entry.get("pci"), dict) else {}
        used = _bytes(mem.get("used")) or 0
        total = _bytes(mem.get("total")) or 0
        power_uw = _num(entry.get("card_power"))
        util = _num(entry.get("util"))
        npus.append(RebellionsNPUStat(
            npu_index=idx,
            name=str(entry.get("name") or "Rebellions NPU"),
            mem_used_mb=used // _MIB,
            mem_total_mb=total // _MIB,
            util_pct=int(round(util)) if util is not None else None,
            temp_c=_int(entry.get("temperature")),
            power_w=round(power_uw / 1e6, 1) if power_uw is not None else None,
            pstate=_int(entry.get("pstate")),
            status=str(entry["status"]) if entry.get("status") else None,
            uuid=str(entry["uuid"]) if entry.get("uuid") else None,
            device_node=str(entry["device"]) if entry.get("device") else None,
            bus_id=str(pci["bus_id"]) if pci.get("bus_id") else None,
            fw_version=str(entry["fw_ver"]) if entry.get("fw_ver") else None,
            location=_int(entry.get("location")),
        ))

    by_idx = {n.npu_index: n for n in npus}
    for ctx in data.get("contexts") or []:
        if not isinstance(ctx, dict):
            continue
        idx = _int(ctx.get("npu"))
        pid = _int(ctx.get("pid"))
        if idx is None or pid is None or idx not in by_idx:
            continue
        mem_b = _bytes(ctx.get("memalloc") if ctx.get("memalloc") is not None else ctx.get("memory"))
        util = _num(ctx.get("util_info"))
        name = str(ctx.get("process") or ctx.get("cmd") or "?")
        by_idx[idx].processes.append(enrich_full(
            pid, name, gpu_mem_mb=mem_b // _MIB if mem_b is not None else None,
            sm_pct=int(round(util)) if util is not None else None))
    return npus


if __name__ == "__main__":
    print(json.dumps(RebellionsNPUStatCollection.new_query().to_dict(), indent=2))
