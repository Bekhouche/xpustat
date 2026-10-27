"""
furiosa.py — FuriosaAI RNGD / Warboy NPUs via ``furiosa-smi``.

Three subcommands, all with JSON output (``--format json`` on RNGD SDKs,
``--output json`` on older Warboy SDKs; the backend tries both):

* ``furiosa-smi info``: one object per device with ``arch``, ``dev_name``
  (``npu0``), ``device_uuid``, ``device_sn``, ``firmware``, ``temperature``,
  ``power``, ``core_clock``, ``governor``, ``pci_bdf``.
* ``furiosa-smi status``: ``device``, ``liveness``, ``cores`` and
  ``memory.DRAM`` (``used_size``, ``total_size`` in bytes) plus
  ``pe_utilizations`` (``pe_core``, ``utilization``) per processing element.
* ``furiosa-smi ps``: ``dev_name`` (``npu0:[0, 7]``), ``pid``, ``cmdline``.

Values in ``info`` are strings with units (``"40°C"``, ``"38 W"``,
``"2000 MHz"``) and are parsed numerically. The field names follow the
``furiosa-smi`` JSON as consumed by the all-smi project; the backend has not
been run on hardware.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from ._device import DeviceMixin
from ._process import ProcessInfo, enrich_full

_MIB = 1024 * 1024
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_DEV = re.compile(r"npu(\d+)")


class FuriosaNPUStat(DeviceMixin):
    """Stats for a single FuriosaAI NPU."""

    vendor = "furiosa"
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
        clock_mhz: Optional[int] = None,
        governor: Optional[str] = None,
        arch: Optional[str] = None,
        uuid: Optional[str] = None,
        serial: Optional[str] = None,
        firmware: Optional[str] = None,
        bus_id: Optional[str] = None,
        liveness: Optional[str] = None,
        pe_util_pct: Optional[List[int]] = None,
        processes: Optional[List[ProcessInfo]] = None,
    ) -> None:
        self.npu_index = npu_index
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb
        #: Mean utilization over the processing elements.
        self.util_pct = util_pct
        self.temp_c = temp_c
        self.power_w = power_w
        self.clock_mhz = clock_mhz
        self.governor = governor
        self.arch = arch
        self.uuid = uuid
        self.serial = serial
        self.firmware = firmware
        self.bus_id = bus_id
        self.liveness = liveness
        #: Utilization per processing element, in ``pe_core`` order.
        self.pe_util_pct: List[int] = list(pe_util_pct or [])
        self.processes: List[ProcessInfo] = processes or []

    @property
    def healthy(self) -> Optional[bool]:
        if self.liveness is None:
            return None
        return self.liveness.lower() in ("alive", "ok", "normal", "true")

    def to_dict(self) -> dict:
        return {
            "npu_index": self.npu_index,
            "name": self.name,
            "arch": self.arch,
            "uuid": self.uuid,
            "serial": self.serial,
            "firmware": self.firmware,
            "bus_id": self.bus_id,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
            "mem_free_mb": self.mem_free_mb,
            "util_pct": self.util_pct,
            "pe_util_pct": self.pe_util_pct,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "clock_mhz": self.clock_mhz,
            "governor": self.governor,
            "liveness": self.liveness,
            "healthy": self.healthy,
            "processes": [p.to_dict() for p in self.processes],
        }

    def __repr__(self) -> str:
        return (
            f"FuriosaNPUStat(index={self.npu_index}, name={self.name!r}, "
            f"mem_used={self.mem_used_mb}MB, mem_total={self.mem_total_mb}MB)"
        )


class FuriosaNPUStatCollection:
    """Collects stats for all FuriosaAI NPUs via ``furiosa-smi``."""

    _json_flag: Optional[str] = None   # "--format" (RNGD) or "--output" (Warboy), learned once

    def __init__(self, npus: List[FuriosaNPUStat]) -> None:
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
        return f"FuriosaNPUStatCollection({self.npus!r})"

    @classmethod
    def _run(cls, subcommand: str) -> Optional[str]:
        flags = [cls._json_flag] if cls._json_flag else ["--format", "--output"]
        for flag in flags:
            try:
                r = subprocess.run(["furiosa-smi", subcommand, flag, "json"],
                                   capture_output=True, text=True, timeout=10)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return None
            if r.returncode == 0 and r.stdout.strip():
                cls._json_flag = flag
                return r.stdout
        return None

    @classmethod
    def new_query(cls) -> "FuriosaNPUStatCollection":
        if shutil.which("furiosa-smi") is None:
            return cls([])
        with ThreadPoolExecutor(max_workers=3) as pool:
            f_info = pool.submit(cls._run, "info")
            f_status = pool.submit(cls._run, "status")
            f_ps = pool.submit(cls._run, "ps")
        info = f_info.result()
        if info is None:
            return cls([])
        return cls(_parse_furiosa(info, f_status.result(), f_ps.result()))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _num(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = _NUM.search(value)
        if m:
            return float(m.group(0))
    return None


def _int(value: Any) -> Optional[int]:
    v = _num(value)
    return int(round(v)) if v is not None else None


def _dev_index(name: Any) -> Optional[int]:
    m = _DEV.search(str(name or ""))
    return int(m.group(1)) if m else None


def _load_list(text: Optional[str]) -> List[dict]:
    if not text:
        return []
    try:
        data = json.loads(text[text.find("[") if "[" in text else 0:]) if text.lstrip().startswith("[") else json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data.get("devices") or data.get("data") or [data]
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


def _parse_furiosa(info_text: str, status_text: Optional[str], ps_text: Optional[str]) -> List[FuriosaNPUStat]:
    status_by_idx: Dict[int, dict] = {}
    for st in _load_list(status_text):
        idx = _dev_index(st.get("device"))
        if idx is not None:
            status_by_idx[idx] = st

    npus: List[FuriosaNPUStat] = []
    for entry in _load_list(info_text):
        idx = _dev_index(entry.get("dev_name"))
        if idx is None:
            idx = _int(entry.get("index"))
        if idx is None:
            continue
        st = status_by_idx.get(idx, {})
        memory = st.get("memory") if isinstance(st.get("memory"), dict) else {}
        dram = memory.get("DRAM") or memory.get("dram") or {}
        used = int(dram.get("used_size") or 0) if isinstance(dram, dict) else 0
        total = int(dram.get("total_size") or 0) if isinstance(dram, dict) else 0
        pe = []
        for u in st.get("pe_utilizations") or []:
            if isinstance(u, dict):
                v = _num(u.get("utilization") if u.get("utilization") is not None else u.get("pe_utilization"))
                if v is not None:
                    pe.append(int(round(v)))
        arch = str(entry.get("arch") or "").strip() or None
        npus.append(FuriosaNPUStat(
            npu_index=idx,
            name=f"FuriosaAI {arch.upper()}" if arch else "FuriosaAI NPU",
            mem_used_mb=used // _MIB,
            mem_total_mb=total // _MIB,
            util_pct=int(round(sum(pe) / len(pe))) if pe else None,
            temp_c=_int(entry.get("temperature")),
            power_w=_num(entry.get("power")),
            clock_mhz=_int(entry.get("core_clock")),
            governor=str(entry["governor"]) if entry.get("governor") else None,
            arch=arch,
            uuid=str(entry["device_uuid"]) if entry.get("device_uuid") else None,
            serial=str(entry["device_sn"]) if entry.get("device_sn") else None,
            firmware=str(entry["firmware"]) if entry.get("firmware") else None,
            bus_id=str(entry["pci_bdf"]) if entry.get("pci_bdf") else None,
            liveness=str(st["liveness"]) if st.get("liveness") else None,
            pe_util_pct=pe,
        ))

    by_idx = {n.npu_index: n for n in npus}
    for p in _load_list(ps_text):
        idx = _dev_index(p.get("dev_name") if p.get("dev_name") is not None else p.get("npu"))
        pid = _int(p.get("pid"))
        if idx is None or pid is None or idx not in by_idx:
            continue
        cmd = str(p.get("cmdline") or p.get("cmd") or "")
        fallback = cmd.split()[0].rsplit("/", 1)[-1] if cmd.strip() else "?"
        by_idx[idx].processes.append(enrich_full(pid, fallback, gpu_mem_mb=None))
    return npus


if __name__ == "__main__":
    print(json.dumps(FuriosaNPUStatCollection.new_query().to_dict(), indent=2))
