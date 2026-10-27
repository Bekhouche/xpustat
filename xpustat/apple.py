"""
apple.py — Apple Silicon (and Intel Mac) GPU stats via IOKit.

macOS ships no vendor CLI comparable to ``nvidia-smi``.  ``powermetrics``
exposes GPU power and residency but needs root, so this backend reads the
``IOAccelerator`` registry entries instead, which every user can query.

Query priority:
  1. IOKit via ``ctypes`` — in-process, ~2 ms, no subprocess.
  2. ``ioreg -r -c IOAccelerator -a -d 1`` — subprocess fallback, ~25 ms.

Each accelerator entry carries a ``PerformanceStatistics`` dictionary with
``Device Utilization %`` and ``In use system memory`` (Apple Silicon) or
``vramUsedBytes`` / ``vramFreeBytes`` (discrete GPUs on Intel Macs).

Apple Silicon uses unified memory: ``mem_total_mb`` is the machine's total
RAM and ``mem_used_mb`` is the slice currently wired by the GPU driver.
GPU temperature and power come from the SMC and IOReport counters read
in-process (see ``_apple_sensors``), without root. Per-process attribution
is root-only on macOS and stays empty.
"""

from __future__ import annotations

import plistlib
import shutil
import subprocess
import sys
from typing import Any, List, Optional

from ._process import ProcessInfo
from ._device import DeviceMixin

_MIB = 1024 * 1024


class AppleGPUStat(DeviceMixin):
    """Stats for a single Apple GPU (Apple Silicon or discrete GPU on a Mac)."""

    vendor = "apple"
    _index_attr = "gpu_index"

    def __init__(
        self,
        gpu_index: int,
        name: str,
        mem_used_mb: int,
        mem_total_mb: int,
        util_pct: Optional[int] = None,
        gpu_cores: Optional[int] = None,
        unified_memory: bool = True,
        processes: Optional[List[ProcessInfo]] = None,
        temp_c: Optional[int] = None,
        power_w: Optional[float] = None,
        ane_power_w: Optional[float] = None,
    ) -> None:
        self.gpu_index = gpu_index
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb
        self.util_pct = util_pct
        #: Number of GPU cores, e.g. 10 for a base M-series chip.
        self.gpu_cores = gpu_cores
        #: ``True`` when memory is shared with the CPU (Apple Silicon). In that
        #: case ``mem_total_mb`` is the total system RAM.
        self.unified_memory = unified_memory
        #: macOS does not expose per-process GPU memory without root.
        self.processes: List[ProcessInfo] = processes or []
        #: Mean of the SMC GPU sensors (``Tg..`` keys); None when unreadable.
        self.temp_c = temp_c
        #: GPU power from the IOReport ``Energy Model`` counters.
        self.power_w = power_w
        #: Apple Neural Engine power, same source.
        self.ane_power_w = ane_power_w

    def to_dict(self) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "name": self.name,
            "gpu_cores": self.gpu_cores,
            "unified_memory": self.unified_memory,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
            "mem_free_mb": self.mem_free_mb,
            "util_pct": self.util_pct,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "ane_power_w": self.ane_power_w,
            "processes": [p.to_dict() for p in self.processes],
        }

    def __repr__(self) -> str:
        return (
            f"AppleGPUStat(index={self.gpu_index}, name={self.name!r}, "
            f"cores={self.gpu_cores}, mem_used={self.mem_used_mb}MB, "
            f"mem_total={self.mem_total_mb}MB)"
        )


class AppleGPUStatCollection:
    """
    Collects stats for all GPUs on a Mac.

    Query priority:
      1. IOKit via ``ctypes`` — in-process, no subprocess (fast).
      2. ``ioreg`` — subprocess fallback.
    """

    #: Which backend was used: ``"iokit"``, ``"subprocess"``, or ``"none"``.
    backend: str = "none"

    def __init__(self, gpus: List[AppleGPUStat], backend: str = "none") -> None:
        self.gpus = gpus
        self.backend = backend

    def to_dict(self) -> dict:
        return {"gpus": [gpu.to_dict() for gpu in self.gpus]}

    def __len__(self) -> int:
        return len(self.gpus)

    def __iter__(self):
        return iter(self.gpus)

    def __getitem__(self, index):
        return self.gpus[index]

    def __repr__(self) -> str:
        return f"AppleGPUStatCollection({self.gpus!r})"

    @classmethod
    def new_query(cls) -> "AppleGPUStatCollection":
        """
        Query IOKit (or ``ioreg``) and return an AppleGPUStatCollection.
        Returns an empty collection on non-macOS hosts or if both paths fail.
        """
        if sys.platform != "darwin":
            return cls([])

        entries = _iokit_accelerator_entries()
        if entries is not None:
            col = cls(_parse_entries(entries, _system_memory_bytes(), _system_memory_available_bytes()), backend="iokit")
        else:
            col = cls._query_subprocess()
        _attach_sensors(col)
        return col

    @classmethod
    def _query_subprocess(cls) -> "AppleGPUStatCollection":
        if shutil.which("ioreg") is None:
            return cls([])

        try:
            result = subprocess.run(
                ["ioreg", "-r", "-c", "IOAccelerator", "-a", "-d", "1"],
                capture_output=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return cls([])

        if result.returncode != 0 or not result.stdout.strip():
            return cls([])

        return cls(_parse_ioreg(result.stdout, _system_memory_bytes(), _system_memory_available_bytes()), backend="subprocess")


def _attach_sensors(col: "AppleGPUStatCollection") -> None:
    """Fill temperature and power on unified-memory (Apple Silicon) GPUs from the in-process sensors."""
    if not col.gpus or not any(g.unified_memory for g in col.gpus):
        return
    try:
        from . import _apple_sensors
        watts = _apple_sensors.energy_watts()
        temps = _apple_sensors.temperatures()
    except Exception:
        return
    for g in col.gpus:
        if not g.unified_memory:
            continue
        if temps.get("gpu") is not None:
            g.temp_c = int(round(temps["gpu"]))
        if "gpu" in watts:
            g.power_w = watts["gpu"]
        if "ane" in watts:
            g.ane_power_w = watts["ane"]


# ---------------------------------------------------------------------------
# Fast path — IOKit via ctypes
# ---------------------------------------------------------------------------

_iokit: Any = None
_cf: Any = None
_kCFPropertyListBinaryFormat_v1_0 = 200


def _load_iokit() -> bool:
    """Bind the handful of IOKit / CoreFoundation symbols we need. Cached."""
    global _iokit, _cf
    if _iokit is not None:
        return True
    try:
        import ctypes
        import ctypes.util

        iokit = ctypes.CDLL(ctypes.util.find_library("IOKit"))
        cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))

        iokit.IOServiceMatching.restype = ctypes.c_void_p
        iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
        iokit.IOServiceGetMatchingServices.restype = ctypes.c_int
        iokit.IOServiceGetMatchingServices.argtypes = [
            ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        iokit.IOIteratorNext.restype = ctypes.c_uint32
        iokit.IOIteratorNext.argtypes = [ctypes.c_uint32]
        iokit.IORegistryEntryCreateCFProperties.restype = ctypes.c_int
        iokit.IORegistryEntryCreateCFProperties.argtypes = [
            ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint32]
        iokit.IOObjectRelease.restype = ctypes.c_int
        iokit.IOObjectRelease.argtypes = [ctypes.c_uint32]

        cf.CFPropertyListCreateData.restype = ctypes.c_void_p
        cf.CFPropertyListCreateData.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p]
        cf.CFDataGetLength.restype = ctypes.c_long
        cf.CFDataGetLength.argtypes = [ctypes.c_void_p]
        cf.CFDataGetBytePtr.restype = ctypes.c_void_p
        cf.CFDataGetBytePtr.argtypes = [ctypes.c_void_p]
        cf.CFRelease.restype = None
        cf.CFRelease.argtypes = [ctypes.c_void_p]

        _iokit, _cf = iokit, cf
        return True
    except (OSError, AttributeError, TypeError):
        return False


def _iokit_accelerator_entries() -> Optional[List[dict]]:
    """
    Return the property dict of every ``IOAccelerator`` service, read
    in-process through IOKit. Returns ``None`` if IOKit is unavailable so the
    caller can fall back to ``ioreg``.
    """
    if not _load_iokit():
        return None
    import ctypes

    iokit, cf = _iokit, _cf
    iterator = ctypes.c_uint32(0)
    # The matching dict is consumed (released) by IOServiceGetMatchingServices.
    matching = iokit.IOServiceMatching(b"IOAccelerator")
    if not matching or iokit.IOServiceGetMatchingServices(0, matching, ctypes.byref(iterator)) != 0:
        return None

    entries: List[dict] = []
    try:
        while True:
            service = iokit.IOIteratorNext(iterator.value)
            if not service:
                break
            try:
                props = ctypes.c_void_p()
                if iokit.IORegistryEntryCreateCFProperties(
                        service, ctypes.byref(props), None, 0) != 0 or not props.value:
                    continue
                try:
                    data = cf.CFPropertyListCreateData(
                        None, props, _kCFPropertyListBinaryFormat_v1_0, 0, None)
                    if not data:
                        continue
                    try:
                        raw = ctypes.string_at(cf.CFDataGetBytePtr(data), cf.CFDataGetLength(data))
                    finally:
                        cf.CFRelease(data)
                finally:
                    cf.CFRelease(props)
                try:
                    entry = plistlib.loads(raw)
                except Exception:
                    continue
                if isinstance(entry, dict):
                    entries.append(entry)
            finally:
                iokit.IOObjectRelease(service)
    finally:
        iokit.IOObjectRelease(iterator.value)
    return entries


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _system_memory_bytes() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        return 0


def _system_memory_available_bytes() -> Optional[int]:
    """Memory the OS can still hand to the GPU on a unified-memory machine."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _to_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.rstrip(b"\x00").decode("utf-8", "replace")
    return str(value)


def _parse_ioreg(plist_bytes: bytes, system_memory_bytes: int = 0,
                 system_memory_available_bytes: Optional[int] = None) -> List[AppleGPUStat]:
    """Parse the plist produced by ``ioreg -r -c IOAccelerator -a -d 1``."""
    try:
        entries = plistlib.loads(plist_bytes)
    except Exception:
        return []
    if not isinstance(entries, list):
        return []
    return _parse_entries(entries, system_memory_bytes, system_memory_available_bytes)


def _parse_entries(entries: List[Any], system_memory_bytes: int = 0,
                   system_memory_available_bytes: Optional[int] = None) -> List[AppleGPUStat]:
    """
    Build stat objects from ``IOAccelerator`` property dicts.

    Args:
        entries:             One property dict per accelerator service.
        system_memory_bytes: Total RAM, used as ``mem_total`` for unified-memory
                             GPUs (Apple Silicon).
    """
    gpus: List[AppleGPUStat] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        stats = entry.get("PerformanceStatistics")
        if not isinstance(stats, dict):
            continue

        name = _to_str(entry.get("model") or entry.get("IOClass") or "Apple GPU")
        cores = entry.get("gpu-core-count")
        gpu_cores = int(cores) if isinstance(cores, int) else None

        util = stats.get("Device Utilization %")
        util_pct = int(util) if isinstance(util, (int, float)) else None

        # Discrete GPU (Intel Mac with AMD/NVIDIA card): dedicated VRAM.
        vram_used = stats.get("vramUsedBytes")
        vram_free = stats.get("vramFreeBytes")
        if isinstance(vram_used, int) and isinstance(vram_free, int):
            mem_used_mb = vram_used // _MIB
            mem_total_mb = (vram_used + vram_free) // _MIB
            unified = False
        else:
            used = stats.get("In use system memory", 0)
            mem_used_mb = int(used) // _MIB if isinstance(used, int) else 0
            mem_total_mb = system_memory_bytes // _MIB
            unified = True

        stat = AppleGPUStat(
            gpu_index=len(gpus),
            name=name,
            mem_used_mb=mem_used_mb,
            mem_total_mb=mem_total_mb,
            util_pct=util_pct,
            gpu_cores=gpu_cores,
            unified_memory=unified,
        )
        if unified and system_memory_available_bytes is not None:
            stat._mem_free_mb = system_memory_available_bytes // _MIB
        gpus.append(stat)
    return gpus


if __name__ == "__main__":
    import json
    collection = AppleGPUStatCollection.new_query()
    print(f"backend: {collection.backend}")
    print(json.dumps(collection.to_dict(), indent=2))
