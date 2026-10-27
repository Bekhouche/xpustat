"""
jetson.py — NVIDIA Jetson modules (Nano, Xavier, Orin) via ``tegrastats``.

Jetson boards run the integrated GPU through the Tegra driver and ship no
``nvidia-smi``; the stock reporting tool is ``tegrastats``, which prints one
line per interval::

    RAM 2567/7772MB (lfb 1x4MB) SWAP 0/3886MB (cached 0MB) CPU [1%@729,...]
    GR3D_FREQ 12%@624 cpu@44.5C gpu@44.3C tj@44.5C VDD_GPU_SOC 1234mW/1234mW

The backend runs ``tegrastats`` once, reads the first line and stops it.
Memory is the unified RAM pool shared with the CPU, ``GR3D_FREQ`` is the GPU
load, ``gpu@..C`` the temperature and the ``VDD_GPU_SOC`` /
``VDD_CPU_GPU_CV`` / ``POM_5V_GPU`` rail the power, depending on the module.
The module name comes from ``/proc/device-tree/model``.

The backend yields nothing when ``nvidia-smi`` is present, so a host is never
listed twice; on such a host the ``nvidia`` backend reports the GPU.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import List, Optional

from ._device import DeviceMixin
from ._process import ProcessInfo

_RAM = re.compile(r"\bRAM\s+(\d+)/(\d+)MB")
_GR3D = re.compile(r"\bGR3D_FREQ\s+(\d+)%")
_GPU_TEMP = re.compile(r"\b(?:gpu|GPU)@([\d.]+)C")
# Power rails in order of preference; value is instantaneous mW ("cur/avg").
_POWER_RAILS = ("VDD_GPU_SOC", "VDD_CPU_GPU_CV", "POM_5V_GPU", "GPU")
_POWER = {rail: re.compile(rf"\b{rail}\s+(\d+)(?:mW)?/(\d+)") for rail in _POWER_RAILS}

_MODEL_PATH = "/proc/device-tree/model"
_TEGRA_RELEASE = "/etc/nv_tegra_release"


class JetsonGPUStat(DeviceMixin):
    """Stats for the integrated GPU of an NVIDIA Jetson module."""

    vendor = "jetson"
    _index_attr = "gpu_index"

    def __init__(
        self,
        gpu_index: int,
        name: str,
        mem_used_mb: int,
        mem_total_mb: int,
        util_pct: Optional[int] = None,
        temp_c: Optional[int] = None,
        power_w: Optional[float] = None,
        power_rail: Optional[str] = None,
        processes: Optional[List[ProcessInfo]] = None,
    ) -> None:
        self.gpu_index = gpu_index
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb
        self.util_pct = util_pct
        self.temp_c = temp_c
        self.power_w = power_w
        #: Which tegrastats rail ``power_w`` came from (``VDD_GPU_SOC`` on Orin
        #: AGX, ``VDD_CPU_GPU_CV`` on Orin Nano where GPU and CPU share a rail).
        self.power_rail = power_rail
        #: tegrastats reports no per-process GPU memory.
        self.processes: List[ProcessInfo] = processes or []

    def to_dict(self) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "name": self.name,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
            "mem_free_mb": self.mem_free_mb,
            "unified_memory": True,
            "util_pct": self.util_pct,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "power_rail": self.power_rail,
            "processes": [p.to_dict() for p in self.processes],
        }

    def __repr__(self) -> str:
        return (
            f"JetsonGPUStat(index={self.gpu_index}, name={self.name!r}, "
            f"mem_used={self.mem_used_mb}MB, mem_total={self.mem_total_mb}MB)"
        )


class JetsonGPUStatCollection:
    """Collects stats for the Jetson integrated GPU via ``tegrastats``."""

    def __init__(self, gpus: List[JetsonGPUStat]) -> None:
        self.gpus = gpus

    def to_dict(self) -> dict:
        return {"gpus": [g.to_dict() for g in self.gpus]}

    def __len__(self) -> int:
        return len(self.gpus)

    def __iter__(self):
        return iter(self.gpus)

    def __getitem__(self, index):
        return self.gpus[index]

    def __repr__(self) -> str:
        return f"JetsonGPUStatCollection({self.gpus!r})"

    @staticmethod
    def is_jetson() -> bool:
        return sys.platform.startswith("linux") and (
            os.path.exists(_TEGRA_RELEASE) or "jetson" in _read_model().lower())

    @classmethod
    def new_query(cls) -> "JetsonGPUStatCollection":
        """
        Run ``tegrastats`` once and return a JetsonGPUStatCollection.
        Returns an empty collection off Jetson hosts, when ``tegrastats`` is
        missing, or when ``nvidia-smi`` exists (the ``nvidia`` backend then
        owns the GPU).
        """
        if shutil.which("nvidia-smi") is not None:
            return cls([])
        binary = shutil.which("tegrastats") or (
            "/usr/bin/tegrastats" if os.path.exists("/usr/bin/tegrastats") else None)
        if binary is None or not cls.is_jetson():
            return cls([])

        line = _tegrastats_line(binary)
        if not line:
            return cls([])
        stat = _parse_tegrastats(line, _read_model() or "NVIDIA Jetson")
        return cls([stat] if stat else [])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_model() -> str:
    try:
        with open(_MODEL_PATH, "rb") as f:
            return f.read().rstrip(b"\x00").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def _tegrastats_line(binary: str, interval_ms: int = 200, timeout: float = 3.0) -> str:
    """Start tegrastats, take its first output line, stop it."""
    try:
        proc = subprocess.Popen(
            [binary, "--interval", str(interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
    except (FileNotFoundError, OSError):
        return ""
    try:
        import threading
        result: List[str] = []

        def _read() -> None:
            try:
                result.append(proc.stdout.readline())  # type: ignore[union-attr]
            except Exception:
                pass

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        t.join(timeout)
        return result[0].strip() if result else ""
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _parse_tegrastats(line: str, name: str = "NVIDIA Jetson") -> Optional[JetsonGPUStat]:
    """Parse one tegrastats line. Returns ``None`` if it carries no RAM field."""
    m = _RAM.search(line)
    if not m:
        return None
    used, total = int(m.group(1)), int(m.group(2))

    util_m = _GR3D.search(line)
    util = int(util_m.group(1)) if util_m else None

    temp_m = _GPU_TEMP.search(line)
    temp = int(round(float(temp_m.group(1)))) if temp_m else None

    power: Optional[float] = None
    rail: Optional[str] = None
    for r in _POWER_RAILS:
        pm = _POWER[r].search(line)
        if pm:
            power = int(pm.group(1)) / 1000.0
            rail = r
            break

    return JetsonGPUStat(
        gpu_index=0, name=name, mem_used_mb=used, mem_total_mb=total,
        util_pct=util, temp_c=temp, power_w=power, power_rail=rail,
    )


if __name__ == "__main__":
    import json
    print(json.dumps(JetsonGPUStatCollection.new_query().to_dict(), indent=2))
