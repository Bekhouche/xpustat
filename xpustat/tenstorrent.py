"""
tenstorrent.py — Tenstorrent Wormhole / Blackhole boards via ``tt-smi``.

``tt-smi -s --snapshot_no_tty`` prints a JSON snapshot::

    {
      "time": "...",
      "host_info": {...},
      "device_info": [
        {
          "board_info": {"bus_id": "0000:01:00.0", "board_type": "n150 L", "board_id": "...", ...},
          "telemetry":  {"voltage": "0.80", "current": " 30.0", "power": " 24.0",
                         "board_power": " 60.0", "aiclk": "1000", "asic_temperature": " 45.2"},
          "firmwares":  {...},
          "limits":     {...}
        }
      ],
      "processes": [{"pid": 1234, "user": "salah", "device": 0, "cmdline": "python train.py"}]
    }

Telemetry values are strings. ``tt-smi`` exposes no DRAM usage, so
``mem_used_mb`` and ``mem_total_mb`` are ``0`` and the CLI shows the memory
column as N/A.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, List, Optional

from ._device import DeviceMixin
from ._process import ProcessInfo, enrich_full


class TenstorrentDeviceStat(DeviceMixin):
    """Stats for a single Tenstorrent device."""

    vendor = "tenstorrent"
    _index_attr = "tt_device"

    def __init__(
        self,
        tt_device: int,
        name: str,
        mem_used_mb: int = 0,
        mem_total_mb: int = 0,
        temp_c: Optional[int] = None,
        power_w: Optional[float] = None,
        board_power_w: Optional[float] = None,
        aiclk_mhz: Optional[int] = None,
        bus_id: Optional[str] = None,
        board_id: Optional[str] = None,
        processes: Optional[List[ProcessInfo]] = None,
    ) -> None:
        self.tt_device = tt_device
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb
        self.temp_c = temp_c
        self.power_w = power_w
        self.board_power_w = board_power_w
        self.aiclk_mhz = aiclk_mhz
        self.bus_id = bus_id
        self.board_id = board_id
        self.processes: List[ProcessInfo] = processes or []

    @property
    def util_pct(self) -> Optional[int]:
        return None

    def to_dict(self) -> dict:
        return {
            "tt_device": self.tt_device,
            "name": self.name,
            "bus_id": self.bus_id,
            "board_id": self.board_id,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
            "mem_free_mb": self.mem_free_mb,
            "util_pct": self.util_pct,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "board_power_w": self.board_power_w,
            "aiclk_mhz": self.aiclk_mhz,
            "processes": [p.to_dict() for p in self.processes],
        }

    def __repr__(self) -> str:
        return (
            f"TenstorrentDeviceStat(index={self.tt_device}, name={self.name!r}, "
            f"temp={self.temp_c}C, power={self.power_w}W)"
        )


class TenstorrentDeviceStatCollection:
    """Collects stats for all Tenstorrent devices via ``tt-smi``."""

    def __init__(self, devices: List[TenstorrentDeviceStat]) -> None:
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
        return f"TenstorrentDeviceStatCollection({self.devices!r})"

    @classmethod
    def new_query(cls) -> "TenstorrentDeviceStatCollection":
        """
        Query ``tt-smi`` and return a TenstorrentDeviceStatCollection.
        Returns an empty collection if ``tt-smi`` is not installed or fails.
        """
        if shutil.which("tt-smi") is None:
            return cls([])

        try:
            result = subprocess.run(
                ["tt-smi", "-s", "--snapshot_no_tty"],
                capture_output=True, text=True, timeout=20,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return cls([])

        if result.returncode != 0 or not result.stdout.strip():
            return cls([])

        return cls(_parse_snapshot(result.stdout))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _num(value: Any) -> Optional[float]:
    """tt-smi emits telemetry as padded strings such as ``" 45.2"``."""
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _is_tt_smi(cmdline: str) -> bool:
    """True when ``cmdline`` is a ``tt-smi`` invocation.

    ``tt-smi`` opens every device to take its snapshot and lists itself in
    ``processes``, so without this filter each device would show the
    ``xpustat`` query itself as a user. A ``tt-smi`` only watches telemetry,
    so dropping every one of them does not hide a workload.
    """
    return any(os.path.basename(tok) == "tt-smi" for tok in cmdline.split())


def _parse_snapshot(output: str) -> List[TenstorrentDeviceStat]:
    start = output.find("{")
    if start < 0:
        return []
    try:
        data = json.loads(output[start:])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []

    procs_by_dev: dict = {}
    for p in data.get("processes") or []:
        if not isinstance(p, dict) or p.get("pid") is None:
            continue
        try:
            pid = int(p["pid"]); dev = int(p.get("device", 0))
        except (TypeError, ValueError):
            continue
        cmdline = str(p.get("cmdline") or "")
        if _is_tt_smi(cmdline):
            continue
        fallback = os.path.basename(cmdline.split()[0]) if cmdline.strip() else "?"
        info = enrich_full(pid, fallback, gpu_mem_mb=None)
        if info.username == "?" and p.get("user"):
            info.username = str(p["user"])
        procs_by_dev.setdefault(dev, []).append(info)

    devices: List[TenstorrentDeviceStat] = []
    for i, entry in enumerate(data.get("device_info") or []):
        if not isinstance(entry, dict):
            continue
        board = entry.get("board_info") if isinstance(entry.get("board_info"), dict) else {}
        telem = entry.get("telemetry") if isinstance(entry.get("telemetry"), dict) else {}

        board_type = str(board.get("board_type") or "Tenstorrent")
        temp = _num(telem.get("asic_temperature"))
        power = _num(telem.get("power"))
        board_power = _num(telem.get("board_power"))
        aiclk = _num(telem.get("aiclk"))

        devices.append(TenstorrentDeviceStat(
            tt_device=i,
            name=f"Tenstorrent {board_type}",
            temp_c=int(round(temp)) if temp is not None else None,
            power_w=power,
            board_power_w=board_power,
            aiclk_mhz=int(aiclk) if aiclk is not None else None,
            bus_id=str(board["bus_id"]) if board.get("bus_id") else None,
            board_id=str(board["board_id"]) if board.get("board_id") else None,
            processes=procs_by_dev.get(i, []),
        ))
    return devices


if __name__ == "__main__":
    collection = TenstorrentDeviceStatCollection.new_query()
    print(json.dumps(collection.to_dict(), indent=2))
