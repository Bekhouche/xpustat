"""
cpu.py — host CPU and system memory, through ``psutil`` and the OS.

The CPU is not an accelerator, so this backend is opt-in: ``query_all()``
skips it unless ``include_cpu=True`` and the CLI shows it only with
``--cpu``. The stat object follows the device contract (memory, utilization,
temperature, power, processes) so that the CLI, the JSON output, the
Prometheus exporter and ``pick_device`` handle it without special cases.

Fields beyond the common ones:

* ``arch``, ``cores_physical``, ``cores_logical``, ``core_layout`` (for
  example ``"4P+6E"`` on Apple Silicon), ``sockets``, ``numa_nodes``
* ``freq_mhz`` and ``freq_max_mhz`` when the OS reports them
* ``load_avg`` (1, 5, 15 minutes) on POSIX
* ``per_core_pct``: utilization per logical core
* ``swap_used_mb`` / ``swap_total_mb``
* ``cgroup_cpu_limit`` and ``cgroup_mem_limit_mb``: the container's cgroup
  v2 quota when running inside one, else ``None``

Memory is the system RAM: ``mem_used_mb`` is what ``psutil`` reports as used,
``mem_free_mb`` is the available figure (what a new process can get).
Temperature comes from ``psutil.sensors_temperatures()`` on Linux; power from
the Intel RAPL counters on Linux when they are readable. Both are ``None``
elsewhere. Processes are the ``top_n`` by resident memory, with their RSS in
``gpu_mem_mb`` (the field name is shared with the accelerator backends).
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

from ._device import DeviceMixin
from ._process import ProcessInfo

_MIB = 1024 * 1024
_RAPL_PACKAGE = "/sys/class/powercap/intel-rapl:0/energy_uj"
_CGROUP_CPU = "/sys/fs/cgroup/cpu.max"
_CGROUP_MEM = "/sys/fs/cgroup/memory.max"
_NUMA_DIR = "/sys/devices/system/node"
_SENSOR_KEYS = ("coretemp", "k10temp", "cpu_thermal", "cpu-thermal", "zenpower", "acpitz", "soc_thermal")


class CPUStat(DeviceMixin):
    """The host CPU and its memory, as one device."""

    vendor = "cpu"
    _index_attr = "cpu_index"

    def __init__(
        self,
        cpu_index: int,
        name: str,
        mem_used_mb: int,
        mem_total_mb: int,
        mem_available_mb: Optional[int] = None,
        util_pct: Optional[int] = None,
        temp_c: Optional[int] = None,
        power_w: Optional[float] = None,
        arch: Optional[str] = None,
        cores_physical: Optional[int] = None,
        cores_logical: Optional[int] = None,
        core_layout: Optional[str] = None,
        sockets: Optional[int] = None,
        numa_nodes: Optional[int] = None,
        freq_mhz: Optional[int] = None,
        freq_max_mhz: Optional[int] = None,
        load_avg: Optional[Tuple[float, float, float]] = None,
        per_core_pct: Optional[List[int]] = None,
        swap_used_mb: Optional[int] = None,
        swap_total_mb: Optional[int] = None,
        cgroup_cpu_limit: Optional[float] = None,
        cgroup_mem_limit_mb: Optional[int] = None,
        processes: Optional[List[ProcessInfo]] = None,
    ) -> None:
        self.cpu_index = cpu_index
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb
        if mem_available_mb is not None:
            self._mem_free_mb = mem_available_mb
        self.util_pct = util_pct
        self.temp_c = temp_c
        self.power_w = power_w
        self.arch = arch
        self.cores_physical = cores_physical
        self.cores_logical = cores_logical
        self.core_layout = core_layout
        self.sockets = sockets
        self.numa_nodes = numa_nodes
        self.freq_mhz = freq_mhz
        self.freq_max_mhz = freq_max_mhz
        self.load_avg = tuple(load_avg) if load_avg else None
        self.per_core_pct = list(per_core_pct) if per_core_pct else None
        self.swap_used_mb = swap_used_mb
        self.swap_total_mb = swap_total_mb
        self.cgroup_cpu_limit = cgroup_cpu_limit
        self.cgroup_mem_limit_mb = cgroup_mem_limit_mb
        self.processes: List[ProcessInfo] = processes or []

    @property
    def torch_device(self) -> Optional[str]:
        return "cpu"

    @property
    def framework_index(self) -> Optional[int]:
        return 0

    @property
    def ray_resource(self) -> Optional[str]:
        return "CPU"

    @property
    def k8s_resource(self) -> Optional[str]:
        return "cpu"

    def to_dict(self) -> dict:
        return {
            "cpu_index": self.cpu_index,
            "name": self.name,
            "arch": self.arch,
            "cores_physical": self.cores_physical,
            "cores_logical": self.cores_logical,
            "core_layout": self.core_layout,
            "sockets": self.sockets,
            "numa_nodes": self.numa_nodes,
            "freq_mhz": self.freq_mhz,
            "freq_max_mhz": self.freq_max_mhz,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
            "mem_free_mb": self.mem_free_mb,
            "swap_used_mb": self.swap_used_mb,
            "swap_total_mb": self.swap_total_mb,
            "util_pct": self.util_pct,
            "per_core_pct": self.per_core_pct,
            "load_avg": list(self.load_avg) if self.load_avg else None,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "cgroup_cpu_limit": self.cgroup_cpu_limit,
            "cgroup_mem_limit_mb": self.cgroup_mem_limit_mb,
            "processes": [p.to_dict() for p in self.processes],
        }

    def __repr__(self) -> str:
        return (
            f"CPUStat(name={self.name!r}, cores={self.cores_physical}/{self.cores_logical}, "
            f"mem_used={self.mem_used_mb}MB, mem_total={self.mem_total_mb}MB)"
        )


class CPUStatCollection:
    """Collects the host CPU as a one-element collection."""

    def __init__(self, cpus: List[CPUStat]) -> None:
        self.cpus = cpus

    def to_dict(self) -> dict:
        return {"cpus": [c.to_dict() for c in self.cpus]}

    def __len__(self) -> int:
        return len(self.cpus)

    def __iter__(self):
        return iter(self.cpus)

    def __getitem__(self, index):
        return self.cpus[index]

    def __repr__(self) -> str:
        return f"CPUStatCollection({self.cpus!r})"

    @classmethod
    def new_query(cls, top_n: int = 5, sample: float = 0.1) -> "CPUStatCollection":
        """
        Read the host CPU through ``psutil``.

        ``sample`` is the interval used for the first utilization reading in
        a process; later readings use the delta since the previous call and
        cost nothing. Returns an empty collection if ``psutil`` is missing.
        """
        try:
            import psutil
        except ImportError:
            return cls([])
        try:
            return cls([_query(psutil, top_n=top_n, sample=sample)])
        except Exception:
            return cls([])


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

_SAMPLED = False


def _query(psutil, top_n: int = 5, sample: float = 0.1) -> CPUStat:
    global _SAMPLED
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()

    # Utilization: the first reading in a process needs an interval; after
    # that psutil keeps the previous CPU times and the call is immediate.
    interval = None if _SAMPLED else max(sample, 0.0)
    per_core = psutil.cpu_percent(interval=interval, percpu=True)
    _SAMPLED = True
    per_core_pct = [int(round(x)) for x in per_core] if per_core else None
    util = int(round(sum(per_core) / len(per_core))) if per_core else None

    freq = freq_max = None
    try:
        f = psutil.cpu_freq()
        if f:
            freq = _mhz(f.current)
            freq_max = _mhz(f.max)
    except Exception:
        pass

    load = None
    try:
        load = tuple(round(x, 2) for x in psutil.getloadavg())
    except Exception:
        pass

    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    cpu_quota, mem_quota = _cgroup_limits()

    return CPUStat(
        cpu_index=0,
        name=_model_name(),
        mem_used_mb=int(vm.used) // _MIB,
        mem_total_mb=int(vm.total) // _MIB,
        mem_available_mb=int(vm.available) // _MIB,
        util_pct=util,
        temp_c=_temperature(psutil) if sys.platform != "darwin" else _mac_cpu_temp(),
        power_w=_rapl_power() if sys.platform != "darwin" else _mac_cpu_power(),
        arch=platform.machine() or None,
        cores_physical=physical,
        cores_logical=logical,
        core_layout=_core_layout(),
        sockets=_sockets(),
        numa_nodes=_numa_nodes(),
        freq_mhz=freq,
        freq_max_mhz=freq_max,
        load_avg=load,
        per_core_pct=per_core_pct,
        swap_used_mb=int(sw.used) // _MIB,
        swap_total_mb=int(sw.total) // _MIB,
        cgroup_cpu_limit=cpu_quota,
        cgroup_mem_limit_mb=mem_quota,
        processes=_top_processes(psutil, top_n),
    )


# ---------------------------------------------------------------------------
# Helpers, each readable from a string or file so they can be unit-tested
# ---------------------------------------------------------------------------

def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r") as f:
            return f.read()
    except OSError:
        return None


def _sysctl(key: str) -> Optional[str]:
    try:
        r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=2)
        return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _mac_cpu_temp() -> Optional[int]:
    try:
        from . import _apple_sensors
        t = _apple_sensors.temperatures().get("cpu")
        return int(round(t)) if t is not None else None
    except Exception:
        return None


def _mac_cpu_power() -> Optional[float]:
    try:
        from . import _apple_sensors
        return _apple_sensors.energy_watts().get("cpu")
    except Exception:
        return None


def _windows_model_name() -> Optional[str]:
    """``ProcessorNameString`` from the registry; ``platform.processor()`` only gives the family string."""
    try:
        import winreg  # type: ignore[import-not-found]
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            value, _ = winreg.QueryValueEx(k, "ProcessorNameString")
            return str(value).strip() or None
    except Exception:
        return None


def _model_name() -> str:
    name = None
    if sys.platform == "win32":
        name = _windows_model_name()
    elif sys.platform.startswith("linux"):
        info = _read("/proc/cpuinfo") or ""
        name = _cpuinfo_field(info, "model name") or _cpuinfo_field(info, "Model") or _cpuinfo_field(info, "Hardware")
    elif sys.platform == "darwin":
        name = _sysctl("machdep.cpu.brand_string")
    return _clean_model_name(name or platform.processor() or platform.machine() or "CPU")


_NOISE = re.compile(r"\((?:R|TM|r|tm)\)|\b\d+-Core Processor\b|\bProcessor\b|\bCPU\b|@\s*[\d.]+\s*GHz|\bwith Radeon Graphics\b", re.IGNORECASE)


def _clean_model_name(name: str) -> str:
    """``"Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz"`` -> ``"Intel Xeon Silver 4314"``."""
    cleaned = " ".join(_NOISE.sub(" ", name).split())
    return cleaned or name


def _mhz(value) -> Optional[int]:
    """psutil reports Apple Silicon frequencies in GHz (e.g. 4.0); no CPU runs below 100 MHz."""
    if not value:
        return None
    v = float(value)
    return int(round(v * 1000)) if v < 100 else int(round(v))


def _cpuinfo_field(info: str, key: str) -> Optional[str]:
    for line in info.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            if k.strip() == key and v.strip():
                return " ".join(v.split())
    return None


def _sockets() -> Optional[int]:
    if sys.platform.startswith("linux"):
        info = _read("/proc/cpuinfo") or ""
        ids = {l.split(":", 1)[1].strip() for l in info.splitlines() if l.startswith("physical id")}
        return len(ids) or None
    if sys.platform == "darwin":
        v = _sysctl("hw.packages")
        return int(v) if v and v.isdigit() else None
    return None


def _numa_nodes() -> Optional[int]:
    try:
        return len([d for d in os.listdir(_NUMA_DIR) if d.startswith("node") and d[4:].isdigit()]) or None
    except OSError:
        return None


def _core_layout() -> Optional[str]:
    """``"4P+6E"`` on Apple Silicon, from the perflevel sysctls; ``None`` elsewhere."""
    if sys.platform != "darwin":
        return None
    p = _sysctl("hw.perflevel0.physicalcpu")
    e = _sysctl("hw.perflevel1.physicalcpu")
    if p and p.isdigit() and e and e.isdigit():
        return f"{p}P+{e}E"
    return None


def _temperature(psutil) -> Optional[int]:
    try:
        temps = psutil.sensors_temperatures()
    except Exception:
        return None
    if not temps:
        return None
    for key in _SENSOR_KEYS:
        entries = temps.get(key)
        if entries:
            vals = [e.current for e in entries if e.current is not None]
            if vals:
                # Package reading if labelled, else the hottest core.
                pkg = [e.current for e in entries if (e.label or "").lower().startswith(("package", "tctl", "tdie"))]
                return int(round(pkg[0] if pkg else max(vals)))
    return None


def _rapl_power(interval: float = 0.05) -> Optional[float]:
    """Package power from the RAPL energy counter, sampled over *interval* seconds."""
    first = _read(_RAPL_PACKAGE)
    if first is None or not first.strip().isdigit():
        return None
    t0 = time.perf_counter()
    time.sleep(interval)
    second = _read(_RAPL_PACKAGE)
    dt = time.perf_counter() - t0
    if second is None or not second.strip().isdigit() or dt <= 0:
        return None
    delta = int(second) - int(first)
    if delta < 0:  # counter wrapped
        return None
    return round(delta / 1e6 / dt, 1)


def _cgroup_limits(cpu_max: Optional[str] = None, mem_max: Optional[str] = None) -> Tuple[Optional[float], Optional[int]]:
    """Parse cgroup v2 ``cpu.max`` (``"200000 100000"`` → 2.0 CPUs) and ``memory.max`` (bytes or ``max``)."""
    if cpu_max is None:
        cpu_max = _read(_CGROUP_CPU)
    if mem_max is None:
        mem_max = _read(_CGROUP_MEM)
    cpu_limit: Optional[float] = None
    if cpu_max:
        parts = cpu_max.split()
        if len(parts) == 2 and parts[0] != "max" and parts[0].isdigit() and parts[1].isdigit() and int(parts[1]) > 0:
            cpu_limit = round(int(parts[0]) / int(parts[1]), 2)
    mem_limit: Optional[int] = None
    if mem_max and mem_max.strip().isdigit():
        mem_limit = int(mem_max.strip()) // _MIB
    return cpu_limit, mem_limit


def _top_processes(psutil, top_n: int) -> List[ProcessInfo]:
    if top_n <= 0:
        return []
    rows: List[Tuple[int, int, str, str]] = []
    for p in psutil.process_iter(["pid", "username", "name", "memory_info"]):
        try:
            mi = p.info.get("memory_info")
            rss = int(mi.rss) if mi else 0
            if rss <= 0:
                continue
            rows.append((rss, int(p.info["pid"]), str(p.info.get("username") or "?"), str(p.info.get("name") or "?")))
        except Exception:
            continue
    rows.sort(reverse=True)
    from ._process import container_of
    return [ProcessInfo(pid=pid, username=user, command=name, gpu_mem_mb=rss // _MIB,
                        rss_mb=rss // _MIB, container=container_of(pid))
            for rss, pid, user, name in rows[:top_n]]


if __name__ == "__main__":
    import json
    print(json.dumps(CPUStatCollection.new_query().to_dict(), indent=2))
