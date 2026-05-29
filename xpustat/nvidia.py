from __future__ import annotations

import shutil
import subprocess
from typing import Dict, List, Optional

from ._process import ProcessInfo, enrich_process

# ---------------------------------------------------------------------------
# Optional fast-path: pynvml (direct NVML C library — no subprocess)
# ---------------------------------------------------------------------------
try:
    import pynvml as _pynvml  # type: ignore[import-untyped]
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False


class NvidiaGPUStat:
    """Stats for a single NVIDIA GPU."""

    def __init__(
        self,
        gpu_index: int,
        uuid: str,
        name: str,
        mem_used_mb: int,
        mem_total_mb: int,
        util_pct: Optional[int] = None,
        temp_c: Optional[int] = None,
        power_w: Optional[float] = None,
        power_limit_w: Optional[float] = None,
        processes: Optional[List[ProcessInfo]] = None,
        cuda_cc: Optional[str] = None,
    ) -> None:
        self.gpu_index = gpu_index
        self.uuid = uuid
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb
        self.util_pct = util_pct
        self.temp_c = temp_c
        self.power_w = power_w
        self.power_limit_w = power_limit_w
        self.processes: List[ProcessInfo] = processes or []
        #: CUDA compute capability, e.g. ``"8.6"`` (RTX 3090) or ``"9.0"`` (H100).
        self.cuda_cc = cuda_cc

    def to_dict(self) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "uuid": self.uuid,
            "name": self.name,
            "cuda_cc": self.cuda_cc,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
            "util_pct": self.util_pct,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "power_limit_w": self.power_limit_w,
            "processes": [p.to_dict() for p in self.processes],
        }

    def __repr__(self) -> str:
        return (
            f"NvidiaGPUStat(index={self.gpu_index}, name={self.name!r}, "
            f"mem_used={self.mem_used_mb}MB, mem_total={self.mem_total_mb}MB)"
        )


class NvidiaGPUStatCollection:
    """
    Collects stats for all NVIDIA GPUs.

    Query priority:
      1. ``pynvml``   — direct NVML library calls (fast, no subprocess).
                        Install with:  pip install pynvml
      2. ``nvidia-smi`` — subprocess fallback (always available with drivers).
    """

    #: Which backend was used: ``"pynvml"``, ``"subprocess"``, or ``"none"``.
    backend: str = "none"

    def __init__(self, gpus: List[NvidiaGPUStat], backend: str = "none") -> None:
        self.gpus = gpus
        self.backend = backend

    def to_dict(self) -> dict:
        return {"gpus": [gpu.to_dict() for gpu in self.gpus]}

    def __len__(self) -> int:
        return len(self.gpus)

    def __iter__(self):
        return iter(self.gpus)

    def __repr__(self) -> str:
        return f"NvidiaGPUStatCollection({self.gpus!r})"

    # ------------------------------------------------------------------
    # Public factory
    # ------------------------------------------------------------------

    @classmethod
    def new_query(cls) -> "NvidiaGPUStatCollection":
        """
        Query NVIDIA GPUs.  Uses ``pynvml`` when available, falls back to
        ``nvidia-smi`` subprocess.  Returns an empty collection if neither
        is installed or the query fails.
        """
        if _HAS_PYNVML:
            try:
                return cls._query_pynvml()
            except Exception:
                pass  # fall through to subprocess

        return cls._query_subprocess()

    # ------------------------------------------------------------------
    # Fast path — pynvml (direct NVML calls)
    # ------------------------------------------------------------------

    @classmethod
    def _query_pynvml(cls) -> "NvidiaGPUStatCollection":
        _pynvml.nvmlInit()
        try:
            count = _pynvml.nvmlDeviceGetCount()
            gpus: List[NvidiaGPUStat] = []

            for i in range(count):
                handle = _pynvml.nvmlDeviceGetHandleByIndex(i)

                name = _pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()

                uuid = _pynvml.nvmlDeviceGetUUID(handle)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode()

                mem = _pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used_mb = mem.used // (1024 * 1024)
                mem_total_mb = mem.total // (1024 * 1024)

                try:
                    util = _pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util_pct: Optional[int] = int(util.gpu)
                except _pynvml.NVMLError:
                    util_pct = None

                try:
                    temp_c: Optional[int] = int(
                        _pynvml.nvmlDeviceGetTemperature(
                            handle, _pynvml.NVML_TEMPERATURE_GPU
                        )
                    )
                except _pynvml.NVMLError:
                    temp_c = None

                try:
                    power_w: Optional[float] = (
                        _pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    )
                except _pynvml.NVMLError:
                    power_w = None

                try:
                    power_limit_w: Optional[float] = (
                        _pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                    )
                except _pynvml.NVMLError:
                    power_limit_w = None

                cuda_cc: Optional[str] = None
                try:
                    major, minor = _pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    cuda_cc = f"{major}.{minor}"
                except _pynvml.NVMLError:
                    pass

                processes: List[ProcessInfo] = []
                try:
                    for p in _pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        gmem = (
                            p.usedGpuMemory // (1024 * 1024)
                            if p.usedGpuMemory
                            else None
                        )
                        username, command = enrich_process(p.pid, "")
                        processes.append(
                            ProcessInfo(
                                pid=p.pid,
                                username=username,
                                command=command,
                                gpu_mem_mb=gmem,
                            )
                        )
                except _pynvml.NVMLError:
                    pass

                gpus.append(
                    NvidiaGPUStat(
                        gpu_index=i,
                        uuid=uuid,
                        name=name,
                        cuda_cc=cuda_cc,
                        mem_used_mb=mem_used_mb,
                        mem_total_mb=mem_total_mb,
                        util_pct=util_pct,
                        temp_c=temp_c,
                        power_w=power_w,
                        power_limit_w=power_limit_w,
                        processes=processes,
                    )
                )

            return cls(gpus, backend="pynvml")
        finally:
            try:
                _pynvml.nvmlShutdown()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Fallback — nvidia-smi subprocess
    # ------------------------------------------------------------------

    @staticmethod
    def _smi(*args: str, timeout: int = 10) -> Optional[str]:
        try:
            r = subprocess.run(
                ["nvidia-smi"] + list(args),
                capture_output=True, text=True, timeout=timeout,
            )
            return r.stdout if r.returncode == 0 and r.stdout.strip() else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    @classmethod
    def _query_subprocess(cls) -> "NvidiaGPUStatCollection":
        if shutil.which("nvidia-smi") is None:
            return cls([], backend="none")

        gpu_out = cls._smi(
            "--query-gpu=index,uuid,name,compute_cap,memory.used,memory.total,"
            "utilization.gpu,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        )
        if gpu_out is None:
            return cls([], backend="none")

        gpus: List[NvidiaGPUStat] = []
        uuid_to_idx: Dict[str, int] = {}

        for line in gpu_out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                idx      = int(parts[0])
                uuid     = parts[1]
                name     = parts[2]
                cc       = parts[3]       if len(parts) > 3 else None
                mem_used = int(parts[4])
                mem_tot  = int(parts[5])
                util     = _to_int(parts[6])   if len(parts) > 6 else None
                temp     = _to_int(parts[7])   if len(parts) > 7 else None
                power    = _to_float(parts[8]) if len(parts) > 8 else None
                plimit   = _to_float(parts[9]) if len(parts) > 9 else None

                gpus.append(NvidiaGPUStat(
                    gpu_index=idx, uuid=uuid, name=name,
                    cuda_cc=cc,
                    mem_used_mb=mem_used, mem_total_mb=mem_tot,
                    util_pct=util, temp_c=temp,
                    power_w=power, power_limit_w=plimit,
                ))
                uuid_to_idx[uuid] = idx
            except (ValueError, IndexError):
                continue

        if not gpus:
            return cls([], backend="none")

        proc_out = cls._smi(
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        )
        if proc_out:
            gpu_by_idx = {g.gpu_index: g for g in gpus}
            for line in proc_out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue
                try:
                    puuid = parts[0]
                    pid   = int(parts[1])
                    cmd   = parts[2]
                    gmem  = _to_int(parts[3]) if len(parts) > 3 else None
                    gidx  = uuid_to_idx.get(puuid)
                    if gidx is None:
                        continue
                    username, command = enrich_process(pid, cmd)
                    gpu_by_idx[gidx].processes.append(
                        ProcessInfo(pid=pid, username=username,
                                    command=command, gpu_mem_mb=gmem)
                    )
                except (ValueError, IndexError):
                    continue

        return cls(gpus, backend="subprocess")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    import json
    col = NvidiaGPUStatCollection.new_query()
    print(f"backend: {col.backend}")
    print(json.dumps(col.to_dict(), indent=2))
