from __future__ import annotations

import json
import shutil
import subprocess
from typing import List, Optional

from ._process import ProcessInfo, enrich_process

# ---------------------------------------------------------------------------
# Optional fast-path: amdsmi (Python bindings shipped with ROCm)
# ---------------------------------------------------------------------------
try:
    import amdsmi as _amdsmi  # type: ignore[import-untyped]
    _HAS_AMDSMI = True
except ImportError:
    _HAS_AMDSMI = False


class AMDGPUStat:
    """Stats for a single AMD GPU."""

    def __init__(
        self,
        gpu_index: int,
        name: str,
        mem_used_mb: int,
        mem_total_mb: int,
        util_pct: Optional[int] = None,
        temp_c: Optional[int] = None,
        power_w: Optional[float] = None,
        gfx_target: Optional[str] = None,
        processes: Optional[List[ProcessInfo]] = None,
    ) -> None:
        self.gpu_index = gpu_index
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb
        self.util_pct = util_pct
        self.temp_c = temp_c
        self.power_w = power_w
        #: ROCm GFX target, e.g. ``"gfx1100"`` (RDNA3) or ``"gfx942"`` (MI300X).
        self.gfx_target = gfx_target
        self.processes: List[ProcessInfo] = processes or []

    def to_dict(self) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "name": self.name,
            "gfx_target": self.gfx_target,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
            "util_pct": self.util_pct,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "processes": [p.to_dict() for p in self.processes],
        }

    def __repr__(self) -> str:
        return (
            f"AMDGPUStat(index={self.gpu_index}, name={self.name!r}, "
            f"mem_used={self.mem_used_mb}MB, mem_total={self.mem_total_mb}MB)"
        )


class AMDGPUStatCollection:
    """
    Collects stats for all AMD GPUs.

    Query priority:
      1. ``amdsmi``  — Python bindings shipped with ROCm (fast, no subprocess).
                       Available after installing ROCm; also: pip install amdsmi
      2. ``amd-smi`` — subprocess fallback (always available with ROCm drivers).
    """

    #: Which backend was used: ``"amdsmi"``, ``"subprocess"``, or ``"none"``.
    backend: str = "none"

    def __init__(self, gpus: List[AMDGPUStat], backend: str = "none") -> None:
        self.gpus = gpus
        self.backend = backend

    def to_dict(self) -> dict:
        return {"gpus": [gpu.to_dict() for gpu in self.gpus]}

    def __len__(self) -> int:
        return len(self.gpus)

    def __iter__(self):
        return iter(self.gpus)

    def __repr__(self) -> str:
        return f"AMDGPUStatCollection({self.gpus!r})"

    # ------------------------------------------------------------------
    # Public factory
    # ------------------------------------------------------------------

    @classmethod
    def new_query(cls) -> "AMDGPUStatCollection":
        """
        Query AMD GPUs.  Uses ``amdsmi`` Python bindings when available,
        falls back to ``amd-smi`` subprocess.  Returns an empty collection
        if neither is present or the query fails.
        """
        if _HAS_AMDSMI:
            try:
                return cls._query_amdsmi()
            except Exception:
                pass  # fall through to subprocess

        return cls._query_subprocess()

    # ------------------------------------------------------------------
    # Fast path — amdsmi Python bindings (ships with ROCm)
    # ------------------------------------------------------------------

    @classmethod
    def _query_amdsmi(cls) -> "AMDGPUStatCollection":
        _amdsmi.amdsmi_init()
        try:
            handles = _amdsmi.amdsmi_get_processor_handles()
            gpus: List[AMDGPUStat] = []

            for i, handle in enumerate(handles):
                # ── name + GFX target ────────────────────────────────────────
                name = "Unknown AMD GPU"
                gfx_target: Optional[str] = None
                try:
                    asic = _amdsmi.amdsmi_get_gpu_asic_info(handle)
                    name = (
                        asic.get("market_name")
                        or asic.get("asic_name")
                        or name
                    )
                    gfx_target = asic.get("target_graphics_version") or None
                except Exception:
                    pass

                # ── VRAM ────────────────────────────────────────────────────
                mem_used_mb = 0
                mem_total_mb = 0
                try:
                    vram_type = _amdsmi.AmdSmiMemoryType.VRAM
                    mem_used_mb = (
                        _amdsmi.amdsmi_get_gpu_memory_usage(handle, vram_type)
                        // (1024 * 1024)
                    )
                    mem_total_mb = (
                        _amdsmi.amdsmi_get_gpu_memory_total(handle, vram_type)
                        // (1024 * 1024)
                    )
                except Exception:
                    pass

                # ── utilization ─────────────────────────────────────────────
                util_pct: Optional[int] = None
                try:
                    activity = _amdsmi.amdsmi_get_gpu_activity(handle)
                    val = activity.get("gfx_activity")
                    if val is not None:
                        util_pct = int(val)
                except Exception:
                    pass

                # ── temperature ─────────────────────────────────────────────
                temp_c: Optional[int] = None
                try:
                    temp_c = int(
                        _amdsmi.amdsmi_get_temp_metric(
                            handle,
                            _amdsmi.AmdSmiTemperatureType.EDGE,
                            _amdsmi.AmdSmiTemperatureMetric.CURRENT,
                        )
                    )
                except Exception:
                    pass

                # ── power ───────────────────────────────────────────────────
                power_w: Optional[float] = None
                try:
                    pwr = _amdsmi.amdsmi_get_power_info(handle)
                    raw = pwr.get("average_socket_power") or pwr.get("current_socket_power")
                    if raw is not None:
                        power_w = float(raw)
                except Exception:
                    pass

                # ── processes ───────────────────────────────────────────────
                processes: List[ProcessInfo] = []
                try:
                    procs = _amdsmi.amdsmi_get_gpu_process_list(handle)
                    for p in procs:
                        pid = p.get("pid")
                        if pid is None:
                            continue
                        mem_val = p.get("memory_usage", {}).get("vram_mem", 0)
                        gpu_mem_mb = int(mem_val) // (1024 * 1024) if mem_val else None
                        username, command = enrich_process(pid, "")
                        processes.append(
                            ProcessInfo(
                                pid=pid,
                                username=username,
                                command=command,
                                gpu_mem_mb=gpu_mem_mb,
                            )
                        )
                except Exception:
                    pass

                gpus.append(
                    AMDGPUStat(
                        gpu_index=i,
                        name=name,
                        gfx_target=gfx_target,
                        mem_used_mb=mem_used_mb,
                        mem_total_mb=mem_total_mb,
                        util_pct=util_pct,
                        temp_c=temp_c,
                        power_w=power_w,
                        processes=processes,
                    )
                )

            return cls(gpus, backend="amdsmi")
        finally:
            try:
                _amdsmi.amdsmi_shut_down()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Fallback — amd-smi subprocess
    # ------------------------------------------------------------------

    @staticmethod
    def _run_amd_smi(args: List[str]) -> Optional[dict]:
        try:
            result = subprocess.run(
                ["amd-smi"] + args + ["--json"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None
            return json.loads(result.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            return None

    @classmethod
    def _query_subprocess(cls) -> "AMDGPUStatCollection":
        if shutil.which("amd-smi") is None:
            return cls([], backend="none")

        static_data = cls._run_amd_smi(["static"])
        metric_data = cls._run_amd_smi(["metric"])
        process_data = cls._run_amd_smi(["process"])

        if static_data is None or metric_data is None:
            return cls([], backend="none")

        static_by_index = {
            e["gpu"]: e for e in static_data.get("gpu_data", [])
        }
        metric_by_index = {
            e["gpu"]: e for e in metric_data.get("gpu_data", [])
        }
        process_by_index = {}
        if process_data is not None:
            for p_info in process_data:
                idx = p_info.get("gpu")
                if idx is not None:
                    process_by_index[idx] = p_info.get("process_list", [])

        gpus: List[AMDGPUStat] = []
        for idx, static in static_by_index.items():
            metric = metric_by_index.get(idx, {})
            asic   = static.get("asic", {})

            name = (
                asic.get("market_name")
                or static.get("board", {}).get("model_number")
                or "Unknown AMD GPU"
            )
            gfx_target = asic.get("target_graphics_version") or None

            mem   = metric.get("mem_usage", {})
            usage = metric.get("usage", {})
            pwr   = metric.get("power", {})
            temp  = metric.get("temperature", {})

            processes: List[ProcessInfo] = []
            for p_item in process_by_index.get(idx, []):
                p = p_item.get("process_info", {})
                pid = p.get("pid")
                if pid is None:
                    continue
                vram_mem = p.get("memory_usage", {}).get("vram_mem", {}).get("value", 0)
                mem_mb = int(vram_mem) // (1024 * 1024) if vram_mem else None
                cmd = p.get("name", "")
                username, command = enrich_process(pid, cmd)
                processes.append(
                    ProcessInfo(
                        pid=pid,
                        username=username,
                        command=command,
                        gpu_mem_mb=mem_mb,
                    )
                )

            gpus.append(AMDGPUStat(
                gpu_index=idx,
                name=name,
                gfx_target=gfx_target,
                mem_used_mb=_mb(mem.get("used_vram")),
                mem_total_mb=_mb(mem.get("total_vram")),
                util_pct=_int(usage.get("gfx_activity")),
                temp_c=_int(temp.get("edge") or temp.get("hotspot")),
                power_w=_float(pwr.get("socket_power")),
                processes=processes,
            ))

        return cls(gpus, backend="subprocess")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mb(field) -> int:
    if isinstance(field, dict):
        try:
            return int(field.get("value", 0))
        except (TypeError, ValueError):
            return 0
    return 0


def _int(field) -> Optional[int]:
    if isinstance(field, dict):
        try:
            return int(field.get("value", 0))
        except (TypeError, ValueError):
            return None
    return None


def _float(field) -> Optional[float]:
    if isinstance(field, dict):
        try:
            return float(field.get("value", 0))
        except (TypeError, ValueError):
            return None
    return None


if __name__ == "__main__":
    col = AMDGPUStatCollection.new_query()
    print(f"backend: {col.backend}")
    print(json.dumps(col.to_dict(), indent=2))
