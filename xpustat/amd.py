from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from ._device import DeviceMixin
from ._process import ProcessInfo, enrich_process, enrich_full

# ---------------------------------------------------------------------------
# Optional fast-path: amdsmi (Python bindings shipped with ROCm)
# ---------------------------------------------------------------------------
try:
    import amdsmi as _amdsmi  # type: ignore[import-untyped]
    _HAS_AMDSMI = True
except Exception:  # ImportError, or a binding that fails against the installed driver library
    _HAS_AMDSMI = False


class AMDGPUStat(DeviceMixin):
    """Stats for a single AMD GPU."""

    vendor = "amd"
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
        gfx_target: Optional[str] = None,
        processes: Optional[List[ProcessInfo]] = None,
        fan_pct: Optional[int] = None,
        fan_rpm: Optional[int] = None,
        clock_gfx_mhz: Optional[int] = None,
        clock_max_gfx_mhz: Optional[int] = None,
        clock_mem_mhz: Optional[int] = None,
        mem_util_pct: Optional[int] = None,
        temp_hotspot_c: Optional[int] = None,
        temp_mem_c: Optional[int] = None,
        ecc_uncorrected: Optional[int] = None,
        ecc_corrected: Optional[int] = None,
        throttled: Optional[bool] = None,
        perf_level: Optional[str] = None,
        power_limit_w: Optional[float] = None,
        energy_j: Optional[float] = None,
        pcie_gen: Optional[int] = None,
        pcie_width: Optional[int] = None,
        bdf: Optional[str] = None,
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
        self.fan_pct = fan_pct
        self.fan_rpm = fan_rpm
        self.clock_gfx_mhz = clock_gfx_mhz
        self.clock_max_gfx_mhz = clock_max_gfx_mhz
        self.clock_mem_mhz = clock_mem_mhz
        #: Memory-controller activity, percent (``umc_activity``).
        self.mem_util_pct = mem_util_pct
        self.temp_hotspot_c = temp_hotspot_c
        self.temp_mem_c = temp_mem_c
        self.ecc_uncorrected = ecc_uncorrected
        self.ecc_corrected = ecc_corrected
        #: ``power.throttle_status == "THROTTLED"``; ``None`` when not reported.
        self.throttled = throttled
        #: e.g. ``"auto"`` from ``AMDSMI_DEV_PERF_LEVEL_AUTO``.
        self.perf_level = perf_level
        self.power_limit_w = power_limit_w
        #: Energy counter in joules when the card reports one (MI series).
        self.energy_j = energy_j
        self.pcie_gen = pcie_gen
        self.pcie_width = pcie_width
        self.bdf = bdf

    @property
    def healthy(self) -> Optional[bool]:
        """False on uncorrected ECC errors; ``None`` when ECC is not reported."""
        if self.ecc_uncorrected is None:
            return None
        return self.ecc_uncorrected == 0

    def to_dict(self) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "name": self.name,
            "gfx_target": self.gfx_target,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
            "mem_free_mb": self.mem_free_mb,
            "util_pct": self.util_pct,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "power_limit_w": self.power_limit_w,
            "processes": [p.to_dict() for p in self.processes],
            "fan_pct": self.fan_pct,
            "fan_rpm": self.fan_rpm,
            "clock_gfx_mhz": self.clock_gfx_mhz,
            "clock_max_gfx_mhz": self.clock_max_gfx_mhz,
            "clock_mem_mhz": self.clock_mem_mhz,
            "mem_util_pct": self.mem_util_pct,
            "temp_hotspot_c": self.temp_hotspot_c,
            "temp_mem_c": self.temp_mem_c,
            "ecc_uncorrected": self.ecc_uncorrected,
            "ecc_corrected": self.ecc_corrected,
            "throttled": self.throttled,
            "healthy": self.healthy,
            "perf_level": self.perf_level,
            "energy_j": self.energy_j,
            "pcie_gen": self.pcie_gen,
            "pcie_width": self.pcie_width,
            "bdf": self.bdf,
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

    def __getitem__(self, index):
        return self.gpus[index]

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
                        processes.append(enrich_full(pid, "", gpu_mem_mb=gpu_mem_mb))
                except Exception:
                    pass

                extra = _extra_fields_amdsmi(handle)

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
                        **extra,
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

        # The three amd-smi invocations are independent; run them concurrently
        # (each can take several hundred ms on ROCm).
        with ThreadPoolExecutor(max_workers=3) as pool:
            f_static  = pool.submit(cls._run_amd_smi, ["static"])
            f_metric  = pool.submit(cls._run_amd_smi, ["metric"])
            f_process = pool.submit(cls._run_amd_smi, ["process"])
        static_data  = f_static.result()
        metric_data  = f_metric.result()
        process_data = f_process.result()

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
            asic   = _section(static, "asic")

            name = (
                asic.get("market_name")
                or _section(static, "board").get("model_number")
                or "Unknown AMD GPU"
            )
            gfx_target = asic.get("target_graphics_version") or None

            mem   = _section(metric, "mem_usage")
            usage = _section(metric, "usage")
            pwr   = _section(metric, "power")
            temp  = _section(metric, "temperature")

            processes: List[ProcessInfo] = []
            for p_item in process_by_index.get(idx, []):
                p = p_item.get("process_info", {})
                # Idle GPUs report the string "No running processes detected"
                if not isinstance(p, dict):
                    continue
                pid = p.get("pid")
                if pid is None:
                    continue
                vram_mem = _section(_section(p, "memory_usage"), "vram_mem").get("value", 0)
                mem_mb = int(vram_mem) // (1024 * 1024) if vram_mem else None
                cmd = p.get("name", "")
                processes.append(enrich_full(pid, cmd if cmd != "N/A" else "", gpu_mem_mb=mem_mb))

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
                **_extra_fields_json(static, metric),
            ))

        return cls(gpus, backend="subprocess")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extra_fields_json(static: dict, metric: dict) -> dict:
    """Fan, clocks, ECC, throttle state, PCIe, power limit and energy from amd-smi JSON."""
    fan = _section(metric, "fan")
    clock = _section(metric, "clock")
    gfx = _section(clock, "gfx_0")
    memclk = _section(clock, "mem_0")
    usage = _section(metric, "usage")
    temp = _section(metric, "temperature")
    ecc = _section(metric, "ecc")
    pwr = _section(metric, "power")
    pcie = _section(metric, "pcie")
    energy = _section(metric, "energy")
    bus = _section(static, "bus")
    limit = _section(_section(static, "limit"), "ppt0")

    throttle = pwr.get("throttle_status")
    throttled = None
    if isinstance(throttle, str) and throttle.upper() in ("THROTTLED", "UNTHROTTLED"):
        throttled = throttle.upper() == "THROTTLED"
    perf = metric.get("perf_level")
    perf_level = perf.replace("AMDSMI_DEV_PERF_LEVEL_", "").lower() if isinstance(perf, str) else None
    gen = None
    ver = bus.get("pcie_interface_version")
    if isinstance(ver, str) and ver.lower().startswith("gen"):
        try:
            gen = int(ver.split()[-1])
        except ValueError:
            pass
    ecc_u = ecc.get("total_uncorrectable_count")
    ecc_c = ecc.get("total_correctable_count")
    return dict(
        fan_pct=_int(fan.get("usage")),
        fan_rpm=fan.get("rpm") if isinstance(fan.get("rpm"), int) else None,
        clock_gfx_mhz=_int(gfx.get("clk")),
        clock_max_gfx_mhz=_int(gfx.get("max_clk")),
        clock_mem_mhz=_int(memclk.get("clk")),
        mem_util_pct=_int(usage.get("umc_activity")),
        temp_hotspot_c=_int(temp.get("hotspot")),
        temp_mem_c=_int(temp.get("mem")),
        ecc_uncorrected=ecc_u if isinstance(ecc_u, int) else None,
        ecc_corrected=ecc_c if isinstance(ecc_c, int) else None,
        throttled=throttled,
        perf_level=perf_level,
        power_limit_w=_float(limit.get("socket_power_limit")),
        energy_j=_float(energy.get("total_energy_consumption")),
        pcie_gen=gen,
        pcie_width=pcie.get("width") if isinstance(pcie.get("width"), int) else None,
        bdf=bus.get("bdf") if isinstance(bus.get("bdf"), str) else None,
    )


def _amdsmi_try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _extra_fields_amdsmi(handle) -> dict:
    """The extended fields through the amdsmi bindings; every call is optional."""
    a = _amdsmi
    out: dict = {}
    clk = _amdsmi_try(lambda: a.amdsmi_get_clock_info(handle, a.AmdSmiClkType.GFX)) or {}
    mclk = _amdsmi_try(lambda: a.amdsmi_get_clock_info(handle, a.AmdSmiClkType.MEM)) or {}
    ecc = _amdsmi_try(lambda: a.amdsmi_get_gpu_total_ecc_count(handle)) or {}
    act = _amdsmi_try(lambda: a.amdsmi_get_gpu_activity(handle)) or {}
    cap = _amdsmi_try(lambda: a.amdsmi_get_power_cap_info(handle)) or {}
    out["clock_gfx_mhz"] = _int_or_none(clk.get("clk"))
    out["clock_max_gfx_mhz"] = _int_or_none(clk.get("max_clk"))
    out["clock_mem_mhz"] = _int_or_none(mclk.get("clk"))
    out["ecc_uncorrected"] = _int_or_none(ecc.get("uncorrectable_count"))
    out["ecc_corrected"] = _int_or_none(ecc.get("correctable_count"))
    out["mem_util_pct"] = _int_or_none(act.get("umc_activity"))
    out["temp_hotspot_c"] = _amdsmi_try(lambda: int(a.amdsmi_get_temp_metric(
        handle, a.AmdSmiTemperatureType.HOTSPOT, a.AmdSmiTemperatureMetric.CURRENT)))
    out["temp_mem_c"] = _amdsmi_try(lambda: int(a.amdsmi_get_temp_metric(
        handle, a.AmdSmiTemperatureType.VRAM, a.AmdSmiTemperatureMetric.CURRENT)))
    fan = _amdsmi_try(lambda: a.amdsmi_get_gpu_fan_speed(handle, 0))
    out["fan_rpm"] = int(fan) if isinstance(fan, int) else None
    power_cap = cap.get("power_cap")
    out["power_limit_w"] = round(power_cap / 1e6, 1) if isinstance(power_cap, (int, float)) and power_cap > 1000 else _float_or_none(power_cap)
    energy = _amdsmi_try(lambda: a.amdsmi_get_energy_count(handle)) or {}
    e = energy.get("energy_accumulator") if isinstance(energy, dict) else None
    res = energy.get("counter_resolution") if isinstance(energy, dict) else None
    out["energy_j"] = round(e * res / 1e6, 1) if isinstance(e, (int, float)) and isinstance(res, (int, float)) else None
    bdf = _amdsmi_try(lambda: a.amdsmi_get_gpu_device_bdf(handle))
    out["bdf"] = bdf if isinstance(bdf, str) else None
    return {k: v for k, v in out.items()}


def _int_or_none(v) -> Optional[int]:
    try:
        return int(v) if v is not None and v != "N/A" else None
    except (TypeError, ValueError):
        return None


def _float_or_none(v) -> Optional[float]:
    try:
        return float(v) if v is not None and v != "N/A" else None
    except (TypeError, ValueError):
        return None


def _section(data: dict, key: str) -> dict:
    """Return ``data[key]`` if it is a dict; amd-smi emits ``"N/A"`` for
    sections a device does not support (e.g. ``usage`` on APUs)."""
    value = data.get(key)
    return value if isinstance(value, dict) else {}


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
