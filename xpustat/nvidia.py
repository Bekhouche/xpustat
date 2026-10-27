from __future__ import annotations

import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

from ._device import DeviceMixin
from ._process import ProcessInfo, enrich_process, enrich_full

# ---------------------------------------------------------------------------
# Optional fast-path: pynvml (direct NVML C library — no subprocess)
# ---------------------------------------------------------------------------
try:
    import pynvml as _pynvml  # type: ignore[import-untyped]
    _HAS_PYNVML = True
except Exception:  # ImportError, or a binding that fails against the installed driver library
    _HAS_PYNVML = False


class MIGInstanceStat(DeviceMixin):
    """
    One MIG compute instance on a partitioned NVIDIA GPU.

    A MIG instance is what a CUDA process actually gets on an A100/H100 in
    MIG mode: ``CUDA_VISIBLE_DEVICES`` takes its ``MIG-...`` UUID and a
    process can enumerate only one instance. ``framework_index`` and
    ``torch_device`` are therefore ``None``; use
    ``visible_devices_env([inst], format="uuid")`` to target it.
    """

    vendor = "nvidia"
    _index_attr = "parent_index"

    def __init__(
        self,
        parent_index: int,
        uuid: str,
        mem_used_mb: int,
        mem_total_mb: int,
        gpu_instance_id: Optional[int] = None,
        compute_instance_id: Optional[int] = None,
        profile: Optional[str] = None,
        processes: Optional[List[ProcessInfo]] = None,
    ) -> None:
        self.parent_index = parent_index
        self.uuid = uuid
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb
        self.gpu_instance_id = gpu_instance_id
        self.compute_instance_id = compute_instance_id
        #: Profile name such as ``"1g.10gb"`` when the driver reports it.
        self.profile = profile
        self.processes: List[ProcessInfo] = processes or []

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"MIG {self.profile}" if self.profile else "MIG instance"

    @property
    def index_label(self) -> str:
        gi = self.gpu_instance_id if self.gpu_instance_id is not None else "?"
        ci = self.compute_instance_id if self.compute_instance_id is not None else "?"
        return f"{self.parent_index}/{gi}.{ci}"

    @property
    def framework_index(self) -> Optional[int]:
        return None

    @property
    def torch_device(self) -> Optional[str]:
        return None

    def to_dict(self) -> dict:
        return {
            "parent_index": self.parent_index,
            "uuid": self.uuid,
            "gpu_instance_id": self.gpu_instance_id,
            "compute_instance_id": self.compute_instance_id,
            "profile": self.profile,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
            "mem_free_mb": self.mem_free_mb,
            "processes": [p.to_dict() for p in self.processes],
        }

    def __repr__(self) -> str:
        return f"MIGInstanceStat(gpu={self.parent_index}, uuid={self.uuid!r}, mem_total={self.mem_total_mb}MB)"


class NvidiaGPUStat(DeviceMixin):
    """Stats for a single NVIDIA GPU."""

    vendor = "nvidia"
    _index_attr = "gpu_index"

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
        mig_instances: Optional[List[MIGInstanceStat]] = None,
        fan_pct: Optional[int] = None,
        clock_sm_mhz: Optional[int] = None,
        clock_mem_mhz: Optional[int] = None,
        clock_max_sm_mhz: Optional[int] = None,
        enc_pct: Optional[int] = None,
        dec_pct: Optional[int] = None,
        ecc_uncorrected: Optional[int] = None,
        ecc_corrected: Optional[int] = None,
        pstate: Optional[int] = None,
        throttle_reasons: Optional[List[str]] = None,
        energy_j: Optional[float] = None,
        pcie_gen: Optional[int] = None,
        pcie_width: Optional[int] = None,
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
        #: MIG compute instances when the GPU is in MIG mode (NVML path only).
        self.mig_instances: List[MIGInstanceStat] = mig_instances or []
        self.fan_pct = fan_pct
        self.clock_sm_mhz = clock_sm_mhz
        self.clock_mem_mhz = clock_mem_mhz
        self.clock_max_sm_mhz = clock_max_sm_mhz
        #: Video encoder / decoder utilization, percent.
        self.enc_pct = enc_pct
        self.dec_pct = dec_pct
        #: Volatile ECC error counts since the last driver reload.
        self.ecc_uncorrected = ecc_uncorrected
        self.ecc_corrected = ecc_corrected
        #: Performance state 0 (max) to 15 (idle); ``P8`` is reported as 8.
        self.pstate = pstate
        #: Active throttle reasons: ``idle``, ``app_clocks``, ``power_cap``,
        #: ``hw_slowdown``, ``sync_boost``, ``sw_thermal``, ``hw_thermal``,
        #: ``hw_power_brake``, ``display_clocks``. Empty when unthrottled.
        self.throttle_reasons: List[str] = list(throttle_reasons or [])
        #: Energy consumed since the driver loaded, joules (NVML only).
        self.energy_j = energy_j
        self.pcie_gen = pcie_gen
        self.pcie_width = pcie_width

    @property
    def throttled(self) -> Optional[bool]:
        """True when a reason other than ``idle`` is active; ``None`` when unknown."""
        if self.pstate is None and not self.throttle_reasons:
            return None
        return any(r != "idle" for r in self.throttle_reasons)

    @property
    def healthy(self) -> Optional[bool]:
        """False on uncorrected ECC errors or a hardware slowdown; ``None`` when unknown."""
        if self.ecc_uncorrected is None and not self.throttle_reasons:
            return None
        if self.ecc_uncorrected:
            return False
        return not any(r in ("hw_slowdown", "hw_thermal", "hw_power_brake") for r in self.throttle_reasons)

    def to_dict(self) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "uuid": self.uuid,
            "name": self.name,
            "cuda_cc": self.cuda_cc,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
            "mem_free_mb": self.mem_free_mb,
            "util_pct": self.util_pct,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "power_limit_w": self.power_limit_w,
            "processes": [p.to_dict() for p in self.processes],
            "mig_instances": [m.to_dict() for m in self.mig_instances],
            "fan_pct": self.fan_pct,
            "clock_sm_mhz": self.clock_sm_mhz,
            "clock_mem_mhz": self.clock_mem_mhz,
            "clock_max_sm_mhz": self.clock_max_sm_mhz,
            "enc_pct": self.enc_pct,
            "dec_pct": self.dec_pct,
            "ecc_uncorrected": self.ecc_uncorrected,
            "ecc_corrected": self.ecc_corrected,
            "pstate": self.pstate,
            "throttle_reasons": self.throttle_reasons,
            "throttled": self.throttled,
            "healthy": self.healthy,
            "energy_j": self.energy_j,
            "pcie_gen": self.pcie_gen,
            "pcie_width": self.pcie_width,
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

    def __getitem__(self, index):
        return self.gpus[index]

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

                proc_util = _process_utilization_nvml(handle)
                processes: List[ProcessInfo] = []
                try:
                    for p in _pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        gmem = (
                            p.usedGpuMemory // (1024 * 1024)
                            if p.usedGpuMemory
                            else None
                        )
                        u = proc_util.get(p.pid, {})
                        processes.append(enrich_full(p.pid, "", gpu_mem_mb=gmem, **u))
                except _pynvml.NVMLError:
                    pass

                extra = _extra_fields_nvml(handle)

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
                        mig_instances=_mig_instances_nvml(handle, i),
                        **extra,
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

        # The nvidia-smi invocations are independent; run them concurrently.
        # The extended query is separate so that an old driver that rejects
        # one of its fields does not take the base query down with it.
        with ThreadPoolExecutor(max_workers=4) as pool:
            f_gpu = pool.submit(
                cls._smi,
                "--query-gpu=index,uuid,name,compute_cap,memory.used,memory.total,"
                "utilization.gpu,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            )
            f_proc = pool.submit(
                cls._smi,
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            )
            f_ext = pool.submit(
                cls._smi,
                "--query-gpu=index," + ",".join(_EXT_FIELDS),
                "--format=csv,noheader,nounits",
            )
            f_pmon = pool.submit(cls._smi, "pmon", "-c", "1")
            f_list = pool.submit(cls._smi, "-L")
        gpu_out = f_gpu.result()
        if gpu_out is None:
            return cls([], backend="none")
        ext = _parse_extended(f_ext.result())
        pmon = _parse_pmon(f_pmon.result())
        migs = _parse_smi_list_mig(f_list.result())

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
                    mig_instances=migs.get(idx, []),
                    **ext.get(idx, {}),
                ))
                uuid_to_idx[uuid] = idx
            except (ValueError, IndexError):
                continue

        if not gpus:
            return cls([], backend="none")

        proc_out = f_proc.result()
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
                    gpu_by_idx[gidx].processes.append(
                        enrich_full(pid, cmd, gpu_mem_mb=gmem, **pmon.get((gidx, pid), {}))
                    )
                except (ValueError, IndexError):
                    continue

        return cls(gpus, backend="subprocess")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Throttle-reason bitmask names, in NVML bit order.
_THROTTLE_BITS = [
    (0x1, "idle"), (0x2, "app_clocks"), (0x4, "power_cap"), (0x8, "hw_slowdown"),
    (0x10, "sync_boost"), (0x20, "sw_thermal"), (0x40, "hw_thermal"),
    (0x80, "hw_power_brake"), (0x100, "display_clocks"),
]

# Extra nvidia-smi fields, queried separately from the base set.
_EXT_FIELDS = [
    "fan.speed", "clocks.sm", "clocks.mem", "clocks.max.sm",
    "utilization.encoder", "utilization.decoder",
    "ecc.errors.uncorrected.volatile.total", "ecc.errors.corrected.volatile.total",
    "pstate", "clocks_throttle_reasons.active",
    "pcie.link.gen.current", "pcie.link.width.current",
]


def throttle_reason_names(mask: int) -> List[str]:
    return [name for bit, name in _THROTTLE_BITS if mask & bit]


def _parse_extended(out: Optional[str]) -> Dict[int, dict]:
    """Parse the extended query into ``{gpu_index: {field: value}}``; missing values are None."""
    result: Dict[int, dict] = {}
    if not out:
        return result
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(_EXT_FIELDS) + 1:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        v = parts[1:]
        pstate = _to_int(v[8].lstrip("Pp")) if v[8] else None
        mask = None
        try:
            mask = int(v[9], 16) if v[9].lower().startswith("0x") else int(v[9])
        except (ValueError, TypeError):
            pass
        result[idx] = dict(
            fan_pct=_to_int(v[0]), clock_sm_mhz=_to_int(v[1]), clock_mem_mhz=_to_int(v[2]),
            clock_max_sm_mhz=_to_int(v[3]), enc_pct=_to_int(v[4]), dec_pct=_to_int(v[5]),
            ecc_uncorrected=_to_int(v[6]), ecc_corrected=_to_int(v[7]), pstate=pstate,
            throttle_reasons=throttle_reason_names(mask) if mask is not None else [],
            pcie_gen=_to_int(v[10]), pcie_width=_to_int(v[11]),
        )
    return result


def _parse_pmon(out: Optional[str]) -> Dict[tuple, dict]:
    """
    Parse ``nvidia-smi pmon -c 1`` into ``{(gpu_index, pid): {sm_pct, mem_pct, enc_pct, dec_pct}}``.
    Columns: gpu pid type sm mem enc dec command; ``-`` means no sample.
    """
    result: Dict[tuple, dict] = {}
    if not out:
        return result
    for line in out.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            gidx, pid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        result[(gidx, pid)] = dict(
            sm_pct=_to_int(parts[3]), mem_pct=_to_int(parts[4]),
            enc_pct=_to_int(parts[5]), dec_pct=_to_int(parts[6]),
        )
    return result


_MIG_LINE = re.compile(r"MIG\s+(\S+)\s+Device\s+(\d+):\s+\(UUID:\s+(MIG-[0-9a-f\-]+)\)")
_GPU_LINE = re.compile(r"^GPU\s+(\d+):")
_PROFILE_GB = re.compile(r"\.(\d+)gb", re.IGNORECASE)


def _parse_smi_list_mig(out: Optional[str]) -> Dict[int, List[MIGInstanceStat]]:
    """
    MIG inventory from ``nvidia-smi -L``::

        GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-...)
          MIG 1g.10gb     Device  0: (UUID: MIG-...)

    ``nvidia-smi -L`` gives no usage, so ``mem_used_mb`` is 0 and the total
    comes from the profile name. The NVML path overrides this with real data.
    """
    result: Dict[int, List[MIGInstanceStat]] = {}
    if not out:
        return result
    current: Optional[int] = None
    for line in out.splitlines():
        g = _GPU_LINE.match(line.strip())
        if g:
            current = int(g.group(1))
            continue
        m = _MIG_LINE.search(line)
        if m and current is not None:
            profile, uuid = m.group(1), m.group(3)
            gb = _PROFILE_GB.search(profile)
            total = int(gb.group(1)) * 1024 if gb else 0
            result.setdefault(current, []).append(MIGInstanceStat(
                parent_index=current, uuid=uuid, mem_used_mb=0, mem_total_mb=total, profile=profile))
    return result


def _nvml_try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _extra_fields_nvml(handle) -> dict:
    """The extended fields through NVML; every call is optional."""
    n = _pynvml
    mask = _nvml_try(lambda: int(n.nvmlDeviceGetCurrentClocksThrottleReasons(handle)))
    energy_mj = _nvml_try(lambda: int(n.nvmlDeviceGetTotalEnergyConsumption(handle)))
    return dict(
        fan_pct=_nvml_try(lambda: int(n.nvmlDeviceGetFanSpeed(handle))),
        clock_sm_mhz=_nvml_try(lambda: int(n.nvmlDeviceGetClockInfo(handle, n.NVML_CLOCK_SM))),
        clock_mem_mhz=_nvml_try(lambda: int(n.nvmlDeviceGetClockInfo(handle, n.NVML_CLOCK_MEM))),
        clock_max_sm_mhz=_nvml_try(lambda: int(n.nvmlDeviceGetMaxClockInfo(handle, n.NVML_CLOCK_SM))),
        enc_pct=_nvml_try(lambda: int(n.nvmlDeviceGetEncoderUtilization(handle)[0])),
        dec_pct=_nvml_try(lambda: int(n.nvmlDeviceGetDecoderUtilization(handle)[0])),
        ecc_uncorrected=_nvml_try(lambda: int(n.nvmlDeviceGetTotalEccErrors(
            handle, n.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, n.NVML_VOLATILE_ECC))),
        ecc_corrected=_nvml_try(lambda: int(n.nvmlDeviceGetTotalEccErrors(
            handle, n.NVML_MEMORY_ERROR_TYPE_CORRECTED, n.NVML_VOLATILE_ECC))),
        pstate=_nvml_try(lambda: int(n.nvmlDeviceGetPerformanceState(handle))),
        throttle_reasons=throttle_reason_names(mask) if mask is not None else [],
        energy_j=round(energy_mj / 1000.0, 1) if energy_mj is not None else None,
        pcie_gen=_nvml_try(lambda: int(n.nvmlDeviceGetCurrPcieLinkGeneration(handle))),
        pcie_width=_nvml_try(lambda: int(n.nvmlDeviceGetCurrPcieLinkWidth(handle))),
    )


def _process_utilization_nvml(handle) -> Dict[int, dict]:
    """Per-process SM/memory/codec utilization samples; ``{}`` when the driver has none."""
    try:
        samples = _pynvml.nvmlDeviceGetProcessUtilization(handle, 0)
    except Exception:
        return {}
    out: Dict[int, dict] = {}
    for s in samples or []:
        try:
            out[int(s.pid)] = dict(sm_pct=int(s.smUtil), mem_pct=int(s.memUtil),
                                   enc_pct=int(s.encUtil), dec_pct=int(s.decUtil))
        except Exception:
            continue
    return out


def _mig_instances_nvml(handle, parent_index: int) -> List[MIGInstanceStat]:
    """
    List the MIG compute instances of *handle* through NVML. Returns ``[]``
    when MIG is off, unsupported (every GPU before A100), or on any error.
    """
    try:
        current, _pending = _pynvml.nvmlDeviceGetMigMode(handle)
    except Exception:
        return []
    if current != getattr(_pynvml, "NVML_DEVICE_MIG_ENABLE", 1):
        return []
    try:
        count = _pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)
    except Exception:
        return []

    out: List[MIGInstanceStat] = []
    for j in range(count):
        try:
            mig = _pynvml.nvmlDeviceGetMigDeviceHandleByIndex(handle, j)
        except Exception:
            continue  # unused slot
        try:
            uuid = _pynvml.nvmlDeviceGetUUID(mig)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            mem = _pynvml.nvmlDeviceGetMemoryInfo(mig)
        except Exception:
            continue
        gi = ci = None
        try:
            gi = int(_pynvml.nvmlDeviceGetGpuInstanceId(mig))
            ci = int(_pynvml.nvmlDeviceGetComputeInstanceId(mig))
        except Exception:
            pass
        processes: List[ProcessInfo] = []
        try:
            for p in _pynvml.nvmlDeviceGetComputeRunningProcesses(mig):
                gmem = p.usedGpuMemory // (1024 * 1024) if p.usedGpuMemory else None
                processes.append(enrich_full(p.pid, "", gpu_mem_mb=gmem))
        except Exception:
            pass
        out.append(MIGInstanceStat(
            parent_index=parent_index, uuid=str(uuid),
            mem_used_mb=mem.used // (1024 * 1024), mem_total_mb=mem.total // (1024 * 1024),
            gpu_instance_id=gi, compute_instance_id=ci, processes=processes,
        ))
    return out


def _to_int(s) -> Optional[int]:
    try:
        return int(str(s).strip().strip("[]").split(".")[0])
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
