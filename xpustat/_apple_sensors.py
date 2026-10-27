"""
_apple_sensors.py — Apple Silicon power and temperature without sudo.

``powermetrics`` needs root. Two private but stable interfaces do not:

* **IOReport** (``/usr/lib/libIOReport.dylib``), group ``Energy Model``:
  energy counters per block, ``CPU Energy``, ``GPU``, ``ANE``, ``DRAM``
  (millijoules, some channels in micro- or nanojoules). Two samples a short
  interval apart give watts. The same source macmon and asitop-style tools use.
* **SMC** through the ``AppleSMC`` IOKit user client: float keys ``Tg..``
  are GPU temperature sensors and ``Tp..`` / ``Te..`` / ``Ts..`` CPU ones.
  Enumerating the ~3000 keys takes about 0.4 s, so the matching key names
  are cached on disk per machine model after the first run.

Everything here returns ``{}`` or ``None`` on any failure and never raises,
so the backends degrade to N/A as before. Nothing runs outside macOS.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

_LOCK = threading.Lock()
_ENABLED = sys.platform == "darwin"

# ---------------------------------------------------------------------------
# CoreFoundation helpers
# ---------------------------------------------------------------------------

_cf = None
_kUTF8 = 0x08000100


def _load_cf():
    global _cf
    if _cf is None:
        import ctypes
        import ctypes.util
        cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFStringGetCString.restype = ctypes.c_bool
        cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        cf.CFDictionaryGetValue.restype = ctypes.c_void_p
        cf.CFDictionaryGetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cf.CFArrayGetCount.restype = ctypes.c_long
        cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
        cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
        cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
        cf.CFDictionaryCreateMutableCopy.restype = ctypes.c_void_p
        cf.CFDictionaryCreateMutableCopy.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p]
        cf.CFRelease.restype = None
        cf.CFRelease.argtypes = [ctypes.c_void_p]
        _cf = cf
    return _cf


def _cfstr(s: str):
    return _load_cf().CFStringCreateWithCString(None, s.encode(), _kUTF8)


def _pystr(ref) -> str:
    if not ref:
        return ""
    import ctypes
    buf = ctypes.create_string_buffer(256)
    return buf.value.decode(errors="replace") if _load_cf().CFStringGetCString(ref, buf, 256, _kUTF8) else ""


# ---------------------------------------------------------------------------
# IOReport energy
# ---------------------------------------------------------------------------

_ior = None
_ior_state: Dict[str, object] = {}   # sub, chans, last_sample, last_time, last_watts
_UNIT_DIV = {"mJ": 1e3, "uJ": 1e6, "nJ": 1e9, "J": 1.0}
_CHANNELS = {"CPU Energy": "cpu", "GPU": "gpu", "ANE": "ane", "DRAM": "dram"}


def _load_ioreport():
    global _ior
    if _ior is None:
        import ctypes
        ior = ctypes.CDLL("/usr/lib/libIOReport.dylib")
        ior.IOReportCopyChannelsInGroup.restype = ctypes.c_void_p
        ior.IOReportCopyChannelsInGroup.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        ior.IOReportCreateSubscription.restype = ctypes.c_void_p
        ior.IOReportCreateSubscription.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64, ctypes.c_void_p]
        ior.IOReportCreateSamples.restype = ctypes.c_void_p
        ior.IOReportCreateSamples.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        ior.IOReportCreateSamplesDelta.restype = ctypes.c_void_p
        ior.IOReportCreateSamplesDelta.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        for fn in ("IOReportChannelGetChannelName", "IOReportChannelGetUnitLabel"):
            getattr(ior, fn).restype = ctypes.c_void_p
            getattr(ior, fn).argtypes = [ctypes.c_void_p]
        ior.IOReportSimpleGetIntegerValue.restype = ctypes.c_int64
        ior.IOReportSimpleGetIntegerValue.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        _ior = ior
    return _ior


def _ioreport_init() -> bool:
    if "sub" in _ior_state:
        return True
    import ctypes
    ior = _load_ioreport()
    cf = _load_cf()
    chans = ior.IOReportCopyChannelsInGroup(_cfstr("Energy Model"), None, 0, 0, 0)
    if not chans:
        return False
    mchans = cf.CFDictionaryCreateMutableCopy(None, 0, chans)
    sub_ch = ctypes.c_void_p()
    sub = ior.IOReportCreateSubscription(None, mchans, ctypes.byref(sub_ch), 0, None)
    if not sub:
        return False
    _ior_state.update(sub=sub, chans=mchans, sub_ch=sub_ch)
    return True


def _ioreport_sample():
    ior = _load_ioreport()
    return ior.IOReportCreateSamples(_ior_state["sub"], _ior_state["chans"], None)


def _delta_watts(s1, s2, seconds: float) -> Dict[str, float]:
    ior = _load_ioreport()
    cf = _load_cf()
    delta = ior.IOReportCreateSamplesDelta(s1, s2, None)
    if not delta:
        return {}
    try:
        arr = cf.CFDictionaryGetValue(delta, _cfstr("IOReportChannels"))
        out: Dict[str, float] = {}
        ecpu = pcpu = 0.0
        for i in range(cf.CFArrayGetCount(arr) if arr else 0):
            item = cf.CFArrayGetValueAtIndex(arr, i)
            name = _pystr(ior.IOReportChannelGetChannelName(item))
            unit = _pystr(ior.IOReportChannelGetUnitLabel(item))
            div = _UNIT_DIV.get(unit)
            if div is None:
                continue
            joules = ior.IOReportSimpleGetIntegerValue(item, 0) / div
            key = _CHANNELS.get(name)
            if key:
                out[key] = round(joules / seconds, 2)
            elif name == "ECPU":
                ecpu = joules
            elif name == "PCPU":
                pcpu = joules
        if "cpu" not in out and (ecpu or pcpu):
            out["cpu"] = round((ecpu + pcpu) / seconds, 2)
        return out
    finally:
        cf.CFRelease(delta)


def energy_watts(sample: float = 0.1, min_interval: float = 0.05) -> Dict[str, float]:
    """
    Power per block, watts: ``{"cpu", "gpu", "ane", "dram"}`` (keys present
    only when the chip reports them). The first call in a process samples for
    *sample* seconds; later calls use the delta since the previous call when at
    least *min_interval* has elapsed, else return the previous result.
    """
    if not _ENABLED:
        return {}
    with _LOCK:
        try:
            if not _ioreport_init():
                return {}
            now = time.monotonic()
            last = _ior_state.get("last_sample")
            if last is None:
                s1 = _ioreport_sample()
                time.sleep(max(sample, 0.02))
                s2 = _ioreport_sample()
                seconds = time.monotonic() - now
            else:
                elapsed = now - float(_ior_state["last_time"])  # type: ignore[arg-type]
                if elapsed < min_interval:
                    return dict(_ior_state.get("last_watts", {}))  # type: ignore[arg-type]
                s1, s2, seconds = last, _ioreport_sample(), elapsed
            watts = _delta_watts(s1, s2, max(seconds, 1e-3)) if s1 and s2 else {}
            if s1 and s1 is not last:
                _load_cf().CFRelease(s1)
            if last is not None and last is not s2:
                _load_cf().CFRelease(last)
            _ior_state.update(last_sample=s2, last_time=time.monotonic(), last_watts=watts)
            return dict(watts)
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# SMC temperatures
# ---------------------------------------------------------------------------

_smc: Dict[str, object] = {}
_GPU_PREFIX = ("Tg",)
_CPU_PREFIX = ("Tp", "Te", "Ts")
_CACHE_DIR = os.path.expanduser("~/Library/Caches/xpustat")


def _smc_open() -> bool:
    if "conn" in _smc:
        return True
    import ctypes
    import ctypes.util

    class Vers(ctypes.Structure):
        _fields_ = [("major", ctypes.c_uint8), ("minor", ctypes.c_uint8), ("build", ctypes.c_uint8),
                    ("reserved", ctypes.c_uint8), ("release", ctypes.c_uint16)]

    class PLimit(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint16), ("length", ctypes.c_uint16), ("cpuPLimit", ctypes.c_uint32),
                    ("gpuPLimit", ctypes.c_uint32), ("memPLimit", ctypes.c_uint32)]

    class KeyInfo(ctypes.Structure):
        _fields_ = [("dataSize", ctypes.c_uint32), ("dataType", ctypes.c_uint32), ("dataAttributes", ctypes.c_uint8)]

    class KeyData(ctypes.Structure):
        _fields_ = [("key", ctypes.c_uint32), ("vers", Vers), ("pLimitData", PLimit), ("keyInfo", KeyInfo),
                    ("result", ctypes.c_uint8), ("status", ctypes.c_uint8), ("data8", ctypes.c_uint8),
                    ("data32", ctypes.c_uint32), ("bytes", ctypes.c_uint8 * 32)]

    iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    iokit.IOServiceMatching.restype = ctypes.c_void_p
    iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
    iokit.IOServiceGetMatchingService.restype = ctypes.c_uint32
    iokit.IOServiceGetMatchingService.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
    iokit.IOServiceOpen.restype = ctypes.c_int
    iokit.IOServiceOpen.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    iokit.IOConnectCallStructMethod.restype = ctypes.c_int
    iokit.IOConnectCallStructMethod.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t,
                                                ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    libc.mach_task_self.restype = ctypes.c_uint32
    svc = iokit.IOServiceGetMatchingService(0, iokit.IOServiceMatching(b"AppleSMC"))
    if not svc:
        return False
    conn = ctypes.c_uint32()
    if iokit.IOServiceOpen(svc, libc.mach_task_self(), 0, ctypes.byref(conn)) != 0:
        return False
    _smc.update(conn=conn.value, iokit=iokit, KeyData=KeyData, ctypes=ctypes)
    return True


def _smc_call(inp):
    ctypes = _smc["ctypes"]
    KeyData = _smc["KeyData"]
    out = KeyData()
    size = ctypes.c_size_t(ctypes.sizeof(KeyData))
    rc = _smc["iokit"].IOConnectCallStructMethod(_smc["conn"], 2, ctypes.byref(inp), ctypes.sizeof(KeyData),
                                                 ctypes.byref(out), ctypes.byref(size))
    return rc, out


def _key_code(name: str) -> int:
    return struct.unpack(">I", name.encode("latin-1"))[0]


def _key_name(code: int) -> str:
    return struct.pack(">I", code).decode("latin-1")


def _smc_read_float(name: str) -> Optional[float]:
    KeyData = _smc["KeyData"]
    inp = KeyData()
    inp.key = _key_code(name)
    inp.data8 = 9                       # key info
    rc, info = _smc_call(inp)
    if rc != 0 or info.result != 0 or _key_name(info.keyInfo.dataType) != "flt " or info.keyInfo.dataSize != 4:
        return None
    inp = KeyData()
    inp.key = _key_code(name)
    inp.keyInfo.dataSize = 4
    inp.keyInfo.dataType = info.keyInfo.dataType
    inp.data8 = 5                       # read bytes
    rc, out = _smc_call(inp)
    if rc != 0 or out.result != 0:
        return None
    return struct.unpack("<f", bytes(out.bytes[:4]))[0]


def _smc_read_u32(name: str) -> Optional[int]:
    KeyData = _smc["KeyData"]
    inp = KeyData()
    inp.key = _key_code(name)
    inp.data8 = 9
    rc, info = _smc_call(inp)
    if rc != 0 or info.result != 0:
        return None
    inp = KeyData()
    inp.key = _key_code(name)
    inp.keyInfo.dataSize = info.keyInfo.dataSize
    inp.keyInfo.dataType = info.keyInfo.dataType
    inp.data8 = 5
    rc, out = _smc_call(inp)
    if rc != 0 or out.result != 0:
        return None
    return int.from_bytes(bytes(out.bytes[:info.keyInfo.dataSize]), "big")


def _machine_model() -> str:
    try:
        r = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True, timeout=2)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _cache_path() -> str:
    return os.path.join(_CACHE_DIR, f"smc-temp-keys-{_machine_model()}.json")


def _temperature_keys() -> List[str]:
    """Names of the SMC float keys that are GPU or CPU temperatures; enumerated once, cached on disk."""
    cached = _smc.get("keys")
    if cached is not None:
        return cached  # type: ignore[return-value]
    path = _cache_path()
    try:
        with open(path, "r") as f:
            keys = json.load(f)
        if isinstance(keys, list) and keys:
            _smc["keys"] = keys
            return keys
    except (OSError, ValueError):
        pass
    KeyData = _smc["KeyData"]
    count = _smc_read_u32("#KEY") or 0
    keys: List[str] = []
    prefixes = _GPU_PREFIX + _CPU_PREFIX
    for i in range(count):
        inp = KeyData()
        inp.data8 = 8                   # key by index
        inp.data32 = i
        rc, out = _smc_call(inp)
        if rc != 0:
            continue
        name = _key_name(out.key)
        if name.startswith(prefixes):
            v = _smc_read_float(name)
            if v is not None and 0 < v < 150:
                keys.append(name)
    _smc["keys"] = keys
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(keys, f)
    except OSError:
        pass
    return keys


def temperatures() -> Dict[str, Optional[float]]:
    """``{"gpu": mean of the GPU sensors, "cpu": mean of the CPU sensors}``; a key is None when no sensor matched."""
    if not _ENABLED:
        return {}
    with _LOCK:
        try:
            if not _smc_open():
                return {}
            gpu: List[float] = []
            cpu: List[float] = []
            for name in _temperature_keys():
                v = _smc_read_float(name)
                if v is None or not (0 < v < 150):
                    continue
                (gpu if name.startswith(_GPU_PREFIX) else cpu).append(v)
            return {
                "gpu": round(sum(gpu) / len(gpu), 1) if gpu else None,
                "cpu": round(sum(cpu) / len(cpu), 1) if cpu else None,
            }
        except Exception:
            return {}


if __name__ == "__main__":
    t0 = time.perf_counter()
    print("power:", energy_watts(), "| temps:", temperatures(), f"| {(time.perf_counter() - t0) * 1000:.0f} ms")
    t0 = time.perf_counter()
    time.sleep(0.2)
    print("power:", energy_watts(), "| temps:", temperatures(), f"| {(time.perf_counter() - t0) * 1000:.0f} ms")
