"""
xpustat — unified accelerator stat tool.

Supports NVIDIA, AMD, Intel Gaudi, Intel Arc/GPU,
Huawei Ascend, Hygon DCU, Cambricon MLU, and Moore Threads.

Quick start::

    import xpustat

    # Query all vendors at once
    stats = xpustat.query_all()

    # Iterate every device across all vendors
    for dev in stats:
        print(dev.name, dev.mem_used_mb, "/", dev.mem_total_mb, "MB")

    # Vendor-specific access
    for gpu in stats.nvidia:
        print(gpu.name, gpu.util_pct, "%", gpu.temp_c, "°C")

    # Subscript by vendor key
    for gpu in stats["amd"]:
        print(gpu.power_w, "W")

    # Query a single vendor
    gpus = xpustat.query("nvidia")

    # JSON string (for HTTP responses / files)
    print(stats.to_json(indent=2))

    # JSON with metadata wrapper
    print(stats.to_json(metadata=True))
"""

__version__ = "0.1.0"
__version_tuple__ = (0, 1, 0)

from ._process import ProcessInfo
from .xpu import AnyDeviceStat, XPUStatCollection, query, query_all
from .nvidia import NvidiaGPUStat, NvidiaGPUStatCollection
from .amd import AMDGPUStat, AMDGPUStatCollection
from .gaudi import GaudiAIPStat, GaudiAIPStatCollection
from .intel_gpu import IntelGPUStat, IntelGPUStatCollection
from .huawei import HuaweiNPUStat, HuaweiNPUStatCollection
from .hygon import HygonDCUStat, HygonDCUStatCollection
from .cambricon import CambriconMLUStat, CambriconMLUStatCollection
from .moorethreads import MooreThreadsGPUStat, MooreThreadsGPUStatCollection

__all__ = [
    "__version__",
    "__version_tuple__",
    # Core API
    "query_all",
    "query",
    "XPUStatCollection",
    "AnyDeviceStat",
    # Process info
    "ProcessInfo",
    # Per-vendor stat objects
    "NvidiaGPUStat",
    "NvidiaGPUStatCollection",
    "AMDGPUStat",
    "AMDGPUStatCollection",
    "GaudiAIPStat",
    "GaudiAIPStatCollection",
    "IntelGPUStat",
    "IntelGPUStatCollection",
    "HuaweiNPUStat",
    "HuaweiNPUStatCollection",
    "HygonDCUStat",
    "HygonDCUStatCollection",
    "CambriconMLUStat",
    "CambriconMLUStatCollection",
    "MooreThreadsGPUStat",
    "MooreThreadsGPUStatCollection",
]
