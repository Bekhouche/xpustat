"""
xpustat — unified accelerator stat tool.

Supports NVIDIA, AMD, Intel Gaudi, Intel Arc/GPU,
Huawei Ascend, Hygon DCU, Cambricon MLU, Moore Threads, Apple Silicon,
AWS Neuron, Google Cloud TPU, and Tenstorrent.

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

    # Host CPU and memory, on request
    stats = xpustat.query_all(include_cpu=True)
    print(stats.cpu[0].name, stats.cpu[0].cores_logical, stats.cpu[0].mem_free_mb)

    # Pick the least-loaded device and hand it to PyTorch
    dev = xpustat.pick_device()[0]
    model.to(dev.torch_device)            # e.g. "cuda:2", "npu:0", "mps"
    print(dev.mem_free_mb, dev.framework_index)

    # Prometheus text format
    print(xpustat.to_prometheus(stats))
"""

__version__ = "0.5.0"
__version_tuple__ = tuple(int(x) for x in __version__.split("."))

from ._process import ProcessInfo
from .xpu import AnyDeviceStat, XPUStatCollection, query, query_all, register_backend, unregister_backend
from .nvidia import NvidiaGPUStat, NvidiaGPUStatCollection, MIGInstanceStat
from .amd import AMDGPUStat, AMDGPUStatCollection
from .gaudi import GaudiAIPStat, GaudiAIPStatCollection
from .intel_gpu import IntelGPUStat, IntelGPUStatCollection
from .huawei import HuaweiNPUStat, HuaweiNPUStatCollection
from .hygon import HygonDCUStat, HygonDCUStatCollection
from .cambricon import CambriconMLUStat, CambriconMLUStatCollection
from .moorethreads import MooreThreadsGPUStat, MooreThreadsGPUStatCollection
from .apple import AppleGPUStat, AppleGPUStatCollection
from .neuron import NeuronDeviceStat, NeuronDeviceStatCollection
from .tpu import TPUChipStat, TPUChipStatCollection
from .tenstorrent import TenstorrentDeviceStat, TenstorrentDeviceStatCollection
from .jetson import JetsonGPUStat, JetsonGPUStatCollection
from .cpu import CPUStat, CPUStatCollection
from .rebellions import RebellionsNPUStat, RebellionsNPUStatCollection
from .furiosa import FuriosaNPUStat, FuriosaNPUStatCollection
from ._device import (DeviceMixin, TORCH_DEVICE_TYPE, VISIBLE_DEVICES_ENV, RAY_RESOURCE, K8S_RESOURCE,
                      pick_device, visible_indices, framework_index, torch_device, visible_devices_env)
from .prometheus import to_prometheus

__all__ = [
    "__version__",
    "__version_tuple__",
    # Core API
    "query_all",
    "query",
    "register_backend",
    "unregister_backend",
    "XPUStatCollection",
    "AnyDeviceStat",
    # Process info
    "ProcessInfo",
    # Per-vendor stat objects
    "NvidiaGPUStat",
    "NvidiaGPUStatCollection",
    "MIGInstanceStat",
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
    "AppleGPUStat",
    "AppleGPUStatCollection",
    "NeuronDeviceStat",
    "NeuronDeviceStatCollection",
    "TPUChipStat",
    "TPUChipStatCollection",
    "TenstorrentDeviceStat",
    "TenstorrentDeviceStatCollection",
    "JetsonGPUStat",
    "JetsonGPUStatCollection",
    "CPUStat",
    "CPUStatCollection",
    "RebellionsNPUStat",
    "RebellionsNPUStatCollection",
    "FuriosaNPUStat",
    "FuriosaNPUStatCollection",
    # Device selection / framework helpers
    "DeviceMixin",
    "TORCH_DEVICE_TYPE",
    "VISIBLE_DEVICES_ENV",
    "RAY_RESOURCE",
    "K8S_RESOURCE",
    "pick_device",
    "visible_indices",
    "framework_index",
    "torch_device",
    "visible_devices_env",
    # Prometheus
    "to_prometheus",
]
