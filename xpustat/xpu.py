"""
xpu.py — unified query across all supported accelerator vendors.

Supported backends (silently skipped if the tool is not installed):
    - NVIDIA          nvidia-smi
    - AMD             amd-smi
    - Intel Gaudi     hl-smi
    - Intel Arc/GPU   xpu-smi
    - Huawei Ascend   npu-smi
    - Hygon DCU       hy-smi
    - Cambricon MLU   cnmon
    - Moore Threads   mthreads-gmi
"""

from __future__ import annotations

import datetime
import json
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterator, List, Union

from . import __version__
from .amd import AMDGPUStat, AMDGPUStatCollection
from .cambricon import CambriconMLUStat, CambriconMLUStatCollection
from .gaudi import GaudiAIPStat, GaudiAIPStatCollection
from .huawei import HuaweiNPUStat, HuaweiNPUStatCollection
from .hygon import HygonDCUStat, HygonDCUStatCollection
from .intel_gpu import IntelGPUStat, IntelGPUStatCollection
from .moorethreads import MooreThreadsGPUStat, MooreThreadsGPUStatCollection
from .nvidia import NvidiaGPUStat, NvidiaGPUStatCollection

# Union of all per-device stat types — useful for type hints downstream
AnyDeviceStat = Union[
    NvidiaGPUStat, AMDGPUStat, GaudiAIPStat, IntelGPUStat,
    HuaweiNPUStat, HygonDCUStat, CambriconMLUStat, MooreThreadsGPUStat,
]


class XPUStatCollection:
    """Aggregated stats from every detected accelerator on the system.

    Typical library usage::

        import xpustat

        stats = xpustat.query_all()

        # iterate all devices across every vendor
        for dev in stats:
            print(dev.name, dev.mem_used_mb, dev.mem_total_mb)

        # vendor-specific access
        for gpu in stats.nvidia:
            print(gpu.name, gpu.temp_c)

        # subscript by vendor key
        for gpu in stats["amd"]:
            print(gpu.util_pct)

        # JSON-ready dict (for APIs / databases)
        data = stats.to_dict()

        # JSON string (for HTTP responses / files)
        payload = stats.to_json(indent=2)

        # JSON with hostname + timestamp wrapper
        payload = stats.to_json(metadata=True)
    """

    def __init__(
        self,
        nvidia: NvidiaGPUStatCollection,
        amd: AMDGPUStatCollection,
        gaudi: GaudiAIPStatCollection,
        intel: IntelGPUStatCollection,
        huawei: HuaweiNPUStatCollection,
        hygon: HygonDCUStatCollection,
        cambricon: CambriconMLUStatCollection,
        moorethreads: MooreThreadsGPUStatCollection,
    ) -> None:
        self.nvidia = nvidia
        self.amd = amd
        self.gaudi = gaudi
        self.intel = intel
        self.huawei = huawei
        self.hygon = hygon
        self.cambricon = cambricon
        self.moorethreads = moorethreads

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def all_devices(self) -> List[AnyDeviceStat]:
        """Flat list of every device across all vendors."""
        return (
            list(self.nvidia)
            + list(self.amd)
            + list(self.gaudi)
            + list(self.intel)
            + list(self.huawei)
            + list(self.hygon)
            + list(self.cambricon)
            + list(self.moorethreads)
        )

    def __iter__(self) -> Iterator[AnyDeviceStat]:
        """Iterate every device across all vendors in vendor order."""
        return iter(self.all_devices())

    def __len__(self) -> int:
        return len(self.all_devices())

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, vendor: str):
        """
        Subscript access by vendor key, e.g. ``stats["nvidia"]``.

        Returns the vendor-specific collection object (iterable, has ``len()``).

        Valid keys: nvidia, amd, gaudi, intel, huawei, hygon, cambricon, moorethreads
        """
        try:
            return getattr(self, vendor)
        except AttributeError:
            raise KeyError(
                f"Unknown vendor {vendor!r}. "
                f"Valid keys: {', '.join(_BACKENDS_KEYS)}"
            ) from None

    def __repr__(self) -> str:
        counts = ", ".join(
            f"{v}={len(getattr(self, v))}" for v in _BACKENDS_KEYS
        )
        return f"XPUStatCollection({counts})"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Return a plain JSON-serialisable dict of all devices, keyed by vendor.

        Example::

            {
              "nvidia":  [ {"gpu_index": 0, "name": "...", ...} ],
              "amd":     [ {"gpu_index": 0, "name": "...", ...} ],
              ...
            }
        """
        return {
            "nvidia":       self.nvidia.to_dict()["gpus"],
            "amd":          self.amd.to_dict()["gpus"],
            "gaudi":        self.gaudi.to_dict()["aips"],
            "intel":        self.intel.to_dict()["gpus"],
            "huawei":       self.huawei.to_dict()["npus"],
            "hygon":        self.hygon.to_dict()["dcus"],
            "cambricon":    self.cambricon.to_dict()["mlus"],
            "moorethreads": self.moorethreads.to_dict()["gpus"],
        }

    def to_json(self, indent: int = 2, metadata: bool = False) -> str:
        """
        Serialise to a JSON string.

        Args:
            indent:   JSON indentation (default 2). Pass ``None`` for compact.
            metadata: When ``True``, wraps the device data under a ``"devices"``
                      key and adds ``hostname``, ``query_time``, and
                      ``xpustat_version`` at the top level.

        Example with ``metadata=True``::

            {
              "hostname": "myserver",
              "query_time": "2026-05-22T14:48:00.123456",
              "xpustat_version": "0.2.0",
              "devices": { "nvidia": [...], "amd": [...], ... }
            }
        """
        if metadata:
            payload: dict[str, Any] = {
                "hostname": socket.gethostname(),
                "query_time": datetime.datetime.now().isoformat(),
                "xpustat_version": __version__,
                "devices": self.to_dict(),
            }
        else:
            payload = self.to_dict()
        return json.dumps(payload, indent=indent)


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_BACKENDS: List[tuple] = [
    ("nvidia",       NvidiaGPUStatCollection),
    ("amd",          AMDGPUStatCollection),
    ("gaudi",        GaudiAIPStatCollection),
    ("intel",        IntelGPUStatCollection),
    ("huawei",       HuaweiNPUStatCollection),
    ("hygon",        HygonDCUStatCollection),
    ("cambricon",    CambriconMLUStatCollection),
    ("moorethreads", MooreThreadsGPUStatCollection),
]

_BACKENDS_KEYS: List[str] = [name for name, _ in _BACKENDS]
_BACKENDS_MAP: dict = {name: cls for name, cls in _BACKENDS}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_all(parallel: bool = True, vendors: Optional[List[str]] = None) -> XPUStatCollection:
    """
    Query every supported accelerator backend and return an XPUStatCollection.

    Backends whose management tool is not installed return an empty collection
    and are never reported as errors.

    Args:
        parallel: Run all backend queries concurrently (default ``True``).
                  Pass ``False`` for sequential execution (useful for debugging).
        vendors: Optional list of vendor keys to query. If provided, only these
                 backends will be queried, and others will be empty.

    Returns:
        :class:`XPUStatCollection` populated with results.
    """
    results: dict = {}
    _empty = {name: cls([]) for name, cls in _BACKENDS}

    backends_to_query = _BACKENDS
    if vendors is not None:
        backends_to_query = [(n, c) for n, c in _BACKENDS if n in vendors]

    if parallel:
        with ThreadPoolExecutor(max_workers=max(1, len(backends_to_query))) as pool:
            futures = {
                pool.submit(cls.new_query): (name, cls)
                for name, cls in backends_to_query
            }
            for future in as_completed(futures):
                name, cls = futures[future]
                try:
                    results[name] = future.result()
                except Exception:
                    results[name] = _empty[name]
    else:
        for name, cls in backends_to_query:
            try:
                results[name] = cls.new_query()
            except Exception:
                results[name] = _empty[name]

    for name in _BACKENDS_KEYS:
        if name not in results:
            results[name] = _empty[name]

    return XPUStatCollection(
        nvidia=results["nvidia"],
        amd=results["amd"],
        gaudi=results["gaudi"],
        intel=results["intel"],
        huawei=results["huawei"],
        hygon=results["hygon"],
        cambricon=results["cambricon"],
        moorethreads=results["moorethreads"],
    )


def query(vendor: str):
    """
    Query a single vendor and return its collection object.

    Args:
        vendor: One of: ``nvidia``, ``amd``, ``gaudi``, ``intel``,
                ``huawei``, ``hygon``, ``cambricon``, ``moorethreads``.

    Returns:
        The vendor-specific ``*StatCollection`` (iterable, has ``to_dict()``).

    Raises:
        ValueError: If *vendor* is not a recognised key.

    Example::

        gpus = xpustat.query("nvidia")
        for gpu in gpus:
            print(gpu.name, gpu.mem_used_mb)
    """
    cls = _BACKENDS_MAP.get(vendor)
    if cls is None:
        raise ValueError(
            f"Unknown vendor {vendor!r}. "
            f"Valid options: {', '.join(_BACKENDS_KEYS)}"
        )
    return cls.new_query()


if __name__ == "__main__":
    collection = query_all()
    print(collection.to_json(metadata=True))
    print(f"\nTotal devices found: {len(collection)}")
