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
    - Apple           IOKit / ioreg (macOS only)
    - AWS Neuron      neuron-ls
    - Google TPU      tpu-info
    - Tenstorrent     tt-smi
    - NVIDIA Jetson   tegrastats
    - CPU             psutil (opt-in: include_cpu=True / --cpu)
    - Rebellions      rbln-stat
    - FuriosaAI       furiosa-smi
    - plugins         any package exposing an ``xpustat.backends`` entry point
"""

from __future__ import annotations

import datetime
import json
import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterator, List, Optional, Union

from . import __version__
from .amd import AMDGPUStat, AMDGPUStatCollection
from .apple import AppleGPUStat, AppleGPUStatCollection
from .cambricon import CambriconMLUStat, CambriconMLUStatCollection
from .gaudi import GaudiAIPStat, GaudiAIPStatCollection
from .huawei import HuaweiNPUStat, HuaweiNPUStatCollection
from .hygon import HygonDCUStat, HygonDCUStatCollection
from .intel_gpu import IntelGPUStat, IntelGPUStatCollection
from .moorethreads import MooreThreadsGPUStat, MooreThreadsGPUStatCollection
from .nvidia import NvidiaGPUStat, NvidiaGPUStatCollection
from .jetson import JetsonGPUStat, JetsonGPUStatCollection
from .cpu import CPUStat, CPUStatCollection
from .rebellions import RebellionsNPUStat, RebellionsNPUStatCollection
from .furiosa import FuriosaNPUStat, FuriosaNPUStatCollection
from .neuron import NeuronDeviceStat, NeuronDeviceStatCollection
from .tpu import TPUChipStat, TPUChipStatCollection
from .tenstorrent import TenstorrentDeviceStat, TenstorrentDeviceStatCollection

# Union of all per-device stat types — useful for type hints downstream
AnyDeviceStat = Union[
    NvidiaGPUStat, AMDGPUStat, GaudiAIPStat, IntelGPUStat,
    HuaweiNPUStat, HygonDCUStat, CambriconMLUStat, MooreThreadsGPUStat,
    AppleGPUStat, NeuronDeviceStat, TPUChipStat, TenstorrentDeviceStat, JetsonGPUStat,
    CPUStat, RebellionsNPUStat, FuriosaNPUStat,
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
        apple: AppleGPUStatCollection = None,
        neuron: NeuronDeviceStatCollection = None,
        tpu: TPUChipStatCollection = None,
        tenstorrent: TenstorrentDeviceStatCollection = None,
        jetson: JetsonGPUStatCollection = None,
        cpu: CPUStatCollection = None,
        rebellions: RebellionsNPUStatCollection = None,
        furiosa: FuriosaNPUStatCollection = None,
        **plugins: Any,
    ) -> None:
        self.nvidia = nvidia
        self.amd = amd
        self.gaudi = gaudi
        self.intel = intel
        self.huawei = huawei
        self.hygon = hygon
        self.cambricon = cambricon
        self.moorethreads = moorethreads
        self.apple = apple if apple is not None else AppleGPUStatCollection([])
        self.neuron = neuron if neuron is not None else NeuronDeviceStatCollection([])
        self.tpu = tpu if tpu is not None else TPUChipStatCollection([])
        self.tenstorrent = tenstorrent if tenstorrent is not None else TenstorrentDeviceStatCollection([])
        self.jetson = jetson if jetson is not None else JetsonGPUStatCollection([])
        self.rebellions = rebellions if rebellions is not None else RebellionsNPUStatCollection([])
        self.furiosa = furiosa if furiosa is not None else FuriosaNPUStatCollection([])
        #: The host CPU, filled only by ``query_all(include_cpu=True)``; empty otherwise.
        self.cpu = cpu if cpu is not None else CPUStatCollection([])
        # Plugin backends (entry point group ``xpustat.backends``), keyed by vendor.
        self._plugins: dict = {}
        for name, cls in _PLUGIN_BACKENDS.items():
            self._plugins[name] = plugins.get(name) if plugins.get(name) is not None else cls([])
        for name in plugins:
            if name not in self._plugins:
                raise TypeError(f"Unknown backend {name!r}; register it with xpustat.register_backend() first")

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
            + list(self.apple)
            + list(self.neuron)
            + list(self.tpu)
            + list(self.tenstorrent)
            + list(self.jetson)
            + list(self.rebellions)
            + list(self.furiosa)
            + [d for col in self._plugins.values() for d in col]
            + list(self.cpu)
        )

    def __iter__(self) -> Iterator[AnyDeviceStat]:
        """Iterate every device across all vendors in vendor order."""
        return iter(self.all_devices())

    def __len__(self) -> int:
        return len(self.all_devices())

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getattr__(self, name: str):
        # Plugin collections are reachable as attributes like the built-in ones.
        plugins = self.__dict__.get("_plugins", {})
        if name in plugins:
            return plugins[name]
        raise AttributeError(name)

    def __getitem__(self, vendor: str):
        """
        Subscript access by vendor key, e.g. ``stats["nvidia"]``.

        Returns the vendor-specific collection object (iterable, has ``len()``).

        Valid keys: nvidia, amd, gaudi, intel, huawei, hygon, cambricon, moorethreads, apple,
        neuron, tpu, tenstorrent
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
        out = {
            "nvidia":       self.nvidia.to_dict()["gpus"],
            "amd":          self.amd.to_dict()["gpus"],
            "gaudi":        self.gaudi.to_dict()["aips"],
            "intel":        self.intel.to_dict()["gpus"],
            "huawei":       self.huawei.to_dict()["npus"],
            "hygon":        self.hygon.to_dict()["dcus"],
            "cambricon":    self.cambricon.to_dict()["mlus"],
            "moorethreads": self.moorethreads.to_dict()["gpus"],
            "apple":        self.apple.to_dict()["gpus"],
            "neuron":       self.neuron.to_dict()["devices"],
            "tpu":          self.tpu.to_dict()["chips"],
            "tenstorrent":  self.tenstorrent.to_dict()["devices"],
            "jetson":       self.jetson.to_dict()["gpus"],
            "rebellions":   self.rebellions.to_dict()["npus"],
            "furiosa":      self.furiosa.to_dict()["npus"],
        }
        # The CPU is opt-in: its key appears only when it was queried, so the
        # JSON shape of an accelerator-only query is unchanged.
        if len(self.cpu):
            out["cpu"] = self.cpu.to_dict()["cpus"]
        for name, col in self._plugins.items():
            d = col.to_dict()
            # A plugin's to_dict() holds one list under a vendor-chosen key.
            out[name] = next((v for v in d.values() if isinstance(v, list)), []) if isinstance(d, dict) else list(d)
        return out

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
              "xpustat_version": "0.4.0",
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
    ("apple",        AppleGPUStatCollection),
    ("neuron",       NeuronDeviceStatCollection),
    ("tpu",          TPUChipStatCollection),
    ("tenstorrent",  TenstorrentDeviceStatCollection),
    ("jetson",       JetsonGPUStatCollection),
    ("rebellions",   RebellionsNPUStatCollection),
    ("furiosa",      FuriosaNPUStatCollection),
]

#: Backends added at runtime, by :func:`register_backend` or entry points.
_PLUGIN_BACKENDS: dict = {}
_BUILTIN_KEYS: List[str] = [name for name, _ in _BACKENDS]


def register_backend(name: str, collection_cls) -> None:
    """
    Register an extra backend under vendor key *name*.

    *collection_cls* must follow the backend contract: ``collection_cls([])``
    builds an empty collection, ``collection_cls.new_query()`` queries the
    hardware and never raises, instances support ``len()``, iteration,
    indexing and ``to_dict()`` returning one list under a vendor-chosen key,
    and each device subclasses :class:`xpustat.DeviceMixin`.

    Packages can register automatically with an entry point::

        [project.entry-points."xpustat.backends"]
        acme = "xpustat_acme:AcmeStatCollection"
    """
    if name in _BUILTIN_KEYS:
        raise ValueError(f"{name!r} is a built-in backend")
    if not (name.isidentifier() and name.islower()):
        raise ValueError("backend name must be a lowercase identifier")
    _PLUGIN_BACKENDS[name] = collection_cls
    _refresh_registry()


def unregister_backend(name: str) -> None:
    """Remove a backend added with :func:`register_backend`."""
    _PLUGIN_BACKENDS.pop(name, None)
    _refresh_registry()


def _refresh_registry() -> None:
    global _BACKENDS_KEYS, _BACKENDS_MAP
    _BACKENDS_KEYS = _BUILTIN_KEYS + list(_PLUGIN_BACKENDS)
    _BACKENDS_MAP = {**dict(_BACKENDS), **_PLUGIN_BACKENDS, "cpu": CPUStatCollection}


def _load_entry_points() -> None:
    """Register every ``xpustat.backends`` entry point; a broken plugin is skipped."""
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover
        return
    try:
        eps = entry_points()
        group = eps.select(group="xpustat.backends") if hasattr(eps, "select") else eps.get("xpustat.backends", [])
    except Exception:
        return
    for ep in group:
        try:
            register_backend(ep.name, ep.load())
        except Exception as exc:
            if os.environ.get("XPUSTAT_DEBUG"):
                sys.stderr.write(f"xpustat: plugin {ep.name!r} not loaded: {exc}\n")

_BACKENDS_KEYS: List[str] = list(_BUILTIN_KEYS)
_BACKENDS_MAP: dict = {**{name: cls for name, cls in _BACKENDS}, "cpu": CPUStatCollection}
_load_entry_points()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CACHE_LOCK = threading.Lock()
_CACHE: dict = {"key": None, "time": 0.0, "stats": None}


def query_all(parallel: bool = True, vendors: Optional[List[str]] = None,
              max_age: Optional[float] = None, include_cpu: bool = False) -> XPUStatCollection:
    """
    Query every supported accelerator backend and return an XPUStatCollection.

    Backends whose management tool is not installed return an empty collection
    and are never reported as errors.

    Args:
        parallel: Run all backend queries concurrently (default ``True``).
                  Pass ``False`` for sequential execution (useful for debugging).
        vendors: Optional list of vendor keys to query. If provided, only these
                 backends will be queried, and others will be empty.
        max_age: Reuse the previous result if it is at most this many seconds
                 old (same ``vendors``). ``None`` (default) always queries.
                 Meant for exporters and loops that poll faster than the
                 vendor tools can answer.
        include_cpu: Also query the host CPU and memory (``stats.cpu``). Off by
                 default: the CPU is not an accelerator and its first reading
                 in a process samples utilization for 0.1 s.

    Returns:
        :class:`XPUStatCollection` populated with results.
    """
    cache_key = (None if vendors is None else tuple(sorted(vendors)), bool(include_cpu))
    if max_age is not None:
        with _CACHE_LOCK:
            if (_CACHE["stats"] is not None and _CACHE["key"] == cache_key
                    and time.monotonic() - _CACHE["time"] <= max_age):
                return _CACHE["stats"]

    all_backends = list(_BACKENDS) + list(_PLUGIN_BACKENDS.items())
    if include_cpu:
        all_backends.append(("cpu", CPUStatCollection))
    results: dict = {}
    _empty = {name: cls([]) for name, cls in all_backends}

    backends_to_query = all_backends
    if vendors is not None:
        backends_to_query = [(n, c) for n, c in all_backends if n in vendors]

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

    stats = XPUStatCollection(
        nvidia=results["nvidia"],
        amd=results["amd"],
        gaudi=results["gaudi"],
        intel=results["intel"],
        huawei=results["huawei"],
        hygon=results["hygon"],
        cambricon=results["cambricon"],
        moorethreads=results["moorethreads"],
        apple=results["apple"],
        neuron=results["neuron"],
        tpu=results["tpu"],
        tenstorrent=results["tenstorrent"],
        jetson=results["jetson"],
        rebellions=results["rebellions"],
        furiosa=results["furiosa"],
        cpu=results.get("cpu"),
        **{name: results[name] for name in _PLUGIN_BACKENDS},
    )
    with _CACHE_LOCK:
        _CACHE.update(key=cache_key, time=time.monotonic(), stats=stats)
    return stats


def query(vendor: str):
    """
    Query a single vendor and return its collection object.

    Args:
        vendor: One of: ``nvidia``, ``amd``, ``gaudi``, ``intel``,
                ``huawei``, ``hygon``, ``cambricon``, ``moorethreads``, ``apple``,
                ``neuron``, ``tpu``, ``tenstorrent``, ``jetson``, ``rebellions``, ``furiosa``, ``cpu``, or a registered plugin.

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
