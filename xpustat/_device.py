"""
_device.py — vendor-neutral helpers shared by every per-device stat object.

Provides:

* :class:`DeviceMixin` — mixed into every ``*Stat`` class. Adds
  ``vendor``, ``device_index``, ``index_label``, ``mem_free_mb``,
  ``framework_index`` and ``torch_device`` without touching the vendor
  specific fields.
* :func:`visible_indices` — parse the vendor's ``*_VISIBLE_DEVICES``
  environment variable the way the framework runtime does.
* :func:`framework_index` / :func:`torch_device` — map a physical device to
  the index the framework will see, and to a ``torch.device`` string.
* :func:`pick_device` — choose the least-loaded device(s), for scripts that
  hand-roll "which GPU should I use".

None of this changes existing fields or JSON keys.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

# ---------------------------------------------------------------------------
# Vendor → framework mapping
# ---------------------------------------------------------------------------

#: ``torch.device`` type string per vendor. ROCm and Hygon DTK expose HIP
#: devices as ``cuda`` in PyTorch; Apple is a single un-indexed ``mps`` device.
TORCH_DEVICE_TYPE: Dict[str, Optional[str]] = {
    "nvidia":       "cuda",
    "amd":          "cuda",
    "hygon":        "cuda",
    "apple":        "mps",
    "huawei":       "npu",    # torch_npu
    "gaudi":        "hpu",    # habana_frameworks
    "intel":        "xpu",    # intel_extension_for_pytorch / native xpu
    "cambricon":    "mlu",    # torch_mlu
    "moorethreads": "musa",   # torch_musa
    "neuron":       "xla",    # torch_neuronx
    "tpu":          "xla",    # torch_xla
    "tenstorrent":  None,     # no PyTorch device type (tt-metal / ttnn)
}

#: Environment variables that restrict which physical devices the framework
#: runtime can see, in the order the runtime consults them.
VISIBLE_DEVICES_ENV: Dict[str, Sequence[str]] = {
    "nvidia":       ("CUDA_VISIBLE_DEVICES",),
    "amd":          ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"),
    "hygon":        ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"),
    "huawei":       ("ASCEND_RT_VISIBLE_DEVICES",),
    "gaudi":        ("HABANA_VISIBLE_MODULES",),
    "intel":        ("ZE_AFFINITY_MASK",),
    "cambricon":    ("MLU_VISIBLE_DEVICES",),
    "moorethreads": ("MUSA_VISIBLE_DEVICES",),
    "apple":        (),
    "neuron":       (),
    "tpu":          (),
    "tenstorrent":  (),
}


def visible_indices(
    vendor: str,
    env: Optional[Mapping[str, str]] = None,
    uuids: Optional[Mapping[str, int]] = None,
) -> Optional[List[int]]:
    """
    Return the ordered list of physical device indices the framework can see,
    or ``None`` when no restriction is set (every device is visible).

    Args:
        vendor: vendor key, e.g. ``"nvidia"``.
        env:    mapping to read instead of ``os.environ`` (for tests).
        uuids:  optional ``{uuid: physical_index}`` map so that entries such
                as ``GPU-8f1c...`` in ``CUDA_VISIBLE_DEVICES`` resolve.

    An empty variable (``CUDA_VISIBLE_DEVICES=""``) hides every device and
    yields ``[]``. Entries that cannot be resolved are skipped, matching the
    runtimes' behaviour of ignoring unknown devices.
    """
    env = os.environ if env is None else env
    for var in VISIBLE_DEVICES_ENV.get(vendor, ()):
        if var not in env:
            continue
        raw = env[var].strip()
        if raw == "":
            return []
        out: List[int] = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            # ZE_AFFINITY_MASK uses "device.subdevice" — keep the device part.
            head = item.split(".", 1)[0]
            if head.lstrip("-").isdigit():
                idx = int(head)
                if idx >= 0:
                    out.append(idx)
                continue
            if uuids and item in uuids:
                out.append(uuids[item])
        return out
    return None


def _uuid_map(devices: Iterable[Any]) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for d in devices:
        uuid = getattr(d, "uuid", None)
        if uuid:
            m[str(uuid)] = d.device_index
    return m


def framework_index(
    dev: Any,
    env: Optional[Mapping[str, str]] = None,
    siblings: Optional[Iterable[Any]] = None,
) -> Optional[int]:
    """
    Index the framework will assign to *dev* (``cuda:N`` → ``N``), after
    applying the vendor's visible-devices environment variable.

    Returns ``None`` when the device is hidden by that variable.
    ``siblings`` lets UUID entries resolve; pass the vendor collection.
    """
    vendor = getattr(dev, "vendor", "unknown")
    vis = visible_indices(vendor, env, _uuid_map(siblings) if siblings else None)
    phys = dev.device_index
    if vis is None:
        return phys
    try:
        return vis.index(phys)
    except ValueError:
        return None


def torch_device(
    dev: Any,
    env: Optional[Mapping[str, str]] = None,
    siblings: Optional[Iterable[Any]] = None,
) -> Optional[str]:
    """
    ``torch.device`` string for *dev*, e.g. ``"cuda:1"``, ``"npu:0"``,
    ``"mps"``. ``None`` when the vendor has no PyTorch backend or the device
    is hidden by the visible-devices environment variable.
    """
    kind = TORCH_DEVICE_TYPE.get(getattr(dev, "vendor", "unknown"))
    if kind is None:
        return None
    if kind == "mps":
        return "mps"
    idx = framework_index(dev, env, siblings)
    if idx is None:
        return None
    return f"{kind}:{idx}"


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class DeviceMixin:
    """
    Vendor-neutral view of a device stat object.

    Subclasses set ``vendor`` and ``_index_attr`` (the attribute holding the
    physical index); ``_sub_index_attr`` is set by vendors whose devices are
    addressed as ``card:chip``.
    """

    vendor: str = "unknown"
    _index_attr: str = "gpu_index"
    _sub_index_attr: Optional[str] = None

    # The vendor classes define these; declared here for type checkers.
    mem_used_mb: int
    mem_total_mb: int

    @property
    def device_index(self) -> int:
        """Physical index as the vendor tool reports it."""
        return int(getattr(self, self._index_attr, 0))

    @property
    def index_label(self) -> str:
        """Display form of the index, e.g. ``"0"`` or ``"0:1"`` (card:chip)."""
        if self._sub_index_attr is not None:
            return f"{self.device_index}:{getattr(self, self._sub_index_attr, 0)}"
        return str(self.device_index)

    @property
    def mem_free_mb(self) -> int:
        """
        Free device memory in MB.

        Defaults to ``mem_total_mb - mem_used_mb``. Vendors with a better
        source (Apple unified memory, where "free" is what the OS can still
        hand to the GPU) set ``_mem_free_mb`` explicitly.
        """
        explicit = getattr(self, "_mem_free_mb", None)
        if explicit is not None:
            return int(explicit)
        return max(int(self.mem_total_mb) - int(self.mem_used_mb), 0)

    @property
    def framework_index(self) -> Optional[int]:
        """Index the ML framework sees after ``*_VISIBLE_DEVICES``; ``None`` if hidden."""
        return framework_index(self)

    @property
    def torch_device(self) -> Optional[str]:
        """``torch.device`` string, e.g. ``"cuda:0"``; ``None`` if unsupported/hidden."""
        return torch_device(self)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

_SORT_KEYS = {
    "mem_free":  lambda d: -d.mem_free_mb,
    "mem_used":  lambda d: int(d.mem_used_mb),
    "util":      lambda d: _util_or_zero(d),
    "index":     lambda d: d.device_index,
}


def _util_or_zero(dev: Any) -> int:
    for attr in ("util_pct", "ai_core_pct"):
        v = getattr(dev, attr, None)
        if v is not None:
            return int(v)
    return 0


def pick_device(
    n: int = 1,
    vendor: Union[str, Sequence[str], None] = None,
    by: str = "mem_free",
    stats: Any = None,
    visible_only: bool = True,
    min_free_mb: int = 0,
    env: Optional[Mapping[str, str]] = None,
) -> List[Any]:
    """
    Return the *n* best devices, best first.

    Args:
        n:            how many devices to return (fewer if not enough exist).
        vendor:       restrict to one vendor key or a list of keys.
        by:           ranking: ``"mem_free"`` (default, most free memory
                      first), ``"mem_used"`` (least used), ``"util"``
                      (least busy), ``"index"``.
        stats:        an existing :class:`XPUStatCollection`; queried fresh
                      when omitted.
        visible_only: drop devices hidden by ``CUDA_VISIBLE_DEVICES`` etc.
        min_free_mb:  skip devices with less free memory than this.
        env:          environment mapping override (for tests).

    Example::

        dev = xpustat.pick_device()[0]
        model.to(dev.torch_device)          # "cuda:2"
    """
    if by not in _SORT_KEYS:
        raise ValueError(f"Unknown ranking {by!r}. Valid: {', '.join(_SORT_KEYS)}")
    if stats is None:
        from .xpu import query_all
        stats = query_all()

    vendors: Optional[List[str]] = None
    if vendor is not None:
        vendors = [vendor] if isinstance(vendor, str) else list(vendor)

    candidates: List[Any] = []
    for v in stats.to_dict().keys():
        if vendors is not None and v not in vendors:
            continue
        col = list(stats[v])
        for d in col:
            if visible_only and framework_index(d, env, col) is None:
                continue
            if d.mem_free_mb < min_free_mb:
                continue
            candidates.append(d)

    key = _SORT_KEYS[by]
    # Stable tie-break on physical index so results are deterministic.
    candidates.sort(key=lambda d: (key(d), d.device_index))
    return candidates[:max(n, 0)]
