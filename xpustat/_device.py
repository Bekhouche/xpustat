"""
_device.py — vendor-neutral helpers shared by every per-device stat object.

Provides:

* :class:`DeviceMixin` — mixed into every ``*Stat`` class. Adds
  ``vendor``, ``device_index``, ``index_label``, ``mem_free_mb``,
  ``framework_index``, ``torch_device``, ``ray_resource`` and
  ``k8s_resource`` without touching the vendor specific fields.
* :func:`visible_indices` — parse the vendor's ``*_VISIBLE_DEVICES``
  environment variable the way the framework runtime does.
* :func:`framework_index` / :func:`torch_device` — map a physical device to
  the index the framework will see, and to a ``torch.device`` string.
* :func:`pick_device` — choose the least-loaded device(s), for scripts that
  hand-roll "which GPU should I use".
* :func:`visible_devices_env` — the inverse: turn chosen devices into the
  ``CUDA_VISIBLE_DEVICES=...`` style assignments that restrict a job to them.

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
    "jetson":       "cuda",
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
    "rebellions":   None,     # optimum-rbln compiles ahead of time; no torch device
    "furiosa":      None,     # furiosa-sdk runtime; no torch device
}

#: Environment variables that restrict which physical devices the framework
#: runtime can see, in the order the runtime consults them. The names follow
#: the accelerator managers in Ray, which set the same variables.
VISIBLE_DEVICES_ENV: Dict[str, Sequence[str]] = {
    "nvidia":       ("CUDA_VISIBLE_DEVICES",),
    "jetson":       ("CUDA_VISIBLE_DEVICES",),
    "amd":          ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"),
    "hygon":        ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"),
    "huawei":       ("ASCEND_RT_VISIBLE_DEVICES",),
    "gaudi":        ("HABANA_VISIBLE_MODULES",),
    "intel":        ("ZE_AFFINITY_MASK",),
    "cambricon":    ("MLU_VISIBLE_DEVICES",),
    "moorethreads": ("MUSA_VISIBLE_DEVICES",),
    "tpu":          ("TPU_VISIBLE_CHIPS",),
    "tenstorrent":  ("TT_VISIBLE_DEVICES",),
    "neuron":       ("NEURON_RT_VISIBLE_CORES",),   # NeuronCore ids, not device ids
    "rebellions":   ("RBLN_DEVICES",),
    "furiosa":      ("FURIOSA_DEVICES",),            # npu:<id> tokens are accepted
    "apple":        (),
}

#: Ray resource name per vendor, as returned by ``get_resource_name()`` of
#: the accelerator managers in ``ray/_private/accelerators``.
RAY_RESOURCE: Dict[str, Optional[str]] = {
    "nvidia":       "GPU",
    "jetson":       "GPU",
    "amd":          "GPU",
    "hygon":        "GPU",
    "intel":        "GPU",
    "apple":        "GPU",
    "huawei":       "NPU",
    "gaudi":        "HPU",
    "tpu":          "TPU",
    "neuron":       "neuron_cores",
    "tenstorrent":  "TTNPU",
    "rebellions":   "RBLN",
    "furiosa":      "FURIOSA",
    "cambricon":    None,
    "moorethreads": None,
}

#: Kubernetes extended-resource name per vendor, from the vendors' device
#: plugin documentation. ``None`` where the name could not be confirmed.
K8S_RESOURCE: Dict[str, Optional[str]] = {
    "nvidia":       "nvidia.com/gpu",
    "jetson":       "nvidia.com/gpu",
    "amd":          "amd.com/gpu",
    "gaudi":        "habana.ai/gaudi",
    "huawei":       "huawei.com/Ascend910",   # 310P cards report huawei.com/Ascend310P
    "neuron":       "aws.amazon.com/neuron",  # per device; aws.amazon.com/neuroncore per core
    "hygon":        None,
    "intel":        None,
    "apple":        None,
    "cambricon":    None,
    "moorethreads": None,
    "tpu":          None,
    "tenstorrent":  None,
    "rebellions":   None,
    "furiosa":      None,
}


# ---------------------------------------------------------------------------
# Visible devices
# ---------------------------------------------------------------------------

def _parse_id_list(raw: str, uuids: Optional[Mapping[str, int]] = None) -> List[int]:
    """
    Parse ``"0,2"``, ``"0-3"`` (Neuron ranges), ``"0.1"`` (Level Zero
    ``device.subdevice``) and UUID entries into a list of ints.
    Unknown entries are skipped, as the runtimes do.
    """
    out: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item and item.split(":", 1)[0].isalpha():
            item = item.split(":", 1)[1]
        head = item.split(".", 1)[0]
        if "-" in head and not head.startswith("-") and all(p.isdigit() for p in head.split("-", 1)):
            lo, hi = (int(p) for p in head.split("-", 1))
            if lo <= hi:
                out.extend(range(lo, hi + 1))
            continue
        if head.lstrip("-").isdigit():
            idx = int(head)
            if idx >= 0:
                out.append(idx)
            continue
        if uuids and item in uuids:
            out.append(uuids[item])
    return out


def visible_indices(
    vendor: str,
    env: Optional[Mapping[str, str]] = None,
    uuids: Optional[Mapping[str, int]] = None,
) -> Optional[List[int]]:
    """
    Return the ordered list of ids the framework can see, or ``None`` when no
    restriction is set (every device is visible).

    For every vendor except Neuron the ids are physical device indices. For
    Neuron they are NeuronCore ids (``NEURON_RT_VISIBLE_CORES``); use
    :func:`framework_index` to map them to devices.

    Args:
        vendor: vendor key, e.g. ``"nvidia"``.
        env:    mapping to read instead of ``os.environ`` (for tests).
        uuids:  optional ``{uuid: physical_index}`` map so that entries such
                as ``GPU-8f1c...`` in ``CUDA_VISIBLE_DEVICES`` resolve.

    An empty variable (``CUDA_VISIBLE_DEVICES=""``) hides every device and
    yields ``[]``.
    """
    env = os.environ if env is None else env
    for var in VISIBLE_DEVICES_ENV.get(vendor, ()):
        if var not in env:
            continue
        raw = env[var].strip()
        if raw == "":
            return []
        return _parse_id_list(raw, uuids)
    return None


def _uuid_map(devices: Iterable[Any]) -> Dict[str, int]:
    m: Dict[str, int] = {}
    for d in devices:
        uuid = getattr(d, "uuid", None)
        if uuid:
            m[str(uuid)] = d.device_index
    return m


def _is_visible(dev: Any, vis: List[int]) -> bool:
    ids = getattr(dev, "_visible_ids", None)
    if ids is not None:                      # e.g. Neuron: visible if any core is
        return any(i in vis for i in ids)
    return dev.device_index in vis


def framework_index(
    dev: Any,
    env: Optional[Mapping[str, str]] = None,
    siblings: Optional[Iterable[Any]] = None,
) -> Optional[int]:
    """
    Index the framework will assign to *dev* (``cuda:N`` → ``N``), after
    applying the vendor's visible-devices environment variable.

    Returns ``None`` when the device is hidden by that variable.
    ``siblings`` lets UUID entries resolve and is needed for vendors whose
    variable lists sub-units (Neuron cores); pass the vendor collection.
    """
    vendor = getattr(dev, "vendor", "unknown")
    sib = list(siblings) if siblings is not None else None
    vis = visible_indices(vendor, env, _uuid_map(sib) if sib else None)
    phys = dev.device_index
    if vis is None:
        return phys
    if getattr(dev, "_visible_ids", None) is not None:
        # Sub-unit variable: the framework index is the device's rank among
        # the visible siblings, in physical order.
        if not _is_visible(dev, vis):
            return None
        pool = sib if sib is not None else [dev]
        visible = sorted((d for d in pool if _is_visible(d, vis)), key=lambda d: d.device_index)
        return next((i for i, d in enumerate(visible) if d.device_index == phys), None)
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


def visible_devices_env(
    devices: Iterable[Any],
    format: str = "index",
) -> Dict[str, str]:
    """
    Build the environment assignments that restrict a job to *devices*.

    Returns ``{VARIABLE: value}`` with one entry per vendor present, e.g.
    ``{"CUDA_VISIBLE_DEVICES": "2,0"}``. Devices are listed in the order
    given, which becomes the framework's device order.

    Args:
        devices: device objects, typically the result of :func:`pick_device`.
        format:  ``"index"`` (default) or ``"uuid"``; UUIDs are used only for
                 devices that have one (NVIDIA), which keeps the selection
                 stable when the enumeration order changes.

    Vendors without a variable (Apple) contribute nothing. For Neuron the
    value lists the devices' NeuronCore ids.
    """
    if format not in ("index", "uuid"):
        raise ValueError("format must be 'index' or 'uuid'")
    out: Dict[str, List[str]] = {}
    for dev in devices:
        vendor = getattr(dev, "vendor", "unknown")
        vars_ = VISIBLE_DEVICES_ENV.get(vendor, ())
        if not vars_:
            continue
        var = vars_[0]
        ids = getattr(dev, "_visible_ids", None)
        if ids is not None:
            entries = [str(i) for i in ids]
        elif format == "uuid" and getattr(dev, "uuid", None):
            entries = [str(dev.uuid)]
        else:
            entries = [str(dev.device_index)]
        bucket = out.setdefault(var, [])
        for e in entries:
            if e not in bucket:
                bucket.append(e)
    return {var: ",".join(vals) for var, vals in out.items()}


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

    @property
    def ray_resource(self) -> Optional[str]:
        """Ray resource name for this vendor, e.g. ``"GPU"`` or ``"NPU"``."""
        return RAY_RESOURCE.get(self.vendor)

    @property
    def k8s_resource(self) -> Optional[str]:
        """Kubernetes extended-resource name, e.g. ``"nvidia.com/gpu"``; ``None`` if unknown."""
        return K8S_RESOURCE.get(self.vendor)


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
