# Python API

```python
import xpustat

stats = xpustat.query_all()
for dev in stats:
    print(dev.vendor, dev.name, dev.mem_used_mb, "/", dev.mem_total_mb, "MB")
```

## Querying

### `query_all(parallel=True, vendors=None, max_age=None)`

Queries every backend and returns an `XPUStatCollection`. With `parallel=True` the backends run in a thread pool; `parallel=False` runs them one after another, which is easier to debug. `vendors` restricts the query to a list of vendor keys; the other vendors come back empty. `max_age` returns the previous result when it is at most that many seconds old and was made with the same `vendors`; it is meant for exporters and loops that poll faster than the vendor tools answer.

A backend whose tool is missing, or whose query raises, contributes an empty collection. `query_all` itself does not raise for vendor problems.

### `query(vendor)`

Queries one vendor and returns its collection, for example `NvidiaGPUStatCollection`. Raises `ValueError` for an unknown key.

### `<Vendor>StatCollection.new_query()`

The same, called on the class directly: `xpustat.AppleGPUStatCollection.new_query()`.

## `XPUStatCollection`

| Member | Meaning |
|---|---|
| `stats.nvidia`, `stats.amd`, ... `stats.tenstorrent` | the per-vendor collection |
| `stats["amd"]` | the same by key; `KeyError` for an unknown key |
| `for dev in stats` | every device, in vendor order |
| `len(stats)`, `bool(stats)` | total device count |
| `stats.all_devices()` | the flat list |
| `stats.to_dict()` | JSON-ready dict keyed by vendor |
| `stats.to_json(indent=2, metadata=False)` | the dict as a string; with `metadata=True` it is wrapped with `hostname`, `query_time` and `xpustat_version` |

The constructor takes the eight original vendor collections as positional arguments and those added later (`apple`, `neuron`, `tpu`, `tenstorrent`, `jetson`, and any registered plugin) as keyword arguments that default to empty, so code that built a collection by hand for version 0.2 keeps working.

## Per-vendor collections

Each vendor collection is a list-like object: `len()`, iteration and indexing (`stats.nvidia[0]`) work, and `to_dict()` returns the device list under a vendor-specific key (`gpus`, `npus`, `aips`, `dcus`, `mlus`, `devices`, `chips`). NVIDIA, AMD and Apple collections carry a `backend` attribute naming the path that produced them: `"pynvml"`, `"amdsmi"`, `"iokit"`, `"subprocess"` or `"none"`.

## Device objects

### Common fields

Every device object has these, whatever the vendor. They are provided by `DeviceMixin`.

| Field | Type | Meaning |
|---|---|---|
| `vendor` | `str` | vendor key, e.g. `"nvidia"` |
| `name` | `str` | device name as the vendor tool reports it |
| `device_index` | `int` | physical index as the vendor tool reports it |
| `index_label` | `str` | display form of the index; `"0"` or `"0:1"` for vendors addressed as card:chip (Huawei, Cambricon) |
| `mem_used_mb` | `int` | memory in use, MB |
| `mem_total_mb` | `int` | total memory, MB |
| `mem_free_mb` | `int` | free memory, MB; `mem_total_mb - mem_used_mb` unless the vendor provides a better number (Apple does) |
| `processes` | `list[ProcessInfo]` | processes using the device; empty when the vendor tool does not report them |
| `framework_index` | `int` or `None` | the index the ML framework will see after `*_VISIBLE_DEVICES`; `None` when hidden |
| `torch_device` | `str` or `None` | `torch.device` string, e.g. `"cuda:1"`; `None` when the vendor has no PyTorch backend or the device is hidden |
| `ray_resource` | `str` or `None` | Ray resource name, e.g. `"GPU"`, `"NPU"`, `"neuron_cores"` |
| `k8s_resource` | `str` or `None` | Kubernetes extended-resource name, e.g. `"nvidia.com/gpu"`; `None` where unconfirmed |
| `to_dict()` | `dict` | JSON-ready representation |

`util_pct`, `temp_c` and `power_w` are present on most vendors but not all; read them with `getattr(dev, "util_pct", None)` in vendor-neutral code. Huawei names its utilization `ai_core_pct`.

### `ProcessInfo`

| Field | Meaning |
|---|---|
| `pid` | process id |
| `username` | owner, `"?"` when the process has exited or is not readable |
| `command` | executable basename from `psutil`, or the vendor tool's name for the process when `psutil` cannot read it |
| `gpu_mem_mb` | device memory used by that process, `None` when not reported |

### Vendor-specific fields

The index attribute is the first entry of each row. These are the keys of `to_dict()` beyond the common ones.

| Vendor | Class | Fields |
|---|---|---|
| nvidia | `NvidiaGPUStat` | `gpu_index`, `uuid`, `cuda_cc`, `util_pct`, `temp_c`, `power_w`, `power_limit_w`, `mig_instances` (list of `MIGInstanceStat`: `parent_index`, `uuid`, `gpu_instance_id`, `compute_instance_id`, `profile`, memory, processes; NVML path only) |
| jetson | `JetsonGPUStat` | `gpu_index`, `util_pct`, `temp_c`, `power_w`, `power_rail`, `unified_memory` (always true: RAM is shared with the CPU) |
| amd | `AMDGPUStat` | `gpu_index`, `gfx_target`, `util_pct`, `temp_c`, `power_w` |
| gaudi | `GaudiAIPStat` | `aip_index` |
| intel | `IntelGPUStat` | `device_id` |
| huawei | `HuaweiNPUStat` | `npu_id`, `chip_id`, `health`, `temp_c`, `power_w`, `ai_core_pct`, `card_type` |
| hygon | `HygonDCUStat` | `card_index` |
| cambricon | `CambriconMLUStat` | `card_id`, `chip_id` |
| moorethreads | `MooreThreadsGPUStat` | `gpu_id` |
| apple | `AppleGPUStat` | `gpu_index`, `gpu_cores`, `unified_memory`, `util_pct`; `temp_c` and `power_w` are always `None` |
| neuron | `NeuronDeviceStat` | `neuron_device`, `nc_count`, `neuroncore_ids`, `bdf`; `util_pct` is always `None`, `mem_used_mb` is always 0 |
| tpu | `TPUChipStat` | `device_id`, `device_path`, `util_pct` (TensorCore duty cycle), `hbm_bw_util_pct` |
| tenstorrent | `TenstorrentDeviceStat` | `tt_device`, `bus_id`, `board_id`, `temp_c`, `power_w`, `board_power_w`, `aiclk_mhz`; memory fields are always 0 |

## JSON output

`stats.to_dict()` and `xpustat --json` produce one list per vendor key. Every vendor key is present, with an empty list for vendors not found, so consumers can rely on the shape.

```json
{
  "nvidia": [
    {
      "gpu_index": 0,
      "uuid": "GPU-33e0a3e4-1cd8-aa3c-bbfc-2718461c2a32",
      "name": "NVIDIA L4",
      "cuda_cc": "8.9",
      "mem_used_mb": 5887,
      "mem_total_mb": 23034,
      "mem_free_mb": 17147,
      "util_pct": 34,
      "temp_c": 61,
      "power_w": 47.7,
      "power_limit_w": 72.0,
      "processes": [
        {"pid": 493828, "username": "salah", "command": "python", "gpu_mem_mb": 5864}
      ]
    }
  ],
  "amd": [],
  "gaudi": [],
  "intel": [],
  "huawei": [],
  "hygon": [],
  "cambricon": [],
  "moorethreads": [],
  "apple": [],
  "neuron": [],
  "tpu": [],
  "tenstorrent": []
}
```

The example is assembled from a query on a host with L4 GPUs; the numbers are illustrative. `mem_free_mb` was added in 0.3.0; the other keys are unchanged from 0.2.0.

## Other exports

| Name | Page |
|---|---|
| `pick_device`, `visible_devices_env`, `visible_indices`, `framework_index`, `torch_device`, `TORCH_DEVICE_TYPE`, `VISIBLE_DEVICES_ENV`, `RAY_RESOURCE`, `K8S_RESOURCE` | [Device selection](device-selection.md) |
| `register_backend`, `unregister_backend`, `DeviceMixin` | [Contributing](contributing.md#plugin-backends) |
| `to_prometheus`, `xpustat.prometheus.serve` | [Prometheus](prometheus.md) |
| `__version__` | the version string |
