# Python API

```python
import xpustat

stats = xpustat.query_all()
for dev in stats:
    print(dev.vendor, dev.name, dev.mem_used_mb, "/", dev.mem_total_mb, "MB")
```

## Querying

### `query_all(parallel=True, vendors=None, max_age=None, include_cpu=False)`

Queries every backend and returns an `XPUStatCollection`. With `parallel=True` the backends run in a thread pool; `parallel=False` runs them one after another, which is easier to debug. `vendors` restricts the query to a list of vendor keys; the other vendors come back empty. `max_age` returns the previous result when it is at most that many seconds old and was made with the same `vendors`; it is meant for exporters and loops that poll faster than the vendor tools answer. `include_cpu=True` also reads the host CPU and memory into `stats.cpu`; it is off by default because the CPU is not an accelerator and the first reading in a process samples utilization for 0.1 s.

A backend whose tool is missing, or whose query raises, contributes an empty collection. `query_all` itself does not raise for vendor problems.

### `query(vendor)`

Queries one vendor and returns its collection, for example `NvidiaGPUStatCollection`. Raises `ValueError` for an unknown key.

### `<Vendor>StatCollection.new_query()`

The same, called on the class directly: `xpustat.AppleGPUStatCollection.new_query()`.

## `XPUStatCollection`

| Member | Meaning |
|---|---|
| `stats.nvidia`, `stats.amd`, ... `stats.tenstorrent` | the per-vendor collection |
| `stats.cpu` | the host CPU, one element when queried with `include_cpu=True`, else empty |
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
| `sm_pct`, `mem_pct`, `enc_pct`, `dec_pct` | the process's own compute, memory-bandwidth, encoder and decoder utilization; NVIDIA only, `None` elsewhere |
| `cmdline` | full command line when readable |
| `rss_mb`, `cpu_pct` | host resident memory and CPU usage of the process; `cpu_pct` is measured since the previous query in the same Python process, so the first reading is 0 |
| `container` | `docker:<12 hex>`, `containerd:<id>`, `podman:<id>`, `pod:<uid>` or `slurm:<jobid>` from the process's cgroup; `None` on the bare host or outside Linux |

### Vendor-specific fields

The index attribute is the first entry of each row. These are the keys of `to_dict()` beyond the common ones.

| Vendor | Class | Fields |
|---|---|---|
| nvidia | `NvidiaGPUStat` | `gpu_index`, `uuid`, `cuda_cc`, `util_pct`, `temp_c`, `power_w`, `power_limit_w`, `fan_pct`, `clock_sm_mhz`, `clock_mem_mhz`, `clock_max_sm_mhz`, `enc_pct`, `dec_pct`, `ecc_uncorrected`, `ecc_corrected`, `pstate`, `throttle_reasons` (`idle`, `app_clocks`, `power_cap`, `hw_slowdown`, `sync_boost`, `sw_thermal`, `hw_thermal`, `hw_power_brake`, `display_clocks`), `throttled`, `healthy`, `energy_j` (NVML only), `pcie_gen`, `pcie_width`, `mig_instances` (list of `MIGInstanceStat`: `parent_index`, `uuid`, `gpu_instance_id`, `compute_instance_id`, `profile`, memory, processes; full data on the NVML path, inventory only from `nvidia-smi -L`) |
| jetson | `JetsonGPUStat` | `gpu_index`, `util_pct`, `temp_c`, `power_w`, `power_rail`, `unified_memory` (always true: RAM is shared with the CPU), `engines` (NVENC, NVDEC, NVJPG, DLA, PVA, EMC: frequency, `off`, or load percent), `power_mode` from `nvpmodel` |
| rebellions | `RebellionsNPUStat` | `npu_index`, `uuid`, `device_node`, `bus_id`, `fw_version`, `location`, `util_pct`, `temp_c`, `power_w`, `pstate`, `status`, `healthy`; processes from `contexts` with their allocation and utilization |
| furiosa | `FuriosaNPUStat` | `npu_index`, `arch`, `uuid`, `serial`, `firmware`, `bus_id`, `util_pct` (mean over processing elements), `pe_util_pct`, `temp_c`, `power_w`, `clock_mhz`, `governor`, `liveness`, `healthy` |
| cpu | `CPUStat` | `cpu_index`, `arch`, `cores_physical`, `cores_logical`, `core_layout` (`4P+6E` on Apple Silicon), `sockets`, `numa_nodes`, `freq_mhz`, `freq_max_mhz`, `util_pct`, `per_core_pct`, `load_avg`, `swap_used_mb`, `swap_total_mb`, `temp_c` (Linux sensors), `power_w` (Linux RAPL when readable), `cgroup_cpu_limit`, `cgroup_mem_limit_mb`. Memory is system RAM; `mem_free_mb` is the available figure. Processes are the top five by resident memory, RSS in `gpu_mem_mb`. `torch_device` is `cpu`, `ray_resource` `CPU`, `k8s_resource` `cpu`. |
| amd | `AMDGPUStat` | `gpu_index`, `gfx_target`, `util_pct`, `temp_c`, `power_w`, `power_limit_w`, `fan_pct`, `fan_rpm`, `clock_gfx_mhz`, `clock_max_gfx_mhz`, `clock_mem_mhz`, `mem_util_pct`, `temp_hotspot_c`, `temp_mem_c`, `ecc_uncorrected`, `ecc_corrected`, `throttled`, `healthy`, `perf_level`, `energy_j` (MI series), `pcie_gen`, `pcie_width`, `bdf` |
| gaudi | `GaudiAIPStat` | `aip_index` |
| intel | `IntelGPUStat` | `device_id` |
| huawei | `HuaweiNPUStat` | `npu_id`, `chip_id`, `health`, `temp_c`, `power_w`, `ai_core_pct`, `card_type` |
| hygon | `HygonDCUStat` | `card_index` |
| cambricon | `CambriconMLUStat` | `card_id`, `chip_id` |
| moorethreads | `MooreThreadsGPUStat` | `gpu_id` |
| apple | `AppleGPUStat` | `gpu_index`, `gpu_cores`, `unified_memory`, `util_pct`, `temp_c`, `power_w`, `ane_power_w` (sensors read in-process, no sudo) |
| neuron | `NeuronDeviceStat` | `neuron_device`, `nc_count`, `neuroncore_ids`, `bdf`; `util_pct` is always `None`, `mem_used_mb` is always 0 |
| tpu | `TPUChipStat` | `device_id`, `device_path`, `util_pct` (TensorCore duty cycle), `hbm_bw_util_pct` |
| tenstorrent | `TenstorrentDeviceStat` | `tt_device`, `bus_id`, `board_id`, `temp_c`, `power_w`, `board_power_w`, `aiclk_mhz`; memory fields are always 0 |

## JSON output

`stats.to_dict()` and `xpustat --json` produce one list per vendor key. Every vendor key is present, with an empty list for vendors not found, so consumers can rely on the shape. The one exception is `cpu`, which appears only when the CPU was queried (`include_cpu=True` or `--cpu`), so that accelerator-only output keeps its 0.3.0 shape.

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
| `xpustat.remote.query_hosts`, `query_ssh`, `RemoteHost` | other machines, see [Command line](cli.md#several-machines) |
| `xpustat.doctor.diagnose`, `format_report` | the diagnostics behind `--doctor` |
| `to_prometheus`, `xpustat.prometheus.serve` | [Prometheus](prometheus.md) |
| `__version__` | the version string |
