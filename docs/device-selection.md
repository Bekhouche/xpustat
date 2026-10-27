# Device selection

Training and inference scripts usually contain a hand-written loop that reads `nvidia-smi`, picks the GPU with the most free memory and builds a `cuda:N` string. xpustat provides that loop for every vendor and takes the visible-devices environment variables into account, which is the step the hand-written versions tend to get wrong.

```python
import xpustat

dev = xpustat.pick_device()[0]
model.to(dev.torch_device)          # "cuda:2", "npu:0", "mps", ...
```

## `pick_device(n=1, vendor=None, by="mem_free", stats=None, visible_only=True, min_free_mb=0, env=None)`

Returns a list of the `n` best devices, best first. Fewer than `n` are returned when fewer qualify; an empty list when none does.

| Argument | Meaning |
|---|---|
| `n` | how many devices to return |
| `vendor` | one vendor key, or a list of keys; `None` means every vendor |
| `by` | ranking: `"mem_free"` (most free memory first, the default), `"util"` (least busy), `"mem_used"` (least used), `"index"` |
| `stats` | an existing `XPUStatCollection`; `query_all()` is called when omitted |
| `visible_only` | drop devices hidden by the vendor's visible-devices variable |
| `min_free_mb` | drop devices with less free memory than this |
| `env` | a mapping to read instead of `os.environ`, for tests |

Ties are broken by physical index, so two calls on an idle machine return the same order. A device whose vendor reports no utilization ranks as idle under `by="util"`.

```python
xpustat.pick_device(n=2, by="util")                 # two least-busy devices
xpustat.pick_device(vendor="amd", min_free_mb=8000)
xpustat.pick_device(vendor=["nvidia", "amd"], stats=stats)
```

## Visible devices

Each framework runtime reads an environment variable that restricts and reorders the devices it sees. xpustat parses the same variables so that `framework_index` and `torch_device` match what `torch.cuda.device_count()` and friends will report.

| Vendor | Variables, in the order consulted |
|---|---|
| nvidia | `CUDA_VISIBLE_DEVICES` |
| amd, hygon | `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES` |
| huawei | `ASCEND_RT_VISIBLE_DEVICES` |
| gaudi | `HABANA_VISIBLE_MODULES` |
| intel | `ZE_AFFINITY_MASK` (the device part of `device.subdevice` entries) |
| cambricon | `MLU_VISIBLE_DEVICES` |
| moorethreads | `MUSA_VISIBLE_DEVICES` |
| tpu | `TPU_VISIBLE_CHIPS` |
| tenstorrent | `TT_VISIBLE_DEVICES` |
| neuron | `NEURON_RT_VISIBLE_CORES`, which lists NeuronCore ids (ranges such as `0-3` allowed); a device is visible when any of its cores is |
| rebellions | `RBLN_DEVICES` |
| furiosa | `FURIOSA_DEVICES` (`npu:0` tokens accepted) |
| jetson | `CUDA_VISIBLE_DEVICES` |
| apple | none |

### `visible_indices(vendor, env=None, uuids=None)`

Returns the ordered list of physical indices the runtime can see, or `None` when no variable is set, meaning every device is visible. An empty variable (`CUDA_VISIBLE_DEVICES=""`) returns `[]`. Entries are integers, or, when `uuids` maps them, NVIDIA UUIDs such as `GPU-33e0a3e4-...`. Entries that cannot be resolved are skipped, as the runtimes do.

### `framework_index(dev, env=None, siblings=None)`

The position of `dev` in the visible list, or its physical index when no variable is set, or `None` when the device is hidden. Pass the vendor collection as `siblings` so that UUID entries resolve.

### `torch_device(dev, env=None, siblings=None)`

The `torch.device` string built from the vendor's device type and `framework_index`.

| Vendor | Device type | Backend package |
|---|---|---|
| nvidia, amd, hygon | `cuda` | PyTorch; ROCm and DTK builds expose HIP devices as `cuda` |
| apple | `mps` | PyTorch; no index |
| huawei | `npu` | `torch_npu` |
| gaudi | `hpu` | `habana_frameworks` |
| intel | `xpu` | PyTorch 2.4 and later, or `intel_extension_for_pytorch` |
| cambricon | `mlu` | `torch_mlu` |
| moorethreads | `musa` | `torch_musa` |
| neuron, tpu | `xla` | `torch_neuronx`, `torch_xla` |
| tenstorrent | none | no PyTorch device type; `torch_device` is `None` |

The mapping is the dictionary `xpustat.TORCH_DEVICE_TYPE`; the variables are `xpustat.VISIBLE_DEVICES_ENV`.

### `visible_devices_env(devices, format="index")`

The inverse of the two functions above: given chosen devices, return the assignments that restrict a job to them, one entry per vendor present, e.g. `{"CUDA_VISIBLE_DEVICES": "2,0"}`. The order of `devices` becomes the framework's device order. With `format="uuid"`, devices that have a UUID (NVIDIA GPUs and MIG instances) are listed by it, which survives a change in enumeration order. Vendors without a variable contribute nothing; for Neuron the value lists the devices' NeuronCore ids. The CLI's `--select` prints this mapping.

```python
chosen = xpustat.pick_device(n=2)
os.environ.update(xpustat.visible_devices_env(chosen))   # before importing torch
```

## Scheduler resource names

Every device carries `ray_resource` and `k8s_resource` so a scheduler built on xpustat can emit the right request. The Ray names come from the accelerator managers in Ray's source; the Kubernetes names from the vendors' device-plugin documentation, and are `None` where that could not be confirmed.

| Vendor | Ray | Kubernetes |
|---|---|---|
| nvidia, jetson | `GPU` | `nvidia.com/gpu` |
| amd | `GPU` | `amd.com/gpu` |
| hygon, intel, apple | `GPU` | unconfirmed |
| huawei | `NPU` | `huawei.com/Ascend910` (310P cards report `huawei.com/Ascend310P`) |
| gaudi | `HPU` | `habana.ai/gaudi` |
| tpu | `TPU` | unconfirmed |
| neuron | `neuron_cores` | `aws.amazon.com/neuron` per device (`aws.amazon.com/neuroncore` per core) |
| tenstorrent | `TTNPU` | unconfirmed |
| rebellions | `RBLN` | unconfirmed |
| furiosa | `FURIOSA` | unconfirmed |
| cambricon, moorethreads | none in Ray | unconfirmed |

## Verified behaviour

On a host with seven NVIDIA L4 GPUs and `CUDA_VISIBLE_DEVICES` set to the UUID of GPU 5 followed by `3`, `pick_device(n=5)` returned GPU 3 as `cuda:1` and GPU 5 as `cuda:0` and nothing else, which is the order CUDA assigns. On an AMD host with a Radeon AI PRO R9700 and an integrated GPU, `HIP_VISIBLE_DEVICES=1` made the integrated GPU `cuda:0` and hid the R9700.

## Limitations

- `pick_device` ranks by what the vendor tool reports at the moment of the call. It does not reserve the device; two processes starting at the same time can pick the same one.
- MIG instances are nested under their parent GPU (`gpu.mig_instances`) and are not candidates for `pick_device`; target one with `visible_devices_env([instance], format="uuid")`.
- The `xla` index for TPU and Neuron is the device index as reported by `tpu-info` and `neuron-ls`; it is not checked against the XLA runtime's own ordering.
