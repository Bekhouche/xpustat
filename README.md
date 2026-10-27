# xpustat

[![GitHub](https://img.shields.io/badge/GitHub-Bekhouche%2Fxpustat-blue?logo=github)](https://github.com/Bekhouche/xpustat)
**Author:** Salah Eddine Bekhouche

A unified accelerator stat tool: `gpustat`, but for every vendor.

## Supported backends

| Status | Vendor | Device | CLI tool | Filter key |
|:---:|---|---|---|---|
| ✅ Tested | AMD | Radeon RX 7900 XTX (gfx1100) | `amd-smi` | `amd` |
| ✅ Tested | NVIDIA | NVIDIA L4 (cc8.9), TITAN V (cc7.0) | `nvidia-smi` | `nvidia` |
| 🔲 Untested | Intel | Gaudi / Gaudi 2 / Gaudi 3 | `hl-smi` | `gaudi` |
| 🔲 Untested | Intel | Arc / Data Center GPU | `xpu-smi` | `intel` |
| ✅ Tested | Huawei | Ascend 910B2 | `ascend-dmi` / `npu-smi` | `huawei` |
| 🔲 Untested | Hygon | DCU | `hy-smi` | `hygon` |
| 🔲 Untested | Cambricon | MLU | `cnmon` | `cambricon` |
| 🔲 Untested | Moore Threads | MTT GPU | `mthreads-gmi` | `moorethreads` |
| ✅ Tested | Apple | Apple M5 (10-core GPU) | IOKit (built into macOS) | `apple` |
| 🔲 Untested | AWS | Neuron (Inferentia / Trainium) | `neuron-ls` | `neuron` |
| 🔲 Untested | Google | Cloud TPU (v4 – v6e) | `tpu-info` | `tpu` |
| 🔲 Untested | Tenstorrent | Wormhole / Blackhole | `tt-smi` | `tenstorrent` |
| 🔲 Untested | NVIDIA | Jetson Nano / Xavier / Orin | `tegrastats` | `jetson` |
| 🔲 Untested | Rebellions | ATOM / REBEL | `rbln-stat --json` | `rebellions` |
| 🔲 Untested | FuriosaAI | RNGD / Warboy | `furiosa-smi` | `furiosa` |

> **Legend:** ✅ tested on real hardware; 🔲 implemented, awaiting a hardware test
>
> Backends whose management tool is not installed are silently skipped.
>
> **Apple Silicon note:** the GPU shares unified memory with the CPU, so `mem_total_mb` is the
> machine's total RAM and `mem_used_mb` is the slice currently held by the GPU driver.
> GPU temperature and power are read in-process from the SMC and IOReport counters, no sudo needed;
> per-process GPU usage stays root-only on macOS. Discrete GPUs in Intel Macs report real VRAM.
>
> **Plugins:** any package can add a vendor through the `xpustat.backends` entry point, or at runtime with `xpustat.register_backend()`; see [docs/contributing.md](docs/contributing.md).
>
> **Neuron / Tenstorrent note:** `neuron-ls` and `tt-smi` report inventory, processes and telemetry
> but not live memory usage, so those backends show memory as `0` / `N/A`.

### Running Vendor Hardware Tests

If you have access to specific hardware, you can help us verify the implementation by running the standalone test scripts found in the `vendor_devices_tests/` directory:

```bash
# Example: Testing the Huawei backend on a machine with Ascend NPUs
python vendor_devices_tests/test_huawei.py

# Example: Testing the NVIDIA backend
python vendor_devices_tests/test_nvidia.py
```
If the test successfully outputs your hardware metrics, please let us know so we can update the backend status to **✅ Tested**!

## Documentation

The full documentation is in [`docs/`](docs/index.md): [installation](docs/installation.md), [command line](docs/cli.md), [Python API](docs/python-api.md), [device selection](docs/device-selection.md), [Prometheus](docs/prometheus.md), [backends](docs/backends.md) and [contributing](docs/contributing.md). `mkdocs serve` renders it as a site.

## Install

**pip**
```bash
pip install xpustat
```

**conda**
```bash
conda install -c conda-forge xpustat
```

## CLI usage

```
xpustat                      # coloured table of all detected devices
xpustat -pui                 # watch mode, showing processes (-p), users (-u), refreshing every 1.0s (-i)
xpustat -pui --cpu           # same, plus the host CPU and memory (opt-in; also for --json and --prometheus)
xpustat -i 2                 # watch mode, refresh every 2 seconds
xpustat -i 1 -n 10           # 10 refreshes then exit
xpustat -a                   # every column: util, temp, power, arch, fan, clocks, codec, health, user, pid, cmd, container
xpustat -H -C -F             # health (P-state, throttling, ECC), clocks, fan
xpustat -puck --full-cmd     # processes with container/job and full command line
xpustat --id 0,2 -f nvidia   # only GPUs 0 and 2
xpustat -f nvidia amd        # show only NVIDIA and AMD
xpustat --no-color           # plain text, no ANSI codes
xpustat --json               # raw JSON output
export $(xpustat --select 2) # restrict a job to the 2 freest devices (CUDA_VISIBLE_DEVICES=..., HIP_..., ...)
xpustat --doctor             # why is a vendor missing? tools, bindings, env vars, errors
xpustat --hosts http://n1:9400 http://n2:9400   # cluster view over the exporters' /json
xpustat --ssh n1 n2 -a       # agentless cluster view over ssh
xpustat --record run.ndjson.gz -i 1 -n 60; xpustat --replay run.ndjson.gz -i 0.2
xpustat --prometheus         # Prometheus text format, once
xpustat --serve-prometheus 9400   # serve http://0.0.0.0:9400/metrics (fresh query per scrape)
xpustat --version
```

Example output:

```
xpustat 0.1.0  ·  myhost  ·  Fri May 22  11:24:00  2026
──────────────────────────────────────────────────────────────────────────
[NVIDIA] #0    Tesla V100                          │   1,024 / 32,768 MB   ███░░░░░░░░░░░░░    3%
[AMD   ] #0    Radeon RX 7900 XTX                  │      26 / 24,560 MB   ░░░░░░░░░░░░░░░░    0%
[APPLE ] #0    Apple M5                            │     375 / 32,768 MB   ░░░░░░░░░░░░░░░░    1%
[HUAWEI] #0:0  Ascend 310                          │   2,703 /  8,192 MB   █████░░░░░░░░░░░   33%
```

Example output with processes, users, and watch mode (`xpustat -pui`):

```
xpustat 0.1.0  ·  salah-amd  ·  Fri May 29  10:07:57  2026
──────────────────────────────────────────────────────────
[AMD   ] #0    Radeon RX 7900 XTX              │  17,918 / 24,560 MB  ████████████░░░░   72%
              └─ salah/348504(17480M)
  refreshing every 1.0s  ·  Ctrl+C to quit
```

## Python API

```python
import xpustat

# Query everything at once (parallel by default)
stats = xpustat.query_all()

print(stats)                      # XPUStatCollection(nvidia=0, amd=1, ...)
print(stats.amd[0].name)          # "Radeon RX 7900 XTX"
print(stats.amd[0].mem_used_mb)   # 26

# As a dict (JSON-serialisable)
import json
print(json.dumps(stats.to_dict(), indent=2))

# Query a single vendor
from xpustat import NvidiaGPUStatCollection
gpus = NvidiaGPUStatCollection.new_query()

# Host CPU and memory, on request (cores, frequency, load, RAM, swap, cgroup limits, top processes)
cpu = xpustat.query_all(include_cpu=True).cpu[0]
print(cpu.name, cpu.core_layout or cpu.cores_logical, cpu.mem_free_mb)
```

### Picking a device for training / inference

Every device object carries vendor-neutral fields on top of the vendor ones:

| Field | Meaning |
|---|---|
| `vendor` | vendor key, e.g. `"nvidia"` |
| `device_index` | physical index as the vendor tool reports it |
| `mem_free_mb` | free memory (for Apple unified memory: what the OS can still give the GPU) |
| `framework_index` | index the ML framework will see after `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` / `ASCEND_RT_VISIBLE_DEVICES` / … (`None` if hidden) |
| `torch_device` | `torch.device` string: `cuda:N` (NVIDIA, AMD, Hygon), `mps`, `npu:N`, `hpu:N`, `xpu:N`, `mlu:N`, `musa:N`, `xla:N` |
| `ray_resource`, `k8s_resource` | the Ray resource name (`GPU`, `NPU`, `TPU`, `neuron_cores`, ...) and the Kubernetes extended-resource name (`nvidia.com/gpu`, `amd.com/gpu`, ...) |

```python
import xpustat

dev = xpustat.pick_device()[0]            # most free memory first, honours *_VISIBLE_DEVICES
model.to(dev.torch_device)                # "cuda:2", "npu:0", "mps", ...

xpustat.pick_device(n=2, by="util")       # two least-busy devices
xpustat.pick_device(vendor="amd", min_free_mb=8000)
xpustat.visible_indices("nvidia")         # parsed CUDA_VISIBLE_DEVICES, or None when unset
xpustat.visible_devices_env(devs)         # {"CUDA_VISIBLE_DEVICES": "2,0"} for a chosen list
xpustat.query_all(max_age=1.0)            # reuse a result younger than 1 s
```

### Prometheus / Grafana

```bash
xpustat --serve-prometheus 9400           # then scrape http://host:9400/metrics
```

Exposes `xpustat_memory_{used,total,free}_bytes`, `xpustat_utilization_percent`,
`xpustat_temperature_celsius`, `xpustat_power_watts`, `xpustat_process_memory_bytes` and
`xpustat_device_info` (with `arch` and `torch_device` labels) for every vendor from one process.
`xpustat.to_prometheus(stats)` returns the same text for embedding in your own exporter.

## Acknowledgements

Hichem Telli tested some devices and reported the results.

## Version history

### 0.5.0
- Host CPU and memory as an opt-in device: `xpustat --cpu`, `query_all(include_cpu=True)`, `stats.cpu`; the `cpu` JSON key appears only when asked, so existing output is unchanged
- Per-process compute utilization (`sm_pct`, from NVML or `nvidia-smi pmon`), full command line, host RSS and CPU, and container or job attribution (`docker`, `containerd`, `podman`, Kubernetes pod, SLURM job) on every process
- NVIDIA and AMD: fan, SM/graphics and memory clocks, encoder/decoder load, ECC counts, P-state or perf level, throttle reasons, PCIe link, power limit and energy (NVML, MI series); `throttled` and `healthy` on every device
- New columns `-F` fan, `-C` clocks, `-E` codec, `-H` health, `-k` container, plus `--full-cmd`, `--id 0,2` and `--print-completion bash|zsh|fish`
- MIG inventory on the `nvidia-smi` path from `nvidia-smi -L`; Jetson engines (NVENC, NVDEC, DLA, ...) and the `nvpmodel` power mode
- Prometheus: fan, clocks, codec, ECC, throttled, healthy, P-state, energy, per-process SM and RSS, `container` label
- The optional `amdsmi` and `nvidia-ml-py` bindings are skipped when they fail to load against the installed driver, instead of breaking `import xpustat`
- New backends: Rebellions (`rbln-stat --json`) and FuriosaAI (`furiosa-smi info/status/ps`), with `RBLN_DEVICES` and `FURIOSA_DEVICES` honoured; written from the tools' JSON schemas, awaiting a hardware run
- Sole author: Salah Eddine Bekhouche; Hichem Telli is acknowledged for testing the Ascend backend
- Apple Silicon GPU temperature, GPU power and ANE power without sudo (SMC and IOReport, in-process); the CPU row gets temperature and power on macOS too
- `xpustat --doctor` explains every backend: tool path, probe answer, native binding, env vars, device count, errors
- Cluster view: `--hosts` reads other exporters' new `/json` endpoint, `--ssh` runs xpustat over ssh; both render one table with a host column and support `--json` and `-i`
- `--record FILE` writes one JSON frame per line (gzip optional); `--replay FILE` renders a recording
- Windows: verified natively on Windows 11 (CPU row, JSON, doctor); the CPU model comes from the registry and the console is switched to UTF-8 for the table glyphs

### 0.4.0
- NVIDIA Jetson backend (`jetson`) via `tegrastats`; MIG instances on the NVML path (`gpu.mig_instances`)
- `xpustat --select N` prints the vendor's visible-devices assignment; `visible_devices_env()` is the library form
- `TPU_VISIBLE_CHIPS`, `TT_VISIBLE_DEVICES` and `NEURON_RT_VISIBLE_CORES` (core ranges) are honoured
- `ray_resource` and `k8s_resource` on every device
- Plugin backends through the `xpustat.backends` entry point and `register_backend()`
- `query_all(max_age=...)` and `--cache-ttl` for the Prometheus exporter

### 0.3.0
- Apple Silicon GPU support (`apple` backend) read in-process from IOKit (~2 ms, `ioreg` fallback), no root required; tested on Apple M5
- `-A` shows the GPU core count for Apple devices (e.g. `10-core`)
- Fix: vendor collections now support index access (`stats.amd[0]`), as documented
- Faster subprocess fallbacks: the `nvidia-smi` and `amd-smi` invocations now run concurrently
- Watch mode no longer swallows errors silently
- Fix: process lines now render as `user:command/pid` instead of running user and command together
- The `nvidia` / `fast` extras now pull `nvidia-ml-py` (the official NVML bindings) instead of the deprecated `pynvml` package
- New backends: AWS Neuron (`neuron-ls`), Google Cloud TPU (`tpu-info`), Tenstorrent (`tt-smi`)
- Vendor-neutral device fields: `vendor`, `device_index`, `mem_free_mb`, `framework_index`, `torch_device`
- `xpustat.pick_device()` picks the least-loaded device(s), honouring `*_VISIBLE_DEVICES`
- Prometheus export: `--prometheus`, `--serve-prometheus PORT`, and `xpustat.to_prometheus()`

### 0.2.0
- Fix AMD GPUs not being detected when `amd-smi` reports `"N/A"` sections or idle GPUs (e.g. ROCm 10 with an iGPU present)
- Flicker-free watch mode: frames are redrawn in place on the alternate screen

### 0.1.0
- Initial release
- Support for NVIDIA, AMD, Intel Gaudi, Intel Arc/GPU, Huawei Ascend, Hygon DCU, Cambricon MLU, Moore Threads
