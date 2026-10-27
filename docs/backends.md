# Backends

One section per vendor: the command xpustat runs, the fields it fills, what it cannot report, and whether it has been run on real hardware. Every backend follows the same contract: `new_query()` returns an empty collection when the tool is absent or fails, and never raises.

Status legend: **tested** means run on the named hardware by the authors; **untested** means implemented from the tool's documented output and waiting for a hardware report. Run the matching script in `vendor_devices_tests/` and open an issue with its output to move a backend to tested.

## NVIDIA

| | |
|---|---|
| Key | `nvidia` |
| Status | tested on NVIDIA L4 (compute capability 8.9) and TITAN V (7.0) |
| Fast path | `nvidia-ml-py` (NVML) when importable |
| Fallback | `nvidia-smi --query-gpu=index,uuid,name,compute_cap,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits` and `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits`, run concurrently |
| Fields | `uuid`, `cuda_cc`, `util_pct`, `temp_c`, `power_w`, `power_limit_w`, fan, SM and memory clocks, encoder/decoder load, ECC counts, P-state, throttle reasons, PCIe link, energy (NVML only); per process: memory, SM/memory/codec utilization, command line, RSS, container |

On a seven-GPU host, the two `nvidia-smi` calls take about 154 ms and 97 ms; running them concurrently brought the query from 246 ms to 172 ms. The NVML path took 123 ms on the same host. The extended fields come from a third `nvidia-smi --query-gpu` call kept separate from the base one, so a driver that rejects a newer field loses only those columns, and per-process utilization from `nvidia-smi pmon -c 1`; both run in the same concurrent batch. On the NVML path, `nvmlDeviceGetProcessUtilization` returns samples only for recently active processes, so `sm_pct` can be `None` for an idle one.

On the NVML path, a GPU in MIG mode also lists its compute instances in `mig_instances`, each with its `MIG-` UUID, GPU-instance and compute-instance ids, memory and processes. The `nvidia-smi` path lists them from `nvidia-smi -L` with the profile name and UUID only; memory used is 0 and the total is read from the profile name. This was written from the NVML API and has not been run on a MIG-enabled GPU; the L4 used for testing does not support MIG.

## NVIDIA Jetson

| | |
|---|---|
| Key | `jetson` |
| Status | untested; parsers written from published `tegrastats` lines of Nano, Xavier and Orin |
| Command | `tegrastats --interval 200`, first line, then stopped |
| Fields | shared RAM used/total, `util_pct` from `GR3D_FREQ`, `temp_c` from `gpu@..C`, `power_w` from the first of `VDD_GPU_SOC`, `VDD_CPU_GPU_CV`, `POM_5V_GPU`, `GPU` present, recorded in `power_rail`; `engines` from the NVENC, NVDEC, NVJPG, DLA, PVA and EMC tokens; `power_mode` from `nvpmodel -q` |

The module name is read from `/proc/device-tree/model`. The backend only runs on a host with `/etc/nv_tegra_release` or a Jetson model string, and never when `nvidia-smi` exists, so a GPU is listed once. There is no per-process memory. On Orin Nano the GPU shares its rail with the CPU, so `power_w` is the combined figure and `power_rail` says so.

## AMD

| | |
|---|---|
| Key | `amd` |
| Status | tested on Radeon RX 7900 XTX (gfx1100) and Radeon AI PRO R9700 (gfx1201) with amd-smi 27.0.0 on ROCm 10.0.0 |
| Fast path | `amdsmi` Python bindings when importable |
| Fallback | `amd-smi static --json`, `amd-smi metric --json` and `amd-smi process --json`, run concurrently |
| Fields | `gfx_target`, `util_pct`, `temp_c` (edge, else hotspot), `power_w` (socket power), fan percent and rpm, graphics and memory clocks, memory-controller activity, hotspot and VRAM temperatures, ECC counts, `throttled` from `power.throttle_status`, perf level, PCIe generation and width, power limit, energy on cards that report it, BDF; per-process VRAM, command line, RSS, container |

Each `amd-smi` call takes about 150 to 165 ms on ROCm 10; concurrent execution brought the query from 460 ms to 167 ms. Sections that `amd-smi` reports as `"N/A"` (for example `usage` on an integrated GPU) are tolerated, which is why an APU shows `N/A` for utilization rather than being dropped.

## Intel Gaudi

| | |
|---|---|
| Key | `gaudi` |
| Status | untested |
| Command | `hl-smi -Q index,name,memory.used,memory.total -f csv,noheader,nounits` |
| Fields | memory only |

## Intel Arc and Data Center GPU

| | |
|---|---|
| Key | `intel` |
| Status | untested |
| Command | `xpu-smi discovery -j` to list devices, then one sample of metric 18 per device with `xpu-smi dump -d <id> -m 18 -n 1`, and `xpu-smi discovery -d <id>` for the total memory |
| Fields | memory only |

Intel released XPU Manager 2.0 in 2026 with the same `xpu-smi` command name; the backend was written against the 1.x output and has not been checked against 2.0.

## Huawei Ascend

| | |
|---|---|
| Key | `huawei` |
| Status | tested on Ascend 910B2 |
| Preferred | `ascend-dmi -i --format json` when `ascend-dmi` is present |
| Fallback | `npu-smi info`, then `npu-smi info -t <type> -i <id>` per NPU for the detail tables |
| Fields | `npu_id`, `chip_id`, `health`, `temp_c`, `power_w`, `ai_core_pct`, `card_type` |

Devices are addressed as `npu:chip` in the CLI and in `index_label`. A `health` value other than `OK` is shown in red next to the device line.

## Hygon DCU

| | |
|---|---|
| Key | `hygon` |
| Status | untested |
| Command | `hy-smi --showproductname --showmeminfo vram --json`; several candidate paths are searched when `hy-smi` is not on `PATH` |
| Fields | memory only |

## Cambricon MLU

| | |
|---|---|
| Key | `cambricon` |
| Status | untested |
| Command | `cnmon info` (also looked for at `/usr/local/neuware/bin/cnmon`) |
| Fields | `card_id`, `chip_id`, memory |

## Moore Threads

| | |
|---|---|
| Key | `moorethreads` |
| Status | untested |
| Command | `mthreads-gmi` |
| Fields | memory, parsed from the `UsedMiB(TotalMiB)` column |

## Apple

| | |
|---|---|
| Key | `apple` |
| Status | tested on Apple M5 (10-core GPU), macOS 26 |
| Fast path | IOKit through `ctypes`: `IOServiceGetMatchingServices` on the `IOAccelerator` class, properties read with `IORegistryEntryCreateCFProperties` |
| Fallback | `ioreg -r -c IOAccelerator -a -d 1` |
| Fields | `gpu_cores`, `unified_memory`, `util_pct` (`Device Utilization %`), `temp_c` (SMC), `power_w` and `ane_power_w` (IOReport) |

The in-process read takes about 2 ms against 25 ms for `ioreg`; a loop of 5,000 queries showed no memory growth, so the CoreFoundation objects are released correctly. Apple Silicon shares memory with the CPU: `mem_total_mb` is the machine's RAM, `mem_used_mb` is the `In use system memory` figure of the GPU driver, and `mem_free_mb` is the memory the OS can still allocate (`psutil.virtual_memory().available`). A discrete GPU in an Intel Mac reports its VRAM from `vramUsedBytes` and `vramFreeBytes` instead. Temperature and power are read in-process without root: GPU temperature is the mean of the SMC `Tg..` sensors, GPU and ANE power come from the IOReport `Energy Model` energy counters sampled over a short interval (the first call in a process samples for 0.1 s; later calls use the delta since the previous call). The matching SMC key names are enumerated once (about 0.4 s) and cached under `~/Library/Caches/xpustat/`. The CPU row uses the same sources for its temperature and power on macOS. Per-process GPU usage remains root-only. The backend returns empty on any platform other than macOS.

## AWS Neuron

| | |
|---|---|
| Key | `neuron` |
| Status | untested; written from the `neuron-ls --json-output` schema in the Neuron documentation |
| Command | `neuron-ls --json-output` (also looked for at `/opt/aws/neuron/bin/neuron-ls`) |
| Fields | `neuron_device`, `nc_count`, `bdf`, `mem_total_mb` from `memory_size`, processes from `neuron_processes` |

`neuron-ls` reports inventory and attached processes, not live memory usage; that comes only from the streaming `neuron-monitor` daemon. `mem_used_mb` is therefore 0 and `util_pct` is `None`. The device name is built as `AWS Neuron (<n> NeuronCores)` because `neuron-ls` does not include the chip generation in its JSON.

## Google Cloud TPU

| | |
|---|---|
| Key | `tpu` |
| Status | untested; written from the `serialize_dashboard_to_json` output of the `tpu-info` source |
| Command | `tpu-info --format json dashboard` and `tpu-info --version`, run concurrently |
| Fields | `device_id`, `device_path`, memory from `memory_used_bytes` and `memory_total_bytes`, `util_pct` from `duty_cycle_pct`, `hbm_bw_util_pct`, one process per chip |

The name is `TPU <type>` using the `accelerator type` line of `tpu-info --version`, or `Cloud TPU` when that line is absent. `tpu-info` needs a running TPU workload to read metrics from `libtpu`; without one, memory and duty cycle read 0.

## CPU

| | |
|---|---|
| Key | `cpu` |
| Status | tested on Apple M5 (macOS), AMD Ryzen and Intel Xeon hosts (Linux) |
| Source | `psutil`, plus `/proc/cpuinfo`, `sysctl`, `/sys/class/powercap` (RAPL), `/sys/fs/cgroup` and `/sys/devices/system/node` where present |
| Fields | model, ISA, physical and logical cores, `4P+6E` layout on Apple Silicon, sockets, NUMA nodes, frequency, utilization total and per core, load average, RAM used/total/available, swap, temperature (Linux sensors), package power (Linux RAPL when the counter is readable), cgroup v2 CPU and memory limits, top processes by resident memory |

Opt-in only: `query_all(include_cpu=True)` or `xpustat --cpu`. The first utilization reading in a process samples for 0.1 s; later readings use the delta since the previous call. Temperature and power are `None` on macOS and on hosts whose RAPL counter is root-only. On Linux the backend never needs root.

## Rebellions

| | |
|---|---|
| Key | `rebellions` |
| Status | untested; written from the `rbln-stat --json` schema as consumed by the all-smi project |
| Command | `rbln-stat --json` (`rbln-smi` on older SDKs) |
| Fields | name, UUID, device node, bus id, firmware, memory from `memory.used` / `memory.total` (bytes), `util`, temperature, card power (from microwatts), P-state, status; processes from `contexts` with allocation and utilization |

ATOM Max boards list one entry per die; each repeats the card's power figure. Values arrive as strings with units and are parsed numerically.

## FuriosaAI

| | |
|---|---|
| Key | `furiosa` |
| Status | untested; written from the `furiosa-smi` JSON schema as consumed by the all-smi project |
| Command | `furiosa-smi info`, `status` and `ps`, with `--format json` (RNGD) or `--output json` (Warboy), run concurrently |
| Fields | arch, UUID, serial, firmware, bus id, temperature, power, core clock, governor from `info`; DRAM used/total, liveness and per-processing-element utilization from `status`; processes from `ps` |

`util_pct` is the mean over the processing elements; `pe_util_pct` keeps the individual values.

## Plugin backends

Any package can add a vendor by exposing a collection class through the `xpustat.backends` entry point, or at runtime with `xpustat.register_backend(name, cls)`. The class follows the same contract as the built-ins; see [Contributing](contributing.md#plugin-backends). A plugin vendor appears in `query_all`, `to_dict`, the CLI (with a generic label) and the Prometheus exporter without further wiring.

## Tenstorrent

| | |
|---|---|
| Key | `tenstorrent` |
| Status | untested; written from the `tt-smi` log model in the `tt-smi` source |
| Command | `tt-smi -s --snapshot_no_tty` |
| Fields | `tt_device`, `bus_id`, `board_id`, `temp_c` from `asic_temperature`, `power_w`, `board_power_w`, `aiclk_mhz`, processes matched by their `device` field |

`tt-smi` reports telemetry and processes, not DRAM usage, so the memory fields are 0 and the CLI shows the memory column as `N/A`. The device name is `Tenstorrent <board_type>`, for example `Tenstorrent n150 L`.
