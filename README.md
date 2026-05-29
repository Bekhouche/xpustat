# xpustat

[![GitHub](https://img.shields.io/badge/GitHub-Bekhouche%2Fxpustat-blue?logo=github)](https://github.com/Bekhouche/xpustat)
**Authors:** Salah Eddine Bekhouche, Hichem Telli

A unified accelerator stat tool — think `gpustat`, but for every vendor.

## Supported backends

| Status | Vendor | Device | CLI tool | Filter key |
|:---:|---|---|---|---|
| ✅ Tested | AMD | Radeon RX 7900 XTX (gfx1100) | `amd-smi` | `amd` |
| ✅ Tested | NVIDIA | NVIDIA L4 (cc8.9), TITAN V (cc7.0) | `nvidia-smi` | `nvidia` |
| 🔲 Untested | Intel | Gaudi / Gaudi 2 / Gaudi 3 | `hl-smi` | `gaudi` |
| 🔲 Untested | Intel | Arc / Data Center GPU | `xpu-smi` | `intel` |
| 🔲 Untested | Huawei | Ascend NPU | `npu-smi` | `huawei` |
| 🔲 Untested | Hygon | DCU | `hy-smi` | `hygon` |
| 🔲 Untested | Cambricon | MLU | `cnmon` | `cambricon` |
| 🔲 Untested | Moore Threads | MTT GPU | `mthreads-gmi` | `moorethreads` |

> **Legend:** ✅ Tested on real hardware — 🔲 Implemented, awaiting hardware test
>
> Backends whose management tool is not installed are silently skipped.

### Running Vendor Hardware Tests

If you have access to specific hardware, you can help us verify the implementation by running the standalone test scripts found in the `vendor_devices_tests/` directory:

```bash
# Example: Testing the Huawei backend on a machine with Ascend NPUs
python vendor_devices_tests/test_huawei.py

# Example: Testing the NVIDIA backend
python vendor_devices_tests/test_nvidia.py
```
If the test successfully outputs your hardware metrics, please let us know so we can update the backend status to **✅ Tested**!

<details>
<summary>Example test output from an AMD Radeon RX 7900 XTX</summary>

```json
{
  "gpus": [
    {
      "gpu_index": 0,
      "name": "Radeon RX 7900 XTX",
      "gfx_target": "gfx1100",
      "mem_used_mb": 986,
      "mem_total_mb": 24560,
      "util_pct": 0,
      "temp_c": 35,
      "power_w": 7.0,
      "processes": [
        {
          "pid": 352030,
          "username": "salah",
          "command": "python3",
          "gpu_mem_mb": 937
        }
      ]
    }
  ]
}
```
</details>

<details>
<summary>Example test output from an NVIDIA L4</summary>

```json
{
  "gpus": [
    {
      "gpu_index": 0,
      "uuid": "GPU-33e0a3e4-1cd8-aa3c-bbfc-2718461c2a32",
      "name": "NVIDIA L4",
      "cuda_cc": "8.9",
      "mem_used_mb": 556,
      "mem_total_mb": 23034,
      "util_pct": 0,
      "temp_c": 37,
      "power_w": 16.122,
      "power_limit_w": 72.0,
      "processes": []
    }
  ]
}
```
</details>

<details>
<summary>Example test output from an NVIDIA TITAN V</summary>

```json
{
  "gpus": [
    {
      "gpu_index": 0,
      "uuid": "GPU-15375434-548d-bb7a-8c5a-451107921530",
      "name": "NVIDIA TITAN V",
      "cuda_cc": "7.0",
      "mem_used_mb": 322,
      "mem_total_mb": 12288,
      "util_pct": 0,
      "temp_c": 51,
      "power_w": 28.402,
      "power_limit_w": 250.0,
      "processes": []
    }
  ]
}
```
</details>

## Install

```bash
pip install xpustat
```

## CLI usage

```
xpustat                      # coloured table of all detected devices
xpustat -pui                 # watch mode, showing processes (-p), users (-u), refreshing every 1.0s (-i)
xpustat -i 2                 # watch mode, refresh every 2 seconds
xpustat -i 1 -n 10           # 10 refreshes then exit
xpustat -a                   # show everything (util, temp, power, arch, user, pid, cmd)
xpustat -f nvidia amd        # show only NVIDIA and AMD
xpustat --no-color           # plain text, no ANSI codes
xpustat --json               # raw JSON output
xpustat --version
```

Example output:

```
xpustat 0.1.0  ·  myhost  ·  Fri May 22  11:24:00  2026
──────────────────────────────────────────────────────────────────────────
[NVIDIA] #0    Tesla V100                          │   1,024 / 32,768 MB   ███░░░░░░░░░░░░░    3%
[AMD   ] #0    Radeon RX 7900 XTX                  │      26 / 24,560 MB   ░░░░░░░░░░░░░░░░    0%
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
```

## Version history

### 0.1.0
- Initial release
- Support for NVIDIA, AMD, Intel Gaudi, Intel Arc/GPU, Huawei Ascend, Hygon DCU, Cambricon MLU, Moore Threads
