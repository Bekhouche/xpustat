# Installation

xpustat requires Python 3.9 or newer and depends on `psutil` only. The vendor tools are not Python packages; they come with the vendor's driver or SDK and must be on `PATH`.

## pip

```bash
pip install xpustat
```

## conda

```bash
conda install -c conda-forge xpustat
```

## From source

```bash
git clone https://github.com/Bekhouche/xpustat
cd xpustat
pip install -e .
```

The hardware test scripts in `vendor_devices_tests/` add the checkout to `sys.path` themselves, so they can be run from a clone without installing the package.

## Optional native bindings

Two backends can query the driver in-process instead of running the vendor command. The subprocess path remains the fallback, so these packages are optional.

| Extra | Package | Effect |
|---|---|---|
| `pip install xpustat[nvidia]` | `nvidia-ml-py` | NVIDIA through NVML. On a host with seven L4 GPUs, one query took 123 ms against 172 ms for the `nvidia-smi` path. |
| `pip install xpustat[amd]` | `amdsmi` | AMD through the ROCm Python bindings. The PyPI wheel bundles its own library and does not need a system ROCm. |
| `pip install xpustat[fast]` | both | |

`nvidia-ml-py` is the package NVIDIA publishes; the older `pynvml` package on PyPI is deprecated and prints a warning. Both install a module named `pynvml`, which is what xpustat imports.

The CLI prints a one-line tip when it detects NVIDIA or AMD devices and the corresponding binding is missing.

## What each vendor needs

| Vendor | Tool on `PATH` | Ships with |
|---|---|---|
| NVIDIA | `nvidia-smi` | NVIDIA driver |
| AMD | `amd-smi` | ROCm (`amdrocm-amdsmi` package on ROCm 7.14 and later) |
| Intel Gaudi | `hl-smi` | Habana / Gaudi software stack |
| Intel Arc / Data Center GPU | `xpu-smi` | Intel XPU Manager |
| Huawei Ascend | `npu-smi`, optionally `ascend-dmi` | Ascend driver / toolkit |
| Hygon DCU | `hy-smi` | DTK |
| Cambricon MLU | `cnmon` | Neuware |
| Moore Threads | `mthreads-gmi` | MUSA driver |
| Apple | none | macOS; read through IOKit |
| AWS Neuron | `neuron-ls` | AWS Neuron SDK (`/opt/aws/neuron/bin` is also searched) |
| Google Cloud TPU | `tpu-info` | `pip install tpu-info` on a TPU VM |
| Tenstorrent | `tt-smi` | Tenstorrent software stack |

A missing tool is not an error. The backend is skipped and the device simply does not appear, which is the intended behaviour on a machine without that vendor's hardware.
