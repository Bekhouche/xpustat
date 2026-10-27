# xpustat documentation

xpustat is a command-line tool and Python library that reports the state of every AI accelerator on a machine, whatever the vendor. It is the equivalent of `gpustat` for NVIDIA, AMD, Intel, Huawei, Hygon, Cambricon, Moore Threads, Apple Silicon, AWS Neuron, Google Cloud TPU and Tenstorrent devices, with one output format and one Python API.

It is used for checking which devices are free before launching a job, watching memory and utilization during training, picking a device from a script, exporting metrics to Prometheus, watching a whole cluster in a browser, and writing vendor-neutral tooling on top of the Python API, to cite a few.

## How it works

Each query runs in three stages:

1. **Detection.** Every backend looks for its vendor tool on `PATH` (`nvidia-smi`, `amd-smi`, `npu-smi`, ...). A backend whose tool is absent returns an empty collection and is never reported as an error, so the same command works on any machine.
2. **Query.** The backends that found their tool run concurrently. NVIDIA, AMD and Apple use an in-process library when one is available and fall back to the vendor command otherwise. See [Backends](backends.md) for the exact commands.
3. **Normalisation.** Each device becomes a stat object with a common set of fields (`name`, `mem_used_mb`, `mem_total_mb`, `mem_free_mb`, `processes`, `torch_device`, ...) on top of its vendor-specific ones. The CLI, the JSON output and the Prometheus exporter all read from these objects.

## Pages

| Page | Covers |
|---|---|
| [Installation](installation.md) | pip, conda, optional native bindings, what each vendor needs |
| [Command line](cli.md) | every flag, the output columns, watch mode, JSON |
| [Python API](python-api.md) | `query_all`, collections, per-vendor fields, JSON schema |
| [Device selection](device-selection.md) | `pick_device`, `*_VISIBLE_DEVICES`, `torch_device` |
| [Prometheus](prometheus.md) | the exporter, the metrics, a scrape configuration |
| [Web UI](web-ui.md) | `--ui`: the browser page for one machine or a cluster over ssh |
| [Backends](backends.md) | one section per vendor: command used, fields, limits, test status |
| [Contributing](contributing.md) | adding a backend, running the tests, hardware test scripts |

## Limitations

- Of the fifteen accelerator backends, five are tested on hardware (NVIDIA, AMD, Huawei Ascend, Apple, and the CPU row). The others (Intel Gaudi, Intel Arc, Hygon, Cambricon, Moore Threads, Neuron, TPU, Tenstorrent, Jetson, Rebellions, FuriosaAI) are implemented from the vendor tools' documented output and await a hardware run. The [Backends](backends.md) page states the status per vendor.
- Per-process attribution depends on what the vendor tool exposes. Apple, Neuron and Tenstorrent report no per-process memory.
- Per-process GPU usage on Apple Silicon is root-only; temperature and power are read without root through the SMC and IOReport.
