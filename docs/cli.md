# Command line

```
xpustat [-h] [-u] [-p] [-c] [-a] [-e] [-t] [-P] [-A] [-i [INTERVAL]] [-n N]
        [-f VENDOR [VENDOR ...]] [--no-header] [--no-color]
        [-F] [-C] [-E] [-H] [-k] [--full-cmd] [--id IDS] [--print-completion SHELL]
        [--cpu] [--json] [--prometheus] [--select N] [--by RANKING] [--min-free MB] [--uuid]
        [--ui] [--port PORT] [--host ADDR] [--hosts-file FILE] [--no-local]
        [--serve-prometheus PORT] [--cache-ttl SECS] [--doctor]
        [--hosts URL ...] [--ssh HOST ...] [--ssh-command CMD] [--timeout SECS]
        [--record FILE] [--replay FILE] [--version]
```

Without flags, `xpustat` prints one line per device with the memory bar and exits.

```
xpustat 0.2.0  ·  amd-X870E-AORUS-ELITE-WIFI7  ·  Sat Oct 24  00:01:21  2026
────────────────────────────────────────────────────────────────────────────
[AMD   ] #0    AMD Radeon AI PRO R9700           gfx1201            │  18,303 / 32,624 MB  █████████░░░░░░░   56%  100%   78°C   287.0W
              └─ root:python/3654443(17552M)
[AMD   ] #1    AMD Radeon Graphics               gfx1036            │      16 /  2,048 MB  ░░░░░░░░░░░░░░░░    0%  N/A   49°C    N/A
```

The line above was produced with `-a`. From left to right: the vendor tag, the device index, the name, the architecture tag (`-A`), used and total memory in MB, the memory bar and percentage, then utilization (`-e`), temperature (`-t`) and power (`-P`). Processes are listed under the device as `user:command/pid(memory)` when `-u`, `-c` or `-p` is given.

## Flags

### Process columns

| Flag | Effect |
|---|---|
| `-u`, `--show-user` | process owner |
| `-p`, `--show-pid` | process PID |
| `-c`, `--show-cmd` | process command name |
| `-k`, `--show-container` | prefix each process with its container or job: `docker:<id>`, `containerd:<id>`, `podman:<id>`, `pod:<uid>`, `slurm:<jobid>` |
| `--full-cmd` | show the full command line instead of the executable name (with `-c`); truncated at 80 characters |
| `-a`, `--show-all` | every column: `-u -p -c -k -e -t -P -A -F -C -E -H` |

### Hardware columns

| Flag | Effect |
|---|---|
| `-e`, `--show-util` | compute utilization in percent |
| `-t`, `--show-temp` | temperature in degrees Celsius |
| `-P`, `--show-power` | power draw in watts |
| `-A`, `--show-arch` | architecture tag: `cc8.9` for CUDA compute capability, `gfx1201` for the ROCm target, `10-core` for Apple and Neuron core counts, `4P+6E` or `16c/32t` for the CPU row |
| `-F`, `--show-fan` | fan speed in percent, or rpm when only that is reported |
| `-C`, `--show-clocks` | SM or graphics clock and memory clock, MHz |
| `-E`, `--show-codec` | video encoder and decoder utilization |
| `-H`, `--show-health` | NVIDIA P-state (`P0` busy to `P8` idle) or AMD perf level, active throttle reasons (`power`, `thermal`, `HW`, ...), and the uncorrected ECC count, red when non-zero |

Process entries read `user:command/pid(memory, sm%)`: the trailing percentage is the process's own compute utilization when the driver reports it (NVIDIA, from NVML samples or `nvidia-smi pmon`).

A column shows `N/A` when the vendor tool does not report the value. Temperature and power are `N/A` on Apple Silicon, since macOS exposes them to root only.

### Watch mode

| Flag | Effect |
|---|---|
| `-i [SECS]`, `--interval`, `--watch` | refresh every SECS seconds; `-i` alone means 1.0 |
| `-n N`, `--count N` | stop after N refreshes; requires `-i` |

On a terminal, watch mode draws on the alternate screen and redraws each frame in place, so the display does not flicker while the vendor tools are queried. Only the vendors found in the first frame are queried again, which keeps the refresh cost to the devices actually present. Ctrl+C restores the screen. An exception raised during a refresh is reported, not swallowed.

### Filtering and layout

| Flag | Effect |
|---|---|
| `--id IDS` | show only devices with these physical indices, e.g. `--id 0,2`; combine with `-f` |
| `-f VENDOR ...`, `--filter` | show only these vendors; keys: `nvidia amd gaudi intel huawei hygon cambricon moorethreads apple neuron tpu tenstorrent jetson rebellions furiosa cpu` plus any registered plugin |
| `--cpu` | also show the host CPU and memory: one `[CPU]` row with RAM as the memory bar, CPU utilization, temperature and power where readable, a detail line (cores and threads, frequency, load average, available memory, swap, cgroup limits), and the top processes by resident memory under `-u`, `-p`, `-c`. Applies to `--json`, `--prometheus` and `--serve-prometheus` as well. `-f cpu` implies it. |
| `--no-header` | omit the hostname and time line |
| `--no-color` | plain text; colour is also disabled automatically when stdout is not a terminal |

### Output formats

| Flag | Effect |
|---|---|
| `--json` | the `to_dict()` structure as JSON, see [Python API](python-api.md#json-output); cannot be combined with `-i` |
| `--prometheus` | the Prometheus text format, once |
| `--ui` | serve the browser page at `http://127.0.0.1:9400/` until interrupted, see [Web UI](web-ui.md); `/json` and `/metrics` too |
| `--port PORT` | port for `--ui` (default 9400) |
| `--host ADDR` | bind address: `127.0.0.1` for `--ui`, `0.0.0.0` for `--serve-prometheus` by default |
| `--serve-prometheus PORT` | serve `/metrics` on `0.0.0.0:PORT` until interrupted, see [Prometheus](prometheus.md) |
| `--cache-ttl SECS` | with `--ui` or `--serve-prometheus`: reuse a local result up to SECS old between requests |
| `--print-completion SHELL` | print a completion script for `bash`, `zsh` or `fish`: `eval "$(xpustat --print-completion bash)"` |
| `--version` | print the version |

### Device selection

`--select N` picks the N best devices and prints the environment assignment that restricts a job to them, one line per vendor present, for example `CUDA_VISIBLE_DEVICES=1,2` or `HIP_VISIBLE_DEVICES=0`. The output is meant for the shell:

```bash
export $(xpustat --select 2)            # two freest devices, any vendor
export $(xpustat --select 1 -f nvidia --by util --uuid)
```

| Flag | Effect |
|---|---|
| `--select N` | number of devices to pick; honours `-f` |
| `--by` | `mem_free` (default), `mem_used`, `util`, `index` |
| `--min-free MB` | skip devices with less free memory |
| `--uuid` | emit UUIDs instead of indices where the vendor has them (NVIDIA, MIG) |

Devices already hidden by the variable are never picked. The command exits 1 when nothing matches, or when the picked devices belong to a vendor without such a variable (Apple).

### Diagnostics

`xpustat --doctor` prints one block per backend: the executables looked for and where they were found, what a version probe answered, whether the native binding imports (and why not), the visible-devices variables set, how many devices `new_query()` returned, how long it took, and any error. `--doctor --json` gives the same as data. On Linux it also says whether the RAPL energy counter is readable.

### Several machines

```bash
xpustat --hosts http://node1:9400 http://node2:9400      # exporters started with --serve-prometheus
xpustat --ssh node1 node2 -a                              # runs `xpustat --json` there over ssh
xpustat --ssh node3 --ssh-command 'cd ~/xpustat && PYTHONPATH=. python3 -m xpustat.cli --json'
```

| Flag | Effect |
|---|---|
| `--hosts URL ...` | read each exporter's `/json` endpoint |
| `--ssh HOST ...` | run xpustat over `ssh -o BatchMode=yes` on each host |
| `--ssh-command CMD` | the command to run instead of `xpustat --json` |
| `--timeout SECS` | per-host timeout, default 10 |

The table gets a host column; a host that does not answer shows `unreachable: <reason>` and the others still render. `--json` prints one object per host with `host`, `hostname`, `query_time`, `error` and `devices`; `-i` refreshes.

Add `--ui` to get the same hosts as a browser page instead of a table; `--hosts-file FILE` reads the host list from a file and `--no-local` leaves this machine out. See [Web UI](web-ui.md).

### Record and replay

```bash
xpustat --record run.ndjson.gz -i 1 -n 60    # one frame per second for a minute
xpustat --replay run.ndjson.gz -i 0.2 -e     # play it back five frames a second
xpustat --replay run.ndjson.gz --json        # the frames as a JSON list
```

Each frame is `to_json(metadata=True)` on one line; a `.gz` suffix compresses. Replay uses the same column flags as a live view.

## Examples

```bash
xpustat                      # snapshot
xpustat -a                   # everything
xpustat -pui                 # processes with PIDs and users, refreshing every second
xpustat -pui --cpu           # the same, plus the host CPU and memory
xpustat -i 2 -n 10           # ten refreshes, two seconds apart
xpustat -f nvidia amd        # only NVIDIA and AMD
xpustat -A                   # architecture column
xpustat --json | jq '.nvidia[].mem_free_mb'
export $(xpustat --select 2)
xpustat --ui --ssh node1 node2 --cpu -i 5   # browser page for the cluster
xpustat --serve-prometheus 9400
```

## Exit status

The command exits 0 whether or not devices are found; "no accelerators detected" is a normal result. It exits 1 when `--json` is combined with `-i`, and when `--select` finds no device.
