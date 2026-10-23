# Command line

```
xpustat [-h] [-u] [-p] [-c] [-a] [-e] [-t] [-P] [-A] [-i [INTERVAL]] [-n N]
        [-f VENDOR [VENDOR ...]] [--no-header] [--no-color]
        [--json] [--prometheus] [--serve-prometheus PORT] [--version]
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
| `-a`, `--show-all` | all of the above plus `-e -t -P -A` |

### Hardware columns

| Flag | Effect |
|---|---|
| `-e`, `--show-util` | compute utilization in percent |
| `-t`, `--show-temp` | temperature in degrees Celsius |
| `-P`, `--show-power` | power draw in watts |
| `-A`, `--show-arch` | architecture tag: `cc8.9` for CUDA compute capability, `gfx1201` for the ROCm target, `10-core` for Apple and Neuron core counts |

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
| `-f VENDOR ...`, `--filter` | show only these vendors; keys: `nvidia amd gaudi intel huawei hygon cambricon moorethreads apple neuron tpu tenstorrent` |
| `--no-header` | omit the hostname and time line |
| `--no-color` | plain text; colour is also disabled automatically when stdout is not a terminal |

### Output formats

| Flag | Effect |
|---|---|
| `--json` | the `to_dict()` structure as JSON, see [Python API](python-api.md#json-output); cannot be combined with `-i` |
| `--prometheus` | the Prometheus text format, once |
| `--serve-prometheus PORT` | serve `/metrics` on `0.0.0.0:PORT` until interrupted, see [Prometheus](prometheus.md) |
| `--version` | print the version |

## Examples

```bash
xpustat                      # snapshot
xpustat -a                   # everything
xpustat -pui                 # processes with PIDs and users, refreshing every second
xpustat -i 2 -n 10           # ten refreshes, two seconds apart
xpustat -f nvidia amd        # only NVIDIA and AMD
xpustat -A                   # architecture column
xpustat --json | jq '.nvidia[].mem_free_mb'
xpustat --serve-prometheus 9400
```

## Exit status

The command exits 0 whether or not devices are found; "no accelerators detected" is a normal result. It exits 1 when `--json` is combined with `-i`.
