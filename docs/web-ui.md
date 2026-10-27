# Web UI

`xpustat --ui` serves a browser page that shows every device the CLI knows about, for this machine or for a whole cluster, with no dependency beyond the standard library. The page is one HTML file with its CSS and JavaScript inline; it loads nothing from the network, so it works on air-gapped nodes.

## Running it

```bash
xpustat --ui                               # http://127.0.0.1:9400/
xpustat --ui --cpu -i 2 --port 8080        # with the CPU row, refreshed every 2 s, on port 8080
xpustat --ui --host 0.0.0.0                # reachable from other machines
```

The same server also answers `/json` (the device JSON with metadata), `/metrics` (Prometheus text) and `/cluster` (what the page reads), so one process serves the browser, scripts and Prometheus.

The UI binds to `127.0.0.1` by default because the page shows process owners and command lines. Use `--host 0.0.0.0` to expose it, or reach a remote machine's page through an ssh tunnel:

```bash
ssh -L 9400:127.0.0.1:9400 node1 'xpustat --ui --cpu'
# then open http://127.0.0.1:9400/ locally
```

## Many machines on one page

```bash
xpustat --ui --ssh node1 node2 node3 --cpu -i 5        # xpustat installed on each node
xpustat --ui --hosts http://node4:9400 http://node5:9400 # nodes running --serve-prometheus or --ui
xpustat --ui --hosts-file lab.txt                        # one host per line, # comments
xpustat --ui --ssh node1 --ssh-command '~/.local/bin/xpustat --json --cpu'
xpustat --ui --ssh node1 node2 --no-local                # leave this machine out
```

`--ssh`, `--hosts`, `--hosts-file`, `--ssh-command` and `--timeout` work as for the [terminal cluster view](cli.md#several-machines). A hosts file may mix exporter URLs (`http://...` or `host:port`) and ssh hosts (`node`, `user@node`). This machine is the first host on the page unless `--no-local` is given.

In cluster mode the server polls the hosts itself, every `-i` seconds (default 5), with one ssh connection per host in parallel, and keeps the latest answer. Every open browser tab reads that cached answer, so ten tabs cost one fan-out, not ten. A host that stops answering keeps its last good snapshot on the page, marked stale with the error; a host that never answered is shown as unreachable. The other hosts are unaffected.

## What the page shows

The summary bar across all hosts: hosts up and down, devices and how many are free (under 5 % memory used, under 5 % utilization, no processes), device memory used over total (the CPU row is left out of this sum), processes, the hottest device, and how many devices are throttled or report a fault.

Then one block per host with a table of its devices. The columns are the `-a` set of the CLI: vendor tag and index, name with the architecture (`cuda_cc`, `gfx_target`, Apple GPU cores), memory bar with used and total, utilization, temperature (orange from 70 °C, red from 85 °C), power with its limit, fan, SM or graphics clock with its maximum and the memory clock, P-state or performance level, health (`ok`, `fault`, `throttled`, `ECC n`), process count, and two sparklines of utilization and memory for the time the tab has been open. MIG instances are listed under their parent GPU.

Clicking a device row opens its detail: UUID, free memory, memory utilization, hotspot and memory temperatures, encoder and decoder load, ECC counts, throttle reasons, energy, PCIe link, ANE power on Apple, and the processes with PID, user, device memory, SM share, host RSS, CPU share, container or SLURM job, and the full command line. For the CPU row the detail holds cores, sockets, core layout, load average, frequency, swap, NUMA nodes, cgroup limits and per-core utilization.

Controls in the header: a host filter, a vendor filter, a text filter over device names, users, commands, containers and PIDs, a compact mode that hides the fan, clock, state and sparkline columns so forty devices fit on one screen, expand all, and pause. Opening `/?expand` starts with every row expanded. The compact setting is remembered in the browser.

## From Python

```python
from xpustat.ui import ClusterCollector, serve_ui

serve_ui(port=9400, host="127.0.0.1")                      # local page; blocks

collector = ClusterCollector(ssh=["node1", "node2"], include_cpu=True, interval=5)
serve_ui(port=9400, cluster=collector)                      # cluster page; starts the poller
```

`serve_ui` takes a `query` callable like `xpustat.prometheus.serve`. `ClusterCollector(fetch=...)` accepts a callable returning `RemoteHost` objects for a custom source, and `local=` a callable for this machine; `snapshot()` returns the `/cluster` payload.

## Limitations

- No authentication and no TLS: keep it on `127.0.0.1`, behind an ssh tunnel or a reverse proxy.
- History lives in the browser tab; closing it loses the sparklines. Use the Prometheus exporter for retention.
- `/metrics` and `/json` always describe the local machine, also in cluster mode; `/cluster` is the multi-host payload.
- Hosts queried over ssh with `xpustat --json` carry no hostname or query time, so the page labels them by the ssh name.
