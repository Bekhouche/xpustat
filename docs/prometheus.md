# Prometheus

xpustat can act as a Prometheus exporter for every vendor from one process, which avoids running `dcgm-exporter` for NVIDIA and a second exporter for AMD on a mixed host. There are no extra dependencies; the server is the standard library's `http.server`.

## Running it

```bash
xpustat --serve-prometheus 9400
```

This serves `http://0.0.0.0:9400/metrics` until interrupted. Every scrape runs a fresh `query_all()`, so the values are as current as the vendor tools allow; on the machines used for testing a query takes between 4 ms (Apple, in-process) and 170 ms (AMD or NVIDIA through the vendor command). A vendor query that raises produces an HTTP 500 with the error in the body, and the server stays up.

Port 9400 is the dcgm-exporter convention; any port works. Any path other than `/metrics` and `/` returns 404.

```bash
xpustat --prometheus          # print the same text once and exit
```

From Python:

```python
import xpustat

text = xpustat.to_prometheus(xpustat.query_all())   # the exposition text
xpustat.prometheus.serve(port=9400, host="0.0.0.0")  # blocks
```

`serve` accepts a `query` callable for embedding in a larger exporter.

## Scrape configuration

```yaml
scrape_configs:
  - job_name: xpustat
    scrape_interval: 15s
    static_configs:
      - targets: ["gpu-host-1:9400", "gpu-host-2:9400"]
```

## Metrics

All metrics are gauges. The device labels are `vendor`, `index` and `name`; `index` is the display index, `0:1` for vendors addressed as card:chip.

| Metric | Labels | Meaning |
|---|---|---|
| `xpustat_device_info` | device labels plus `arch`, `torch_device`, `uuid` (NVIDIA) | always 1; carries the static labels |
| `xpustat_memory_used_bytes` | device | memory in use |
| `xpustat_memory_total_bytes` | device | total memory; system RAM for Apple unified memory |
| `xpustat_memory_free_bytes` | device | free memory |
| `xpustat_utilization_percent` | device | compute utilization, 0 to 100; omitted when the vendor does not report it |
| `xpustat_temperature_celsius` | device | omitted when not reported |
| `xpustat_power_watts` | device | omitted when not reported |
| `xpustat_process_memory_bytes` | device plus `pid`, `user`, `command` | per-process device memory; only processes with a known memory figure |
| `xpustat_process_count` | device | number of processes on the device, including those without a memory figure |
| `xpustat_scrape_devices` | `vendor` | devices seen per vendor, including 0 for vendors not present |

Memory is converted from the MB values of the API to bytes. Utilization is a percentage rather than a ratio in order to match `DCGM_FI_DEV_GPU_UTIL`.

Example output from an NVIDIA host:

```
# HELP xpustat_device_info Static device information; value is always 1.
# TYPE xpustat_device_info gauge
xpustat_device_info{vendor="nvidia",index="0",name="NVIDIA L4",arch="8.9",torch_device="cuda:0",uuid="GPU-33e0a3e4-1cd8-aa3c-bbfc-2718461c2a32"} 1
# HELP xpustat_memory_used_bytes Device memory in use.
# TYPE xpustat_memory_used_bytes gauge
xpustat_memory_used_bytes{vendor="nvidia",index="0",name="NVIDIA L4"} 6173163520
```

## Grafana

A per-device memory panel is `xpustat_memory_used_bytes / xpustat_memory_total_bytes`. A per-user panel is `sum by (user) (xpustat_process_memory_bytes)`. Join `xpustat_device_info` on `vendor, index, name` to display the architecture or the `torch_device` string next to a series.

## Limitations

- The exporter is a snapshot per scrape. There is no caching, so a very short `scrape_interval` on a host where the vendor command is slow adds that much load to the vendor tool.
- There is no TLS and no authentication. Put the port behind a reverse proxy or a firewall rule if the host is reachable from outside the monitoring network.
- The `name` label is the vendor's device string and can contain spaces; it is escaped per the exposition format.
