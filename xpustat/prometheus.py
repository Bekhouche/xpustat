"""
prometheus.py — Prometheus text exposition for every detected accelerator.

Two entry points, no third-party dependencies:

* :func:`to_prometheus` renders an :class:`XPUStatCollection` as the
  Prometheus text format (``# HELP`` / ``# TYPE`` / samples).
* :func:`serve` runs a tiny HTTP server that answers ``/metrics`` with a
  fresh query on every scrape, so one process can replace per-vendor
  exporters on a mixed-vendor host.

CLI::

    xpustat --prometheus            # print metrics once and exit
    xpustat --serve-prometheus 9400 # serve http://0.0.0.0:9400/metrics

Metric names follow Prometheus conventions (base units: bytes, celsius,
watts; utilization as ``_percent`` to match dcgm-exporter).
"""

from __future__ import annotations

import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

_MIB = 1024 * 1024


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(pairs: Dict[str, Any]) -> str:
    return "{" + ",".join(f'{k}="{_escape(v)}"' for k, v in pairs.items() if v is not None) + "}"


def _fmt(value: float) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def _util(dev: Any) -> Optional[int]:
    for attr in ("util_pct", "ai_core_pct"):
        v = getattr(dev, attr, None)
        if v is not None:
            return int(v)
    return None


def _arch(dev: Any) -> Optional[str]:
    for attr in ("cuda_cc", "gfx_target"):
        v = getattr(dev, attr, None)
        if v:
            return str(v)
    cores = getattr(dev, "gpu_cores", None)
    return f"{cores}-core" if cores else None


_METRICS: List[Tuple[str, str, str]] = [
    # name, type, help
    ("xpustat_device_info",         "gauge", "Static device information; value is always 1."),
    ("xpustat_memory_used_bytes",   "gauge", "Device memory in use."),
    ("xpustat_memory_total_bytes",  "gauge", "Total device memory (system RAM for unified-memory devices)."),
    ("xpustat_memory_free_bytes",   "gauge", "Device memory available."),
    ("xpustat_utilization_percent", "gauge", "Compute utilization (0-100)."),
    ("xpustat_temperature_celsius", "gauge", "Device temperature."),
    ("xpustat_power_watts",         "gauge", "Power draw."),
    ("xpustat_process_memory_bytes", "gauge", "Device memory used per process."),
    ("xpustat_process_count",       "gauge", "Number of processes using the device."),
    ("xpustat_scrape_devices",      "gauge", "Number of devices seen in this scrape, per vendor."),
]


def to_prometheus(collection: Any) -> str:
    """Render *collection* (an ``XPUStatCollection``) in Prometheus text format."""
    samples: Dict[str, List[str]] = {name: [] for name, _, _ in _METRICS}
    vendor_counts: Dict[str, int] = {}

    for vendor in collection.to_dict().keys():
        devices = list(collection[vendor])
        vendor_counts[vendor] = len(devices)
        # MIG instances are exported as devices of their own, labelled parent/gi.ci.
        devices = devices + [m for d in devices for m in (getattr(d, "mig_instances", None) or [])]
        for dev in devices:
            base = {"vendor": vendor, "index": dev.index_label, "name": dev.name}
            info = dict(base)
            info["arch"] = _arch(dev)
            info["torch_device"] = dev.torch_device
            uuid = getattr(dev, "uuid", None)
            if uuid:
                info["uuid"] = uuid
            samples["xpustat_device_info"].append(f"xpustat_device_info{_labels(info)} 1")

            lbl = _labels(base)
            samples["xpustat_memory_used_bytes"].append(
                f"xpustat_memory_used_bytes{lbl} {int(dev.mem_used_mb) * _MIB}")
            samples["xpustat_memory_total_bytes"].append(
                f"xpustat_memory_total_bytes{lbl} {int(dev.mem_total_mb) * _MIB}")
            samples["xpustat_memory_free_bytes"].append(
                f"xpustat_memory_free_bytes{lbl} {int(dev.mem_free_mb) * _MIB}")

            util = _util(dev)
            if util is not None:
                samples["xpustat_utilization_percent"].append(
                    f"xpustat_utilization_percent{lbl} {util}")
            temp = getattr(dev, "temp_c", None)
            if temp is not None:
                samples["xpustat_temperature_celsius"].append(
                    f"xpustat_temperature_celsius{lbl} {_fmt(temp)}")
            power = getattr(dev, "power_w", None)
            if power is not None:
                samples["xpustat_power_watts"].append(
                    f"xpustat_power_watts{lbl} {_fmt(power)}")

            procs = getattr(dev, "processes", []) or []
            samples["xpustat_process_count"].append(
                f"xpustat_process_count{lbl} {len(procs)}")
            for p in procs:
                if p.gpu_mem_mb is None:
                    continue
                plbl = _labels({**base, "pid": p.pid, "user": p.username, "command": p.command})
                samples["xpustat_process_memory_bytes"].append(
                    f"xpustat_process_memory_bytes{plbl} {int(p.gpu_mem_mb) * _MIB}")

    for vendor, count in vendor_counts.items():
        samples["xpustat_scrape_devices"].append(
            f'xpustat_scrape_devices{{vendor="{_escape(vendor)}"}} {count}')

    out: List[str] = []
    for name, mtype, help_text in _METRICS:
        lines = samples[name]
        if not lines:
            continue
        out.append(f"# HELP {name} {help_text}")
        out.append(f"# TYPE {name} {mtype}")
        out.extend(lines)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    query = None  # set by serve()

    def do_GET(self) -> None:  # noqa: N802  (http.server API)
        if self.path.split("?", 1)[0] not in ("/metrics", "/"):
            self.send_response(404)
            self.end_headers()
            return
        try:
            body = to_prometheus(type(self).query()).encode("utf-8")
            status = 200
        except Exception as exc:  # never take the exporter down on a vendor hiccup
            body = f"# xpustat error: {exc}\n".encode("utf-8")
            status = 500
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        return


def serve(port: int = 9400, host: str = "0.0.0.0", query=None, max_age: float = 0.0) -> None:
    """
    Serve ``/metrics`` forever. Each scrape triggers a fresh ``query_all()``.

    Args:
        port:  TCP port (default 9400, the dcgm-exporter convention).
        host:  bind address.
        query: callable returning an ``XPUStatCollection``; defaults to
               :func:`xpustat.query_all`.
        max_age: with the default query, serve a result up to this many
               seconds old instead of re-querying on every scrape.
    """
    if query is None:
        from .xpu import query_all
        query = (lambda: query_all(max_age=max_age)) if max_age > 0 else query_all

    handler = type("XPUStatHandler", (_Handler,), {"query": staticmethod(query)})
    server = ThreadingHTTPServer((host, port), handler)
    sys.stderr.write(f"xpustat: serving Prometheus metrics on http://{host}:{port}/metrics\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
