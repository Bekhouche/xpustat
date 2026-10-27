"""
remote.py — query other hosts: through their xpustat exporter, or over SSH.

Two agentless-friendly paths:

* :func:`query_hosts` fetches ``/json`` from running ``xpustat
  --serve-prometheus`` exporters (``http://host:9400``).
* :func:`query_ssh` runs ``xpustat --json`` on each host over ``ssh`` and
  needs nothing but xpustat installed there (or a checkout and
  ``--ssh-command``).

Both return a list of :class:`RemoteHost` objects holding the parsed JSON
and an error string when the host did not answer, so one dead node never
hides the others. The CLI's ``--hosts`` and ``--ssh`` render them as the
local table with a host column.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional


class RemoteHost:
    """One queried host: its label, the device JSON (``to_dict()`` shape) and metadata."""

    def __init__(self, label: str, devices: Optional[Dict[str, list]] = None,
                 hostname: Optional[str] = None, query_time: Optional[str] = None,
                 version: Optional[str] = None, error: Optional[str] = None) -> None:
        self.label = label
        self.devices = devices or {}
        self.hostname = hostname
        self.query_time = query_time
        self.version = version
        self.error = error

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {"host": self.label, "hostname": self.hostname, "query_time": self.query_time,
                "xpustat_version": self.version, "error": self.error, "devices": self.devices}

    def __repr__(self) -> str:
        n = sum(len(v) for v in self.devices.values()) if self.devices else 0
        return f"RemoteHost({self.label!r}, devices={n}, error={self.error!r})"


def _from_payload(label: str, payload: Any) -> RemoteHost:
    """Accept both the metadata wrapper (``to_json(metadata=True)``) and the bare ``to_dict()``."""
    if not isinstance(payload, dict):
        return RemoteHost(label, error="response is not a JSON object")
    if "devices" in payload and isinstance(payload["devices"], dict):
        return RemoteHost(label, devices=payload["devices"], hostname=payload.get("hostname"),
                          query_time=payload.get("query_time"), version=payload.get("xpustat_version"))
    return RemoteHost(label, devices=payload)


def _fetch(url: str, timeout: float) -> RemoteHost:
    target = url.rstrip("/")
    if not target.startswith(("http://", "https://")):
        target = "http://" + target
    if not target.endswith("/json"):
        target += "/json"
    try:
        with urllib.request.urlopen(target, timeout=timeout) as r:
            return _from_payload(url, json.loads(r.read().decode("utf-8")))
    except urllib.error.HTTPError as exc:
        return RemoteHost(url, error=f"HTTP {exc.code}")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return RemoteHost(url, error=str(getattr(exc, "reason", exc))[:120])


def query_hosts(urls: List[str], timeout: float = 5.0, workers: int = 16) -> List[RemoteHost]:
    """Fetch the device JSON from each exporter URL concurrently, in the given order."""
    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=min(workers, len(urls))) as pool:
        return list(pool.map(lambda u: _fetch(u, timeout), urls))


def _ssh(host: str, command: str, timeout: float) -> RemoteHost:
    argv = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(max(timeout, 1))}", host, command]
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 10)
    except subprocess.TimeoutExpired:
        return RemoteHost(host, error="ssh timed out")
    except OSError as exc:
        return RemoteHost(host, error=str(exc))
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip().splitlines()
        return RemoteHost(host, error=(err[-1] if err else f"ssh exit {r.returncode}")[:120])
    text = r.stdout
    start = text.find("{")
    if start < 0:
        return RemoteHost(host, error="no JSON in output")
    try:
        return _from_payload(host, json.loads(text[start:]))
    except ValueError as exc:
        return RemoteHost(host, error=f"bad JSON: {exc}")


def query_ssh(hosts: List[str], command: Optional[str] = None, include_cpu: bool = False,
              timeout: float = 15.0, workers: int = 8) -> List[RemoteHost]:
    """
    Run xpustat on each host over SSH and parse its JSON.

    ``command`` defaults to ``xpustat --json`` (plus ``--cpu`` when
    *include_cpu*); pass another string for a checkout, e.g.
    ``"cd ~/xpustat && PYTHONPATH=. python3 -m xpustat.cli --json"``.
    """
    if not hosts:
        return []
    cmd = command or ("xpustat --json" + (" --cpu" if include_cpu else ""))
    with ThreadPoolExecutor(max_workers=min(workers, len(hosts))) as pool:
        return list(pool.map(lambda h: _ssh(h, cmd, timeout), hosts))


def devices_of(host: RemoteHost) -> List[dict]:
    """Flatten a host's device JSON into ``[{"vendor": ..., **device}, ...]``."""
    out: List[dict] = []
    for vendor, devs in (host.devices or {}).items():
        for d in devs or []:
            if isinstance(d, dict):
                out.append({"vendor": vendor, **d})
    return out


def shell_quote(command: str) -> str:
    return " ".join(shlex.quote(p) for p in shlex.split(command))
