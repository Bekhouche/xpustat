"""Tests for the web UI: routes, cluster collector, host lists, CLI wiring."""
import json
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xpustat import ui  # noqa: E402
from xpustat.remote import RemoteHost  # noqa: E402
from xpustat.xpu import XPUStatCollection  # noqa: E402
from xpustat.nvidia import NvidiaGPUStat, NvidiaGPUStatCollection  # noqa: E402
from xpustat.amd import AMDGPUStatCollection  # noqa: E402
from xpustat.gaudi import GaudiAIPStatCollection  # noqa: E402
from xpustat.intel_gpu import IntelGPUStatCollection  # noqa: E402
from xpustat.huawei import HuaweiNPUStatCollection  # noqa: E402
from xpustat.hygon import HygonDCUStatCollection  # noqa: E402
from xpustat.cambricon import CambriconMLUStatCollection  # noqa: E402
from xpustat.moorethreads import MooreThreadsGPUStatCollection  # noqa: E402
from xpustat._process import ProcessInfo  # noqa: E402


def _collection():
    gpu = NvidiaGPUStat(gpu_index=0, uuid="GPU-abc", name="L4", cuda_cc="8.9", mem_used_mb=1024,
                        mem_total_mb=23034, util_pct=42, temp_c=61, power_w=70.5, power_limit_w=72,
                        processes=[ProcessInfo(pid=4194303, username="salah", command="python",
                                               gpu_mem_mb=1000, cmdline="python train.py")],
                        throttle_reasons=["idle"], pstate=0)
    return XPUStatCollection(
        nvidia=NvidiaGPUStatCollection([gpu], backend="test"), amd=AMDGPUStatCollection([]),
        gaudi=GaudiAIPStatCollection([]), intel=IntelGPUStatCollection([]), huawei=HuaweiNPUStatCollection([]),
        hygon=HygonDCUStatCollection([]), cambricon=CambriconMLUStatCollection([]),
        moorethreads=MooreThreadsGPUStatCollection([]))


def _serve(query=None, cluster=None, interval=2.0):
    handler = ui.make_handler(query or _collection, cluster, interval)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read()


def test_page_is_self_contained_html():
    server, base = _serve()
    try:
        status, ctype, body = _get(base + "/")
        assert status == 200 and ctype.startswith("text/html")
        text = body.decode("utf-8")
        assert text.startswith("<!doctype html>") and "</html>" in text
        # nothing loaded from the network: no external script, stylesheet or font
        assert "<script src" not in text and "<link " not in text and "@import" not in text
        assert "http://" not in text.replace("http://HOST", "") and "https://" not in text
        assert _get(base + "/index.html")[0] == 200
    finally:
        server.shutdown(); server.server_close()


def test_cluster_route_local_mode_wraps_one_host():
    server, base = _serve(interval=3.5)
    try:
        status, ctype, body = _get(base + "/cluster")
        assert status == 200 and ctype.startswith("application/json")
        payload = json.loads(body)
        assert payload["cluster"] is False and payload["interval"] == 3.5
        assert payload["version"]
        (host,) = payload["hosts"]
        assert host["host"] and host["stale"] is False and host["error"] is None
        assert host["devices"]["nvidia"][0]["name"] == "L4"
        assert host["devices"]["nvidia"][0]["processes"][0]["cmdline"] == "python train.py"
        # the exporter routes still answer on the same port
        assert _get(base + "/json")[1].startswith("application/json")
        assert b"xpustat_memory_used_bytes" in _get(base + "/metrics")[2]
    finally:
        server.shutdown(); server.server_close()


def test_cluster_route_reports_query_errors_as_json():
    def boom():
        raise RuntimeError("vendor tool hung")
    server, base = _serve(query=boom)
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(base + "/cluster")
        assert exc.value.code == 500
        assert "vendor tool hung" in json.loads(exc.value.read())["error"]
    finally:
        server.shutdown(); server.server_close()


def test_collector_keeps_last_snapshot_for_dead_hosts():
    devices = {"nvidia": [{"gpu_index": 0, "name": "L4", "mem_used_mb": 1, "mem_total_mb": 2, "processes": []}]}
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        if calls["n"] == 1:
            return [RemoteHost("a", devices=devices, hostname="node-a", version="0.5.0"),
                    RemoteHost("b", error="ssh timed out")]
        return [RemoteHost("a", error="Connection refused"),
                RemoteHost("b", devices=devices, hostname="node-b")]

    col = ui.ClusterCollector(ssh=["a", "b"], interval=1, fetch=fetch)
    col.refresh()
    snap = col.snapshot()
    assert snap["cluster"] is True and snap["refreshes"] == 1
    a, b = snap["hosts"]
    assert a["host"] == "a" and a["stale"] is False and a["devices"] == devices and a["hostname"] == "node-a"
    assert b["error"] == "ssh timed out" and b["stale"] is False and b["devices"] == {}

    col.refresh()
    a, b = col.snapshot()["hosts"]
    # a died: last good devices kept, flagged stale, with the new error
    assert a["stale"] is True and a["error"] == "Connection refused" and a["devices"] == devices
    # b came back
    assert b["stale"] is False and b["error"] is None and b["hostname"] == "node-b"


def test_collector_survives_fetch_exceptions_and_runs_in_background():
    def fetch():
        raise OSError("ssh binary missing")
    col = ui.ClusterCollector(ssh=["x"], interval=0.5, fetch=fetch)
    col.start()
    deadline = time.time() + 5
    while col.refreshes < 1 and time.time() < deadline:
        time.sleep(0.05)
    col.stop()
    (host,) = col.snapshot()["hosts"]
    assert host["host"] == "x" and "ssh binary missing" in host["error"]


def test_cluster_route_serves_collector_snapshot():
    col = ui.ClusterCollector(urls=["http://n1:9400"], interval=7,
                              fetch=lambda: [RemoteHost("http://n1:9400", devices={"amd": []}, hostname="n1")])
    col.refresh()
    server, base = _serve(cluster=col)
    try:
        payload = json.loads(_get(base + "/cluster")[2])
        assert payload["cluster"] is True and payload["interval"] == 7
        assert payload["hosts"][0]["hostname"] == "n1"
    finally:
        server.shutdown(); server.server_close()


def test_hosts_file_and_split():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("# lab nodes\nhttp://node1:9400\n\nnode2:9400  # exporter\nuser@node3\nnode4\n")
        path = f.name
    try:
        entries = ui.read_hosts_file(path)
    finally:
        os.unlink(path)
    assert entries == ["http://node1:9400", "node2:9400", "user@node3", "node4"]
    urls, ssh = ui.split_hosts(entries)
    assert urls == ["http://node1:9400", "node2:9400"]
    assert ssh == ["user@node3", "node4"]


def test_cli_ui_flags(monkeypatch, tmp_path):
    from xpustat import cli
    seen = {}

    def fake_serve_ui(**kw):
        seen.update(kw)
    monkeypatch.setattr("xpustat.ui.serve_ui", fake_serve_ui)
    hosts = tmp_path / "hosts.txt"
    hosts.write_text("http://n1:9400\nn2\n")
    monkeypatch.setattr(sys, "argv", ["xpustat", "--ui", "--port", "8080", "--host", "0.0.0.0", "--cpu", "-i", "3",
                                      "--ssh", "n3", "--hosts-file", str(hosts), "--ssh-command", "xpustat --json"])
    cli.main()
    assert seen["port"] == 8080 and seen["host"] == "0.0.0.0" and seen["include_cpu"] is True
    col = seen["cluster"]
    assert col.urls == ["http://n1:9400"] and col.ssh == ["n3", "n2"]
    assert col.interval == 3 and col.command == "xpustat --json" and col.include_cpu is True
    assert col.local is True

    seen.clear()
    monkeypatch.setattr(sys, "argv", ["xpustat", "--ui", "--ssh", "n3", "--no-local"])
    cli.main()
    assert seen["cluster"].local is None and seen["cluster"].labels == ["n3"]

    seen.clear()
    monkeypatch.setattr(sys, "argv", ["xpustat", "--ui"])
    cli.main()
    assert seen["port"] == 9400 and seen["host"] == "127.0.0.1" and seen["cluster"] is None and seen["interval"] == 2.0


def test_collector_includes_local_machine_first():
    col = ui.ClusterCollector(ssh=["b"], interval=1, local=_collection,
                              fetch=lambda: [RemoteHost("b", devices={"amd": []}, hostname="node-b")])
    assert col.labels == ["local", "b"]
    col.refresh()
    local, b = col.snapshot()["hosts"]
    assert local["host"] == "local" and local["devices"]["nvidia"][0]["name"] == "L4" and local["xpustat_version"]
    assert b["hostname"] == "node-b"


def test_collector_local_query_failure_is_one_dead_host():
    def boom():
        raise RuntimeError("no tools")
    col = ui.ClusterCollector(interval=1, local=boom, fetch=lambda: [])
    col.refresh()
    (local,) = col.snapshot()["hosts"]
    assert local["error"] == "RuntimeError: no tools" and local["devices"] == {}
