import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from xpustat import to_prometheus, XPUStatCollection
from xpustat._process import ProcessInfo
from xpustat.nvidia import NvidiaGPUStat, NvidiaGPUStatCollection
from xpustat.amd import AMDGPUStatCollection
from xpustat.gaudi import GaudiAIPStatCollection
from xpustat.intel_gpu import IntelGPUStatCollection
from xpustat.huawei import HuaweiNPUStat, HuaweiNPUStatCollection
from xpustat.hygon import HygonDCUStatCollection
from xpustat.cambricon import CambriconMLUStatCollection
from xpustat.moorethreads import MooreThreadsGPUStatCollection
from xpustat.apple import AppleGPUStat, AppleGPUStatCollection
from xpustat import prometheus as prom


def _collection():
    nv = NvidiaGPUStat(gpu_index=0, uuid="GPU-1", name='L4 "quoted"', mem_used_mb=1024, mem_total_mb=4096,
                       util_pct=33, temp_c=61, power_w=47.5, cuda_cc="8.9",
                       processes=[ProcessInfo(pid=42, username="salah", command="python", gpu_mem_mb=512),
                                  ProcessInfo(pid=43, username="x", command="y", gpu_mem_mb=None)])
    npu = HuaweiNPUStat(npu_id=0, chip_id=1, name="Ascend 910B", mem_used_mb=10, mem_total_mb=20, ai_core_pct=7)
    ap = AppleGPUStat(gpu_index=0, name="Apple M5", mem_used_mb=300, mem_total_mb=32768, util_pct=5, gpu_cores=10)
    return XPUStatCollection(
        nvidia=NvidiaGPUStatCollection([nv]), amd=AMDGPUStatCollection([]), gaudi=GaudiAIPStatCollection([]),
        intel=IntelGPUStatCollection([]), huawei=HuaweiNPUStatCollection([npu]), hygon=HygonDCUStatCollection([]),
        cambricon=CambriconMLUStatCollection([]), moorethreads=MooreThreadsGPUStatCollection([]),
        apple=AppleGPUStatCollection([ap]),
    )


def test_to_prometheus_format_and_values():
    out = to_prometheus(_collection())
    lines = out.splitlines()
    assert out.endswith("\n")
    assert "# HELP xpustat_memory_used_bytes" in out
    assert "# TYPE xpustat_memory_used_bytes gauge" in out
    # bytes conversion
    assert 'xpustat_memory_used_bytes{vendor="nvidia",index="0",name="L4 \\"quoted\\""} 1073741824' in lines
    assert 'xpustat_memory_free_bytes{vendor="nvidia",index="0",name="L4 \\"quoted\\""} 3221225472' in lines
    # optional metrics present only when the vendor reports them
    assert 'xpustat_utilization_percent{vendor="nvidia",index="0",name="L4 \\"quoted\\""} 33' in lines
    assert 'xpustat_utilization_percent{vendor="huawei",index="0:1",name="Ascend 910B"} 7' in lines
    assert 'xpustat_temperature_celsius{vendor="nvidia",index="0",name="L4 \\"quoted\\""} 61' in lines
    assert 'xpustat_power_watts{vendor="nvidia",index="0",name="L4 \\"quoted\\""} 47.5' in lines
    assert not any(l.startswith('xpustat_temperature_celsius{vendor="apple"') for l in lines)
    # processes: only those with known memory get a sample; count includes all
    assert 'xpustat_process_memory_bytes{vendor="nvidia",index="0",name="L4 \\"quoted\\"",pid="42",user="salah",command="python"} 536870912' in lines
    assert 'xpustat_process_count{vendor="nvidia",index="0",name="L4 \\"quoted\\""} 2' in lines
    # device info carries arch + torch device
    assert any(l.startswith('xpustat_device_info{') and 'arch="8.9"' in l and 'torch_device="cuda:0"' in l and 'uuid="GPU-1"' in l for l in lines)
    assert any('vendor="apple"' in l and 'arch="10-core"' in l and 'torch_device="mps"' in l for l in lines)
    # per-vendor device counts, including empty vendors
    assert 'xpustat_scrape_devices{vendor="amd"} 0' in lines
    assert 'xpustat_scrape_devices{vendor="nvidia"} 1' in lines


def test_to_prometheus_empty_collection():
    empty = XPUStatCollection(
        nvidia=NvidiaGPUStatCollection([]), amd=AMDGPUStatCollection([]), gaudi=GaudiAIPStatCollection([]),
        intel=IntelGPUStatCollection([]), huawei=HuaweiNPUStatCollection([]), hygon=HygonDCUStatCollection([]),
        cambricon=CambriconMLUStatCollection([]), moorethreads=MooreThreadsGPUStatCollection([]),
    )
    out = to_prometheus(empty)
    assert "xpustat_memory_used_bytes" not in out
    assert 'xpustat_scrape_devices{vendor="tenstorrent"} 0' in out


def test_http_server_serves_metrics_and_404():
    handler = type("H", (prom._Handler,), {"query": staticmethod(_collection)})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
            body = r.read().decode()
            assert r.status == 200
            assert r.headers["Content-Type"].startswith("text/plain")
            assert "xpustat_memory_used_bytes" in body
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=5)
            assert False, "expected 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown(); server.server_close()


def test_http_server_reports_query_errors_without_dying():
    def boom():
        raise RuntimeError("vendor tool exploded")
    handler = type("H", (prom._Handler,), {"query": staticmethod(boom)})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True); t.start()
    try:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5)
            assert False, "expected 500"
        except urllib.error.HTTPError as e:
            assert e.code == 500 and b"vendor tool exploded" in e.read()
    finally:
        server.shutdown(); server.server_close()


def test_prometheus_exports_mig_instances_as_devices():
    from xpustat.nvidia import MIGInstanceStat
    col = _collection()
    gpu = col.nvidia[0]
    gpu.mig_instances = [MIGInstanceStat(parent_index=0, uuid="MIG-1", mem_used_mb=100, mem_total_mb=10240,
                                         gpu_instance_id=3, compute_instance_id=0, profile="1g.10gb")]
    lines = to_prometheus(col).splitlines()
    assert 'xpustat_memory_total_bytes{vendor="nvidia",index="0/3.0",name="MIG 1g.10gb"} 10737418240' in lines
    assert any('index="0/3.0"' in l and 'uuid="MIG-1"' in l for l in lines if l.startswith("xpustat_device_info"))
    assert 'xpustat_scrape_devices{vendor="nvidia"} 1' in lines      # parent GPU count unchanged
