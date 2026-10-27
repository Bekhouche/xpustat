import json
import sys

import pytest

import xpustat
from xpustat import cpu as cpu_mod
from xpustat.cpu import CPUStat, CPUStatCollection, _cgroup_limits, _cpuinfo_field
from xpustat._process import ProcessInfo


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_cgroup_limits_parsing():
    assert _cgroup_limits("200000 100000\n", "8589934592\n") == (2.0, 8192)
    assert _cgroup_limits("max 100000\n", "max\n") == (None, None)
    assert _cgroup_limits("50000 100000", "garbage") == (0.5, None)
    assert _cgroup_limits("", "") == (None, None)


def test_cpuinfo_field():
    info = "processor\t: 0\nmodel name\t: AMD Ryzen 9 9950X   16-Core Processor\nphysical id\t: 0\n"
    assert _cpuinfo_field(info, "model name") == "AMD Ryzen 9 9950X 16-Core Processor"
    assert _cpuinfo_field(info, "missing") is None


def test_cpustat_fields_and_dict():
    c = CPUStat(cpu_index=0, name="Test CPU", mem_used_mb=4000, mem_total_mb=16000, mem_available_mb=11000,
                util_pct=12, cores_physical=8, cores_logical=16, core_layout="4P+4E", load_avg=(1.0, 0.5, 0.25),
                per_core_pct=[1, 2], swap_used_mb=0, swap_total_mb=2048, cgroup_cpu_limit=2.0,
                processes=[ProcessInfo(pid=1, username="u", command="python", gpu_mem_mb=512)])
    assert c.vendor == "cpu" and c.device_index == 0 and c.index_label == "0"
    assert c.mem_free_mb == 11000                       # "available", not total - used
    assert c.torch_device == "cpu" and c.framework_index == 0
    assert c.ray_resource == "CPU" and c.k8s_resource == "cpu"
    d = c.to_dict()
    assert d["load_avg"] == [1.0, 0.5, 0.25] and d["core_layout"] == "4P+4E" and d["cgroup_cpu_limit"] == 2.0
    assert d["processes"][0]["gpu_mem_mb"] == 512
    json.dumps(d)
    assert CPUStatCollection([c])[0] is c and CPUStatCollection([]).to_dict() == {"cpus": []}
    no_avail = CPUStat(cpu_index=0, name="x", mem_used_mb=10, mem_total_mb=100)
    assert no_avail.mem_free_mb == 90


# ---------------------------------------------------------------------------
# Real query on this host (psutil is a dependency, works everywhere)
# ---------------------------------------------------------------------------

def test_new_query_on_host():
    col = CPUStatCollection.new_query(top_n=3, sample=0.05)
    assert len(col) == 1
    c = col[0]
    assert c.name and c.mem_total_mb > 0 and 0 <= c.mem_used_mb <= c.mem_total_mb
    assert c.cores_logical and c.cores_logical >= 1
    assert c.util_pct is None or 0 <= c.util_pct <= 100
    assert c.per_core_pct is None or len(c.per_core_pct) == c.cores_logical
    assert len(c.processes) <= 3 and all(p.gpu_mem_mb is not None for p in c.processes)
    # second call in the process uses the delta since the first: no sampling pause
    import time
    t = time.perf_counter(); CPUStatCollection.new_query(top_n=0); dt = time.perf_counter() - t
    assert dt < 1.0


def test_new_query_never_raises(monkeypatch):
    monkeypatch.setattr(cpu_mod, "_query", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert len(CPUStatCollection.new_query()) == 0


# ---------------------------------------------------------------------------
# Opt-in semantics: nothing changes unless asked
# ---------------------------------------------------------------------------

def test_query_all_cpu_is_opt_in(monkeypatch):
    from xpustat import xpu
    for name, cls in xpu._BACKENDS:
        monkeypatch.setattr(cls, "new_query", classmethod(lambda c: c([])))
    s = xpustat.query_all(parallel=False)
    assert len(s.cpu) == 0 and "cpu" not in s.to_dict() and len(s) == 0
    assert xpustat.pick_device(stats=s) == []
    c = xpustat.query_all(parallel=False, include_cpu=True)
    assert len(c.cpu) == 1 and "cpu" in c.to_dict() and len(c) == 1
    assert list(c)[0].vendor == "cpu"
    assert xpustat.pick_device(stats=c)[0].vendor == "cpu"
    assert xpustat.pick_device(stats=c, vendor="nvidia") == []
    assert "cpu" in xpustat.to_prometheus(c) and 'vendor="cpu"' not in xpustat.to_prometheus(s)
    # the cache keys the two apart
    assert xpustat.query_all(parallel=False, max_age=10) is not xpustat.query_all(parallel=False, include_cpu=True, max_age=10)
    assert xpustat.query("cpu")[0].vendor == "cpu"


def test_cli_cpu_flag(monkeypatch, capsys):
    from xpustat import cli, xpu
    for name, cls in xpu._BACKENDS:
        monkeypatch.setattr(cls, "new_query", classmethod(lambda c: c([])))
    monkeypatch.setattr(sys, "argv", ["xpustat", "--json"])
    cli.main()
    assert "cpu" not in json.loads(capsys.readouterr().out)
    monkeypatch.setattr(sys, "argv", ["xpustat", "--json", "--cpu"])
    cli.main()
    out = json.loads(capsys.readouterr().out)
    assert out["cpu"][0]["cores_logical"] >= 1 and out["nvidia"] == []
    monkeypatch.setattr(sys, "argv", ["xpustat", "--cpu", "--no-color", "--no-header", "-A", "-u"])
    cli.main()
    text = capsys.readouterr().out
    assert text.startswith("[CPU   ]") and "threads" in text and "avail" in text
    monkeypatch.setattr(sys, "argv", ["xpustat", "--no-color", "--no-header"])
    cli.main()
    assert "no accelerators detected" in capsys.readouterr().out


def test_mhz_normalisation():
    from xpustat.cpu import _mhz
    assert _mhz(4.0) == 4000 and _mhz(3500.0) == 3500 and _mhz(0) is None and _mhz(None) is None


def test_clean_model_name():
    from xpustat.cpu import _clean_model_name
    assert _clean_model_name("Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz") == "Intel Xeon Silver 4314"
    assert _clean_model_name("AMD Ryzen 9 9950X3D 16-Core Processor") == "AMD Ryzen 9 9950X3D"
    assert _clean_model_name("AMD Ryzen 7 7840U w/ Radeon 780M Graphics") == "AMD Ryzen 7 7840U w/ Radeon 780M Graphics"
    assert _clean_model_name("Apple M5") == "Apple M5"
    assert _clean_model_name("(R)") == "(R)"          # never return an empty name
