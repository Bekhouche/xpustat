import sys
import pytest
from unittest.mock import patch, MagicMock
from xpustat import cli
import argparse

@patch("xpustat.cli._render")
def test_print_xpustat_snapshot(mock_render):
    cli.print_xpustat()
    mock_render.assert_called_once()
    args = mock_render.call_args[1]
    assert args["filter_vendors"] is None
    assert args["show_util"] is False

@patch("xpustat.cli._render")
@patch("time.sleep", return_value=None)
@patch("time.monotonic", side_effect=[0, 1, 2, 3])
def test_print_xpustat_watch_mode(mock_monotonic, mock_sleep, mock_render):
    mock_render.return_value = ([], 0)   # _render returns (active_vendors, line_count)
    cli.print_xpustat(interval=1.0, count=2)
    assert mock_render.call_count == 2


@patch("xpustat.cli._render", side_effect=RuntimeError("boom"))
def test_print_xpustat_watch_mode_surfaces_errors(mock_render):
    # errors inside the watch loop must not be swallowed silently
    with pytest.raises(RuntimeError):
        cli.print_xpustat(interval=1.0, count=1)

def test_fields_extract():
    class DummyDev:
        name = "TestDev"
        processes = []
    
    idx, name, arch, used, tot, util, temp, pwr, procs = cli._fields("unknown", DummyDev())
    assert idx == "?"
    assert name.startswith("<") or "TestDev" in name
    assert used == 0
    assert tot == 0

def test_mem_bar():
    # used=500, total=1000 => 50%
    bar = cli._mem_bar("32", 500, 1000)
    assert "█" in bar
    assert "░" in bar

def test_fmt_cc():
    assert cli._fmt_cc("8.6") == "cc8.6"
    assert cli._fmt_cc(None) is None

def test_fmt_gfx():
    assert cli._fmt_gfx("gfx1100") == "gfx1100"
    assert cli._fmt_gfx(None) is None


def test_fmt_cores():
    assert cli._fmt_cores(10) == "10-core"
    assert cli._fmt_cores(None) is None


def test_fields_apple():
    from xpustat.apple import AppleGPUStat
    dev = AppleGPUStat(gpu_index=0, name="Apple M5", mem_used_mb=375,
                       mem_total_mb=32768, util_pct=15, gpu_cores=10)
    idx, name, arch, used, tot, util, temp, pwr, procs = cli._fields("apple", dev)
    assert idx == "0"
    assert name == "Apple M5"
    assert arch == "10-core"
    assert (used, tot, util) == (375, 32768, 15)
    assert temp is None and pwr is None
    assert procs == []


def test_proc_repr_separates_user_and_command():
    from xpustat._process import ProcessInfo
    p = ProcessInfo(pid=42, username="root", command="python", gpu_mem_mb=1024)
    cli._USE_COLOR = False
    assert cli._proc_repr(p, show_user=True, show_pid=True, show_cmd=True) == "root:python/42(1024M)"
    assert cli._proc_repr(p, show_user=True, show_pid=False, show_cmd=False) == "root(1024M)"
    assert cli._proc_repr(p, show_user=False, show_pid=True, show_cmd=True) == "python/42(1024M)"


def test_fields_generic_fallback_uses_mixin():
    from xpustat._device import DeviceMixin
    class PluginDev(DeviceMixin):
        vendor = "acme"; _index_attr = "idx"
        def __init__(self):
            self.idx = 3; self.name = "Acme X1"; self.mem_used_mb = 10; self.mem_total_mb = 100
            self.util_pct = 7; self.temp_c = 40; self.power_w = 12.5; self.processes = []
    idx, name, arch, used, tot, util, temp, pwr, procs = cli._fields("acme", PluginDev())
    assert (idx, name, arch, used, tot, util, temp, pwr, procs) == ("3", "Acme X1", None, 10, 100, 7, 40, 12.5, [])


def test_cli_select_prints_env(monkeypatch, capsys):
    from xpustat.nvidia import NvidiaGPUStat, NvidiaGPUStatCollection
    from xpustat.xpu import XPUStatCollection
    from xpustat.amd import AMDGPUStatCollection
    from xpustat.gaudi import GaudiAIPStatCollection
    from xpustat.intel_gpu import IntelGPUStatCollection
    from xpustat.huawei import HuaweiNPUStatCollection
    from xpustat.hygon import HygonDCUStatCollection
    from xpustat.cambricon import CambriconMLUStatCollection
    from xpustat.moorethreads import MooreThreadsGPUStatCollection
    gpus = [NvidiaGPUStat(gpu_index=i, uuid=f"GPU-{i}", name="L4", mem_used_mb=u, mem_total_mb=24000)
            for i, u in ((0, 20000), (1, 100), (2, 5000))]
    stats = XPUStatCollection(NvidiaGPUStatCollection(gpus), AMDGPUStatCollection([]), GaudiAIPStatCollection([]),
                              IntelGPUStatCollection([]), HuaweiNPUStatCollection([]), HygonDCUStatCollection([]),
                              CambriconMLUStatCollection([]), MooreThreadsGPUStatCollection([]))
    import xpustat._device as dev_mod
    monkeypatch.setattr(dev_mod, "query_all", lambda *a, **k: stats, raising=False)
    monkeypatch.setattr("xpustat.xpu.query_all", lambda *a, **k: stats)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    monkeypatch.setattr(sys, "argv", ["xpustat", "--select", "2"])
    cli.main()
    assert capsys.readouterr().out == "CUDA_VISIBLE_DEVICES=1,2\n"

    monkeypatch.setattr(sys, "argv", ["xpustat", "--select", "1", "--uuid", "--by", "mem_used"])
    cli.main()
    assert capsys.readouterr().out == "CUDA_VISIBLE_DEVICES=GPU-1\n"

    monkeypatch.setattr(sys, "argv", ["xpustat", "--select", "1", "--min-free", "99999"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1
