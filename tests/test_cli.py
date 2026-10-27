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


def test_new_columns_and_process_repr():
    from xpustat.nvidia import NvidiaGPUStat
    from xpustat._process import ProcessInfo
    cli._USE_COLOR = False
    g = NvidiaGPUStat(gpu_index=0, uuid="u", name="n", mem_used_mb=0, mem_total_mb=1, fan_pct=35,
                      clock_sm_mhz=1980, clock_mem_mhz=6250, enc_pct=12, dec_pct=3, pstate=0,
                      throttle_reasons=["power_cap", "hw_thermal"], ecc_uncorrected=2)
    assert cli._col_fan(g).strip() == "35%"
    assert cli._col_clocks(g).strip() == "1980/6250MHz"
    assert cli._col_codec(g).strip() == "enc 12% dec  3%"
    assert cli._col_health(g).strip() == "P0 power,HWtherm ECC:2"
    bare = NvidiaGPUStat(gpu_index=0, uuid="u", name="n", mem_used_mb=0, mem_total_mb=1)
    assert "N/A" in cli._col_fan(bare) and "N/A" in cli._col_health(bare)
    p = ProcessInfo(pid=7, username="salah", command="python", gpu_mem_mb=512, sm_pct=99,
                    cmdline="python train.py --epochs 3", container="docker:35dc31c32251")
    assert cli._proc_repr(p, True, True, True) == "salah:python/7(512M, 99%)"
    assert cli._proc_repr(p, True, True, True, full_cmd=True, show_container=True) == \
        "[docker:35dc31c32251] salah:python train.py --epochs 3/7(512M, 99%)"


def test_completion_scripts_and_id_filter(monkeypatch, capsys):
    import argparse
    for shell in ("bash", "zsh", "fish"):
        monkeypatch.setattr(sys, "argv", ["xpustat", "--print-completion", shell])
        cli.main()
        out = capsys.readouterr().out
        assert "--select" in out and "nvidia" in out and "xpustat" in out
    monkeypatch.setattr(sys, "argv", ["xpustat", "--id", "0,x"])
    with pytest.raises(SystemExit) as e:
        cli.main()
    assert e.value.code == 1


def test_doctor_runs_and_reports_each_backend(monkeypatch, capsys):
    from xpustat import doctor, xpu
    for name, cls in xpu._BACKENDS:
        monkeypatch.setattr(cls, "new_query", classmethod(lambda c: c([])))
    results = doctor.diagnose()
    names = [r["backend"] for r in results]
    assert "nvidia" in names and "cpu" in names and all("query_ms" in r for r in results)
    nv = next(r for r in results if r["backend"] == "nvidia")
    assert nv["binding"]["module"] == "pynvml" and nv["tools"][0]["name"] == "nvidia-smi"
    text = doctor.format_report(results)
    assert "[nvidia]" in text and "doctor" in text
    monkeypatch.setattr(sys, "argv", ["xpustat", "--doctor", "--json"])
    cli.main()
    import json
    assert isinstance(json.loads(capsys.readouterr().out), list)


def test_record_and_replay_roundtrip(tmp_path, monkeypatch, capsys):
    from xpustat import xpu
    from xpustat.apple import AppleGPUStat, AppleGPUStatCollection
    for name, cls in xpu._BACKENDS:
        monkeypatch.setattr(cls, "new_query", classmethod(lambda c: c([])))
    monkeypatch.setattr(AppleGPUStatCollection, "new_query",
                        classmethod(lambda c: c([AppleGPUStat(gpu_index=0, name="Apple M5", mem_used_mb=300, mem_total_mb=32768, util_pct=5)])))
    rec = tmp_path / "run.ndjson.gz"
    monkeypatch.setattr(sys, "argv", ["xpustat", "--record", str(rec), "-i", "0.01", "-n", "3"])
    cli.main()
    import gzip, json
    with gzip.open(rec, "rt") as f:
        frames = [json.loads(l) for l in f if l.strip()]
    assert len(frames) == 3 and frames[0]["devices"]["apple"][0]["name"] == "Apple M5" and "hostname" in frames[0]
    monkeypatch.setattr(sys, "argv", ["xpustat", "--replay", str(rec), "--no-color", "--no-header", "-e"])
    cli.main()
    out = capsys.readouterr().out
    assert out.count("[APPLE ]") == 3 and "Apple M5" in out and "5%" in out
    monkeypatch.setattr(sys, "argv", ["xpustat", "--replay", str(rec), "--json"])
    cli.main()
    assert len(json.loads(capsys.readouterr().out)) == 3
    empty = tmp_path / "empty.ndjson"; empty.write_text("")
    monkeypatch.setattr(sys, "argv", ["xpustat", "--replay", str(empty)])
    with pytest.raises(SystemExit):
        cli.main()


def test_remote_hosts_render_and_errors(monkeypatch, capsys):
    from xpustat import remote
    payload = {"hostname": "node1", "query_time": "t", "xpustat_version": "0.4.0",
               "devices": {"nvidia": [{"gpu_index": 2, "name": "NVIDIA L4", "mem_used_mb": 100, "mem_total_mb": 23034, "util_pct": 7,
                                        "processes": [{"pid": 1, "username": "u", "command": "python", "gpu_mem_mb": 90, "sm_pct": 3}]}]}}
    monkeypatch.setattr(remote, "_fetch", lambda url, timeout: remote._from_payload(url, payload) if "good" in url else remote.RemoteHost(url, error="refused"))
    hosts = remote.query_hosts(["http://good:9400", "bad:9400"])
    assert hosts[0].ok and hosts[0].hostname == "node1" and hosts[1].error == "refused"
    assert remote.devices_of(hosts[0])[0]["vendor"] == "nvidia"
    monkeypatch.setattr(sys, "argv", ["xpustat", "--hosts", "http://good:9400", "bad:9400", "--no-color", "--no-header", "-e", "-pu"])
    cli.main()
    out = capsys.readouterr().out
    assert "node1" in out and "[NVIDIA] #2" in out and "7%" in out and "u/1(90M, 3%)" in out and "unreachable: refused" in out
    monkeypatch.setattr(sys, "argv", ["xpustat", "--hosts", "http://good:9400", "--json"])
    cli.main()
    import json
    assert json.loads(capsys.readouterr().out)[0]["hostname"] == "node1"


def test_remote_ssh_parses_output(monkeypatch):
    from xpustat import remote
    class R:
        def __init__(self, rc, out, err=""): self.returncode, self.stdout, self.stderr = rc, out, err
    def fake_run(argv, **kw):
        host = argv[-2]
        if host == "ok":
            return R(0, 'warning line\n{"apple": [{"gpu_index": 0, "name": "M5", "mem_used_mb": 1, "mem_total_mb": 2}]}\n')
        return R(255, "", "ssh: connect to host bad: Connection refused")
    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    hosts = remote.query_ssh(["ok", "bad"], include_cpu=True)
    assert hosts[0].ok and hosts[0].devices["apple"][0]["name"] == "M5"
    assert hosts[1].error.startswith("ssh: connect")
