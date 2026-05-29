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
    cli.print_xpustat(interval=1.0, count=2)
    assert mock_render.call_count == 2

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
