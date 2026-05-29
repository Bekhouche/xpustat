import pytest
from unittest.mock import patch, MagicMock
from xpustat._process import ProcessInfo, enrich_process
import psutil

def test_process_info():
    p = ProcessInfo(pid=1234, username="testuser", command="python script.py", gpu_mem_mb=500)
    assert p.pid == 1234
    assert p.username == "testuser"
    assert p.command == "python script.py"
    assert p.gpu_mem_mb == 500
    
    d = p.to_dict()
    assert d["pid"] == 1234
    assert d["username"] == "testuser"
    assert d["command"] == "python script.py"
    assert d["gpu_mem_mb"] == 500
    
    assert repr(p) == "testuser:python script.py/1234(500M)"

@patch("psutil.Process")
def test_enrich_process_success(mock_process_cls):
    mock_proc = MagicMock()
    mock_proc.username.return_value = "mockuser"
    mock_proc.cmdline.return_value = ["/usr/bin/python", "app.py"]
    mock_process_cls.return_value = mock_proc
    
    username, command = enrich_process(9999, "fallback")
    assert username == "mockuser"
    assert command == "python"

@patch("psutil.Process")
def test_enrich_process_access_denied(mock_process_cls):
    mock_proc = MagicMock()
    mock_proc.username.return_value = "mockuser"
    mock_proc.cmdline.side_effect = psutil.AccessDenied()
    mock_process_cls.return_value = mock_proc
    
    username, command = enrich_process(9999, "fallback")
    assert username == "mockuser"
    assert command == "fallback"

@patch("psutil.Process")
def test_enrich_process_no_process(mock_process_cls):
    mock_process_cls.side_effect = psutil.NoSuchProcess(9999)
    
    username, command = enrich_process(9999, "fallback")
    assert username == "?"
    assert command == "fallback"
