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


# ---------------------------------------------------------------------------
# container attribution and host details
# ---------------------------------------------------------------------------

from xpustat._process import parse_cgroup, host_details, enrich_full, ProcessInfo


def test_parse_cgroup_variants():
    assert parse_cgroup("0::/system.slice/docker-35dc31c322519c413bf3512496b34984540f63c431d6aaf5136462d42b34d7e1.scope\n") == "docker:35dc31c32251"
    assert parse_cgroup("12:memory:/docker/0124dcd024a7f3e2\n") == "docker:0124dcd024a7"
    assert parse_cgroup("0::/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod1a2b3c4d_5e6f_7a8b_9c0d_1e2f3a4b5c6d.slice/cri-containerd-abcdef1234567890abcdef.scope") == "containerd:abcdef123456"
    assert parse_cgroup("0::/kubepods/burstable/pod1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d/other") == "pod:1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d"
    assert parse_cgroup("0::/machine.slice/libpod-9f8e7d6c5b4a3f2e1d0c.scope") == "podman:9f8e7d6c5b4a"
    assert parse_cgroup("3:cpuset:/slurm/uid_1000/job_123456/step_0") == "slurm:123456"
    assert parse_cgroup("1:freezer:/\n0::/system.slice/slurmd.service\n") is None
    assert parse_cgroup("") is None


def test_host_details_on_self_and_missing_pid():
    import os
    cmdline, rss, cpu, container = host_details(os.getpid())
    assert cmdline and "python" in cmdline.lower()
    assert rss and rss > 0
    assert cpu is None or cpu >= 0
    assert host_details(4194303) == (None, None, None, None)


def test_enrich_full_and_to_dict():
    import os
    p = enrich_full(os.getpid(), "fallback", gpu_mem_mb=42, sm_pct=7)
    assert p.pid == os.getpid() and p.gpu_mem_mb == 42 and p.sm_pct == 7 and p.rss_mb
    d = p.to_dict()
    assert set(d) == {"pid", "username", "command", "gpu_mem_mb", "sm_pct", "mem_pct", "enc_pct", "dec_pct",
                      "cmdline", "rss_mb", "cpu_pct", "container"}
    gone = enrich_full(4194303, "ghost", gpu_mem_mb=None)
    assert gone.username == "?" and gone.command == "ghost" and gone.cmdline is None and gone.container is None
    assert ProcessInfo(pid=1, username="u", command="c", gpu_mem_mb=None).to_dict()["sm_pct"] is None
