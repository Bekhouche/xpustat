"""
Shared process info helpers across all vendors.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import psutil  # type: ignore[import-untyped]

_MIB = 1024 * 1024

# cgroup paths that identify a container or a scheduler job, cgroup v1 and v2.
_DOCKER = re.compile(r"docker[-/]([0-9a-f]{12,64})")
_CONTAINERD = re.compile(r"(?:cri-containerd[-:]|containerd[-/])([0-9a-f]{12,64})")
_PODMAN = re.compile(r"libpod-([0-9a-f]{12,64})")
_K8S_POD = re.compile(r"kubepods[^/]*/(?:[^/]+/)*pod([0-9a-f\-_]{36})")
_SLURM = re.compile(r"slurm[^/]*/uid_\d+/job_(\d+)")


class ProcessInfo:
    """One running process using accelerator memory."""

    __slots__ = ("pid", "username", "command", "gpu_mem_mb",
                 "sm_pct", "mem_pct", "enc_pct", "dec_pct",
                 "cmdline", "rss_mb", "cpu_pct", "container")

    def __init__(
        self,
        pid: int,
        username: str,
        command: str,
        gpu_mem_mb: Optional[int],
        sm_pct: Optional[int] = None,
        mem_pct: Optional[int] = None,
        enc_pct: Optional[int] = None,
        dec_pct: Optional[int] = None,
        cmdline: Optional[str] = None,
        rss_mb: Optional[int] = None,
        cpu_pct: Optional[float] = None,
        container: Optional[str] = None,
    ) -> None:
        self.pid = pid
        self.username = username
        self.command = command
        self.gpu_mem_mb = gpu_mem_mb
        #: Device compute (SM) utilization of this process, percent; NVIDIA.
        self.sm_pct = sm_pct
        #: Device memory-bandwidth utilization of this process, percent; NVIDIA.
        self.mem_pct = mem_pct
        self.enc_pct = enc_pct
        self.dec_pct = dec_pct
        #: Full command line, when readable.
        self.cmdline = cmdline
        #: Host resident memory, MB.
        self.rss_mb = rss_mb
        #: Host CPU usage of the process, percent of one core.
        self.cpu_pct = cpu_pct
        #: ``docker:<12 hex>``, ``containerd:<id>``, ``pod:<uid>``, ``slurm:<jobid>``
        #: or ``None`` when the process runs on the bare host.
        self.container = container

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "username": self.username,
            "command": self.command,
            "gpu_mem_mb": self.gpu_mem_mb,
            "sm_pct": self.sm_pct,
            "mem_pct": self.mem_pct,
            "enc_pct": self.enc_pct,
            "dec_pct": self.dec_pct,
            "cmdline": self.cmdline,
            "rss_mb": self.rss_mb,
            "cpu_pct": self.cpu_pct,
            "container": self.container,
        }

    def __repr__(self) -> str:
        mem = f"{self.gpu_mem_mb}M" if self.gpu_mem_mb is not None else "?"
        return f"{self.username}:{self.command}/{self.pid}({mem})"


def enrich_process(pid: int, fallback_cmd: str) -> Tuple[str, str]:
    """
    Return (username, command) for a PID.
    Falls back to ('?', fallback_cmd) if the process has already exited
    or access is denied.
    """
    try:
        p = psutil.Process(pid)
        username = p.username()
        try:
            cmdline = p.cmdline()
            command = os.path.basename(cmdline[0]) if cmdline else fallback_cmd
        except psutil.AccessDenied:
            command = fallback_cmd
        return username, command
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return "?", fallback_cmd


def host_details(pid: int) -> Tuple[Optional[str], Optional[int], Optional[float], Optional[str]]:
    """
    Return (cmdline, rss_mb, cpu_pct, container) for a PID, each ``None`` when
    unreadable. Cheap: one ``psutil.Process`` and one ``/proc/<pid>/cgroup`` read.
    ``cpu_pct`` is measured since the previous call for that PID in this
    process, so the first reading is 0.
    """
    cmdline = None
    rss_mb = None
    cpu_pct = None
    try:
        p = psutil.Process(pid)
        with p.oneshot():
            try:
                args = p.cmdline()
                cmdline = " ".join(args) if args else None
            except (psutil.AccessDenied, psutil.ZombieProcess):
                pass
            try:
                rss_mb = int(p.memory_info().rss) // _MIB
            except (psutil.AccessDenied, psutil.ZombieProcess):
                pass
            try:
                cpu_pct = round(p.cpu_percent(interval=None), 1)
            except (psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        pass
    return cmdline, rss_mb, cpu_pct, container_of(pid)


def container_of(pid: int) -> Optional[str]:
    """Identify the container or scheduler job of *pid* from its cgroup (Linux only)."""
    try:
        with open(f"/proc/{pid}/cgroup", "r") as f:
            text = f.read()
    except OSError:
        return None
    return parse_cgroup(text)


def parse_cgroup(text: str) -> Optional[str]:
    """``parse_cgroup("0::/system.slice/docker-35dc31c3...scope")`` -> ``"docker:35dc31c32251"``."""
    for pattern, kind in ((_DOCKER, "docker"), (_CONTAINERD, "containerd"), (_PODMAN, "podman")):
        m = pattern.search(text)
        if m:
            return f"{kind}:{m.group(1)[:12]}"
    m = _K8S_POD.search(text)
    if m:
        return f"pod:{m.group(1)}"
    m = _SLURM.search(text)
    if m:
        return f"slurm:{m.group(1)}"
    return None


def enrich_full(pid: int, fallback_cmd: str, **kw) -> ProcessInfo:
    """Build a ProcessInfo with user, command and the host details in one go."""
    username, command = enrich_process(pid, fallback_cmd)
    cmdline, rss_mb, cpu_pct, container = host_details(pid)
    return ProcessInfo(pid=pid, username=username, command=command,
                       cmdline=cmdline, rss_mb=rss_mb, cpu_pct=cpu_pct, container=container, **kw)
