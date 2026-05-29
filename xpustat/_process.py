"""
Shared process info helpers across all vendors.
"""

from __future__ import annotations

import os
from typing import Optional

import psutil  # type: ignore[import-untyped]


class ProcessInfo:
    """One running process using accelerator memory."""

    __slots__ = ("pid", "username", "command", "gpu_mem_mb")

    def __init__(
        self,
        pid: int,
        username: str,
        command: str,
        gpu_mem_mb: Optional[int],
    ) -> None:
        self.pid = pid
        self.username = username
        self.command = command
        self.gpu_mem_mb = gpu_mem_mb

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "username": self.username,
            "command": self.command,
            "gpu_mem_mb": self.gpu_mem_mb,
        }

    def __repr__(self) -> str:
        mem = f"{self.gpu_mem_mb}M" if self.gpu_mem_mb is not None else "?"
        return f"{self.username}:{self.command}/{self.pid}({mem})"


def enrich_process(pid: int, fallback_cmd: str) -> tuple[str, str]:
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
