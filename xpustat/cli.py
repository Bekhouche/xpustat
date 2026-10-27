"""
xpustat CLI — display all detected accelerators in the terminal.
"""

from __future__ import annotations

import contextlib
import datetime
import io
import os
import signal
import socket
import sys
import time
from typing import List, Optional

from . import __version__
from .nvidia import _HAS_PYNVML
from .amd import _HAS_AMDSMI
from .xpu import query_all

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.name != "nt"


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _bold(text: str) -> str:
    return _c("1", text)


def _dim(text: str) -> str:
    return _c("2", text)


# ---------------------------------------------------------------------------
# Vendor config  (ANSI color, short label — padded to _LBL_W)
# ---------------------------------------------------------------------------

_VENDORS: dict = {
    "nvidia":       ("32",  "NVIDIA"),    # green
    "amd":          ("31",  "AMD"),       # red
    "gaudi":        ("36",  "GAUDI"),     # cyan
    "intel":        ("34",  "INTEL"),     # blue
    "huawei":       ("35",  "HUAWEI"),    # magenta
    "hygon":        ("33",  "HYGON"),     # yellow
    "cambricon":    ("96",  "CAMBR."),    # bright cyan
    "moorethreads": ("95",  "MTT"),       # bright magenta
    "apple":        ("37",  "APPLE"),     # white
    "neuron":       ("94",  "NEURON"),    # bright blue
    "tpu":          ("92",  "TPU"),       # bright green
    "tenstorrent":  ("93",  "TENST."),    # bright yellow
    "jetson":       ("32",  "JETSON"),    # green, like NVIDIA
    "cpu":          ("90",  "CPU"),       # grey; shown only with --cpu
    "rebellions":   ("91",  "RBLN"),      # bright red
    "furiosa":      ("96",  "FURIOS"),    # bright cyan
}

from .xpu import _BACKENDS_KEYS as _ALL_KEYS  # noqa: E402  (built-ins + plugins)
for _key in _ALL_KEYS:
    if _key not in _VENDORS:
        _VENDORS[_key] = ("37", _key[:6].upper())

_LBL_W = max(len(lbl) for _, lbl in _VENDORS.values())


def _vendor_tag(color: str, label: str) -> str:
    return _c(f"1;{color}", f"[{label:<{_LBL_W}}]")


# ---------------------------------------------------------------------------
# Per-vendor field extractor
# ---------------------------------------------------------------------------

def _fields(vendor: str, dev):
    """Return (index_str, name, arch_tag, mem_used_mb, mem_total_mb, util_pct, temp_c, power_w, processes)."""
    processes = getattr(dev, "processes", [])

    if vendor == "nvidia":
        return str(dev.gpu_index), dev.name, _fmt_cc(dev.cuda_cc), \
               dev.mem_used_mb, dev.mem_total_mb, \
               dev.util_pct, dev.temp_c, dev.power_w, processes
    if vendor == "amd":
        return str(dev.gpu_index), dev.name, _fmt_gfx(dev.gfx_target), \
               dev.mem_used_mb, dev.mem_total_mb, \
               dev.util_pct, dev.temp_c, dev.power_w, processes
    if vendor == "gaudi":
        return str(dev.aip_index), dev.name, None, \
               dev.mem_used_mb, dev.mem_total_mb, None, None, None, processes
    if vendor == "intel":
        return str(dev.device_id), dev.name, None, \
               dev.mem_used_mb, dev.mem_total_mb, None, None, None, processes
    if vendor == "huawei":
        return (f"{dev.npu_id}:{dev.chip_id}", dev.name, None,
                dev.mem_used_mb, dev.mem_total_mb,
                dev.ai_core_pct, dev.temp_c, dev.power_w, processes)
    if vendor == "hygon":
        return str(dev.card_index), dev.name, None, \
               dev.mem_used_mb, dev.mem_total_mb, None, None, None, processes
    if vendor == "cambricon":
        return (f"{dev.card_id}:{dev.chip_id}", dev.name, None,
                dev.mem_used_mb, dev.mem_total_mb,
                None, None, None, processes)
    if vendor == "moorethreads":
        return str(dev.gpu_id), dev.name, None, \
               dev.mem_used_mb, dev.mem_total_mb, None, None, None, processes
    if vendor == "apple":
        return str(dev.gpu_index), dev.name, _fmt_cores(dev.gpu_cores), \
               dev.mem_used_mb, dev.mem_total_mb, \
               dev.util_pct, dev.temp_c, dev.power_w, processes
    if vendor == "neuron":
        return str(dev.neuron_device), dev.name, _fmt_cores(dev.nc_count), \
               dev.mem_used_mb, dev.mem_total_mb, None, None, None, processes
    if vendor == "tpu":
        return str(dev.device_id), dev.name, None, \
               dev.mem_used_mb, dev.mem_total_mb, dev.util_pct, None, None, processes
    if vendor == "tenstorrent":
        return str(dev.tt_device), dev.name, None, \
               dev.mem_used_mb, dev.mem_total_mb, None, dev.temp_c, dev.power_w, processes
    if vendor == "jetson":
        return str(dev.gpu_index), dev.name, None, \
               dev.mem_used_mb, dev.mem_total_mb, dev.util_pct, dev.temp_c, dev.power_w, processes
    if vendor == "cpu":
        return str(dev.cpu_index), dev.name, _fmt_cpu_arch(dev), \
               dev.mem_used_mb, dev.mem_total_mb, dev.util_pct, dev.temp_c, dev.power_w, processes
    # Generic fallback (plugin backends): read the vendor-neutral fields.
    util = getattr(dev, "util_pct", None)
    if util is None:
        util = getattr(dev, "ai_core_pct", None)
    return (getattr(dev, "index_label", "?"), getattr(dev, "name", str(dev)), None,
            getattr(dev, "mem_used_mb", 0), getattr(dev, "mem_total_mb", 0),
            util, getattr(dev, "temp_c", None), getattr(dev, "power_w", None), processes)


def _fmt_cc(cc: Optional[str]) -> Optional[str]:
    """Format CUDA compute capability for display, e.g. 'cc8.6'."""
    return f"cc{cc}" if cc else None


def _fmt_gfx(gfx: Optional[str]) -> Optional[str]:
    """Format ROCm GFX target for display, e.g. 'gfx1100'."""
    return gfx if gfx else None


def _fmt_cores(cores: Optional[int]) -> Optional[str]:
    """Format an Apple GPU core count for display, e.g. '10-core'."""
    return f"{cores}-core" if cores else None


def _fmt_cpu_arch(dev) -> Optional[str]:
    """Arch tag for the CPU row: '4P+6E' when known, else '16c/32t', else the ISA."""
    if dev.core_layout:
        return dev.core_layout
    if dev.cores_physical and dev.cores_logical:
        return f"{dev.cores_physical}c/{dev.cores_logical}t"
    return dev.arch


def _cpu_detail(dev) -> str:
    """One dim line under the CPU row with the figures that have no column."""
    parts: List[str] = []
    if dev.cores_logical:
        cores = f"{dev.cores_physical or '?'} cores / {dev.cores_logical} threads"
        if dev.sockets and dev.sockets > 1:
            cores += f" on {dev.sockets} sockets"
        parts.append(cores)
    if dev.freq_mhz:
        parts.append(f"{dev.freq_mhz / 1000:.2f} GHz" + (f" (max {dev.freq_max_mhz / 1000:.2f})" if dev.freq_max_mhz else ""))
    if dev.load_avg:
        parts.append("load " + " ".join(f"{x:.2f}" for x in dev.load_avg))
    parts.append(f"avail {dev.mem_free_mb:,} MB")
    if dev.swap_total_mb:
        parts.append(f"swap {dev.swap_used_mb:,}/{dev.swap_total_mb:,} MB")
    if dev.cgroup_cpu_limit or dev.cgroup_mem_limit_mb:
        lim = []
        if dev.cgroup_cpu_limit:
            lim.append(f"{dev.cgroup_cpu_limit:g} CPUs")
        if dev.cgroup_mem_limit_mb:
            lim.append(f"{dev.cgroup_mem_limit_mb:,} MB")
        parts.append("cgroup limit " + ", ".join(lim))
    return "  ·  ".join(parts)


# ---------------------------------------------------------------------------
# Memory bar
# ---------------------------------------------------------------------------

_BAR_W = 16


def _mem_bar(color: str, used: int, total: int) -> str:
    if total <= 0:
        return _dim("─" * _BAR_W)
    ratio  = min(used / total, 1.0)
    filled = round(ratio * _BAR_W)
    return _c(f"1;{color}", "█" * filled) + _dim("░" * (_BAR_W - filled))


# ---------------------------------------------------------------------------
# Optional column renderers
# ---------------------------------------------------------------------------

def _col_util(util_pct) -> str:
    if util_pct is None:
        return _dim("  N/A")
    color = "31" if util_pct >= 90 else "33" if util_pct >= 60 else "32"
    return f"  {_c(color, f'{util_pct:3d}%')}"


def _col_temp(temp_c) -> str:
    if temp_c is None:
        return _dim("   N/A")
    color = "31" if temp_c >= 85 else "33" if temp_c >= 70 else "32"
    return f"  {_c(color, f'{temp_c:3d}°C')}"


def _col_power(power_w) -> str:
    if power_w is None:
        return _dim("    N/A")
    return f"  {_c('36', f'{power_w:6.1f}W')}"


def _col_fan(dev) -> str:
    fan = getattr(dev, "fan_pct", None)
    if fan is None:
        rpm = getattr(dev, "fan_rpm", None)
        return f"  {_c('36', f'{rpm:5d}rpm')}" if rpm is not None else _dim("     N/A")
    return f"  {_c('36', f'{fan:3d}%')}" + " " * 3


def _col_clocks(dev) -> str:
    sm = getattr(dev, "clock_sm_mhz", None)
    if sm is None:
        sm = getattr(dev, "clock_gfx_mhz", None)
    mem = getattr(dev, "clock_mem_mhz", None)
    if sm is None and mem is None:
        return _dim("          N/A")
    a = f"{sm}" if sm is not None else "-"
    b = f"{mem}" if mem is not None else "-"
    return f"  {_c('35', f'{a:>4}/{b:<4}MHz')}"


def _col_codec(dev) -> str:
    enc = getattr(dev, "enc_pct", None)
    dec = getattr(dev, "dec_pct", None)
    if enc is None and dec is None:
        return _dim("        N/A")
    return f"  {_dim('enc')}{enc if enc is not None else '-':>3}% {_dim('dec')}{dec if dec is not None else '-':>3}%"


_THROTTLE_SHORT = {"idle": "idle", "app_clocks": "appclk", "power_cap": "power", "hw_slowdown": "HW",
                   "sync_boost": "sync", "sw_thermal": "thermal", "hw_thermal": "HWtherm",
                   "hw_power_brake": "HWbrake", "display_clocks": "display"}


def _col_health(dev) -> str:
    """P-state / perf level, active throttle reasons, ECC uncorrected count."""
    bits: List[str] = []
    pstate = getattr(dev, "pstate", None)
    if pstate is not None:
        bits.append(f"P{pstate}")
    elif getattr(dev, "perf_level", None):
        bits.append(str(dev.perf_level))
    reasons = [r for r in (getattr(dev, "throttle_reasons", None) or []) if r != "idle"]
    if reasons:
        bits.append(_c("33", ",".join(_THROTTLE_SHORT.get(r, r) for r in reasons)))
    elif getattr(dev, "throttled", None) is True:
        bits.append(_c("33", "throttled"))
    ecc = getattr(dev, "ecc_uncorrected", None)
    if ecc:
        bits.append(_c("31", f"ECC:{ecc}"))
    elif ecc == 0:
        bits.append(_c("32", "ECC:0"))
    if not bits:
        return _dim("  N/A")
    return "  " + " ".join(bits)


def _proc_repr(p, show_user: bool, show_pid: bool, show_cmd: bool,
               full_cmd: bool = False, show_container: bool = False) -> str:
    parts: List[str] = []
    if show_container and p.container:
        parts.append(_c("34", f"[{p.container}] "))
    if show_user:
        parts.append(_c("32", p.username))
    if show_cmd:
        if show_user:
            parts.append(_dim(":"))      # user:command, gpustat style
        cmd = p.command
        if full_cmd and p.cmdline:
            cmd = p.cmdline if len(p.cmdline) <= 80 else p.cmdline[:79] + "…"
        parts.append(_c("1", cmd))
    if show_pid:
        parts.append(_dim(f"/{p.pid}"))
    mem = f"{p.gpu_mem_mb}M" if p.gpu_mem_mb is not None else ""
    if p.sm_pct is not None:
        mem += f"{', ' if mem else ''}{p.sm_pct}%"
    parts.append(_c("33", f"({mem})") if mem else "")
    return "".join(parts)


# ---------------------------------------------------------------------------
# SDK tip helper
# ---------------------------------------------------------------------------

def _sdk_tips(collection) -> List[str]:
    """
    Return tip strings for SDKs that would speed up detected vendors but are
    not currently installed.
    """
    tips: List[str] = []
    if len(collection.nvidia) > 0 and not _HAS_PYNVML:
        tips.append("pip install nvidia-ml-py  → faster NVIDIA queries (direct NVML, no subprocess)")
    if len(collection.amd) > 0 and not _HAS_AMDSMI:
        tips.append("pip install amdsmi        → faster AMD queries (ROCm Python bindings, no subprocess)")
    return tips


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

_NAME_W = 32
_MEM_W  = 6
_ARCH_W = 8   # max width of arch tag without ANSI (e.g. "gfx1100x" is 8)


def _render(
    filter_vendors: Optional[List[str]] = None,
    show_util: bool = False,
    show_temp: bool = False,
    show_power: bool = False,
    show_user: bool = False,
    show_pid: bool = False,
    show_cmd: bool = False,
    show_arch: bool = False,
    no_header: bool = False,
    active_vendors: Optional[List[str]] = None,
    include_cpu: bool = False,
    show_fan: bool = False,
    show_clocks: bool = False,
    show_codec: bool = False,
    show_health: bool = False,
    full_cmd: bool = False,
    show_container: bool = False,
    ids: Optional[List[int]] = None,
) -> tuple:
    n = 0

    def _p(s: str = "") -> None:
        nonlocal n
        print(s)
        n += 1

    collection = query_all(parallel=True, vendors=active_vendors, include_cpu=include_cpu)

    # ── header ─────────────────────────────────────────────────────────────
    if not no_header:
        hostname  = socket.gethostname()
        now       = datetime.datetime.now().strftime("%a %b %d  %H:%M:%S  %Y")
        title     = _bold("xpustat") + f" {__version__}"
        dots      = _dim("  ·  ")
        header    = f"{title}{dots}{_bold(hostname)}{dots}{now}"
        plain_len = len(f"xpustat {__version__}  ·  {hostname}  ·  {now}")
        _p(header)
        _p(_dim("─" * plain_len))

    visible = [
        (v, s) for v, s in _VENDORS.items()
        if (filter_vendors is None or v in filter_vendors)
        and list(getattr(collection, v))
    ]

    if not visible:
        _p(_dim("  no accelerators detected"))
        return [], n

    show_procs = show_user or show_pid or show_cmd

    for vendor, (color, label) in visible:
        for dev in getattr(collection, vendor):
            if ids is not None and getattr(dev, "device_index", None) not in ids:
                continue
            idx, name, arch_tag, used, total, util, temp, power, procs = _fields(vendor, dev)

            if len(name) > _NAME_W:
                name = name[:_NAME_W - 1] + "…"

            # ── arch tag (cc8.6 / gfx1100) ─────────────────────────────────
            arch_col = ""
            if show_arch:
                arch_str = arch_tag or "N/A"
                arch_col = f"  {_c('2;36', arch_str):<{_ARCH_W + 9}}"

            # ── core columns ───────────────────────────────────────────────
            tag      = _vendor_tag(color, label)
            index    = _bold(f" #{idx:<3}")
            name_col = f"  {name:<{_NAME_W}}"
            sep      = _dim("  │  ")
            mem_used = _c("1;33", f"{used:>{_MEM_W},}")
            mem_tot  = _dim(f"{total:>{_MEM_W},} MB")
            bar      = f"  {_mem_bar(color, used, total)}"
            pct_str  = f"  {int(used / total * 100):3d}%" if total > 0 else "    N/A"

            line = f"{tag}{index}{name_col}{arch_col}{sep}{mem_used} {_dim('/')} {mem_tot}{bar}{pct_str}"

            # ── optional columns ────────────────────────────────────────────
            if show_util:  line += _col_util(util)
            if show_temp:  line += _col_temp(temp)
            if show_power: line += _col_power(power)
            if show_fan:    line += _col_fan(dev)
            if show_clocks: line += _col_clocks(dev)
            if show_codec:  line += _col_codec(dev)
            if show_health: line += _col_health(dev)

            # ── health (Huawei only) ────────────────────────────────────────
            health = getattr(dev, "health", None)
            if health and health != "OK":
                line += f"  {_c('31', health)}"

            _p(line)

            # ── CPU detail line ─────────────────────────────────────────────
            if vendor == "cpu":
                _p(" " * (_LBL_W + 8) + _dim(_cpu_detail(dev)))

            # ── MIG instances (NVIDIA in MIG mode) ──────────────────────────
            for m in getattr(dev, "mig_instances", []) or []:
                indent = " " * (_LBL_W + 4)
                m_used = _c("1;33", f"{m.mem_used_mb:>{_MEM_W},}")
                m_tot  = _dim(f"{m.mem_total_mb:>{_MEM_W},} MB")
                m_pct  = f"  {int(m.mem_used_mb / m.mem_total_mb * 100):3d}%" if m.mem_total_mb > 0 else "    N/A"
                _p(f"{indent}{_dim('├─')} {_bold(m.index_label):<10} {m.name:<{_NAME_W - 8}}{sep}"
                   f"{m_used} {_dim('/')} {m_tot}  {_mem_bar(color, m.mem_used_mb, m.mem_total_mb)}{m_pct}  {_dim(m.uuid)}")

            # ── process sub-lines ───────────────────────────────────────────
            if show_procs and procs:
                indent = " " * (_LBL_W + 8)   # align under device name
                proc_str = "  ".join(
                    _proc_repr(p, show_user, show_pid, show_cmd, full_cmd, show_container)
                    for p in procs
                )
                _p(f"{indent}{_dim('└─')} {proc_str}")

            # ── SDK speed tips (only shown when header is visible) ─────────────────
    if not no_header:
        tips = _sdk_tips(collection)
        if tips:
            _p(_dim("  tip: install native SDK for faster queries:"))
            for tip in tips:
                _p(_dim(f"       {tip}"))

    return [v for v, s in visible], n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def print_xpustat(
    use_color: Optional[bool] = None,
    interval: Optional[float] = None,
    count: Optional[int] = None,
    filter_vendors: Optional[List[str]] = None,
    show_util: bool = False,
    show_temp: bool = False,
    show_power: bool = False,
    show_user: bool = False,
    show_pid: bool = False,
    show_cmd: bool = False,
    show_arch: bool = False,
    no_header: bool = False,
    include_cpu: bool = False,
    show_fan: bool = False,
    show_clocks: bool = False,
    show_codec: bool = False,
    show_health: bool = False,
    full_cmd: bool = False,
    show_container: bool = False,
    ids: Optional[List[int]] = None,
) -> None:
    global _USE_COLOR
    if use_color is not None:
        _USE_COLOR = use_color

    kwargs = dict(
        filter_vendors=filter_vendors,
        show_util=show_util, show_temp=show_temp, show_power=show_power,
        show_user=show_user, show_pid=show_pid, show_cmd=show_cmd,
        show_arch=show_arch, no_header=no_header, include_cpu=include_cpu,
        show_fan=show_fan, show_clocks=show_clocks, show_codec=show_codec, show_health=show_health,
        full_cmd=full_cmd, show_container=show_container, ids=ids,
    )

    if interval is None or interval <= 0:
        _render(**kwargs)  # return value (vendors, lines) unused in single-shot
        return

    # ── watch loop ──────────────────────────────────────────────────────────
    # On a POSIX tty, draw each frame in place on the alternate screen:
    # render into a buffer first, then overwrite the previous frame in a
    # single write, so the screen is never blank while devices are queried.
    in_place = sys.stdout.isatty() and os.name != "nt"
    _CLEAR = "cls" if os.name == "nt" else "clear"
    iteration = 0
    _stop = False
    active_vendors = None

    def _handle_sigint(*_):
        nonlocal _stop
        _stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    if in_place:
        sys.stdout.write("\033[?1049h\033[?25l")  # alt screen, hide cursor

    try:
        while not _stop:
            kwargs["active_vendors"] = active_vendors
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                active_vendors, _ = _render(**kwargs)
                print(_dim(f"  refreshing every {interval}s  ·  Ctrl+C to quit"))
            frame = buf.getvalue()

            if in_place:
                # home, then clear to end of each line and below the frame
                lines = frame.rstrip("\n").split("\n")
                sys.stdout.write("\033[H" + "\033[K\n".join(lines) + "\033[K\033[J")
            else:
                if iteration > 0:
                    sys.stdout.flush()
                    os.system(_CLEAR)
                sys.stdout.write(frame)
            sys.stdout.flush()

            iteration += 1
            if count and iteration >= count:
                break
            deadline = time.monotonic() + interval
            while not _stop and time.monotonic() < deadline:
                time.sleep(0.1)
    finally:
        if in_place:
            sys.stdout.write("\033[?25h\033[?1049l")  # show cursor, main screen
            sys.stdout.flush()


# ---------------------------------------------------------------------------
# Remote hosts, record and replay
# ---------------------------------------------------------------------------

def _frame_lines(devices: dict, host: str, args, width: int = 0) -> List[str]:
    """Render a ``to_dict()`` payload (local or remote) as table lines with a host prefix."""
    from .xpu import XPUStatCollection
    lines: List[str] = []
    show_all = args.show_all
    tag_w = max(width, len(host))
    prefix = _c("1;34", f"{host:<{tag_w}} ") if host else ""
    show_procs = show_all or args.show_user or args.show_pid or args.show_cmd
    for vendor, devs in devices.items():
        if args.filter and vendor not in args.filter:
            continue
        color, label = _VENDORS.get(vendor, ("37", vendor[:6].upper()))
        for d in devs or []:
            name = str(d.get("name", "?"))
            if len(name) > _NAME_W:
                name = name[:_NAME_W - 1] + "…"
            used, total = int(d.get("mem_used_mb") or 0), int(d.get("mem_total_mb") or 0)
            idx = _index_label_from_dict(vendor, d)
            line = (f"{prefix}{_vendor_tag(color, label)}{_bold(f' #{idx:<3}')}  {name:<{_NAME_W}}"
                    f"{_dim('  │  ')}{_c('1;33', f'{used:>{_MEM_W},}')} {_dim('/')} {_dim(f'{total:>{_MEM_W},} MB')}"
                    f"  {_mem_bar(color, used, total)}" + (f"  {int(used / total * 100):3d}%" if total > 0 else "    N/A"))
            util = d.get("util_pct", d.get("ai_core_pct"))
            if show_all or args.show_util:  line += _col_util(util)
            if show_all or args.show_temp:  line += _col_temp(d.get("temp_c"))
            if show_all or args.show_power: line += _col_power(d.get("power_w"))
            if show_all or args.show_health:
                line += _col_health(_DictView(d))
            lines.append(line)
            if show_procs and d.get("processes"):
                from ._process import ProcessInfo
                procs = [ProcessInfo(**{k: p.get(k) for k in ProcessInfo.__slots__}) for p in d["processes"] if isinstance(p, dict)]
                proc_str = "  ".join(_proc_repr(p, show_all or args.show_user, show_all or args.show_pid,
                                                show_all or args.show_cmd, args.full_cmd, show_all or args.show_container)
                                     for p in procs)
                lines.append(" " * (tag_w + 1 + _LBL_W + 8) + f"{_dim('└─')} {proc_str}")
    return lines


class _DictView:
    """Attribute access over a device dict, for the column renderers."""
    def __init__(self, d: dict) -> None:
        self._d = d
    def __getattr__(self, name):
        return self._d.get(name)


def _index_label_from_dict(vendor: str, d: dict) -> str:
    for key in ("gpu_index", "aip_index", "device_id", "card_index", "gpu_id", "neuron_device", "tt_device", "npu_index", "cpu_index"):
        if key in d:
            return str(d[key])
    if "npu_id" in d:
        return f"{d['npu_id']}:{d.get('chip_id', 0)}"
    if "card_id" in d:
        return f"{d['card_id']}:{d.get('chip_id', 0)}"
    return "?"


def _run_remote(args, include_cpu: bool) -> None:
    from .remote import query_hosts, query_ssh
    import json

    def fetch():
        hosts = []
        if args.hosts:
            hosts += query_hosts(args.hosts, timeout=args.timeout)
        if args.ssh:
            hosts += query_ssh(args.ssh, command=args.ssh_command, include_cpu=include_cpu, timeout=args.timeout)
        return hosts

    def render(hosts) -> List[str]:
        width = max((len(h.hostname or h.label) for h in hosts), default=0)
        out: List[str] = []
        for h in hosts:
            label = h.hostname or h.label
            if h.error:
                out.append(_c("1;34", f"{label:<{width}} ") + _c("31", f"unreachable: {h.error}"))
                continue
            lines = _frame_lines(h.devices, label, args, width)
            out.extend(lines or [_c("1;34", f"{label:<{width}} ") + _dim("no accelerators detected")])
        return out

    if args.json:
        print(json.dumps([h.to_dict() for h in fetch()], indent=2))
        return
    interval = args.interval
    if interval is None:
        interval = 1.0
    count = 0
    while True:
        hosts = fetch()
        if not args.no_header:
            now = datetime.datetime.now().strftime("%a %b %d  %H:%M:%S  %Y")
            print(f"{_bold('xpustat')} {__version__}{_dim('  ·  ')}{len(hosts)} host(s){_dim('  ·  ')}{now}")
        for line in render(hosts):
            print(line)
        count += 1
        if not interval or interval <= 0 or (args.count and count >= args.count):
            break
        print(_dim(f"  refreshing every {interval}s  ·  Ctrl+C to quit"))
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            break
        if sys.stdout.isatty():
            os.system("cls" if os.name == "nt" else "clear")


def _open_record(path: str, mode: str):
    import gzip
    return gzip.open(path, mode + "t", encoding="utf-8") if path.endswith(".gz") else open(path, mode, encoding="utf-8")


def _run_record(args, include_cpu: bool) -> None:
    """Write one ``to_json(metadata=True)`` line per frame; render nothing unless -i."""
    interval = args.interval if args.interval and args.interval > 0 else 0.0
    n = 0
    with _open_record(args.record, "a") as f:
        while True:
            stats = query_all(parallel=True, include_cpu=include_cpu)
            f.write(stats.to_json(indent=None, metadata=True) + "\n")
            f.flush()
            n += 1
            sys.stderr.write(f"\rxpustat: recorded {n} frame(s) to {args.record}")
            if interval <= 0 or (args.count and n >= args.count):
                break
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                break
    sys.stderr.write("\n")


def _run_replay(args) -> None:
    import json
    frames = []
    with _open_record(args.replay, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    frames.append(json.loads(line))
                except ValueError:
                    continue
    if not frames:
        sys.stderr.write(f"xpustat: no frames in {args.replay}\n")
        sys.exit(1)
    if args.json:
        print(json.dumps(frames, indent=2))
        return
    interval = args.interval if args.interval and args.interval > 0 else 0.0
    for i, fr in enumerate(frames):
        devices = fr.get("devices", fr)
        if not args.no_header:
            print(f"{_bold('xpustat')} {fr.get('xpustat_version', '')}{_dim('  ·  ')}{_bold(str(fr.get('hostname', '')))}"
                  f"{_dim('  ·  ')}{fr.get('query_time', '')}{_dim('  ·  ')}frame {i + 1}/{len(frames)}")
        lines = _frame_lines(devices, "", args)
        for line in lines or [_dim("  no accelerators detected")]:
            print(line)
        if interval > 0 and i + 1 < len(frames):
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                break
            if sys.stdout.isatty():
                os.system("cls" if os.name == "nt" else "clear")


# ---------------------------------------------------------------------------
# Shell completion
# ---------------------------------------------------------------------------

def _completion_script(parser, shell: str) -> str:
    """A static completion script built from the parser's options."""
    opts = sorted({o for a in parser._actions for o in a.option_strings})
    longs = [o for o in opts if o.startswith("--")]
    shorts = [o for o in opts if not o.startswith("--")]
    vendors = " ".join(_VENDOR_KEYS)
    if shell == "bash":
        return f"""# bash completion for xpustat: eval "$(xpustat --print-completion bash)"
_xpustat() {{
    local cur prev
    cur="${{COMP_WORDS[COMP_CWORD]}}"; prev="${{COMP_WORDS[COMP_CWORD-1]}}"
    case "$prev" in
        -f|--filter) COMPREPLY=( $(compgen -W "{vendors}" -- "$cur") ); return ;;
        --by) COMPREPLY=( $(compgen -W "mem_free mem_used util index" -- "$cur") ); return ;;
        --print-completion) COMPREPLY=( $(compgen -W "bash zsh fish" -- "$cur") ); return ;;
    esac
    COMPREPLY=( $(compgen -W "{' '.join(opts)}" -- "$cur") )
}}
complete -F _xpustat xpustat"""
    if shell == "zsh":
        lines = "\n".join(f"    '{o}[{_help_for(parser, o)}]'" for o in longs + shorts)
        return f"""#compdef xpustat
# zsh completion for xpustat: eval "$(xpustat --print-completion zsh)"
_xpustat() {{
  _arguments -s \\
    '(-f --filter)'{{-f,--filter}}'[vendors]:vendor:({vendors})' \\
    '--by[ranking]:ranking:(mem_free mem_used util index)' \\
    '--print-completion[shell]:shell:(bash zsh fish)' \\
{lines}
}}
compdef _xpustat xpustat"""
    # fish
    out = ["# fish completion for xpustat: xpustat --print-completion fish > ~/.config/fish/completions/xpustat.fish"]
    for o in longs:
        out.append(f"complete -c xpustat -l {o[2:]} -d '{_help_for(parser, o)}'")
    for o in shorts:
        out.append(f"complete -c xpustat -s {o[1:]} -d '{_help_for(parser, o)}'")
    out.append(f"complete -c xpustat -s f -l filter -x -a '{vendors}'")
    out.append("complete -c xpustat -l by -x -a 'mem_free mem_used util index'")
    return "\n".join(out)


def _help_for(parser, option: str) -> str:
    for a in parser._actions:
        if option in a.option_strings and a.help:
            return (a.help.split(".")[0]).replace("'", "").replace("%%", "%")[:60]
    return ""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

_VENDOR_KEYS = list(_VENDORS.keys())


def _ensure_console_encoding() -> None:
    """
    The table uses box-drawing and block characters. Windows consoles often
    default to a legacy code page that cannot encode them; switch stdout to
    UTF-8 with replacement so the command never dies on a glyph.
    """
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", None) or "").lower()
        if enc.replace("-", "") != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main() -> None:
    import argparse
    _ensure_console_encoding()

    parser = argparse.ArgumentParser(
        prog="xpustat",
        description="Show all detected AI accelerators and their memory usage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "vendor keys:  " + "  ".join(_VENDOR_KEYS) + "\n\n"
            "examples:\n"
            "  xpustat                      # snapshot\n"
            "  xpustat -a                   # show everything (util, temp, power, user, pid, cmd)\n"
            "  xpustat -pui --cpu           # same, plus the host CPU and memory\n"
            "  xpustat -u -p                # show users and PIDs\n"
            "  xpustat -e -t -P             # show util, temp, power\n"
            "  xpustat -H -C -F             # health (P-state, throttling, ECC), clocks, fan\n"
            "  xpustat -puck --full-cmd     # processes with container and full command line\n"
            "  xpustat --id 0,2 -f nvidia   # only GPUs 0 and 2\n"
            "  xpustat -i 2                 # watch mode, refresh every 2 s\n"
            "  xpustat -i 1 -n 10           # 10 refreshes then exit\n"
            "  xpustat -f nvidia amd        # show only NVIDIA and AMD\n"
            "  xpustat --json               # raw JSON\n"
            "  export $(xpustat --select 2) # restrict a job to the 2 freest devices\n"
            "  xpustat --doctor             # why is a vendor missing? tools, bindings, env, errors\n"
            "  xpustat --hosts http://n1:9400 http://n2:9400   # cluster view over exporters\n"
            "  xpustat --ssh n1 n2 -a       # agentless cluster view over ssh\n"
            "  xpustat --record run.ndjson.gz -i 1 -n 60; xpustat --replay run.ndjson.gz -i 0.2\n"
            "  xpustat --prometheus         # Prometheus text format, once\n"
            "  xpustat --serve-prometheus 9400   # /metrics endpoint for Prometheus\n"
        ),
    )

    # ── process / annotation flags (gpustat-style) ──────────────────────────
    parser.add_argument("-u", "--show-user",  action="store_true", help="Show process owner username.")
    parser.add_argument("-p", "--show-pid",   action="store_true", help="Show process PID.")
    parser.add_argument("-c", "--show-cmd",   action="store_true", help="Show process command name.")
    parser.add_argument("-a", "--show-all",   action="store_true",
                        help="Show every column: util, temp, power, arch, fan, clocks, codec, health, user, pid, cmd, container.")

    # ── hardware metric flags ────────────────────────────────────────────────
    parser.add_argument("-e", "--show-util",  action="store_true", help="Show GPU/NPU utilization %%.")
    parser.add_argument("-t", "--show-temp",  action="store_true", help="Show chip temperature (°C).")
    parser.add_argument("-P", "--show-power", action="store_true", help="Show power draw (W).")
    parser.add_argument("-A", "--show-arch",  action="store_true",
                        help="Show compute architecture (CUDA cc / ROCm gfx target / Apple GPU cores).")
    parser.add_argument("-F", "--show-fan",   action="store_true", help="Show fan speed (%% or rpm).")
    parser.add_argument("-C", "--show-clocks", action="store_true", help="Show SM/graphics and memory clocks (MHz).")
    parser.add_argument("-E", "--show-codec", action="store_true", help="Show video encoder/decoder utilization.")
    parser.add_argument("-H", "--show-health", action="store_true",
                        help="Show P-state or perf level, active throttle reasons and uncorrected ECC count.")
    parser.add_argument("-k", "--show-container", action="store_true",
                        help="Prefix processes with their container or job (docker, containerd, pod, slurm).")
    parser.add_argument("--full-cmd", action="store_true", help="Show the full command line of processes (with -c).")
    parser.add_argument("--id", metavar="IDS", default=None,
                        help="Show only devices with these physical indices, e.g. --id 0,2 (combine with -f).")

    # ── layout / filtering ───────────────────────────────────────────────────
    parser.add_argument("-i", "--interval", "--watch", nargs="?", type=float, default=0,
                        help="Watch mode: refresh every SECS seconds (default: 1.0).")
    parser.add_argument("-n", "--count",    metavar="N",    type=int,   default=None,
                        help="Stop after N refreshes (requires -i).")
    parser.add_argument("-f", "--filter",   metavar="VENDOR", nargs="+", dest="filter",
                        choices=_VENDOR_KEYS, default=None,
                        help="Show only the specified vendor(s).")
    parser.add_argument("--cpu", action="store_true",
                        help="Also show the host CPU and memory (one row plus a detail line); applies to --json, "
                             "--prometheus and --serve-prometheus too.")
    parser.add_argument("--no-header", action="store_true", help="Suppress hostname/time header.")
    parser.add_argument("--no-color",  action="store_true", help="Disable ANSI colour output.")

    # ── output format ────────────────────────────────────────────────────────
    parser.add_argument("--print-completion", metavar="SHELL", choices=["bash", "zsh", "fish"], default=None,
                        help="Print a shell completion script for bash, zsh or fish and exit.")
    # ── remote hosts ─────────────────────────────────────────────────────────
    rem = parser.add_argument_group("remote hosts (adds a host column; works with --json and -i)")
    rem.add_argument("--hosts", metavar="URL", nargs="+", default=None,
                     help="Read other machines' exporters: http://node:9400 (their /json endpoint).")
    rem.add_argument("--ssh", metavar="HOST", nargs="+", default=None,
                     help="Run xpustat --json on these hosts over ssh (needs xpustat installed there).")
    rem.add_argument("--ssh-command", metavar="CMD", default=None,
                     help="Command to run over ssh instead of 'xpustat --json', e.g. a checkout.")
    rem.add_argument("--timeout", metavar="SECS", type=float, default=10.0, help="Per-host timeout (default 10).")

    # ── record / replay ──────────────────────────────────────────────────────
    rec = parser.add_argument_group("record and replay")
    rec.add_argument("--record", metavar="FILE", default=None,
                     help="Append one JSON line per frame to FILE (gzip when it ends in .gz); use with -i and -n.")
    rec.add_argument("--replay", metavar="FILE", default=None,
                     help="Render frames from a --record file instead of querying (with -i to pace them).")

    parser.add_argument("--doctor",  action="store_true",
                        help="Explain each backend: tool found, probe result, native binding, env vars, devices, errors.")
    parser.add_argument("--json",    action="store_true", help="Print raw JSON.")
    parser.add_argument("--prometheus", action="store_true",
                        help="Print metrics in Prometheus text format and exit.")

    # ── device selection ─────────────────────────────────────────────────────
    sel = parser.add_argument_group("device selection (prints VAR=ids lines for the shell)")
    sel.add_argument("--select", metavar="N", type=int, default=None,
                     help="Pick the N best devices and print e.g. CUDA_VISIBLE_DEVICES=2,0. "
                          "Use as: export $(xpustat --select 2). Honours -f.")
    sel.add_argument("--by", choices=["mem_free", "mem_used", "util", "index"], default="mem_free",
                     help="Ranking for --select (default: mem_free).")
    sel.add_argument("--min-free", metavar="MB", type=int, default=0,
                     help="Skip devices with less than MB free (with --select).")
    sel.add_argument("--uuid", action="store_true",
                     help="Emit UUIDs instead of indices where the vendor has them (with --select).")
    parser.add_argument("--serve-prometheus", metavar="PORT", type=int, default=None,
                        help="Serve Prometheus metrics at http://0.0.0.0:PORT/metrics (fresh query per scrape).")
    parser.add_argument("--cache-ttl", metavar="SECS", type=float, default=0.0,
                        help="With --serve-prometheus: reuse a result up to SECS old between scrapes.")
    parser.add_argument("--version", action="version",   version=f"xpustat {__version__}")

    args = parser.parse_args()
    include_cpu = bool(args.cpu or (args.filter and "cpu" in args.filter))

    if args.print_completion:
        print(_completion_script(parser, args.print_completion))
        return

    if args.hosts or args.ssh:
        _run_remote(args, include_cpu)
        return

    if args.replay:
        _run_replay(args)
        return

    if args.record:
        _run_record(args, include_cpu)
        return

    if args.doctor:
        from .doctor import diagnose, format_report
        results = diagnose()
        if args.json:
            import json
            print(json.dumps(results, indent=2))
        else:
            print(format_report(results))
        return

    ids: Optional[List[int]] = None
    if args.id:
        try:
            ids = [int(x) for x in args.id.split(",") if x.strip() != ""]
        except ValueError:
            sys.stderr.write("Error: --id takes a comma-separated list of integers.\n")
            sys.exit(1)

    if args.select is not None:
        from ._device import pick_device, visible_devices_env
        chosen = pick_device(n=args.select, vendor=args.filter, by=args.by, min_free_mb=args.min_free,
                             stats=query_all(include_cpu=include_cpu) if include_cpu else None)
        if not chosen:
            sys.stderr.write("xpustat: no device matches the selection\n")
            sys.exit(1)
        env = visible_devices_env(chosen, format="uuid" if args.uuid else "index")
        if not env:
            sys.stderr.write("xpustat: selected devices have no visible-devices variable "
                             f"({', '.join(sorted({d.vendor for d in chosen}))})\n")
            sys.exit(1)
        for var, value in env.items():
            print(f"{var}={value}")
        return

    if args.prometheus:
        from .prometheus import to_prometheus
        sys.stdout.write(to_prometheus(query_all(parallel=True, include_cpu=include_cpu)))
        return

    if args.serve_prometheus is not None:
        from .prometheus import serve
        serve(port=args.serve_prometheus, max_age=max(args.cache_ttl, 0.0), include_cpu=include_cpu)
        return

    if args.json:
        import json
        if args.interval is None or args.interval > 0:
            sys.stderr.write("Error: --json and --interval/-i can't be used together.\n")
            sys.exit(1)
        print(json.dumps(query_all(parallel=True, include_cpu=include_cpu).to_dict(), indent=2))
        return

    show_all = args.show_all
    
    if args.interval is None:
        args.interval = 1.0
    elif args.interval > 0:
        args.interval = max(0.1, args.interval)
    print_xpustat(
        use_color=not args.no_color,
        interval=args.interval,
        count=args.count,
        filter_vendors=args.filter,
        show_util=show_all  or args.show_util,
        show_temp=show_all  or args.show_temp,
        show_power=show_all or args.show_power,
        show_arch=show_all  or args.show_arch,
        show_user=show_all  or args.show_user,
        show_pid=show_all   or args.show_pid,
        show_cmd=show_all   or args.show_cmd,
        no_header=args.no_header,
        include_cpu=include_cpu,
        show_fan=show_all or args.show_fan,
        show_clocks=show_all or args.show_clocks,
        show_codec=show_all or args.show_codec,
        show_health=show_all or args.show_health,
        show_container=show_all or args.show_container,
        full_cmd=args.full_cmd,
        ids=ids,
    )


if __name__ == "__main__":
    main()
