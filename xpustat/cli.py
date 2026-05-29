"""
xpustat CLI — display all detected accelerators in the terminal.
"""

from __future__ import annotations

import datetime
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
}

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
    return "?", str(dev), None, 0, 0, None, None, None, []


def _fmt_cc(cc: Optional[str]) -> Optional[str]:
    """Format CUDA compute capability for display, e.g. 'cc8.6'."""
    return f"cc{cc}" if cc else None


def _fmt_gfx(gfx: Optional[str]) -> Optional[str]:
    """Format ROCm GFX target for display, e.g. 'gfx1100'."""
    return gfx if gfx else None


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


def _proc_repr(p, show_user: bool, show_pid: bool, show_cmd: bool) -> str:
    parts: List[str] = []
    if show_user:
        parts.append(_c("32", p.username))
    if show_cmd:
        parts.append(_c("1", p.command))
    if show_pid:
        parts.append(_dim(f"/{p.pid}"))
    mem = f"({p.gpu_mem_mb}M)" if p.gpu_mem_mb is not None else ""
    parts.append(_c("33", mem))
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
        tips.append("pip install pynvml    → faster NVIDIA queries (direct NVML, no subprocess)")
    if len(collection.amd) > 0 and not _HAS_AMDSMI:
        tips.append("pip install amdsmi    → faster AMD queries (ROCm Python bindings, no subprocess)")
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
) -> List[str]:
    collection = query_all(parallel=True, vendors=active_vendors)

    # ── header ─────────────────────────────────────────────────────────────
    if not no_header:
        hostname  = socket.gethostname()
        now       = datetime.datetime.now().strftime("%a %b %d  %H:%M:%S  %Y")
        title     = _bold("xpustat") + f" {__version__}"
        dots      = _dim("  ·  ")
        header    = f"{title}{dots}{_bold(hostname)}{dots}{now}"
        plain_len = len(f"xpustat {__version__}  ·  {hostname}  ·  {now}")
        print(header)
        print(_dim("─" * plain_len))

    visible = [
        (v, s) for v, s in _VENDORS.items()
        if (filter_vendors is None or v in filter_vendors)
        and list(getattr(collection, v))
    ]

    if not visible:
        print(_dim("  no accelerators detected"))
        return

    show_procs = show_user or show_pid or show_cmd

    for vendor, (color, label) in visible:
        for dev in getattr(collection, vendor):
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

            # ── health (Huawei only) ────────────────────────────────────────
            health = getattr(dev, "health", None)
            if health and health != "OK":
                line += f"  {_c('31', health)}"

            print(line)

            # ── process sub-lines ───────────────────────────────────────────
            if show_procs and procs:
                indent = " " * (_LBL_W + 8)   # align under device name
                proc_str = "  ".join(
                    _proc_repr(p, show_user, show_pid, show_cmd)
                    for p in procs
                )
                print(f"{indent}{_dim('└─')} {proc_str}")

            # ── SDK speed tips (only shown when header is visible) ─────────────────
    if not no_header:
        tips = _sdk_tips(collection)
        if tips:
            print(_dim("  tip: install native SDK for faster queries:"))
            for tip in tips:
                print(_dim(f"       {tip}"))

    return [v for v, s in visible]


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
) -> None:
    global _USE_COLOR
    if use_color is not None:
        _USE_COLOR = use_color

    kwargs = dict(
        filter_vendors=filter_vendors,
        show_util=show_util, show_temp=show_temp, show_power=show_power,
        show_user=show_user, show_pid=show_pid, show_cmd=show_cmd,
        show_arch=show_arch, no_header=no_header,
    )

    if interval is None or interval <= 0:
        _render(**kwargs)
        return

    # ── watch loop ──────────────────────────────────────────────────────────
    iteration = 0
    _stop = False
    active_vendors = None

    def _handle_sigint(*_):
        nonlocal _stop
        _stop = True

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        while not _stop:
            if iteration > 0:
                sys.stdout.write("\033[H\033[J")
                sys.stdout.flush()
            
            kwargs["active_vendors"] = active_vendors
            active_vendors = _render(**kwargs)
            
            print(_dim(f"  refreshing every {interval}s  ·  Ctrl+C to quit"))
            iteration += 1
            if count and iteration >= count:
                break
            deadline = time.monotonic() + interval
            while not _stop and time.monotonic() < deadline:
                time.sleep(0.1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

_VENDOR_KEYS = list(_VENDORS.keys())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="xpustat",
        description="Show all detected AI accelerators and their memory usage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "vendor keys:  " + "  ".join(_VENDOR_KEYS) + "\n\n"
            "examples:\n"
            "  xpustat                      # snapshot\n"
            "  xpustat -a                   # show everything (util, temp, power, user, pid, cmd)\n"
            "  xpustat -u -p                # show users and PIDs\n"
            "  xpustat -e -t -P             # show util, temp, power\n"
            "  xpustat -i 2                 # watch mode, refresh every 2 s\n"
            "  xpustat -i 1 -n 10           # 10 refreshes then exit\n"
            "  xpustat -f nvidia amd        # show only NVIDIA and AMD\n"
            "  xpustat --json               # raw JSON\n"
        ),
    )

    # ── process / annotation flags (gpustat-style) ──────────────────────────
    parser.add_argument("-u", "--show-user",  action="store_true", help="Show process owner username.")
    parser.add_argument("-p", "--show-pid",   action="store_true", help="Show process PID.")
    parser.add_argument("-c", "--show-cmd",   action="store_true", help="Show process command name.")
    parser.add_argument("-a", "--show-all",   action="store_true", help="Show util, temp, power, arch, user, pid, cmd.")

    # ── hardware metric flags ────────────────────────────────────────────────
    parser.add_argument("-e", "--show-util",  action="store_true", help="Show GPU/NPU utilization %%.")
    parser.add_argument("-t", "--show-temp",  action="store_true", help="Show chip temperature (°C).")
    parser.add_argument("-P", "--show-power", action="store_true", help="Show power draw (W).")
    parser.add_argument("-A", "--show-arch",  action="store_true",
                        help="Show compute architecture (CUDA cc / ROCm gfx target).")

    # ── layout / filtering ───────────────────────────────────────────────────
    parser.add_argument("-i", "--interval", "--watch", nargs="?", type=float, default=0,
                        help="Watch mode: refresh every SECS seconds (default: 1.0).")
    parser.add_argument("-n", "--count",    metavar="N",    type=int,   default=None,
                        help="Stop after N refreshes (requires -i).")
    parser.add_argument("-f", "--filter",   metavar="VENDOR", nargs="+", dest="filter",
                        choices=_VENDOR_KEYS, default=None,
                        help="Show only the specified vendor(s).")
    parser.add_argument("--no-header", action="store_true", help="Suppress hostname/time header.")
    parser.add_argument("--no-color",  action="store_true", help="Disable ANSI colour output.")

    # ── output format ────────────────────────────────────────────────────────
    parser.add_argument("--json",    action="store_true", help="Print raw JSON.")
    parser.add_argument("--version", action="version",   version=f"xpustat {__version__}")

    args = parser.parse_args()

    if args.json:
        import json
        if args.interval is None or args.interval > 0:
            sys.stderr.write("Error: --json and --interval/-i can't be used together.\n")
            sys.exit(1)
        print(json.dumps(query_all(parallel=True).to_dict(), indent=2))
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
    )


if __name__ == "__main__":
    main()
