"""
doctor.py — explain why each backend did or did not report devices.

``xpustat --doctor`` runs the checks and prints one block per backend: the
tool looked for, where it was found, what it answered, whether the native
binding loaded, which visible-devices variables are set, and what
``new_query()`` returned. It never changes anything.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

from ._device import VISIBLE_DEVICES_ENV

#: Per backend: the executables looked for, the arguments that print a
#: version or a cheap probe, the optional Python binding, platform notes.
_CHECKS: Dict[str, dict] = {
    "nvidia":       {"tools": ["nvidia-smi"], "probe": ["--version"], "binding": "pynvml", "binding_pkg": "nvidia-ml-py"},
    "amd":          {"tools": ["amd-smi"], "probe": ["version"], "binding": "amdsmi", "binding_pkg": "amdsmi"},
    "gaudi":        {"tools": ["hl-smi"], "probe": ["--version"]},
    "intel":        {"tools": ["xpu-smi"], "probe": ["--version"]},
    "huawei":       {"tools": ["ascend-dmi", "npu-smi"], "probe": ["info", "-h"]},
    "hygon":        {"tools": ["hy-smi"], "probe": ["--version"]},
    "cambricon":    {"tools": ["cnmon", "/usr/local/neuware/bin/cnmon"], "probe": ["version"]},
    "moorethreads": {"tools": ["mthreads-gmi"], "probe": ["--version"]},
    "apple":        {"tools": ["ioreg"], "probe": None, "platform": "darwin"},
    "neuron":       {"tools": ["neuron-ls", "/opt/aws/neuron/bin/neuron-ls"], "probe": ["--version"]},
    "tpu":          {"tools": ["tpu-info"], "probe": ["--version"]},
    "tenstorrent":  {"tools": ["tt-smi"], "probe": ["-v"]},
    "jetson":       {"tools": ["tegrastats"], "probe": None, "platform": "linux",
                     "note": "runs only when /etc/nv_tegra_release exists and nvidia-smi is absent"},
    "rebellions":   {"tools": ["rbln-stat", "rbln-smi"], "probe": ["--version"]},
    "furiosa":      {"tools": ["furiosa-smi"], "probe": ["--version"]},
    "cpu":          {"tools": [], "probe": None, "binding": "psutil", "binding_pkg": "psutil", "opt_in": True},
}


def _probe(binary: str, args: Optional[List[str]]) -> str:
    if not args:
        return ""
    try:
        r = subprocess.run([binary] + args, capture_output=True, text=True, timeout=8)
    except subprocess.TimeoutExpired:
        return "probe timed out after 8 s"
    except OSError as exc:
        return f"probe failed: {exc}"
    out = (r.stdout or r.stderr or "").strip().splitlines()
    first = out[0].strip() if out else ""
    return f"exit {r.returncode}: {first[:100]}" if first else f"exit {r.returncode}"


def _binding_status(module: str) -> str:
    try:
        __import__(module)
        return "importable"
    except ImportError:
        return "not installed"
    except Exception as exc:  # loads but fails against the driver library
        return f"import error: {type(exc).__name__}: {str(exc)[:80]}"


def diagnose(include_cpu: bool = True) -> List[dict]:
    """Return one dict per backend with the findings; see :func:`format_report`."""
    from . import xpu
    results: List[dict] = []
    env = os.environ
    for name, cls in list(xpu._BACKENDS) + list(xpu._PLUGIN_BACKENDS.items()) + ([("cpu", xpu.CPUStatCollection)] if include_cpu else []):
        check = _CHECKS.get(name, {"tools": [], "probe": None})
        entry: Dict[str, Any] = {"backend": name, "tools": [], "binding": None, "env": {}, "notes": []}
        plat = check.get("platform")
        if plat and not sys.platform.startswith(plat):
            entry["notes"].append(f"only on {plat}; this is {sys.platform}")
        for tool in check.get("tools", []):
            path = shutil.which(tool) or (tool if os.path.isabs(tool) and os.path.exists(tool) else None)
            entry["tools"].append({"name": tool, "path": path, "probe": _probe(path, check.get("probe")) if path else ""})
        if check.get("binding"):
            entry["binding"] = {"module": check["binding"], "package": check.get("binding_pkg"),
                                "status": _binding_status(check["binding"])}
        for var in VISIBLE_DEVICES_ENV.get(name, ()):
            if var in env:
                entry["env"][var] = env[var]
        if check.get("note"):
            entry["notes"].append(check["note"])
        if check.get("opt_in"):
            entry["notes"].append("opt-in: --cpu or include_cpu=True")
        t0 = time.perf_counter()
        try:
            col = cls.new_query()
            entry["devices"] = len(col)
            entry["backend_path"] = getattr(col, "backend", None)
            entry["error"] = None
        except Exception as exc:
            entry["devices"] = 0
            entry["backend_path"] = None
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["query_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        results.append(entry)
    return results


def format_report(results: List[dict]) -> str:
    from . import __version__
    lines = [f"xpustat {__version__} doctor  ·  {platform.platform()}  ·  Python {platform.python_version()}", ""]
    rapl = "/sys/class/powercap/intel-rapl:0/energy_uj"
    if sys.platform.startswith("linux") and os.path.exists(rapl):
        lines.append(f"RAPL energy counter: {'readable' if os.access(rapl, os.R_OK) else 'root-only (CPU power stays N/A)'}")
        lines.append("")
    for r in results:
        head = f"[{r['backend']}] {r['devices']} device(s)"
        if r.get("backend_path"):
            head += f" via {r['backend_path']}"
        head += f"  ({r['query_ms']} ms)"
        lines.append(head)
        for t in r["tools"]:
            if t["path"]:
                lines.append(f"    tool {t['name']}: {t['path']}" + (f"  →  {t['probe']}" if t["probe"] else ""))
            else:
                lines.append(f"    tool {t['name']}: not found on PATH")
        if r["binding"]:
            b = r["binding"]
            hint = "" if b["status"] == "importable" else f"  (pip install {b['package']} for the native path)" if b["package"] and b["status"] == "not installed" else ""
            lines.append(f"    binding {b['module']}: {b['status']}{hint}")
        for var, val in r["env"].items():
            lines.append(f"    env {var}={val!r}")
        for n in r["notes"]:
            lines.append(f"    note: {n}")
        if r["error"]:
            lines.append(f"    error: {r['error']}")
        lines.append("")
    plugins = [r["backend"] for r in results if r["backend"] not in _CHECKS]
    if plugins:
        lines.append(f"plugins registered: {', '.join(plugins)}")
    lines.append("set XPUSTAT_DEBUG=1 to see why a plugin failed to load")
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_report(diagnose()))
