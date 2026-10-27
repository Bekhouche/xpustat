#!/usr/bin/env python3
import os
import sys
import json

# Allow running straight from a git checkout without installing xpustat.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xpustat import query  # noqa: E402

def main():
    vendor = "jetson"
    print(f"Testing {vendor.upper()} query...")
    try:
        stats = query(vendor)
    except Exception as e:
        print(f"Failed to query {vendor}: {e}")
        sys.exit(1)

    if not stats:
        print(f"No {vendor.upper()} devices found or management tool not installed.")
        sys.exit(0)

    print(f"Found {len(stats)} {vendor.upper()} device(s):")
    for dev in stats:
        idx = getattr(dev, "gpu_index", "?")
        print(f" - [{idx}] {dev.name} ({dev.mem_used_mb}MB / {dev.mem_total_mb}MB shared, util {dev.util_pct}%, {dev.temp_c}C, {dev.power_w}W via {dev.power_rail}, torch={dev.torch_device})")
        if hasattr(dev, 'processes') and dev.processes:
            print(f"   Processes: {len(dev.processes)}")
            for p in dev.processes:
                print(f"    * PID {p.pid} ({p.username}) - {p.command} - {p.gpu_mem_mb}MB")

    print("\nJSON Output:")
    print(json.dumps(stats.to_dict(), indent=2))

if __name__ == "__main__":
    main()
