import json
import re
import shutil
import subprocess
from typing import Optional


class HygonDCUStat:
    """Parse DCU info from hy-smi for a single Hygon Deep Computing Unit."""

    def __init__(self, card_index: int, name: str, mem_used_mb: int, mem_total_mb: int):
        self.card_index = card_index
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb

    def to_dict(self) -> dict:
        return {
            "card_index": self.card_index,
            "name": self.name,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
        }

    def __repr__(self) -> str:
        return (
            f"HygonDCUStat(card_index={self.card_index}, name={self.name!r}, "
            f"mem_used={self.mem_used_mb}MB, mem_total={self.mem_total_mb}MB)"
        )


class HygonDCUStatCollection:
    """Collects stats for all Hygon DCUs via hy-smi.

    hy-smi is ROCm-compatible, installed at /opt/hyhal/bin/hy-smi or on PATH.
    Uses JSON output:

        hy-smi --showproductname --showmeminfo vram --json

    Output example:
        {
          "card0": {
            "Card Series": "Z100L",
            "vram Total Memory (MiB)": "32752",
            "vram Total Used Memory (MiB)": "2"
          }
        }
    """

    def __init__(self, dcus: list[HygonDCUStat]):
        self.dcus = dcus

    def to_dict(self) -> dict:
        return {"dcus": [dcu.to_dict() for dcu in self.dcus]}

    def __len__(self) -> int:
        return len(self.dcus)

    def __iter__(self):
        return iter(self.dcus)

    def __repr__(self) -> str:
        return f"HygonDCUStatCollection({self.dcus!r})"

    @staticmethod
    def _find_binary() -> Optional[str]:
        """Locate hy-smi on PATH or common install paths."""
        if shutil.which("hy-smi"):
            return "hy-smi"
        for candidate in ("/opt/hyhal/bin/hy-smi", "/opt/dtk/bin/hy-smi"):
            if shutil.which(candidate):
                return candidate
        return None

    @classmethod
    def new_query(cls) -> "HygonDCUStatCollection":
        """
        Query hy-smi and return a HygonDCUStatCollection.
        Returns an empty collection if hy-smi is not installed or fails.
        """
        binary = cls._find_binary()
        if binary is None:
            return cls([])

        try:
            result = subprocess.run(
                [binary, "--showproductname", "--showmeminfo", "vram", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return cls([])

        if result.returncode != 0 or not result.stdout.strip():
            return cls([])

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return cls([])

        # card keys are "card0", "card1", ...
        _CARD_KEY = re.compile(r"^card(\d+)$")

        dcus: list[HygonDCUStat] = []
        for key, info in data.items():
            m = _CARD_KEY.match(key)
            if not m or not isinstance(info, dict):
                continue
            card_index = int(m.group(1))

            name = (
                info.get("Card Series")
                or info.get("Card model")
                or "Unknown Hygon DCU"
            )

            try:
                mem_total_mb = int(info.get("vram Total Memory (MiB)", 0))
            except (TypeError, ValueError):
                mem_total_mb = 0

            try:
                mem_used_mb = int(info.get("vram Total Used Memory (MiB)", 0))
            except (TypeError, ValueError):
                mem_used_mb = 0

            dcus.append(HygonDCUStat(
                card_index=card_index,
                name=name,
                mem_used_mb=mem_used_mb,
                mem_total_mb=mem_total_mb,
            ))

        dcus.sort(key=lambda d: d.card_index)
        return cls(dcus)


if __name__ == "__main__":
    collection = HygonDCUStatCollection.new_query()
    print(json.dumps(collection.to_dict(), indent=2))
