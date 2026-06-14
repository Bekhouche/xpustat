from __future__ import annotations

import json
import re
import shutil
import subprocess
from typing import Dict, List, Optional


class HuaweiNPUStat:
    """Stats for a single Huawei Ascend NPU chip."""

    def __init__(
        self,
        npu_id: int,
        chip_id: int,
        name: str,
        mem_used_mb: int,
        mem_total_mb: int,
        health: Optional[str] = None,
        temp_c: Optional[int] = None,
        power_w: Optional[float] = None,
        ai_core_pct: Optional[int] = None,
        card_type: Optional[str] = None,
    ) -> None:
        self.npu_id = npu_id
        self.chip_id = chip_id
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb
        self.health = health
        self.temp_c = temp_c
        self.power_w = power_w
        self.ai_core_pct = ai_core_pct
        self.card_type = card_type

    def to_dict(self) -> dict:
        return {
            "npu_id": self.npu_id,
            "chip_id": self.chip_id,
            "name": self.name,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
            "health": self.health,
            "temp_c": self.temp_c,
            "power_w": self.power_w,
            "ai_core_pct": self.ai_core_pct,
            "card_type": self.card_type,
        }

    def __repr__(self) -> str:
        return (
            f"HuaweiNPUStat(npu_id={self.npu_id}, chip_id={self.chip_id}, "
            f"name={self.name!r}, mem_used={self.mem_used_mb}MB, "
            f"mem_total={self.mem_total_mb}MB)"
        )


class HuaweiNPUStatCollection:
    """Collects stats for all Huawei Ascend NPUs.

    Backend selection (mirrors npustat):
      1. If ``ascend-dmi`` is available → parse its JSON output (richer data).
      2. Otherwise fall back to ``npu-smi info`` ASCII table.
    """

    def __init__(self, npus: List[HuaweiNPUStat]) -> None:
        self.npus = npus

    def to_dict(self) -> dict:
        return {"npus": [npu.to_dict() for npu in self.npus]}

    def __len__(self) -> int:
        return len(self.npus)

    def __iter__(self):
        return iter(self.npus)

    def __repr__(self) -> str:
        return f"HuaweiNPUStatCollection({self.npus!r})"

    @staticmethod
    def _run(cmd: List[str], timeout: int = 10) -> Optional[str]:
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout,
            )
            return r.stdout if r.returncode == 0 and r.stdout.strip() else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    @classmethod
    def new_query(cls) -> "HuaweiNPUStatCollection":
        """
        Query Huawei NPU stats. Returns empty collection if no tool found.
        """
        # ── Try ascend-dmi (preferred) ──────────────────────────────────────
        if shutil.which("ascend-dmi"):
            probe = cls._run(["ascend-dmi", "-i"])
            if probe:
                npus = _query_ascend_dmi(cls)
                if npus is not None:
                    return cls(npus)

        # ── Fall back to npu-smi ────────────────────────────────────────────
        if shutil.which("npu-smi") is None:
            return cls([])

        out = cls._run(["npu-smi", "info"])
        if out is None:
            return cls([])

        npus = _parse_npu_smi_info(out)

        # Enrich with card type (one extra call per NPU ID, cached)
        card_types: Dict[int, str] = {}
        for npu in npus:
            if npu.npu_id not in card_types:
                ct = _fetch_card_type(cls, npu.npu_id)
                if ct:
                    card_types[npu.npu_id] = ct
            npu.card_type = card_types.get(npu.npu_id)

        return cls(npus)


# ---------------------------------------------------------------------------
# ascend-dmi path
# ---------------------------------------------------------------------------

def _query_ascend_dmi(cls: type) -> Optional[List[HuaweiNPUStat]]:
    """Parse `ascend-dmi -i --format json` into HuaweiNPUStat objects.

    Handles two JSON shapes emitted by different ascend-dmi versions:

    • Server format (ascend-dmi ≥ 7.x):
        hardware_brief.server.devices[]  — memory as "65536 MB" strings,
        power under power_information.realtime_power, temp as "50 C".

    • Card-list format (older ascend-dmi):
        hardware_brief[]  — a list of cards, each with a chip_list[].
    """
    out = cls._run(["ascend-dmi", "-i", "--format", "json"])
    if out is None:
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None

    hw = data.get("hardware_brief", {})
    npus: List[HuaweiNPUStat] = []

    # ── Server format: hardware_brief is a dict ──────────────────────────────
    if isinstance(hw, dict):
        server = hw.get("server", {})
        card_type = server.get("type") or None
        for dev in server.get("devices", []):
            ai_info  = dev.get("ai_core_information", {})
            mem_info = dev.get("memory_information", {})
            pwr_info = dev.get("power_information", {})
            npus.append(HuaweiNPUStat(
                npu_id=int(dev.get("device_id", 0)),
                chip_id=int(dev.get("chip_id", 0)),
                name=dev.get("chip_name", "Ascend NPU"),
                mem_used_mb=_parse_mb(mem_info.get("used", "0 MB")),
                mem_total_mb=_parse_mb(mem_info.get("total", "0 MB")),
                health=dev.get("health"),
                temp_c=_parse_int_prefix(dev.get("temperature", "")),
                power_w=_parse_float_prefix(pwr_info.get("realtime_power", "")),
                ai_core_pct=_safe_int(ai_info.get("ai_core_usage")),
                card_type=card_type,
            ))

    # ── Card-list format: hardware_brief is a list ───────────────────────────
    elif isinstance(hw, list):
        for card in hw:
            npu_id     = int(card.get("card_id", 0))
            card_type  = card.get("type", "") or None
            card_power = _safe_float(card.get("power"))
            for chip in card.get("chip_list", []):
                ai_info  = chip.get("ai_core_information", {})
                mem_info = chip.get("memory_information", {})
                npus.append(HuaweiNPUStat(
                    npu_id=npu_id,
                    chip_id=int(chip.get("chip_id", 0)),
                    name=f"Ascend {chip.get('chip_name', '')}".strip(),
                    mem_used_mb=_safe_int(mem_info.get("memory_used")),
                    mem_total_mb=_safe_int(mem_info.get("memory_total")),
                    health=chip.get("health"),
                    temp_c=_safe_int(chip.get("temperature")),
                    power_w=_safe_float(chip.get("realtime_power")) or card_power,
                    ai_core_pct=_safe_int(ai_info.get("ai_core_usage")),
                    card_type=card_type,
                ))

    return npus if npus else None


# ---------------------------------------------------------------------------
# npu-smi info table parser
# ---------------------------------------------------------------------------

_DATA_ROW  = re.compile(r"^\|(.+)\|(.+)\|(.+)\|$")
_MEM_FIELD = re.compile(r"(\d+)\s*/\s*(\d+)")
_POWER_TEMP = re.compile(r"([\d.]+)\s+(\d+)")


def _parse_npu_smi_info(output: str) -> List[HuaweiNPUStat]:
    """
    Parse the ASCII table from `npu-smi info`.

    Each device occupies two consecutive data rows:
      Row 1: | NPU_ID  Name | Health | Power(W)  Temp(C) |
      Row 2: | Chip_ID Dev  | Bus-Id | AICore(%) Used/Total |
    """
    data_rows: List[tuple] = []

    for raw in output.splitlines():
        m = _DATA_ROW.match(raw.strip())
        if not m:
            continue
        c1, c2, c3 = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        first = c1.split()[0] if c1.split() else ""
        if not first.isdigit():
            continue
        data_rows.append((c1, c2, c3))

    npus: List[HuaweiNPUStat] = []
    for i in range(0, len(data_rows) - 1, 2):
        r1c1, r1c2, r1c3 = data_rows[i]
        r2c1, _,   r2c3  = data_rows[i + 1]
        try:
            t1 = r1c1.split()
            npu_id = int(t1[0])
            name   = f"Ascend {t1[1]}" if len(t1) > 1 else "Ascend NPU"

            chip_id = int(r2c1.split()[0])
            health  = r1c2.strip() or None

            # Power and temperature from row-1 col-3: "12.8  56"
            pt = _POWER_TEMP.search(r1c3)
            power_w = float(pt.group(1)) if pt else None
            temp_c  = int(pt.group(2))   if pt else None

            # AI core % from row-2 col-3 before the memory field
            ai_core_pct: Optional[int] = None
            mem_match = _MEM_FIELD.search(r2c3)
            if mem_match:
                # Everything before the memory field is AICore%
                pre = r2c3[:mem_match.start()].strip()
                ai_m = re.search(r"(\d+)", pre)
                if ai_m:
                    ai_core_pct = int(ai_m.group(1))
                mem_used_mb  = int(mem_match.group(1))
                mem_total_mb = int(mem_match.group(2))
            else:
                mem_used_mb = mem_total_mb = 0

            npus.append(HuaweiNPUStat(
                npu_id=npu_id, chip_id=chip_id, name=name,
                mem_used_mb=mem_used_mb, mem_total_mb=mem_total_mb,
                health=health, temp_c=temp_c, power_w=power_w,
                ai_core_pct=ai_core_pct,
            ))
        except (IndexError, ValueError):
            continue

    return npus


def _fetch_card_type(cls: type, npu_id: int) -> Optional[str]:
    """Return a card type string from npu-smi, trying 'product' then 'board'."""
    for qtype in ("product", "board"):
        out = cls._run(["npu-smi", "info", "-t", qtype, "-i", str(npu_id)])
        if not out:
            continue
        for line in out.splitlines():
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            v = val.strip()
            if not v or v in ("N/A", "NA", ""):
                continue
            if qtype == "board":
                if "product name" in key.lower():
                    return v
            else:
                return v
    return None


# ---------------------------------------------------------------------------
# Safe parsers
# ---------------------------------------------------------------------------

def _safe_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_mb(v) -> int:
    """Parse '65536 MB' or plain integer → int MB."""
    try:
        return int(str(v).split()[0])
    except (IndexError, ValueError):
        return 0


def _parse_int_prefix(v) -> Optional[int]:
    """Parse '50 C' or '50' → 50."""
    m = re.match(r"(\d+)", str(v).strip())
    return int(m.group(1)) if m else None


def _parse_float_prefix(v) -> Optional[float]:
    """Parse '104.900002 W' or '104.9' → 104.9."""
    m = re.match(r"([\d.]+)", str(v).strip())
    return float(m.group(1)) if m else None


if __name__ == "__main__":
    col = HuaweiNPUStatCollection.new_query()
    print(json.dumps(col.to_dict(), indent=2))
