import re
import shutil
import subprocess


class CambriconMLUStat:
    """Parse MLU info from cnmon for a single Cambricon MLU accelerator."""

    def __init__(self, card_id: int, chip_id: int, name: str, mem_used_mb: int, mem_total_mb: int):
        self.card_id = card_id
        self.chip_id = chip_id
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb

    def to_dict(self) -> dict:
        return {
            "card_id": self.card_id,
            "chip_id": self.chip_id,
            "name": self.name,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
        }

    def __repr__(self) -> str:
        return (
            f"CambriconMLUStat(card_id={self.card_id}, chip_id={self.chip_id}, "
            f"name={self.name!r}, mem_used={self.mem_used_mb}MB, mem_total={self.mem_total_mb}MB)"
        )


class CambriconMLUStatCollection:
    """Collects stats for all Cambricon MLUs via cnmon.

    Parses the ASCII table from ``cnmon info``, which has two data rows per
    device entry (similar layout to npu-smi):

        Row 1:  | CardID  Model   | Health | Power(W)  Temp(C)         |
        Row 2:  | ChipID  Device  | Bus-Id | Util(%)   MemUsed/Total   |

    Memory values are in MB.
    """

    def __init__(self, mlus: list[CambriconMLUStat]):
        self.mlus = mlus

    def to_dict(self) -> dict:
        return {"mlus": [mlu.to_dict() for mlu in self.mlus]}

    def __len__(self) -> int:
        return len(self.mlus)

    def __iter__(self):
        return iter(self.mlus)

    def __repr__(self) -> str:
        return f"CambriconMLUStatCollection({self.mlus!r})"

    @classmethod
    def new_query(cls) -> "CambriconMLUStatCollection":
        """
        Query cnmon and return a CambriconMLUStatCollection.
        Returns an empty collection if cnmon is not installed or fails.
        """
        binary = shutil.which("cnmon") or shutil.which("/usr/local/neuware/bin/cnmon")
        if binary is None:
            return cls([])

        try:
            result = subprocess.run(
                [binary, "info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return cls([])

        if result.returncode != 0 or not result.stdout.strip():
            return cls([])

        return cls(_parse_cnmon_info(result.stdout))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_DATA_ROW = re.compile(r"^\|(.+)\|(.+)\|(.+)\|$")
_MEM_FIELD = re.compile(r"(\d+)\s*/\s*(\d+)")


def _parse_cnmon_info(output: str) -> list[CambriconMLUStat]:
    """
    Parse the ASCII table from ``cnmon info``.

    Each device occupies two consecutive data rows:
    Row 1 → card_id, model name
    Row 2 → chip_id, mem_used / mem_total
    """
    mlus: list[CambriconMLUStat] = []
    data_rows: list[tuple[str, str, str]] = []

    for raw_line in output.splitlines():
        line = raw_line.strip()
        m = _DATA_ROW.match(line)
        if not m:
            continue
        col1 = m.group(1).strip()
        col2 = m.group(2).strip()
        col3 = m.group(3).strip()

        first_token = col1.split()[0] if col1.split() else ""
        if not first_token.isdigit():
            continue

        data_rows.append((col1, col2, col3))

    for i in range(0, len(data_rows) - 1, 2):
        row1_col1, _, _ = data_rows[i]
        row2_col1, _, row2_col3 = data_rows[i + 1]

        try:
            tokens1 = row1_col1.split()
            card_id = int(tokens1[0])
            name = tokens1[1] if len(tokens1) > 1 else "Unknown MLU"

            chip_id = int(row2_col1.split()[0])

            mem_match = _MEM_FIELD.search(row2_col3)
            if mem_match:
                mem_used_mb = int(mem_match.group(1))
                mem_total_mb = int(mem_match.group(2))
            else:
                mem_used_mb, mem_total_mb = 0, 0

            mlus.append(CambriconMLUStat(
                card_id=card_id,
                chip_id=chip_id,
                name=name,
                mem_used_mb=mem_used_mb,
                mem_total_mb=mem_total_mb,
            ))
        except (IndexError, ValueError):
            continue

    return mlus


if __name__ == "__main__":
    import json
    collection = CambriconMLUStatCollection.new_query()
    print(json.dumps(collection.to_dict(), indent=2))
