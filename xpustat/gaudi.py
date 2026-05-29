import shutil
import subprocess


class GaudiAIPStat:
    """Parse AIP info from hl-smi for a single Intel Gaudi accelerator."""

    def __init__(self, aip_index: int, name: str, mem_used_mb: int, mem_total_mb: int):
        self.aip_index = aip_index
        self.name = name
        self.mem_used_mb = mem_used_mb
        self.mem_total_mb = mem_total_mb

    def to_dict(self) -> dict:
        return {
            "aip_index": self.aip_index,
            "name": self.name,
            "mem_used_mb": self.mem_used_mb,
            "mem_total_mb": self.mem_total_mb,
        }

    def __repr__(self) -> str:
        return (
            f"GaudiAIPStat(index={self.aip_index}, name={self.name!r}, "
            f"mem_used={self.mem_used_mb}MB, mem_total={self.mem_total_mb}MB)"
        )


class GaudiAIPStatCollection:
    """Collects stats for all Intel Gaudi AIPs via hl-smi.

    hl-smi shares the same --query-aip / --format interface as nvidia-smi,
    so parsing is identical to the NVIDIA implementation.

    Example command:
        hl-smi -Q index,name,memory.used,memory.total -f csv,noheader,nounits
    Output (one line per device):
        0, HL-225, 512, 32768
    """

    def __init__(self, aips: list[GaudiAIPStat]):
        self.aips = aips

    def to_dict(self) -> dict:
        return {"aips": [aip.to_dict() for aip in self.aips]}

    def __len__(self) -> int:
        return len(self.aips)

    def __iter__(self):
        return iter(self.aips)

    def __repr__(self) -> str:
        return f"GaudiAIPStatCollection({self.aips!r})"

    @classmethod
    def new_query(cls) -> "GaudiAIPStatCollection":
        """
        Query hl-smi and return a GaudiAIPStatCollection.
        Returns an empty collection if hl-smi is not installed or fails.
        """
        if shutil.which("hl-smi") is None:
            return cls([])

        try:
            result = subprocess.run(
                [
                    "hl-smi",
                    "-Q", "index,name,memory.used,memory.total",
                    "-f", "csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return cls([])

        if result.returncode != 0 or not result.stdout.strip():
            return cls([])

        aips: list[GaudiAIPStat] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                aips.append(GaudiAIPStat(
                    aip_index=int(parts[0]),
                    name=parts[1],
                    mem_used_mb=int(parts[2]),
                    mem_total_mb=int(parts[3]),
                ))
            except ValueError:
                continue

        return cls(aips)


if __name__ == "__main__":
    import json
    collection = GaudiAIPStatCollection.new_query()
    print(json.dumps(collection.to_dict(), indent=2))
