import pytest
import datetime
import json
from xpustat import xpu
from xpustat.xpu import XPUStatCollection
from xpustat.nvidia import NvidiaGPUStatCollection
from xpustat.amd import AMDGPUStatCollection
from xpustat.gaudi import GaudiAIPStatCollection
from xpustat.intel_gpu import IntelGPUStatCollection
from xpustat.huawei import HuaweiNPUStatCollection
from xpustat.hygon import HygonDCUStatCollection
from xpustat.cambricon import CambriconMLUStatCollection
from xpustat.moorethreads import MooreThreadsGPUStatCollection
from xpustat.apple import AppleGPUStatCollection

def create_mock_collection():
    return XPUStatCollection(
        nvidia=NvidiaGPUStatCollection([]),
        amd=AMDGPUStatCollection([]),
        gaudi=GaudiAIPStatCollection([]),
        intel=IntelGPUStatCollection([]),
        huawei=HuaweiNPUStatCollection([]),
        hygon=HygonDCUStatCollection([]),
        cambricon=CambriconMLUStatCollection([]),
        moorethreads=MooreThreadsGPUStatCollection([]),
        apple=AppleGPUStatCollection([]),
    )

def test_xpustat_collection_empty():
    col = create_mock_collection()
    assert len(col) == 0
    assert not bool(col)
    assert len(list(col)) == 0

def test_xpustat_collection_get_item():
    col = create_mock_collection()
    assert isinstance(col["nvidia"], NvidiaGPUStatCollection)
    assert isinstance(col["amd"], AMDGPUStatCollection)
    with pytest.raises(KeyError):
        _ = col["unknown"]

def test_xpustat_collection_to_dict():
    col = create_mock_collection()
    d = col.to_dict()
    assert "nvidia" in d
    assert "amd" in d
    assert "moorethreads" in d
    assert "apple" in d
    for k in ("neuron", "tpu", "tenstorrent"):
        assert d[k] == []
    assert d["nvidia"] == []

def test_xpustat_collection_to_json():
    col = create_mock_collection()
    j = col.to_json()
    assert "nvidia" in json.loads(j)
    
    j_meta = col.to_json(metadata=True)
    d_meta = json.loads(j_meta)
    assert "hostname" in d_meta
    assert "query_time" in d_meta
    assert "xpustat_version" in d_meta
    assert "devices" in d_meta

def test_query_all(monkeypatch):
    # Mocking new_query on all collections to return empty for testing
    for name, cls in xpu._BACKENDS:
        monkeypatch.setattr(cls, "new_query", lambda: cls([]))
        
    col = xpu.query_all(parallel=False)
    assert isinstance(col, XPUStatCollection)
    assert len(col) == 0

    col_parallel = xpu.query_all(parallel=True)
    assert isinstance(col_parallel, XPUStatCollection)

def test_query(monkeypatch):
    monkeypatch.setattr(NvidiaGPUStatCollection, "new_query", lambda: NvidiaGPUStatCollection([]))
    res = xpu.query("nvidia")
    assert isinstance(res, NvidiaGPUStatCollection)
    
    with pytest.raises(ValueError):
        xpu.query("unknown")


# ---------------------------------------------------------------------------
# 0.4.0: jetson key, plugins, cache
# ---------------------------------------------------------------------------

from xpustat._device import DeviceMixin


class _AcmeDev(DeviceMixin):
    vendor = "acme"; _index_attr = "idx"
    def __init__(self, idx, used=1, total=8):
        self.idx = idx; self.name = f"Acme {idx}"; self.mem_used_mb = used; self.mem_total_mb = total; self.processes = []
    def to_dict(self):
        return {"idx": self.idx, "name": self.name, "mem_used_mb": self.mem_used_mb, "mem_total_mb": self.mem_total_mb}


class _AcmeCollection:
    def __init__(self, devs): self.devs = devs
    def __len__(self): return len(self.devs)
    def __iter__(self): return iter(self.devs)
    def __getitem__(self, i): return self.devs[i]
    def to_dict(self): return {"accelerators": [d.to_dict() for d in self.devs]}
    @classmethod
    def new_query(cls): return cls([_AcmeDev(0), _AcmeDev(1, used=7)])


def test_to_dict_has_jetson_key():
    assert create_mock_collection().to_dict()["jetson"] == []


def test_register_backend_roundtrip(monkeypatch):
    import xpustat
    for name, cls in xpu._BACKENDS:
        monkeypatch.setattr(cls, "new_query", classmethod(lambda c: c([])))
    xpustat.register_backend("acme", _AcmeCollection)
    try:
        assert "acme" in xpu._BACKENDS_KEYS and xpu._BACKENDS_MAP["acme"] is _AcmeCollection
        s = xpu.query_all(parallel=False)
        assert len(s.acme) == 2 and s["acme"][1].name == "Acme 1" and len(s) == 2
        assert s.to_dict()["acme"][1]["mem_used_mb"] == 7
        assert "acme=2" in repr(s)
        assert xpustat.pick_device(stats=s)[0].idx == 0            # most free
        assert xpustat.query("acme")[0].idx == 0
        assert "acme" in xpustat.to_prometheus(s)
        empty = create_mock_collection()                          # no kwarg → empty plugin slot
        assert len(empty.acme) == 0 and empty.to_dict()["acme"] == []
        with pytest.raises(ValueError):
            xpustat.register_backend("nvidia", _AcmeCollection)   # built-in name
        with pytest.raises(ValueError):
            xpustat.register_backend("Not-Valid", _AcmeCollection)
    finally:
        xpustat.unregister_backend("acme")
    assert "acme" not in xpu._BACKENDS_KEYS
    with pytest.raises(TypeError):
        XPUStatCollection(*[c([]) for _, c in xpu._BACKENDS[:8]], acme=_AcmeCollection([]))


def test_query_all_max_age_cache(monkeypatch):
    for name, cls in xpu._BACKENDS:
        monkeypatch.setattr(cls, "new_query", classmethod(lambda c: c([])))
    a = xpu.query_all(parallel=False)
    b = xpu.query_all(parallel=False, max_age=10)
    assert b is a
    c = xpu.query_all(parallel=False)                     # no max_age → fresh
    assert c is not a
    monkeypatch.setattr(xpu.time, "monotonic", lambda: xpu._CACHE["time"] + 100)
    assert xpu.query_all(parallel=False, max_age=10) is not c
    assert xpu.query_all(parallel=False, vendors=["nvidia"], max_age=10) is not xpu.query_all(parallel=False, max_age=10)
