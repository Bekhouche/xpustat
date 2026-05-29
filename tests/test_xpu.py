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
