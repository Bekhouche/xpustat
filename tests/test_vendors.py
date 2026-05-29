import pytest
from xpustat.nvidia import NvidiaGPUStat, NvidiaGPUStatCollection
from xpustat.amd import AMDGPUStat, AMDGPUStatCollection
from xpustat.gaudi import GaudiAIPStat, GaudiAIPStatCollection
from xpustat.intel_gpu import IntelGPUStat, IntelGPUStatCollection
from xpustat.huawei import HuaweiNPUStat, HuaweiNPUStatCollection
from xpustat.hygon import HygonDCUStat, HygonDCUStatCollection
from xpustat.cambricon import CambriconMLUStat, CambriconMLUStatCollection
from xpustat.moorethreads import MooreThreadsGPUStat, MooreThreadsGPUStatCollection

def test_nvidia_stat():
    stat = NvidiaGPUStat(
        gpu_index=0,
        uuid="GPU-12345",
        name="Mock Nvidia",
        mem_used_mb=1000,
        mem_total_mb=8000,
        util_pct=50,
        temp_c=45,
        power_w=120.0,
        cuda_cc="8.6",
        processes=[]
    )
    col = NvidiaGPUStatCollection([stat])
    assert len(col) == 1
    d = col.to_dict()
    assert len(d["gpus"]) == 1
    assert d["gpus"][0]["name"] == "Mock Nvidia"

def test_amd_stat():
    stat = AMDGPUStat(
        gpu_index=0,
        name="Mock AMD",
        mem_used_mb=2000,
        mem_total_mb=16000,
        util_pct=80,
        temp_c=65,
        power_w=200.0,
        gfx_target="gfx1100"
    )
    col = AMDGPUStatCollection([stat])
    assert len(col) == 1
    d = col.to_dict()
    assert len(d["gpus"]) == 1

def test_gaudi_stat():
    stat = GaudiAIPStat(
        aip_index=0,
        name="Gaudi2",
        mem_used_mb=5000,
        mem_total_mb=32000
    )
    col = GaudiAIPStatCollection([stat])
    assert len(col) == 1

def test_intel_stat():
    stat = IntelGPUStat(
        device_id=0,
        name="Intel Arc",
        mem_used_mb=1000,
        mem_total_mb=8000
    )
    col = IntelGPUStatCollection([stat])
    assert len(col) == 1

def test_huawei_stat():
    stat = HuaweiNPUStat(
        npu_id=0,
        chip_id=0,
        name="Ascend 910B",
        mem_used_mb=2000,
        mem_total_mb=32000,
        ai_core_pct=50,
        temp_c=40,
        power_w=150.0,
        health="OK"
    )
    col = HuaweiNPUStatCollection([stat])
    assert len(col) == 1

def test_hygon_stat():
    stat = HygonDCUStat(
        card_index=0,
        name="Hygon DCU",
        mem_used_mb=1000,
        mem_total_mb=16000
    )
    col = HygonDCUStatCollection([stat])
    assert len(col) == 1

def test_cambricon_stat():
    stat = CambriconMLUStat(
        card_id=0,
        chip_id=0,
        name="Cambricon MLU",
        mem_used_mb=1000,
        mem_total_mb=16000
    )
    col = CambriconMLUStatCollection([stat])
    assert len(col) == 1

def test_moorethreads_stat():
    stat = MooreThreadsGPUStat(
        gpu_id=0,
        name="MTT S80",
        mem_used_mb=1000,
        mem_total_mb=16000
    )
    col = MooreThreadsGPUStatCollection([stat])
    assert len(col) == 1
