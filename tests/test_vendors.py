import pytest
from xpustat.nvidia import NvidiaGPUStat, NvidiaGPUStatCollection
from xpustat.amd import AMDGPUStat, AMDGPUStatCollection
from xpustat.gaudi import GaudiAIPStat, GaudiAIPStatCollection
from xpustat.intel_gpu import IntelGPUStat, IntelGPUStatCollection
from xpustat.huawei import HuaweiNPUStat, HuaweiNPUStatCollection
from xpustat.hygon import HygonDCUStat, HygonDCUStatCollection
from xpustat.cambricon import CambriconMLUStat, CambriconMLUStatCollection
from xpustat.moorethreads import MooreThreadsGPUStat, MooreThreadsGPUStatCollection
from xpustat.apple import AppleGPUStat, AppleGPUStatCollection, _parse_ioreg

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

def test_apple_stat():
    stat = AppleGPUStat(
        gpu_index=0,
        name="Apple M5",
        mem_used_mb=375,
        mem_total_mb=32768,
        util_pct=15,
        gpu_cores=10,
    )
    col = AppleGPUStatCollection([stat])
    assert len(col) == 1
    d = col.to_dict()
    assert d["gpus"][0]["name"] == "Apple M5"
    assert d["gpus"][0]["gpu_cores"] == 10
    assert d["gpus"][0]["unified_memory"] is True
    assert d["gpus"][0]["temp_c"] is None
    assert d["gpus"][0]["power_w"] is None


_IOREG_APPLE_SILICON = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<array>
  <dict>
    <key>IOClass</key><string>AGXAcceleratorG17G</string>
    <key>model</key><string>Apple M5</string>
    <key>gpu-core-count</key><integer>10</integer>
    <key>PerformanceStatistics</key>
    <dict>
      <key>Device Utilization %</key><integer>15</integer>
      <key>Renderer Utilization %</key><integer>15</integer>
      <key>In use system memory</key><integer>393560064</integer>
      <key>Alloc system memory</key><integer>5243404288</integer>
    </dict>
  </dict>
</array>
</plist>
"""

_IOREG_INTEL_MAC_DGPU = b"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<array>
  <dict>
    <key>IOClass</key><string>AMDRadeonX6000_AMDRDNAGraphicsAccelerator</string>
    <key>model</key><data>QU1EIFJhZGVvbiBQcm8gNTUwME0A</data>
    <key>PerformanceStatistics</key>
    <dict>
      <key>Device Utilization %</key><integer>42</integer>
      <key>vramUsedBytes</key><integer>1073741824</integer>
      <key>vramFreeBytes</key><integer>3221225472</integer>
    </dict>
  </dict>
  <dict>
    <key>IOClass</key><string>SomethingWithoutStats</string>
  </dict>
</array>
</plist>
"""


def test_apple_parse_ioreg_apple_silicon():
    gpus = _parse_ioreg(_IOREG_APPLE_SILICON, system_memory_bytes=34359738368)
    assert len(gpus) == 1
    g = gpus[0]
    assert g.gpu_index == 0
    assert g.name == "Apple M5"
    assert g.gpu_cores == 10
    assert g.util_pct == 15
    assert g.unified_memory is True
    assert g.mem_used_mb == 393560064 // (1024 * 1024)
    assert g.mem_total_mb == 32768
    assert g.processes == []


def test_apple_parse_ioreg_discrete_gpu():
    gpus = _parse_ioreg(_IOREG_INTEL_MAC_DGPU, system_memory_bytes=17179869184)
    # the entry without PerformanceStatistics is skipped
    assert len(gpus) == 1
    g = gpus[0]
    assert g.name == "AMD Radeon Pro 5500M"      # bytes model decoded, NUL stripped
    assert g.gpu_cores is None
    assert g.util_pct == 42
    assert g.unified_memory is False
    assert g.mem_used_mb == 1024
    assert g.mem_total_mb == 4096


def test_apple_parse_ioreg_garbage():
    assert _parse_ioreg(b"not a plist") == []
    assert _parse_ioreg(b"<plist version=\"1.0\"><dict/></plist>") == []


def test_apple_new_query_off_macos(monkeypatch):
    import xpustat.apple as apple_mod
    monkeypatch.setattr(apple_mod.sys, "platform", "linux")
    assert len(AppleGPUStatCollection.new_query()) == 0


def test_all_collections_support_index_access():
    """README promises ``stats.amd[0].name``; every vendor collection must index."""
    cases = [
        (NvidiaGPUStatCollection, NvidiaGPUStat(gpu_index=0, uuid="u", name="n", mem_used_mb=1, mem_total_mb=2)),
        (AMDGPUStatCollection, AMDGPUStat(gpu_index=0, name="n", mem_used_mb=1, mem_total_mb=2)),
        (GaudiAIPStatCollection, GaudiAIPStat(aip_index=0, name="n", mem_used_mb=1, mem_total_mb=2)),
        (IntelGPUStatCollection, IntelGPUStat(device_id=0, name="n", mem_used_mb=1, mem_total_mb=2)),
        (HuaweiNPUStatCollection, HuaweiNPUStat(npu_id=0, chip_id=0, name="n", mem_used_mb=1, mem_total_mb=2)),
        (HygonDCUStatCollection, HygonDCUStat(card_index=0, name="n", mem_used_mb=1, mem_total_mb=2)),
        (CambriconMLUStatCollection, CambriconMLUStat(card_id=0, chip_id=0, name="n", mem_used_mb=1, mem_total_mb=2)),
        (MooreThreadsGPUStatCollection, MooreThreadsGPUStat(gpu_id=0, name="n", mem_used_mb=1, mem_total_mb=2)),
        (AppleGPUStatCollection, AppleGPUStat(gpu_index=0, name="n", mem_used_mb=1, mem_total_mb=2)),
    ]
    for cls, stat in cases:
        col = cls([stat])
        assert col[0] is stat, cls.__name__
        assert col[-1] is stat, cls.__name__
        with pytest.raises(IndexError):
            _ = col[1]


def test_apple_iokit_and_ioreg_paths_agree():
    """On macOS the in-process IOKit path must see the same devices as ioreg."""
    import sys
    if sys.platform != "darwin":
        pytest.skip("macOS only")
    import xpustat.apple as apple_mod
    fast = AppleGPUStatCollection.new_query()
    slow = AppleGPUStatCollection._query_subprocess()
    assert fast.backend == "iokit"
    assert slow.backend == "subprocess"
    assert len(fast) == len(slow)
    for a, b in zip(fast, slow):
        assert (a.name, a.gpu_cores, a.unified_memory, a.mem_total_mb) == \
               (b.name, b.gpu_cores, b.unified_memory, b.mem_total_mb)


def test_apple_falls_back_to_ioreg_when_iokit_unavailable(monkeypatch):
    import xpustat.apple as apple_mod
    monkeypatch.setattr(apple_mod.sys, "platform", "darwin")
    monkeypatch.setattr(apple_mod, "_iokit_accelerator_entries", lambda: None)
    monkeypatch.setattr(apple_mod.shutil, "which", lambda *_: "/usr/sbin/ioreg")

    class _R:
        returncode = 0
        stdout = _IOREG_APPLE_SILICON
    monkeypatch.setattr(apple_mod.subprocess, "run", lambda *a, **k: _R())
    monkeypatch.setattr(apple_mod, "_system_memory_bytes", lambda: 34359738368)

    col = AppleGPUStatCollection.new_query()
    assert col.backend == "subprocess"
    assert len(col) == 1 and col[0].name == "Apple M5" and col[0].mem_total_mb == 32768


def test_apple_iokit_entries_used_directly(monkeypatch):
    import xpustat.apple as apple_mod
    monkeypatch.setattr(apple_mod.sys, "platform", "darwin")
    entries = [{"model": "Apple M3 Max", "gpu-core-count": 40,
                "PerformanceStatistics": {"Device Utilization %": 77, "In use system memory": 8 * 1024 * 1024 * 1024}}]
    monkeypatch.setattr(apple_mod, "_iokit_accelerator_entries", lambda: entries)
    monkeypatch.setattr(apple_mod, "_system_memory_bytes", lambda: 128 * 1024 ** 3)
    col = AppleGPUStatCollection.new_query()
    assert col.backend == "iokit"
    g = col[0]
    assert (g.name, g.gpu_cores, g.util_pct, g.mem_used_mb, g.mem_total_mb) == ("Apple M3 Max", 40, 77, 8192, 131072)


# ---------------------------------------------------------------------------
# AWS Neuron
# ---------------------------------------------------------------------------

from xpustat.neuron import NeuronDeviceStat, NeuronDeviceStatCollection, _parse_neuron_ls

_NEURON_LS = """[
  {"neuron_device": 0, "bdf": "cc:00.0", "cpu_affinity": "48-95", "numa_node": "1",
   "connected_to": [12, 3], "nc_count": 4, "logical_neuroncore_config": 2,
   "memory_size": 103079215104, "neuroncore_ids": [0, 1, 2, 3],
   "neuron_processes": [{"pid": 4194303, "command": "neuron-bench exec --run-as-cc-neff x", "neuron_runtime_version": "2.0.0"}]},
  {"neuron_device": 1, "bdf": "cd:00.0", "nc_count": 4, "memory_size": 103079215104,
   "neuroncore_ids": [4, 5, 6, 7], "neuron_processes": []},
  {"garbage": true}
]"""


def test_neuron_parse():
    devs = _parse_neuron_ls(_NEURON_LS)
    assert len(devs) == 2
    d = devs[0]
    assert d.neuron_device == 0 and d.vendor == "neuron" and d.device_index == 0
    assert d.nc_count == 4 and d.bdf == "cc:00.0"
    assert d.mem_total_mb == 98304 and d.mem_used_mb == 0 and d.util_pct is None
    assert d.name == "AWS Neuron (4 NeuronCores)"
    assert len(d.processes) == 1 and d.processes[0].pid == 4194303
    assert d.processes[0].command == "neuron-bench"       # nonexistent PID → fallback basename
    assert d.torch_device == "xla:0"
    col = NeuronDeviceStatCollection(devs)
    assert col[1].neuron_device == 1 and col.to_dict()["devices"][0]["mem_free_mb"] == 98304
    assert _parse_neuron_ls("not json") == [] and _parse_neuron_ls("{}") == []


# ---------------------------------------------------------------------------
# Google Cloud TPU
# ---------------------------------------------------------------------------

from xpustat.tpu import TPUChipStat, TPUChipStatCollection, _parse_dashboard, _accelerator_type

_TPU_DASH = """WARNING: something before the json
{
  "timestamp": "2026-10-24T00:00:00+00:00", "tpu_info_version": "0.5.1", "libtpu_version": "0.0.18",
  "topology": "2x4",
  "devices": [
    {"device_id": 0, "device_path": "/dev/vfio/0", "memory_used_bytes": 18538430464,
     "memory_total_bytes": 33554432000, "duty_cycle_pct": 99.6, "hbm_bandwidth_util_pct": 12.5,
     "process": {"pid": 4194303, "name": "python3"}},
    {"device_id": 1, "device_path": "/dev/vfio/1", "memory_used_bytes": 0,
     "memory_total_bytes": 33554432000, "duty_cycle_pct": 0.0, "hbm_bandwidth_util_pct": 0.0, "process": null}
  ]
}"""


def test_tpu_parse():
    assert _accelerator_type("- tpu-info version: 0.5.1\n- libtpu version: 0.0.18\n- accelerator type: v6e\n") == "v6e"
    assert _accelerator_type(None) is None
    chips = _parse_dashboard(_TPU_DASH, "v6e")
    assert len(chips) == 2
    c = chips[0]
    assert c.vendor == "tpu" and c.device_id == 0 and c.name == "TPU v6e"
    assert c.mem_used_mb == 17679 and c.mem_total_mb == 32000 and c.mem_free_mb == 14321
    assert c.util_pct == 100 and c.hbm_bw_util_pct == 12.5 and c.device_path == "/dev/vfio/0"
    assert len(c.processes) == 1 and c.processes[0].pid == 4194303
    assert chips[1].processes == [] and chips[1].util_pct == 0
    assert c.torch_device == "xla:0"
    assert _parse_dashboard(_TPU_DASH)[0].name == "Cloud TPU"
    assert _parse_dashboard("no json here") == [] and _parse_dashboard("[1,2]") == []
    assert TPUChipStatCollection(chips)[1].device_id == 1


# ---------------------------------------------------------------------------
# Tenstorrent
# ---------------------------------------------------------------------------

from xpustat.tenstorrent import TenstorrentDeviceStat, TenstorrentDeviceStatCollection, _parse_snapshot

_TT_SNAP = """{
  "time": "2026-10-24T00:00:00", "host_info": {"OS": "Linux"},
  "device_info": [
    {"board_info": {"bus_id": "0000:01:00.0", "board_type": "n150 L", "board_id": "0100014511706018"},
     "telemetry": {"voltage": "0.80", "current": " 30.0", "power": " 24.0", "board_power": " 60.0",
                   "aiclk": "1000", "asic_temperature": " 45.6"},
     "firmwares": {}, "limits": {}},
    {"board_info": {"bus_id": "0000:02:00.0", "board_type": "n300 R"},
     "telemetry": {"power": "N/A", "asic_temperature": null}}
  ],
  "processes": [{"pid": 4194303, "user": "salah", "device": 0, "cmdline": "/usr/bin/python3 train.py"}]
}"""


def test_tenstorrent_parse():
    devs = _parse_snapshot(_TT_SNAP)
    assert len(devs) == 2
    d = devs[0]
    assert d.vendor == "tenstorrent" and d.tt_device == 0 and d.device_index == 0 and d.name == "Tenstorrent n150 L"
    assert d.temp_c == 46 and d.power_w == 24.0 and d.board_power_w == 60.0 and d.aiclk_mhz == 1000
    assert d.bus_id == "0000:01:00.0" and d.board_id == "0100014511706018"
    assert d.mem_used_mb == 0 and d.mem_total_mb == 0 and d.mem_free_mb == 0 and d.util_pct is None
    assert len(d.processes) == 1 and d.processes[0].pid == 4194303 and d.processes[0].command == "python3"
    assert devs[1].temp_c is None and devs[1].power_w is None and devs[1].processes == []
    assert d.torch_device is None
    assert TenstorrentDeviceStatCollection(devs)[1].name == "Tenstorrent n300 R"
    assert _parse_snapshot("nope") == [] and _parse_snapshot("[]") == []


def test_new_backends_off_host_return_empty(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda *_: None)
    assert len(NeuronDeviceStatCollection.new_query()) == 0
    assert len(TPUChipStatCollection.new_query()) == 0
    assert len(TenstorrentDeviceStatCollection.new_query()) == 0


# ---------------------------------------------------------------------------
# NVIDIA Jetson
# ---------------------------------------------------------------------------

from xpustat.jetson import JetsonGPUStat, JetsonGPUStatCollection, _parse_tegrastats

_TEGRA_ORIN = ("10-24-2026 10:00:00 RAM 2567/7772MB (lfb 1x4MB) SWAP 0/3886MB (cached 0MB) "
               "CPU [1%@729,0%@729,0%@729,0%@729,off,off] EMC_FREQ 0%@2133 GR3D_FREQ 12%@[624,624] "
               "VIC_FREQ 0%@115 APE 174 cpu@44.5C soc2@42.093C soc0@43.5C gpu@44.343C tj@44.5C soc1@43.5C "
               "VDD_IN 4522mW/4522mW VDD_CPU_GPU_CV 486mW/486mW VDD_SOC 1296mW/1296mW")
_TEGRA_XAVIER = ("RAM 1234/15817MB (lfb 3000x4MB) SWAP 0/7908MB (cached 0MB) CPU [0%@1190,off,off,0%@1190] "
                 "EMC_FREQ 0% GR3D_FREQ 0% AO@36.5C GPU@36C PMIC@100C AUX@36C CPU@37.5C thermal@36.5C "
                 "GPU 0/0 CPU 155/155 SOC 931/931 CV 0/0 VDDRQ 155/155 SYS5V 1555/1555")
_TEGRA_NANO = ("RAM 1800/3964MB (lfb 20x4MB) SWAP 0/1982MB (cached 0MB) IRAM 0/252kB(lfb 252kB) "
               "CPU [10%@1479,8%@1479,7%@1479,8%@1479] EMC_FREQ 3%@1600 GR3D_FREQ 25%@76 "
               "PLL@35C CPU@37.5C PMIC@100C GPU@36.5C AO@42.5C thermal@36.75C POM_5V_IN 1500/1500 POM_5V_GPU 120/120 POM_5V_CPU 500/500")


def test_jetson_parse_orin():
    g = _parse_tegrastats(_TEGRA_ORIN, "NVIDIA Jetson Orin Nano Developer Kit")
    assert g.vendor == "jetson" and g.name.startswith("NVIDIA Jetson Orin")
    assert (g.mem_used_mb, g.mem_total_mb, g.mem_free_mb) == (2567, 7772, 5205)
    assert g.util_pct == 12 and g.temp_c == 44
    assert g.power_w == 0.486 and g.power_rail == "VDD_CPU_GPU_CV"
    assert g.torch_device == "cuda:0" and g.k8s_resource == "nvidia.com/gpu"
    assert g.to_dict()["unified_memory"] is True


def test_jetson_parse_xavier_and_nano():
    x = _parse_tegrastats(_TEGRA_XAVIER)
    assert (x.mem_used_mb, x.mem_total_mb, x.util_pct, x.temp_c) == (1234, 15817, 0, 36)
    assert x.power_w == 0.0 and x.power_rail == "GPU"          # Xavier rail named "GPU", not the GPU@..C temp
    n = _parse_tegrastats(_TEGRA_NANO)
    assert (n.util_pct, n.temp_c, n.power_w, n.power_rail) == (25, 36, 0.12, "POM_5V_GPU")
    assert _parse_tegrastats("garbage without ram") is None
    assert JetsonGPUStatCollection([n])[0] is n


def test_jetson_backend_yields_to_nvidia_smi(monkeypatch):
    import xpustat.jetson as jm
    monkeypatch.setattr(jm.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    assert len(JetsonGPUStatCollection.new_query()) == 0
    monkeypatch.setattr(jm.shutil, "which", lambda name: "/usr/bin/tegrastats" if name == "tegrastats" else None)
    monkeypatch.setattr(JetsonGPUStatCollection, "is_jetson", staticmethod(lambda: True))
    monkeypatch.setattr(jm, "_tegrastats_line", lambda *a, **k: _TEGRA_NANO)
    monkeypatch.setattr(jm, "_read_model", lambda: "NVIDIA Jetson Nano Developer Kit")
    col = JetsonGPUStatCollection.new_query()
    assert len(col) == 1 and col[0].name == "NVIDIA Jetson Nano Developer Kit" and col[0].util_pct == 25


# ---------------------------------------------------------------------------
# NVIDIA MIG
# ---------------------------------------------------------------------------

from xpustat.nvidia import MIGInstanceStat
from xpustat import visible_devices_env


def test_mig_instance_stat():
    m = MIGInstanceStat(parent_index=1, uuid="MIG-abc", mem_used_mb=100, mem_total_mb=10240,
                        gpu_instance_id=3, compute_instance_id=0, profile="1g.10gb")
    assert m.vendor == "nvidia" and m.device_index == 1 and m.index_label == "1/3.0"
    assert m.name == "MIG 1g.10gb" and m.mem_free_mb == 10140
    assert m.framework_index is None and m.torch_device is None
    assert visible_devices_env([m], format="uuid") == {"CUDA_VISIBLE_DEVICES": "MIG-abc"}
    g = NvidiaGPUStat(gpu_index=1, uuid="GPU-x", name="H100", mem_used_mb=0, mem_total_mb=81920, mig_instances=[m])
    assert g.to_dict()["mig_instances"][0]["uuid"] == "MIG-abc"
    assert NvidiaGPUStat(gpu_index=0, uuid="u", name="n", mem_used_mb=0, mem_total_mb=1).to_dict()["mig_instances"] == []


def test_mig_nvml_helper_handles_unsupported(monkeypatch):
    import xpustat.nvidia as nv
    class FakeNVML:
        NVML_DEVICE_MIG_ENABLE = 1
        class NVMLError(Exception): pass
        @staticmethod
        def nvmlDeviceGetMigMode(h): raise FakeNVML.NVMLError("not supported")
    monkeypatch.setattr(nv, "_pynvml", FakeNVML, raising=False)
    assert nv._mig_instances_nvml(object(), 0) == []

    class FakeNVMLMig(FakeNVML):
        @staticmethod
        def nvmlDeviceGetMigMode(h): return (1, 1)
        @staticmethod
        def nvmlDeviceGetMaxMigDeviceCount(h): return 2
        @staticmethod
        def nvmlDeviceGetMigDeviceHandleByIndex(h, j):
            if j == 1: raise FakeNVML.NVMLError("unused slot")
            return ("mig", j)
        @staticmethod
        def nvmlDeviceGetUUID(h): return b"MIG-0000"
        @staticmethod
        def nvmlDeviceGetMemoryInfo(h):
            class M: used = 512 * 1024 * 1024; total = 10 * 1024 ** 3
            return M()
        @staticmethod
        def nvmlDeviceGetGpuInstanceId(h): return 5
        @staticmethod
        def nvmlDeviceGetComputeInstanceId(h): return 0
        @staticmethod
        def nvmlDeviceGetComputeRunningProcesses(h): return []
    monkeypatch.setattr(nv, "_pynvml", FakeNVMLMig, raising=False)
    migs = nv._mig_instances_nvml(object(), 2)
    assert len(migs) == 1 and migs[0].uuid == "MIG-0000" and migs[0].index_label == "2/5.0"
    assert (migs[0].mem_used_mb, migs[0].mem_total_mb) == (512, 10240)
