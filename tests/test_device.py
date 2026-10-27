import pytest

from xpustat import pick_device, torch_device, framework_index, visible_indices, XPUStatCollection
from xpustat._device import DeviceMixin, VISIBLE_DEVICES_ENV
from xpustat.nvidia import NvidiaGPUStat, NvidiaGPUStatCollection
from xpustat.amd import AMDGPUStat, AMDGPUStatCollection
from xpustat.huawei import HuaweiNPUStat, HuaweiNPUStatCollection
from xpustat.apple import AppleGPUStat, AppleGPUStatCollection
from xpustat.tenstorrent import TenstorrentDeviceStat, TenstorrentDeviceStatCollection


def _nv(i, used, total=24000, util=None, uuid=None):
    return NvidiaGPUStat(gpu_index=i, uuid=uuid or f"GPU-{i}", name="L4",
                         mem_used_mb=used, mem_total_mb=total, util_pct=util)


def _stats(**kw):
    return XPUStatCollection(
        nvidia=NvidiaGPUStatCollection(kw.get("nvidia", [])),
        amd=AMDGPUStatCollection(kw.get("amd", [])),
        gaudi=__import__("xpustat.gaudi", fromlist=["x"]).GaudiAIPStatCollection([]),
        intel=__import__("xpustat.intel_gpu", fromlist=["x"]).IntelGPUStatCollection([]),
        huawei=HuaweiNPUStatCollection(kw.get("huawei", [])),
        hygon=__import__("xpustat.hygon", fromlist=["x"]).HygonDCUStatCollection([]),
        cambricon=__import__("xpustat.cambricon", fromlist=["x"]).CambriconMLUStatCollection([]),
        moorethreads=__import__("xpustat.moorethreads", fromlist=["x"]).MooreThreadsGPUStatCollection([]),
        apple=AppleGPUStatCollection(kw.get("apple", [])),
    )


# ---------------------------------------------------------------------------
# Mixin fields
# ---------------------------------------------------------------------------

def test_mixin_basic_fields():
    g = _nv(3, used=1000, total=4000)
    assert isinstance(g, DeviceMixin)
    assert g.vendor == "nvidia"
    assert g.device_index == 3
    assert g.index_label == "3"
    assert g.mem_free_mb == 3000
    assert g.to_dict()["mem_free_mb"] == 3000


def test_mixin_sub_index_label():
    n = HuaweiNPUStat(npu_id=2, chip_id=1, name="Ascend", mem_used_mb=1, mem_total_mb=2)
    assert n.device_index == 2
    assert n.index_label == "2:1"


def test_mem_free_never_negative_and_explicit_override():
    g = _nv(0, used=5000, total=4000)
    assert g.mem_free_mb == 0
    a = AppleGPUStat(gpu_index=0, name="M5", mem_used_mb=400, mem_total_mb=32768)
    assert a.mem_free_mb == 32368
    a._mem_free_mb = 12345
    assert a.mem_free_mb == 12345


# ---------------------------------------------------------------------------
# visible devices / framework index / torch device
# ---------------------------------------------------------------------------

def test_visible_indices_unset_means_all():
    assert visible_indices("nvidia", env={}) is None


def test_visible_indices_parses_list_and_empty():
    assert visible_indices("nvidia", env={"CUDA_VISIBLE_DEVICES": "2,0"}) == [2, 0]
    assert visible_indices("nvidia", env={"CUDA_VISIBLE_DEVICES": ""}) == []
    assert visible_indices("nvidia", env={"CUDA_VISIBLE_DEVICES": " 1 , 3 "}) == [1, 3]


def test_visible_indices_uuid_entries_and_unknowns():
    env = {"CUDA_VISIBLE_DEVICES": "GPU-b,bogus,1"}
    assert visible_indices("nvidia", env=env, uuids={"GPU-b": 7}) == [7, 1]
    assert visible_indices("nvidia", env=env) == [1]           # no uuid map → skip


def test_visible_indices_vendor_precedence_and_ze_mask():
    env = {"HIP_VISIBLE_DEVICES": "1", "CUDA_VISIBLE_DEVICES": "0"}
    assert visible_indices("amd", env=env) == [1]              # HIP wins over CUDA for AMD
    assert visible_indices("nvidia", env=env) == [0]
    assert visible_indices("huawei", env={"ASCEND_RT_VISIBLE_DEVICES": "3,2"}) == [3, 2]
    assert visible_indices("intel", env={"ZE_AFFINITY_MASK": "0.0,1.1"}) == [0, 1]
    assert visible_indices("apple", env={"CUDA_VISIBLE_DEVICES": ""}) is None   # not applicable


def test_framework_index_and_torch_device_with_env():
    g0, g1, g2 = _nv(0, 1), _nv(1, 1), _nv(2, 1)
    env = {"CUDA_VISIBLE_DEVICES": "2,0"}
    assert framework_index(g2, env=env) == 0
    assert framework_index(g0, env=env) == 1
    assert framework_index(g1, env=env) is None               # hidden
    assert torch_device(g2, env=env) == "cuda:0"
    assert torch_device(g1, env=env) is None
    assert torch_device(g1, env={}) == "cuda:1"


def test_framework_index_resolves_uuid_via_siblings():
    gpus = [_nv(0, 1, uuid="GPU-aaa"), _nv(1, 1, uuid="GPU-bbb")]
    env = {"CUDA_VISIBLE_DEVICES": "GPU-bbb"}
    assert framework_index(gpus[1], env=env, siblings=gpus) == 0
    assert framework_index(gpus[0], env=env, siblings=gpus) is None


def test_torch_device_per_vendor():
    assert AMDGPUStat(gpu_index=1, name="x", mem_used_mb=0, mem_total_mb=1).torch_device == "cuda:1"
    assert HuaweiNPUStat(npu_id=2, chip_id=0, name="x", mem_used_mb=0, mem_total_mb=1).torch_device == "npu:2"
    assert AppleGPUStat(gpu_index=0, name="x", mem_used_mb=0, mem_total_mb=1).torch_device == "mps"
    assert TenstorrentDeviceStat(tt_device=0, name="x").torch_device is None


def test_mixin_properties_read_real_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    g = _nv(1, 1)
    assert g.framework_index == 0 and g.torch_device == "cuda:0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert g.framework_index is None and g.torch_device is None


# ---------------------------------------------------------------------------
# pick_device
# ---------------------------------------------------------------------------

def test_pick_device_most_free_first():
    stats = _stats(nvidia=[_nv(0, 20000), _nv(1, 1000), _nv(2, 10000)])
    picked = pick_device(n=2, stats=stats, env={})
    assert [d.gpu_index for d in picked] == [1, 2]
    assert pick_device(stats=stats, env={})[0].torch_device == "cuda:1"


def test_pick_device_rankings_and_filters():
    stats = _stats(nvidia=[_nv(0, 100, util=90), _nv(1, 200, util=5)],
                   apple=[AppleGPUStat(gpu_index=0, name="M5", mem_used_mb=0, mem_total_mb=100000)])
    assert pick_device(by="util", vendor="nvidia", stats=stats, env={})[0].gpu_index == 1
    assert pick_device(by="util", stats=stats, env={})[0].vendor == "apple"   # no util reported → treated as idle
    assert pick_device(by="mem_used", stats=stats, env={})[0].vendor == "apple"
    assert pick_device(vendor="nvidia", by="mem_used", stats=stats, env={})[0].gpu_index == 0
    assert [d.vendor for d in pick_device(n=5, vendor=["apple"], stats=stats, env={})] == ["apple"]
    assert pick_device(stats=stats, env={}, min_free_mb=50_000)[0].vendor == "apple"
    assert pick_device(n=0, stats=stats, env={}) == []


def test_pick_device_respects_visible_devices():
    stats = _stats(nvidia=[_nv(0, 0), _nv(1, 20000)])
    env = {"CUDA_VISIBLE_DEVICES": "1"}
    picked = pick_device(stats=stats, env=env)
    assert [d.gpu_index for d in picked] == [1]
    assert pick_device(stats=stats, env=env, visible_only=False)[0].gpu_index == 0
    assert pick_device(stats=stats, env={"CUDA_VISIBLE_DEVICES": ""}) == []


def test_pick_device_rejects_unknown_ranking():
    with pytest.raises(ValueError):
        pick_device(by="nope", stats=_stats())


# ---------------------------------------------------------------------------
# 0.4.0: ranges, sub-unit variables, env string, resource names
# ---------------------------------------------------------------------------

from xpustat import visible_devices_env, RAY_RESOURCE, K8S_RESOURCE
from xpustat.neuron import NeuronDeviceStat, NeuronDeviceStatCollection
from xpustat.tpu import TPUChipStat


def test_visible_indices_ranges_and_new_vendors():
    assert visible_indices("neuron", env={"NEURON_RT_VISIBLE_CORES": "0-3,8"}) == [0, 1, 2, 3, 8]
    assert visible_indices("tpu", env={"TPU_VISIBLE_CHIPS": "1,0"}) == [1, 0]
    assert visible_indices("tenstorrent", env={"TT_VISIBLE_DEVICES": "0"}) == [0]
    assert visible_indices("neuron", env={"NEURON_RT_VISIBLE_CORES": "3-1"}) == []   # bad range skipped


def _neuron(i, cores):
    return NeuronDeviceStat(neuron_device=i, name="n", mem_used_mb=0, mem_total_mb=1, nc_count=len(cores), neuroncore_ids=cores)


def test_neuron_cores_map_to_devices():
    devs = [_neuron(0, [0, 1]), _neuron(1, [2, 3]), _neuron(2, [4, 5])]
    env = {"NEURON_RT_VISIBLE_CORES": "2-5"}
    assert framework_index(devs[0], env=env, siblings=devs) is None
    assert framework_index(devs[1], env=env, siblings=devs) == 0
    assert framework_index(devs[2], env=env, siblings=devs) == 1
    assert torch_device(devs[2], env=env, siblings=devs) == "xla:1"
    assert framework_index(devs[2], env={}, siblings=devs) == 2
    # a device is visible if any of its cores is
    assert framework_index(devs[0], env={"NEURON_RT_VISIBLE_CORES": "1"}, siblings=devs) == 0


def test_visible_devices_env_index_and_uuid():
    gpus = [_nv(2, 1, uuid="GPU-c"), _nv(0, 1, uuid="GPU-a")]
    assert visible_devices_env(gpus) == {"CUDA_VISIBLE_DEVICES": "2,0"}
    assert visible_devices_env(gpus, format="uuid") == {"CUDA_VISIBLE_DEVICES": "GPU-c,GPU-a"}
    mixed = gpus + [AMDGPUStat(gpu_index=1, name="x", mem_used_mb=0, mem_total_mb=1),
                    AppleGPUStat(gpu_index=0, name="x", mem_used_mb=0, mem_total_mb=1),
                    _neuron(1, [2, 3]), TPUChipStat(device_id=3, name="t", mem_used_mb=0, mem_total_mb=1)]
    env = visible_devices_env(mixed, format="uuid")
    assert env == {"CUDA_VISIBLE_DEVICES": "GPU-c,GPU-a", "HIP_VISIBLE_DEVICES": "1",
                   "NEURON_RT_VISIBLE_CORES": "2,3", "TPU_VISIBLE_CHIPS": "3"}
    assert visible_devices_env([]) == {}
    with pytest.raises(ValueError):
        visible_devices_env(gpus, format="nope")


def test_resource_names():
    assert _nv(0, 1).ray_resource == "GPU" and _nv(0, 1).k8s_resource == "nvidia.com/gpu"
    n = HuaweiNPUStat(npu_id=0, chip_id=0, name="x", mem_used_mb=0, mem_total_mb=1)
    assert n.ray_resource == "NPU" and n.k8s_resource == "huawei.com/Ascend910"
    assert _neuron(0, [0]).ray_resource == "neuron_cores"
    assert AppleGPUStat(gpu_index=0, name="x", mem_used_mb=0, mem_total_mb=1).k8s_resource is None
    assert set(RAY_RESOURCE) == set(K8S_RESOURCE) >= set(VISIBLE_DEVICES_ENV)
