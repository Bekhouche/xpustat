# Contributing

## Running the tests

The unit tests need `pytest` and `psutil` and no hardware; vendor outputs are replayed from fixtures.

```bash
pip install -e . pytest
python -m pytest tests -q
```

With `uv`, nothing needs to be installed in the environment:

```bash
uv run --no-project --with pytest --with psutil python -m pytest tests -q
```

One test is macOS-only (it compares the IOKit and `ioreg` paths on the same machine) and is skipped elsewhere.

## Hardware test scripts

`vendor_devices_tests/test_<vendor>.py` queries one backend on the current machine and prints the devices and the JSON. The scripts add the checkout to `sys.path`, so they run from a clone without installing:

```bash
python vendor_devices_tests/test_amd.py
```

If a backend marked untested in [Backends](backends.md) produces correct output on your hardware, open an issue with the script output, the vendor tool version and the device name, so the status can be updated.

## Adding a backend

A backend is one module with two classes and a parser. Use `xpustat/moorethreads.py` as the smallest example and `xpustat/apple.py` as an example with a fast path and a fallback.

1. **The stat class.** Subclass `DeviceMixin` from `xpustat/_device.py`, set `vendor` to the key and `_index_attr` to the name of the attribute holding the physical index (`_sub_index_attr` as well when devices are addressed as card:chip). Store at least `name`, `mem_used_mb` and `mem_total_mb` as integers and `processes` as a list of `ProcessInfo`. Fields the tool cannot report are `None`, not guessed. `to_dict()` lists the vendor fields plus `mem_free_mb`.
2. **The collection class.** Hold the list, implement `to_dict`, `__len__`, `__iter__`, `__getitem__` and `__repr__`, and a `new_query()` classmethod. `new_query` checks `shutil.which` for the tool, runs it with `subprocess.run(..., capture_output=True, timeout=...)`, catches `FileNotFoundError` and `subprocess.TimeoutExpired`, and returns an empty collection on any failure. If the tool needs several calls, submit them to a `ThreadPoolExecutor` so they run concurrently.
3. **The parser.** A module-level function that takes the tool's text and returns a list of stat objects, with no subprocess inside, so it can be tested from a fixture. Skip malformed entries rather than raising.
4. **Registration.** Add the module to `xpustat/xpu.py`: the import, the `AnyDeviceStat` union, a keyword argument with a `None` default on `XPUStatCollection.__init__`, `all_devices`, `to_dict` and the `_BACKENDS` list; then the exports in `xpustat/__init__.py`.
5. **CLI.** Add a colour and a label of at most six characters to `_VENDORS` in `xpustat/cli.py` and a branch in `_fields` returning the index string, name, architecture tag, memory, utilization, temperature, power and processes.
6. **Framework mapping.** Add the vendor to `TORCH_DEVICE_TYPE` and `VISIBLE_DEVICES_ENV` in `xpustat/_device.py`, with `None` and an empty tuple when there is no PyTorch backend or no visible-devices variable.
7. **Tests and docs.** A parser test from a captured output in `tests/test_vendors.py`, the `to_dict` key in `tests/test_xpu.py`, a script in `vendor_devices_tests/`, a row in the README table and a section in [Backends](backends.md) stating the command, the fields and what is not reported.

Keep the existing JSON keys and constructor signatures unchanged; new fields are added, not renamed.

## Plugin backends

A vendor can live outside this repository. Expose the collection class as an entry point:

```toml
[project.entry-points."xpustat.backends"]
acme = "xpustat_acme:AcmeStatCollection"
```

or register it at runtime with `xpustat.register_backend("acme", AcmeStatCollection)`. The key must be a lowercase identifier that is not a built-in vendor. The contract is the one in the steps above: `AcmeStatCollection([])` is empty, `AcmeStatCollection.new_query()` never raises, instances support `len()`, iteration, indexing and `to_dict()` with one list under a key of your choice, and each device subclasses `xpustat.DeviceMixin` with `vendor` and `_index_attr` set. Plugins get a generic CLI label (the first six letters of the key, upper case); add the key to `TORCH_DEVICE_TYPE`, `VISIBLE_DEVICES_ENV`, `RAY_RESOURCE` and `K8S_RESOURCE` from your package if the vendor has them. A plugin that fails to import is skipped; set `XPUSTAT_DEBUG=1` to see why.

## Conventions

- Vendor tools are the source of truth. When a tool does not report a value, the field is `None` or 0 and the documentation says so; the value is not estimated.
- Backends never raise out of `new_query`. A broken vendor tool on one machine must not hide the other vendors.
- Query cost matters because watch mode and the Prometheus exporter call `query_all` repeatedly. Prefer one tool invocation, run independent invocations concurrently, and prefer an in-process library over a subprocess when one exists.
