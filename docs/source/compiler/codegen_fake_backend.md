# Codegen-only fake backend design

## Executive summary

Codegen-only mode now has a deliberately small C++ PrivateUse1 backend slice. It gives PyTorch enough real backend surface area to construct `spyre:0` tensors for compiler capture, while still refusing runtime initialization, streams, device discovery, DMA, and kernel launch.

This is not a simulator and not hardware evidence. It is a frontend/codegen harness whose job is to remove the local-machine blocker where `x.to("spyre")` failed before Inductor/Triton could expose useful compiler artifacts.

## Target problem

The previous Python-only codegen mode could register the `spyre` backend name and import compiler modules, but it could not safely materialize a Spyre tensor in ordinary Python:

```python
import torch
import torch_spyre

x = torch.empty((512, 128), dtype=torch.float16)
y = x.to("spyre")
```

That failed because PyTorch still needs a real PrivateUse1 backend implementation for device guard, allocation, storage, and dispatch plumbing. A Python shim can fake module attributes; it cannot make PyTorch believe the binary was linked with a PrivateUse1 device backend.

The new implementation targets this narrow requirement:

- allow `torch` / `torch_spyre` import after a codegen-only editable install;
- register PrivateUse1 hooks and a guard implementation;
- support local fake/codegen tensors on `spyre:0`;
- attach/query `SpyreTensorLayout` on those tensors;
- preserve public `torch.spyre.is_available() == False` and `torch.spyre.device_count() == 0` so no caller mistakes this for hardware;
- keep launch/runtime APIs fail-loud.

## Implemented architecture

### Build-time selection

When installed with:

```bash
TORCH_SPYRE_CODEGEN_ONLY=1 python -m pip install -e .
```

`setup.py` builds only:

```text
torch_spyre._codegen_backend
```

It does not link Sentient/Flex runtime libraries and does not build the full `_C` / `_hooks` runtime extensions. The persisted `torch.backends` entry point still points at:

```text
torch_spyre:_autoload_codegen_only
```

so the selected Python environment remains in codegen-only mode without requiring the environment variable on every later import.

### C++ extension responsibilities

`torch_spyre._codegen_backend` registers:

- `PrivateUse1HooksInterface`, enough for PyTorch to recognize the backend as present;
- `DeviceGuardImplInterface`, exposing a single synthetic codegen device, `spyre:0`;
- `aten::empty.memory_format` for PrivateUse1;
- `aten::empty_strided` for PrivateUse1;
- `aten::copy_` for a no-value-transfer CPU→Spyre materialization path;
- `aten::set_.source_Storage_storage_offset` for metadata plumbing;
- pybind exports for `DataFormats`, `SpyreTensorLayout`, layout getters/setters, and codegen allocation helpers.

The fake tensor is a real PrivateUse1 tensor from PyTorch's perspective, but its allocation is a host-side token. The storage byte size records the logical tensor footprint; the data pointer is not valid device memory and must not be used for execution.

### Python integration

`torch_spyre._C` remains the import compatibility surface in codegen-only mode. If `_codegen_backend` is available, `_C.py` aliases the compiled `DataFormats`, `SpyreTensorLayout`, `empty_with_layout`, `spyre_empty_with_layout`, `get_spyre_tensor_layout`, `set_spyre_tensor_layout`, `copy_tensor`, and device helpers. If the C++ shim is absent, `_C.py` falls back to the old pure-Python layout-only shim, but allocation helpers continue to fail loudly.

`torch_spyre._autoload()` imports `_codegen_backend` in codegen-only mode before registering the Python `torch.spyre` module. It then imports `_C`, which applies the existing Spyre tensor monkey patch after `_C` has finished binding the codegen backend symbols. This avoids the previous circular import trap.

The monkey patch now treats `x.to("spyre")` in codegen-only mode as compiler materialization: it builds a default `SpyreTensorLayout`, creates a fake Spyre tensor, and invokes the C++ no-op copy hook. No runtime initialization, DMA, or hardware launch occurs.

## Boundary conditions

The public availability contract intentionally remains negative:

```python
assert torch.spyre.is_available() is False
assert torch.spyre.device_count() == 0
```

The internal C++ guard reports one synthetic device only so PyTorch can allocate PrivateUse1 tensors locally. That split is intentional: compiler capture needs a backend-shaped tensor; application/runtime code must still see that no real hardware is available.

Runtime-only entry points still call `require_runtime(...)`, including:

- `start_runtime()`
- `launch_kernel()`
- `launch_kernel_from_bytes()`
- stream APIs
- device synchronization
- full-runtime memory helpers not backed by the codegen shim

## Options considered

### Option A: C++ codegen-only PrivateUse1 backend slice — selected

This is the least-bad local path. It implements the minimum PyTorch backend hooks necessary for `spyre:0` tensor construction while keeping runtime launch impossible.

Advantages:

- exercises PyTorch's real PrivateUse1 dispatch path rather than only Python side tables;
- removes the `PyTorch is not linked with support for spyre devices` class of blocker;
- preserves `SpyreTensorLayout` as tensor-owned metadata;
- does not require Sentient/Flex libraries on a compiler-only workstation;
- keeps the runtime availability boundary explicit.

Risks:

- any fake tensor can be mistaken for hardware readiness unless tests and docs keep the boundary sharp;
- PyTorch dispatch coverage is still minimal, so unsupported ops should fail rather than silently compute on CPU;
- this still does not force Triton TensorDescriptor emission by itself.

### Option B: Python monkeypatch only — rejected as primary design

A Python-only approach can intercept selected calls like `Tensor.to` and attach side metadata. That is useful glue, but it cannot satisfy PyTorch's lower-level PrivateUse1 hooks, device guard, or storage invariants.

Failure mode: compiler paths that consult the dispatcher or device guard still fail before useful IR is captured, or worse, continue as CPU tensors with side-table metadata and produce invalid evidence.

### Option C: Meta/FakeTensor-only capture — rejected as sufficient evidence

Meta tensors and `FakeTensorMode` are attractive for shape-only compiler testing, but they do not represent the Spyre backend layout contract. Prior attempts reached generated Triton source, but produced raw-pointer `tl.load` / `tl.store` artifacts instead of `tt.make_tensor_descriptor`, `tt.descriptor_load`, and `tt.descriptor_store`.

Failure mode: the capture proves only generic Inductor lowering, not the Spyre TensorDescriptor path.

### Option D: Full runtime with no-op launcher — rejected for local codegen-only mode

Running the full backend with a no-op launcher can get farther through the stack, but it still depends on runtime libraries and risks masking hardware/runtime assumptions. It is a useful separate diagnostic on a properly provisioned machine, not the right default for a laptop/compiler-only editable install.

## What this does not prove

This backend does not prove:

- TensorDescriptor TTIR emission;
- descriptor stride/layout compatibility with Sentient packet contracts;
- byte identity against tracked `.bin` goldens;
- runtime ABI correctness;
- DMA correctness;
- hardware KAT correctness.

The next TensorDescriptor milestone still needs a real captured fixture such as:

```text
codegen/tests/fixtures/ttir_descriptor/real_inductor_pointwise_identity_512x128.ttir
```

and then a proof ladder from descriptor structure to pointwise schedule to packet/byte comparison. The fake backend merely gets the local Python frontend far enough to attempt that capture.

## Acceptance tests

Minimum local acceptance checks:

```python
import torch
import torch_spyre

assert torch._C._get_privateuse1_backend_name() == "spyre"
assert torch_spyre.codegen_only_available() is True
assert torch.spyre.codegen_only_available() is True
assert torch.spyre.is_available() is False
assert torch.spyre.device_count() == 0

x = torch.empty((512, 128), dtype=torch.float16)
y = x.to("spyre")
assert y.device.type == "spyre"
assert y.device.index == 0
assert tuple(y.shape) == (512, 128)
assert tuple(y.stride()) == (128, 1)

layout = y.device_tensor_layout()
assert layout.device_size == [512, 2, 64]
assert layout.stride_map == [128, 64, 1]
assert layout.elems_per_stick() == 64
```

Runtime launch must still fail:

```python
import torch_spyre._C as _C
_C.launch_kernel_from_bytes(b"", [])  # raises SpyreRuntimeUnavailableError
```

## Audit posture

Against CUDA/ROCm/oneAPI, this is infrastructure plumbing, not a competitive feature. NVIDIA does not win because `cuda:0` tensors can be constructed; NVIDIA wins because the backend, compiler, runtime, profiler, libraries, and developer contracts all agree. This change only closes one humiliating local compiler-host gap: PyTorch can now hold a backend-shaped Spyre tensor without a device. The hard part remains proving that the compiler then emits TensorDescriptor IR and that the resulting packets match hardware contracts.
