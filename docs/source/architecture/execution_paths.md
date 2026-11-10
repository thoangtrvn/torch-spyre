# Execution Paths and Eager Tensor Semantics

This document states two design invariants of torch-spyre that are easy to
re-derive incorrectly from scattered sources:

1. **Every execution path runs a compiled Sentient program** — the device has
   no interpreter and no per-op kernel library in the CUDA sense. What differs
   between "eager" and "compiled" is *who chooses the program*, not whether
   compilation happened.
2. **Eager-time tensor existence is orthogonal to op execution** — in
   codegen-only mode, tensors must materialize with real metadata and no data,
   and the policy for value-consuming ops (`.item()`) is fail-loud by default
   with an explicit escape hatch.

## The execution-path invariant

There is no un-compiled execution on the spyre device. Three routes exist:

| Route | Program selection | Where |
|---|---|---|
| User-driven `torch.compile` | Inductor fusion across the captured graph | `torch_spyre/_inductor/` backend |
| Eager dispatch | `compile_once(op)(dispatch_to_torch_compile)` — a **cached per-op torch.compile** invoked from a registered PrivateUse1 kernel | `torch_spyre/ops/eager.py` |
| Specialized ops | Hand-written device programs (e.g. softmax, embedding) | `torch_spyre/ops/softmax.py`, `torch_spyre/ops/embedding.py` |

The eager route answers a frequently-asked design question — "should
torch-spyre force all eager ops through torch.compile?" — with *it already
does*: `register_torch_compile_kernel` wraps each supported ATen overload so
that an eager call dispatches to a single-op graph compiled once and cached
(`compile_once`). What torch-spyre deliberately does NOT do is hide that
boundary behind an outer auto-compile wrapper around arbitrary user code:
host-side control flow (`.item()`-driven branches, printing, data-dependent
shapes) is not device execution and must not be silently captured.

The boundary is fail-loud by construction, not by policy check: compute ops
without a registration raise on dispatch (no kernel exists), and unsupported
ops fall back to eager CPU execution with a warning
(`torch_spyre/ops/fallbacks.py`). There is no silent wrong-answer path from
un-compiled device math.

## Eager tensor semantics in codegen-only mode

`TORCH_SPYRE_CODEGEN_ONLY=1` runs without the runtime extension. The device
still must produce tensors at eager time — `torch.empty(..., device='spyre')`,
`.to('spyre')`, model construction, weight placement — because compilation
itself needs input tensors with real shape/stride/layout metadata. This is a
different stage from PyTorch's own `FakeTensorMode`: fake mode exists only
while something is tracing; our materialization exists so that *eager* code
before any trace can run. The two compose — dynamo creates its own fake mode
during capture on top of these tensors.

Design choices in this mode (`torch_spyre/__init__.py`):

- **Allocation** (`empty`/`empty_strided`/factories): fake DataPtr on the
  PrivateUse1 device; no data fidelity is possible or required. `copy_` is a
  checked no-op, consistent with the fake-pointer semantics.
- **Views**: strides are computed by performing the view on a `meta`-device
  tensor with the same size/stride, then re-wrapping the result — the stride
  math is PyTorch's, not re-implemented. `as_strided` with a nonzero
  `storage_offset` raises: there is no backing storage to offset into, and
  pretending otherwise would fabricate a layout.
- **`memory_format`**: `ones`/`new_ones`/factories accept the kwarg and
  assert contiguous-only with a loud message. (Note: stock torch's
  `torch._refs.ones` does not accept `memory_format` at all — an upstream
  limit for composite decompositions, upstream of this layer.)
- **`.item()` / `_local_scalar_dense`**: the fake tensor has no data, so any
  scalar answer is a lie. Policy: return `0` with a warning deduplicated by
  the calling tensor's `(size, dtype)` (keeps incidental uses — logging
  prints — from killing a capture without per-iteration warning spam), OR
  raise `SpyreRuntimeUnavailableError` when `TORCH_SPYRE_STRICT_ITEM` is set
  to a truthy value.

### Why the strict-item escape hatch exists

`TORCH_SPYRE_STRICT_ITEM` is a *policy* knob, not a mode detector — codegen-only
mode is already detected by `codegen_only_available()`, and the fake
`_local_scalar_dense` kernel is only registered in that mode. The knob exists
for **capture-fidelity probing**: run a compile/capture with it set and a clean
exit proves the captured graph is data-independent — no Python-side branch
consumed a tensor value (a fake `0` can otherwise steer Dynamo down a path
real data might never take). This mirrors `FakeTensorMode`'s own posture,
where scalar extraction fails loudly during tracing.

The env-var form (rather than a Python API) was chosen because the primary
consumer is probe scripts executed as subprocesses, where behavior must flip
without editing the script under test.

## Lowering adapters and the Triton path

`enable_spyre_lowerings(*, triton_path=True)` gates the non-Triton clamp
adapters: their inner lowering calls `V.ops.clamp`, a handler that only the
(non-Triton) ops class defines. Installing them on the Triton path (the only
registered scheduling backend) made any clamp-bearing graph die at Triton
codegen with `AttributeError`. The default is therefore `triton_path=True`;
the adapters install only when `triton_path=False`.
