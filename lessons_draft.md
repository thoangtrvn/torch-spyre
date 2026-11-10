# Lessons Ledger - codegen-loader-import-order

## L-DRAFT-loader-import-order: Module-scope import of torch in backend shim triggers circular import via backend autoload

**Class**: Import-order circular dependency in codegen-only shim

**Falsifier**: `test_codegen_only_C_first_import_before_torch` — imports `torch_spyre._C` before `torch` in a fresh subprocess; fails on base (circular import), passes at tip.

**Root cause**: `_C.py` line 109 (codegen-only branch) executed `import torch` at module scope. In a fresh interpreter importing `torch_spyre._C` first:
1. `_C.py:109` runs `import torch`
2. `torch/__init__.py:_import_device_backends()` invokes the `torch.backends` entrypoint
3. Entrypoint calls `torch_spyre.__init__._autoload_codegen_only()`
4. Which calls `_autoload()` → `ts_autoload()` → `torch_spyre._inductor._autoload()`
5. Which imports `choices.py:24` → `from torch_spyre._C import get_elem_in_stick`
6. But `torch_spyre._C` is still frozen at its own `import torch` → `ImportError: partially initialized module`

**Fix**: Move `import torch` and the dtype mapping table into a lazy accessor (`_get_torch_dtype_map()` + `_get_dtype_map()`). The public surface (`__all__`, behavior when torch is already imported) is unchanged.

**General lesson**: Backend shims that may be imported before torch must not execute `import torch` at module scope — it triggers torch's backend autoload recursively. All torch-dependent initialization must be deferred to first-use functions.

**Affected files**: `torch_spyre/_C.py` (codegen-only branch), `tests/test_codegen_only.py` (new regression test)
---

# Lessons Ledger - private-use1-aten-surface

## L-DRAFT-pu1-shadow-device-options: Creating a "shadow CPU tensor" with src.options() from a device tensor silently creates ANOTHER device tensor

**Class**: Silent-wrong channel construction; secondary destructor deadlock

**Falsifier**: `tests/test_codegen_fold_surface.py::test_copy_extraction_and_clone_roundtrip` — spyre→spyre copy_ then clone/`c[1,1].item()` hangs (mutex deadlock) at the buggy commit; passes at tip.

**Root cause**: In the fold-surface `copy_` (spyre→spyre branch), the host-side shadow was allocated with `at::empty_strided(..., src.options())` where `src` is a SPYRE tensor — `options()` carries `device=spyre`, so the "shadow" dispatched back to our own `empty_strided` and was itself a fake-storage spyre tensor. Storing it in the token→shadow registry meant `delete_fake_allocation` → registry erase → destroying that spyre tensor → its fake StorageImpl destructor → `delete_fake_allocation` again → re-locking the already-held non-recursive registry mutex → same-thread deadlock (no diagnostic, pure hang).

**Fix**: Force `.device(c10::kCPU)` on every shadow allocation (fill_ already did), plus a defensive `TORCH_CHECK(!shadow.is_privateuseone())` in `set_shadow` so no spyre tensor can ever enter the registry.

**General lesson**: When building a host-side sidecar for device tensors, ALWAYS pin the device explicitly on the sidecar allocation — `Tensor.options()` inherited from a device tensor silently routes the allocation back through your own backend. Sidecar registries keyed by storage tokens must also guard against their VALUES holding tensors from the same backend: the token deleter runs inside registry mutation, so a same-backend value re-enters the deleter under the held lock (deadlock with no traceback — repro only via native stack sample, `sample <pid>`).

**Affected files**: `torch_spyre/csrc/codegen_backend.cpp` (spyre_codegen_copy_ SpyreToSpyre branch, set_shadow guard)

## L-DRAFT-pu1-shadow-offset-coordinates: Sidecar views must be rebased to the spyre tensor's storage_offset coordinate system

**Class**: Silent-wrong values (wrong element read, off by the view's offset)

**Falsifier**: `c[1,1].item()` on a cloned 2×2 spyre tensor returned `c[0,1]`'s value (1.0 instead of 3.0) at the buggy commit; exact-value round-trip passes at tip.

**Root cause**: `view`/`as_strided` produced a fresh-storage spyre tensor (storage_offset 0) but stored the CPU shadow as a literal `shadow->view(size)` / `shadow->as_strided(...)` — which KEEPS the CPU view's storage_offset. Downstream composites (e.g. `select` → `as_strided`) compute the new offset from the SPYRE tensor's `storage_offset()` (0) plus the index delta, then apply that offset to a shadow indexed from a DIFFERENT origin — every chained select reads the wrong element by exactly the accumulated base offset.

**Fix**: Materialize the shadow rebased to offset 0 with identical sizes/strides (`empty_strided(...).copy_(view)`), maintaining the invariant `spyre.storage_offset() == shadow.storage_offset()` (both 0) for every registry entry.

**General lesson**: A host-side sidecar for a custom-backend tensor is a COORDINATE SYSTEM, not just a value bag: every layout property the dispatcher's composites read off the backend tensor (storage_offset, sizes, strides) must match the sidecar exactly, or offset arithmetic computed on one is applied to the other. When the backend op gives the output a fresh storage, the sidecar must be materialized to match — a lazy CPU view silently diverges on the first chained op.
