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