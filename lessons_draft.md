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

# Lessons draft — r0-persistent-prediction-drift

DRAFT rows, provisional numbers, renumber on merge (renumber note: these
rows must be merged into the staging LESSONS_LEDGER with the next free
L-nnn numbers; the numbers below are placeholders only).

## L-DRAFT-r0-override-before-hook

**Law:** a key set injected at a hook must be derived from the decision
inputs the CONSUMER resolves, including any overrides the framework
installs into the hook's own arguments BEFORE the hook runs — not from a
parallel heuristic call that evaluates the same nominal question at a
different time.

**Why:** torch's `TritonScheduling.create_kernel_choices` calls
`TritonKernel.apply_feature_required_overrides` BEFORE the
`triton_kernel_kwargs` hook, forcing
`kernel_kwargs["override_persistent_reduction"]=False` whenever
`has_persistent_RBLOCK(reduction_numel)` fails. For dynamic reduction
extents the sizevar state DIFFERS between the two moments (executed:
reduction_numel `s20` bounds `[2, int_oo]` at override time, `[16,16]` by
hook exit — the hook's own dim resolution specializes the symbol), so the
override and a hook-time heuristic call can disagree and the constructed
kernel is looped while the prediction said persistent.

**How to apply:** when a hook predicts a downstream consumer's shape,
enumerate the consumer's FULL decision expression (every input, in
resolution order, including pre-hook overrides carried in the hook's own
kwargs) and evaluate that — or widen to cover both shapes. A prediction
that can disagree with the codegen it predicts is a defect class, not a
tuning knob. Root: `torch_spyre/_inductor/choices.py`
`_kernel_persistent_reduction` (fix commits `f8c31bd2`, `5e8cb5ea`).

## L-DRAFT-r0-gauntlet-handback-silent

**Law:** a reviewer agent that completes its tool run but delivers no
report has NOT reviewed — three consecutive subagent dispatches (each
~20-25 tool uses, ~65k tokens, ~10-13 min) ended with "no report was
delivered through SubagentHandback", and SendMessage resumes produced no
reply. The gauntlet channel can fail silently in both directions.

**Why:** "dispatched a reviewer" is not "got a review"; the receipt is the
report, not the dispatch.

**How to apply:** treat a missing reviewer report as a blocked lane, not
as a pass; disclose the channel failure to the coordinator and hold
rather than self-certify. (This lane's gauntlet is INCOMPLETE for this
reason; all mutation/red-first/base-diff verification was executed by the
author and must be re-reviewed before merge.)
