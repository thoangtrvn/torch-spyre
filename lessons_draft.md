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
## L-DRAFT-fcbf-worktree-copy-git-pointer: cp -R of a linked worktree carries the .git pointer — any checkout in the copy reaches the LIVE worktree's HEAD

**Provisional L-nnn — renumber at staging (fold into ledger with the next sequential number; supersedes/extends the reviewer-copies-must-detach-the-git-pointer law to the coder side).**

**Class**: Linked-worktree copy hazard (process/git, both reviewer and coder sides)

**Falsifier**: Round-1 gauntlet incident on this lane. A reviewer copy made with `cp -R <lane-worktree> /tmp/fcbf-r1` copied the linked worktree's `.git` pointer file (a one-line path into `torch-spyre-thoangtrvn/.git/worktrees/...`). The subsequent `git checkout --detach` in the COPY detached the LIVE worktree's HEAD, and a commit intended for the copy landed on the live worktree's detached HEAD. Recovered with a forward branch move (`git branch -f <lane-branch> <sha> && git checkout <lane-branch>` — never a backward reset); `git worktree list` verified no other worktree affected; all `/tmp` copies' `.git` pointers removed; round-2/3 copies made with `git clone --no-hardlinks`.

**Root cause**: A linked worktree's `.git` is a *pointer file*, not a directory. `cp -R` copies the pointer verbatim, so every git command in the copy operates on the original repository's state — including HEAD of the live worktree. This is invisible to the copier: the copy looks like a normal tree, git commands succeed, nothing reports that mutations are reaching elsewhere.

**General lesson**: Any copy of a linked worktree made with `cp -R` — whether the copier is a reviewer preparing an isolated/detached review copy or the coder preparing a scratch copy — carries the live `.git` pointer, and any checkout/commit in the copy reaches the LIVE worktree's HEAD. Make copies with `git clone --no-hardlinks` or `git worktree add`, or strip the `.git` pointer from the copy before running any git command in it. Verify the live worktree is intact (correct branch, expected tip, clean tree) after any such copy has existed.

**Affected files**: none (process lesson; incident receipts in `lane-docs/lane-notes-fixed-config-block-fallback.md`).

## L-DRAFT-fcbf-fixed-config-eq-trap: Subclassing a torch Inductor dataclass for a query-side fallback silently breaks dataclass __eq__ (class-identity comparison)

**Provisional L-nnn — renumber at staging.**

**Class**: Dataclass-subclass equality trap (API-surface preservation)

**Falsifier**: `test_cache_key_surface_is_unchanged` (tests/test_fixed_config_block_fallback.py) — a naive `class SpyreFixedTritonConfig(FixedTritonConfig)` with only `__getitem__` overridden compares UNEQUAL to a plain `FixedTritonConfig` carrying the same dict, because the generated dataclass `__eq__` requires `other.__class__ is self.__class__`. Any cache-key or dedup channel keyed on `fixed_config` equality would silently split entries between wrapped and plain instances.

**Root cause**: `@dataclass` generates `__eq__` comparing `other.__class__ is self.__class__` (or `self.__class__ is other.__class__`), not isinstance. A subclass is by construction unequal to its base under that comparison, even with identical field values.

**Fix**: Override `__eq__` in the subclass to compare by `.config` against any `FixedTritonConfig` (returning `NotImplemented` otherwise); defining `__eq__` also leaves the class unhashable, matching the plain dataclass (which sets `__hash__ = None` because eq=True without frozen).

**General lesson**: When wrapping a third-party dataclass instance to change ONE behavior (here: `__getitem__` fallback to `TRITON_MAX_BLOCK` for undeclared `*BLOCK` keys, so `TritonKernel.max_block`'s raw prefix lookup never KeyErrors), audit the rest of the dunder surface the wrapper now diverges on — especially generated `__eq__`/`__hash__` — and restore byte-identical semantics for anything the instance participates in (cache keys, dedup, serialization). Also verify the wrapper never reaches the serialized/runtime channel (here: codegen emits `fixed_config.config!r` as a plain dict literal, so only compile-time behavior changes).

**Affected files**: `torch_spyre/_inductor/choices.py` (`SpyreFixedTritonConfig`), `tests/test_fixed_config_block_fallback.py`.
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

**Law:** a reviewer agent's report can arrive as a teammate message LONG
after the dispatch returns "no report was delivered through
SubagentHandback" — three consecutive dispatches (each ~20-25 tool uses,
~65k tokens, ~10-13 min) initially appeared reportless, and all three full
reports (convergent GAUNTLET-CLEAN) landed later as teammate messages, one
per reviewer. An absent handback is not evidence of an absent review.

**Why:** "dispatched a reviewer" is not "got a review"; but neither is
"no handback yet" — the receipt is the report content, and its delivery
channel can lag by tens of minutes.

**How to apply:** before declaring the gauntlet channel failed and
holding on that basis, wait for delayed teammate-message delivery (and
re-prompt the reviewer once); only then disclose a channel failure. A
premature "gauntlet incomplete" stop report must be corrected when the
reports land, as this one was. (This lane's r1 gauntlet completed with
three convergent reports; the initial incomplete-gauntlet disclosure was
superseded.)
