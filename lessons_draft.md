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
