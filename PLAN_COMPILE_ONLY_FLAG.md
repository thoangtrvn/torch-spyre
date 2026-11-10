# TORCH_SPYRE_COMPILE_ONLY / TORCH_SPYRE_CODEGEN_ONLY — decision & deferred revert plan

Status: **RETIRED (2026-11-10)** — `TORCH_SPYRE_COMPILE_ONLY` is dead. The
link-deferred build flavor, the `_build_runtime_stub_dylib()` generator, the
loader-side `_load_runtime_stubs()` preload, and `build-compile-only.sh` are
removed from the tree. Kernel-generation-without-a-device environments use the
codegen-only build (`TORCH_SPYRE_CODEGEN_ONLY`, `_codegen_backend.so` +
pure-Python `_C`). Everything below is historical record of the decision; the
"deferred revert" it describes has now been executed.

Historical status: deferred — no removal then. Re-evaluate trigger was the
deeptools removal landing in `csrc/`.

## Context

- `TORCH_SPYRE_CODEGEN_ONLY` (existing, shipped): builds the tiny self-contained
  `torch_spyre/_codegen_backend.so` (fake PrivateUse1 backend, torch/ATen headers
  only). Serves the user-facing "compiler stack without a device" need on any
  platform. Docs + ~10 test files consume it. **Meaning stays untouched.**
- `TORCH_SPYRE_COMPILE_ONLY` (new-runtime, commits c6452434 + e9e2c9c7): builds the
  full real `_C.so`/`_hooks.so` from all of `csrc/` against flex/senlib/deeptools
  headers, link-deferred (`-undefined dynamic_lookup`) with a generated
  `_runtime_stubs.dylib` satisfying the unresolved runtime symbols
  (`ret`-stubs / zeroed vtables, real `flex::getNumDevices -> 0` and real
  `deeptools::FloatToFp16Bin`). macOS-local only.

## Why the flags do NOT conflict (validated)

The two flags are orthogonal knobs:

| CODEGEN_ONLY | COMPILE_ONLY | Result |
|---|---|---|
| 0 | 0 | Normal full build, links `sendnn`/`sendnn_interface`/`flex`. |
| 0 | 1 | Full-source `_C.so`+`_hooks.so`, link-deferred + stub dylib. |
| 1 | 0 | Tiny `_codegen_backend.so` only. |
| 1 | 1 | CODEGEN_ONLY wins on artifact; COMPILE_ONLY degenerates to junk flags; stub generator no-ops (no `_C.so` in `build/lib.*/`). Not broken, just meaningless. |

Setting one at a time works as designed. No merge/rename needed.

## Why removal is deferred, not done now

1. **No necessity:** orthogonality means nothing breaks by coexisting; no CI,
   docs, or tests reference COMPILE_ONLY (only the untracked
   `build-compile-only.sh` and local use).
2. **Still in active use:** it is the only way to exercise the real `csrc/`
   code (real `SpyreTensorLayout`, real DL16 `encode_constant`, T-DC-17 seam)
   on macOS without hardware (e.g. the `encode_constant(1.0) == 0x3E00` oracle).
3. **Its removal trigger hasn't landed:** `torch_spyre/csrc/module.cpp:124`
   still calls `deeptools::FloatToFp16Bin` in the normal build path. The stub
   machinery's flagship real implementation exists *because* deeptools isn't
   linkable on macOS. Remove the flag only together with the deeptools removal
   from `csrc/`.

## Immediate (cheap, do anytime)

- **Stub-preload guard:** `torch_spyre/__init__.py`'s
  `_SpyreImpl._load_runtime_stubs()` preloads `_runtime_stubs.dylib` based
  purely on file presence, regardless of which mode was built (cross-mode
  contamination hazard, esp. with a stale `_codegen_backend.so` sitting in
  `torch_spyre/` from Oct 19). Make the preload conditional on the extensions
  actually being link-deferred (e.g. an explicit marker emitted by the
  compile-only build, not file presence).

## Deferred revert (execute when deeptools removal lands in csrc)

Scope (surgical removal — do NOT `git revert`; e9e2c9c7 was a re-landing and
later commits touch the same regions):

1. `setup.py`:
   - `COMPILE_ONLY_BUILD` + the `if COMPILE_ONLY_BUILD: LIBRARIES = []` branch
     (~lines 101-104)
   - the `EXTRA_CXX_FLAGS` compile-only block (~lines 151-166)
   - `_build_runtime_stub_dylib()` (~lines 189-323)
   - the `if COMPILE_ONLY_BUILD: _build_runtime_stub_dylib()` call in
     `PermanentBuildExtension.run()` (~lines 440-441)
2. `torch_spyre/__init__.py`: `_load_runtime_stubs()` (~lines 85-100), its
   module-level call + comment (~lines 200-203), the `else: pass` branch under
   `device_count() == 0` (~lines 114-119).
3. Delete untracked: `build-compile-only.sh`, `torch_spyre/_runtime_stubs.dylib`,
   `_runtime_stubs.dylib.disabled-gate-env-restore`, stale
   `torch_spyre/_codegen_backend.so` (Oct 19).
4. `build-macos.sh:6`: drop the stale comment referencing COMPILE_ONLY.
5. Keep unchanged: fc5978e3 (T-DC-17 guard), 835e8e0e (its test), ab50db54,
   and the entire TORCH_SPYRE_CODEGEN_ONLY path.

## Unrelated finding (environment, not repo)

`import torch` fails with `AttributeError: module 'triton' has no attribute
'language'` **only when Python is launched from `spyre-thirdparty-repos/`**:
the upstream `triton/` source checkout at that root (repo root, no
`__init__.py`) becomes a namespace package on `sys.path[0]` and shadows the
editable-installed `triton-spyre/python/triton`. Fix: launch from any
subdirectory, or `PYTHONSAFEPATH=1` / `python -P`.
