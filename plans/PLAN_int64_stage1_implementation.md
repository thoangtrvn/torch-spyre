# PLAN — Stage 1 implementation (int64 three-dtype, AIU 1.0)

Status: **PLAN ONLY. No code until reviewed.** Companion to
`plans/PLAN_int64_three_dtype.md` (the design plan); this file is the work plan for
Stage 1 under **Ruling 3 (USER, 2026-11-16): Stage 1 first, then Branch E.**

Lane `torchspyre-int64-graph-policy` (repo `torch-spyre-thoangtrvn`, base `b413c3e3`).
CSRC change → **rebuild on the box under the hardware claim protocol**.

**Lane scope at this revision (after review round 1):**
- **Slice 1 — split (do not conflate): S1.a sendnn removal is LIVE and implementable;
  S1.b one-owner reconciliation is BLOCKED** on P-a/P-b (§1). The carve recommendation
  concerns **S1.b only**; dispatch is USER-only and this lane creates no branch.
- **Slice 2 — live** (owner table is documentation + tests; no build dependency).
- **Slice 3 — live scope, diagnosis NOT established**: no mechanism, no fix. 3a
  classifies by **exception class first** (§3a); the named candidate is the
  **storage-geometry split**, in which case T-DC-121 and Slice 1 are the same defect
  and Slice 3 **dissolves into the (blocked) Slice 1**.
- **Slice 4 — live scope and now load-bearing**: its witness chooses Slice 3's
  mechanism, and the witness must drive the **staging** branch (§4).
- **E entry conditions — planning artefacts only** (§6).

Every `file:line` below was read in this lane's worktree at plan time.

---

## 0. Why Stage 1 is first

Stage 1 is mandatory under **every** route: it is two live defects, not a
pre-requisite that any branch can skip.

- **C1 — `get_device_dtype` answers two ways by build mode.** Codegen-only Python
  returns `IEEE_INT64` (16 elems/stick); the C++ runtime returns the Spyre-side
  `IEEE_INT32` (32 elems/stick). Same question, different answer, selected by build.
- **T-DC-121 — D2H staging overrun.** The staging buffer is sized from *logical*
  element counts with the logical dtype, while the device storage is allocated at the
  *device* width (`get_device_size_in_bytes`). For an int64 tensor that is 8B/element
  of host staging against 4B/element of device storage.

A route that computes exact integers correctly but sizes a readback at the wrong
width is not correct. Both defects are independent of which branch (A/E/D) computes
the values.

---

## 1. Slice 1 — two independent items

> **Status split (do not conflate them):**
> - **S1.a — sendnn removal: LIVE, implementable.** Pure type swap + libs-list edit, full
>   build only; can proceed now, independent of the carve.
> - **S1.b — two-answer one-owner reconciliation: BLOCKED** behind the P-a/P-b
>   precondition. This is the piece that needs the USER's carve call, and it is what
>   gates T-DC-121 *if* the geometry-split candidate holds (§3a).
>
> Sequence: **S1.a now, as its own commit; S1.b after the precondition is chosen.**
> Different review surfaces — never one commit.

**Defect (verified):**

| side | site | int64 answer | elems/stick |
|---|---|---|---|
| C++ runtime | `torch_spyre/csrc/module.cpp:190-195` (`get_elem_in_stick`), `:198-202` (`get_device_dtype`) — both `stringToDTDataFormatPair(str_type).second` | `IEEE_INT32` (`csrc/types_mapping.h:71`, `{"int64", {IEEE_INT64, IEEE_INT32}}`) | **32** |
| **codegen backend (LIVE)** | `torch_spyre/csrc/codegen_backend.cpp:145-167` `spyre::codegen::get_device_dtype` — **its own switch**, `Long → DataFormats::IEEE_INT64`, with its own `enum class DataFormats` at `:52-72`; bound to Python at `:935` | `IEEE_INT64` | **16** (`elems_per_stick_for_format`) |
| codegen-only Python fallback (**dead in a real install**) | `torch_spyre/_C.py:137` → `_get_dtype_map()[dtype]`, map at `:116-121` | `IEEE_INT64` | **16** (`_C.py:99-100`) |

**Three answer-sites, not two — and the one this slice first proposed to edit is not
the live one.** In a real codegen-only build `_C.py:137` is the *no-backend fallback*;
the accessor on the path is `codegen_backend.cpp:145-167`. "Delete the `_C.py` map" is
therefore a **no-op against the live path**.

### S1.b — one-owner reconciliation: BLOCKED (P-a / P-b)

**BLOCKED — precondition before the reconciliation (verified).** Routing
`codegen_backend.cpp` through the C++ pair cannot be done as written:
`csrc/types_mapping.h:24` does `#include <sendnn/tensor/sendatatype.hpp>`, and
`setup.py` adds `SEN_COMMON_HEADERS` (the dir holding it) **only inside
`if not CODEGEN_ONLY:`** (`setup.py:98` guards the block; `INCLUDE_DIRS +=
[os.environ["SEN_COMMON_HEADERS"]]` is inside it). In the codegen-only build that
header is off the include path, so `codegen_backend.cpp` **cannot** include
`types_mapping.h`.

One owner therefore requires **one of two preconditions**, and the plan must not pick
silently:
- **(P-a) Extract a `sendnn`-free header** holding the logical→device pair table and a
  shared `DataFormats` enum, included by both `codegen_backend.cpp` and the runtime.
  Preferred (no build-surface change), but it is a refactor with its own review.
- **(P-b) Add `SEN_COMMON_HEADERS` to the codegen-only include set** in `setup.py`.
  Smaller diff, larger blast radius: it changes what the codegen-only build can pull
  in, and that is a build-system change the USER should see.

**S1.b is BLOCKED until one precondition is chosen.** No worktree or branch is
created for it (dispatch is USER-only); the carve-out recommendation is with the USER.

**After the precondition, the change is:** make the extracted/shared pair table the
single owner of the physical/compute format, and have every accessor — runtime,
`codegen_backend.cpp`, and `_C.py` if it survives — answer from it. The accessors must
not carry independent maps/switches.

### S1.a — sendnn removal (USER-directed; **LIVE, implementable**)

The USER has directed that `sendnn` be removed, adopting the necessary piece from
upstream. It lands here because `types_mapping.h` is already on the int64 critical
path. **Scoped against the canonical tree** (`torch-spyre-thoangtrvn` @ `new-runtime`
`b413c3e3`) **and upstream** (`torch-spyre-upstream` @ `main` `52aeb859`,
sendnn-free); re-verified in this lane:

**Signal (verified):** sendnn appears in exactly two places —
`torch_spyre/csrc/types_mapping.h:24` (`#include <sendnn/tensor/sendatatype.hpp>`) and
`setup.py:96` (`LIBRARIES = ["sendnn", "sendnn_interface", "flex"]`). The enum appears
**138 tokens on 136 lines** (lines 110 and 117 carry two each: both are
`std::pair<sendnn::sen_datatype_enum, sendnn::sen_datatype_enum>`), and **all 138 are
inside one function**, `stringToSenDatatypePair` (`types_mapping.h:110-348`). No other
`sendnn::` symbol is used, so it is a pure *type* dependency and `LIBRARIES` drops to
`["flex"]` with no unresolved-symbol risk.

**This is a DELETION, not a substitution — verified, and it simplifies the brief.**
- `stringToSenDatatypePair` has **zero callers** in the tree (repo-wide grep: the
  definition at `:111` and stale `.worktrees/` copies only). Everything calls
  `stringToDTDataFormatPair` (`:58`), which is **already `DataFormats`-based and
  sendnn-free** and already used by `module.cpp:194,201`, `spyre_tensor_impl.cpp:143`
  and `spyre_mem.cpp:366`.
- So the change is: **delete `types_mapping.h:110-348` and the `:24` include; change
  nothing in `stringToDTDataFormatPair`.** No member renaming, no adopted table.

**Why NOT adopt upstream's `stringToDTDataFormatPair` body (the brief's step 3)** —
two independent reasons, both verified:
1. **It would not compile here.** Upstream maps `uint8 → SNUINT8`-style
   `DataFormats::SENUINT8`, and the installed `deeptools/util/sendefs.h:30-54`
   `enum class DataFormats` has **no `SENUINT8`** (it jumps `SENUINT32`, `SENUINT2`).
2. **Even if it compiled it would change behaviour silently.** The only content
   difference between the local and upstream tables is exactly that row: local
   `uint8 → {SENUINT32, SENUINT32}`, upstream `uint8 → {SENUINT8, SENUINT32}`. Adopting
   upstream would alter the host-side uint8 format — a change with nothing to do with
   removing sendnn, in a lane whose subject is int64.

**The dropped-dtype risk therefore evaporates.** `float64`/`float128`/`float256`/
`decimal*`/`string` exist only in the *dead* sendnn function; deleting it removes them
from the header, and nothing that reaches `stringToDTDataFormatPair` loses a mapping.
Keep the check anyway as a bar: confirm no caller passes one of those spellings
(`torchScalarToString` maps `kDouble → "float64"`, so a caller could in principle ask;
the live table has no `float64` row and today's code already behaves that way, so this
bar guards against *introducing* a reachable path, not against S1.a).

**Cross-check that does still matter:** the two Stage-1 rows must be unchanged after
the deletion — `"int32" → {IEEE_INT32, IEEE_INT32}`, `"int64" → {IEEE_INT64,
IEEE_INT32}` (`types_mapping.h:71` region). Deletion touches neither, and the int64 test
in §1's test list pins it.

**Relation to S1.b — none, and that is the correction.** This removal is **standalone
and unblocked**: it touches only the **full build** (`module.cpp` → `types_mapping.h`);
the codegen-only build never includes `types_mapping.h`, so it is unaffected, and no
P-a/P-b decision is needed. The full build already has `DataFormats` via
`types_mapping.h:22`'s `<util/sendefs.h>`.

It is **not** the same thing as P-a for the reconciliation: making `types_mapping.h`
sendnn-free does not by itself let `codegen_backend.cpp` include it, because the
codegen-only include path would still need `<module.h>` and `<util/sendefs.h>`.
**S1.b's precondition is a separate decision and stays open.** (If, once S1.a lands,
someone wants to test whether the reconciliation became easier, that check is: does
the codegen-only build resolve both headers? — but it is not required by S1.a.)

**Codegen-only build:** unaffected and must stay green — `codegen_backend.cpp` does not
include `types_mapping.h` and keeps its own local `enum class DataFormats` (`:52-72`).
Removing sendnn does **not** fix the two-answer split; that remains S1's separate work.

**Verification bars:** (a) full build compiles with `LIBRARIES=["flex"]` and no sendnn
header on the include path; (b) int64 still maps to `{IEEE_INT64, IEEE_INT32}` with
16/32 elem-per-stick semantics preserved; (c) the codegen-only build compiles unchanged;
(d) `grep -rn sendnn torch_spyre/csrc setup.py` is empty; **(e) `stringToDTDataFormatPair`
is byte-identical before and after** — the deletion must not touch the live table, and a
diff of that function's body is the mechanical proof; **(f) `uint8` still maps to
`{SENUINT32, SENUINT32}`** (the local row that upstream would have changed).

**Two commits, not one:** sendnn removal and the two-answer fix have different review
surfaces (build/type vocabulary vs accessor semantics) and must land separately.

**Do not** change the *logical* dtype: `torch.int64` stays int64 at the eager
`.to("spyre")` boundary (Ruling 2; `tests/test_spyre.py:236`).

**Blast radius before the change:** enumerate every caller of `get_device_dtype` and
`get_elem_in_stick` (host Python and C++) and list them in the commit. This is a
behaviour change to a shared accessor: each caller that sizes an allocation from it
is a latent half-size bug today and a changed number tomorrow. The list is produced
by search, not by memory, and an empty list means the accessor is dead (say so).
**Already known to be on the list: `torch_spyre/_inductor/choices.py:333` —
`elements_per_stick = get_elem_in_stick(self._kernel_dtype(features))` sizes the tiling
blocks, so an int64 kernel currently tiles at 16 elem/stick instead of 32 (verified).**

**Tests (red-first), including the pin that must flip in the SAME commit:**
- `test_device_dtype_one_answer`: `get_device_dtype(torch.int64) == IEEE_INT32` and
  `get_elem_in_stick(torch.int64) == 32` in **both** build modes. Fails today in
  codegen-only (returns `IEEE_INT64` / 16) — that failure is the proof of C1.
- **`tests/test_choices_eps_dtype.py:49` pins `(torch.int64, 16)`** (verified). The
  correct fix makes it 32, so it turns red. It must be flipped `16 → 32` **in the same
  commit** that fixes the accessor (same-commit red-first), not surface at gate time.
- `test_device_dtype_matches_cpp_pair`: for every dtype in
  `torchScalarToString`/`stringToDTDataFormatPair`, every accessor answer equals the
  shared pair's `.second`. This is the anti-drift guard, not just an int64 special
  case.

---

## 2. Slice 2 — the owner table (the acceptance artefact)

Stage 1 is accepted on this table, not on a routing-key fix. Each row has **one**
owner; a row that consults two is the defect class.

| operation | must consult | correct value for an int64-logical tensor | defect if it consults the wrong one |
|---|---|---|---|
| stride (torch API) | `logical_dtype` | int64 elements | — |
| stride → bytes (DMA) | `storage_dtype` | ×4 | ×8 (logical itemsize) ⇒ over-stride / OOB |
| storage size | `storage_dtype` | `numel × 4` | logical-sized ⇒ `setStorage` / T-DC-121 |
| copy H2D / D2H | the CPU/Spyre **pair** (`types_mapping.h:59-61,71`) via `generate_dci` | int64 ↔ INT32 | width mismatch if storage is not INT32-width |
| view / `as_strided(dtype)` | `storage_dtype` | reinterpret at 4B | logical-int64 view mis-strides 4B storage (#4329 `storage_offset=1` clone class) |
| host transfer / readback | `storage_dtype` / `compute_dtype` | size from INT32 (32/stick) | sizing from `get_elem_in_stick(int64)` ⇒ 16 (codegen-only) vs 32 (device) |

**Deliverable:** this table committed with, per row, the `file:line` of the call site
that implements it and the test that pins it. A row whose call site cannot be named
is an open defect, not a complete row.

**How this table fails, stated so the reviewer can check it:**
- **Column headings are not a population.** A row filled with "storage_dtype" but no
  `file:line` is a plan, not a Stage-1 artefact. Every row needs a real call site.
- **`compute_dtype` may have no caller at all.** It is the format the kernel executes
  in; on the host side the readback might legitimately consult only
  `storage_dtype`. If the `compute_dtype` column comes back empty, that is a
  **finding to record** (which accessor, if any, answers the compute question), not a
  row to fill by analogy.
- **The set of callers of `get_device_dtype` / `get_elem_in_stick` must be an
  enumeration, not an example.** Slice 1 changes what those functions answer, so
  every caller's sizing decision changes with it; the enumeration is the blast
  radius and is produced by search, then listed.

**Tests:** one per row (six). The view/`as_strided` and host-transfer rows are the
ones that currently fail or are unexercised; the stride and storage-size rows are the
T-DC-121 neighbourhood.

---

## 3. Slice 3 — T-DC-121: **diagnosis NOT established; do not implement a sizing change**

### 3.0 What the code says, and why the previous diagnosis is refuted

The previous version of this slice asserted a 2× overrun: "`[1,512]` int64 needs
4096 B of host staging from 2048 B of device storage", with the fix "size staging to
device bytes". **That diagnosis is refuted by the code**, for three reasons read at
this tip:

1. **The D2H copy widens int32→int64 on the way out.**
   `generate_dci` (`spyre_mem.cpp:362-372`) sets, for `host2device == false`,
   `dataformat_src_ = dtype_dev` (**INT32** for an int64 tensor,
   `types_mapping.h:71`) and `dataformat_dst_ = dtype_cpu` (**INT64**).
2. **The converter's *output* is int64 elements.**
   `spyre_mem.cpp:418-419`: `input_shape_ = host2device ? cpu_shape : dev_shape`,
   `output_shape_ = host2device ? dev_shape : cpu_shape` — so a D2H emits `numel`
   int64 elements to the host.
3. **The staging buffer is sized for exactly that output.**
   `cpu_alloc = at::empty(dma_sizes, dst.options())` (`:601`) is `dma_numel`
   elements × **int64** itemsize = `numel × 8` = 4096 B for `[1,512]` — the widened
   output size. **It is correctly sized.**

So 2048 B is the correct device **read** size and 4096 B is the correct host **write**
size. They are not two claims about one buffer; they are the two ends of a widening
conversion. **Sizing the staging buffer to `get_device_size_in_bytes`
(2048 B) would under-size the widening destination and manufacture the heap overrun
this slice claims to fix.** Do not make that change.

**Retraction of my own earlier framing.** The "odd one out — join the convention"
argument is a **category error for this site** and is withdrawn. It is true as a
census fact that the staging allocation is the only one in the file not using
`get_device_size_in_bytes` (four neighbours at `:449,:493,:526,:652`, verified). But
those four allocate **device storage**, whose byte size is device geometry; this one
allocates a **host buffer**, whose byte size is the converter's *output* geometry.
Sameness of the helper would be a false analogy here. The census stands; the
conclusion drawn from it does not.

### 3a. Establish the failure before any fix — the output must name the mechanism

**T-DC-121's mechanism is not established.** No receipt in this lane pins a failing
operation to a branch of this D2H path. Until one does, this slice carries **no
mechanism and no fix**, only the measurement plan.

**Classify by EXCEPTION CLASS FIRST — before any branch reasoning.** The recorded
T-DC-121 symptom is `setStorage` / `RecursionError`, and neither is a copy-path
signature. The exception identifies the layer, and it may localize the defect **without
the box**:

| recorded exception | what it is | points at |
|---|---|---|
| `TORCH_CHECK ... setStorage ... sizes/strides/storage_offset must be within bounds` | a **view-bounds** check | an over-declared logical int64 view over storage sized in int32 units — the **dtype/storage-width split (Slice 1)**, not staging |
| `std::bad_array_new_length` | an **allocation-size** throw | a size expression computed negative/huge — also geometry; trace the expression |
| a copy-path `TORCH_CHECK`, or wrong values with no throw | the copy itself | then (a)/(b) below apply |

**Named candidate (INFERENCE, not established): (c) the storage-geometry split.** If
device storage is sized in int32 units (`module.cpp:198` answer, 32 elem/stick ⇒
`[1,512]` = 2048 B) while `self`'s logical int64 sizes/strides imply 4096 B, then any
view / `as_strided` / `set_storage` over that storage throws exactly what `setStorage`
reports — an over-declared view over under-sized storage. **If this holds, T-DC-121 and
the `get_device_dtype` two-answer split are the same defect, the fix is consistent
storage geometry (Slice 1), and Slice 3 dissolves into it** — staging sizing would be a
red herring against the wrong symptom.

Remaining candidates, none asserted:
- **(a) Converter ignores `dtype_dev` and copies raw bytes** → host buffer under-filled,
  wrong values, no throw. A correctness bug, not a heap bug.
- **(b) Over-READ from a device buffer sized in int32 units** → the only overrun-shaped
  candidate consistent with a widening converter.

**3a deliverable, in this order:**
1. **Reproduce host-side / compile-only first** if the exception is reproducible without
   the device — that is the cheaper move, and a view-bounds throw may be reachable that
   way. The box witness is the fallback, not the first step.
2. Record **the exception class and its throw site** before anything else, then the
   command, tensor (shape, dtype, dense-or-slice, `storage_offset`), and a logged
   evaluation of `is_non_overlapping_and_dense_default()`, `dma_numel`, `self.numel()`,
   `physical_exceeds_logical`, the branch taken, and each buffer's byte size.

**If no such receipt can be produced, the slice reports "T-DC-121 not established" and
no code is written.** A plausible mechanism the code contradicts is worse than an open
question.

**Predicate note (unchanged, and still correct):** the staging branch is entered by
`!is_non_overlapping_and_dense_default() || dma_numel > self.numel()`
(`:594-595`). For a **dense** `[1,512]`, `dma_sizes == [1,512]` (`:462,512,543`;
`codegen_backend.cpp:559`) so `dma_numel == numel` and the tensor is dense — **the
staging branch is not taken**; it runs the fast dense path. The size clause
(`512 > 512` = false) is blind to this case but is **not the operative clause**. **If a
fix touches `physical_exceeds_logical`, it is fixing the wrong predicate** and must be
rejected in review.

**Tests (all diagnostic until 3a reports):**
- **`test_d2h_dense_int64_1x512`** — required; its **status is an output of 3a**, not
  an input here. Passes → valid passing control for later work. Overruns → a second
  defect, and the "control" is a failing test.
- `test_d2h_sliced_int64_1x512` — the sliced trigger, same status rule.
- `test_d2h_staging_float16_unchanged` / `test_d2h_staging_int32_unchanged` —
  byte-identity for the non-int64 dtypes. **What this pair cannot catch, stated so no
  reader over-reads a green:** a change that alters **only the int64 path** passes
  both by construction; their green covers fp16 and int32 only.
- **Regression bar (non-negotiable):** every currently-passing int64 path stays
  byte-identical, **with the enumerated path list in the report**. An empty
  enumeration is a **STOP** — byte-identical over an empty set is not evidence.

**No change is specified here.** The fix follows 3a's mechanism, and Slice 4's witness
is a precondition for choosing between (a) and (b).

---

## 4. Slice 4 — witness the int32→int64 D2H conversion (now the load-bearing question)

**The claim:** `generate_dci` sets
`dci.dataformat_src_ = host2device ? dtype_cpu : dtype_dev`
(`spyre_mem.cpp:362-373`), so a D2H copy converts `dtype_dev` (INT32 for an int64
tensor) → `dtype_cpu` (IEEE_INT64) inside the transfer. **That conversion is
runtime-claimed with no committed witness.**

**Important precision the code adds:** `SpyreStream::copyAsync` refuses a copy whose
two tensors have **different torch scalar types**
(`spyre_stream.cpp:199-203`: `TORCH_CHECK(src.scalar_type() == dst.scalar_type(),
"Spyre backend does not support type conversion yet during copy.")`). Both ends of a
D2H are therefore declared `torch.int64`; the `INT32 → INT64` step is a **physical
format** widening driven by `dataformat_src_/dst_`, not a torch-dtype conversion. The
witness must distinguish "the converter widened the physical format" from "it copied
raw bytes and the host reinterpreted".

**Why it is now load-bearing for Slice 3:** Slice 3's candidates (a) converter copies
raw bytes and under-fills, and (b) the converter is asked to read more device bytes
than exist, are **indistinguishable without this witness**. If the DMA widens, the
host destination must be int64-sized (it is, `:601`) and any overrun would be a device
over-read; if it does not widen, the host buffer is merely under-filled and the defect
is correctness, not heap. **The witness lands before Slice 3's mechanism is chosen at
all** — not merely before it is called done.

**Witness required (device, claim protocol):** an int64 tensor with values that
distinguish int32 from int64 width **and sign** — at least one **negative** (`-3` reads
`-3` if the converter sign-extends; huge/garbage if it raw-packs) and one **≥ 2^31**
(width) — e.g. `[-3, -1, 0, 1, 2^20, 2^31 + 5]` on device — copied D2H and asserted
exactly. It must be a committed receipt, not a verbal result.

**The witness must drive the STAGING branch, not the fast path.** A dense tensor never
enters staging (§3a predicate), so a non-staging witness proves only that the fast path
converts — it cannot observe how the staging branch sizes or fills its buffer. The
witness tensor must be **non-dense or `physical_exceeds_logical`** (e.g. a strided /
sliced view) so the staging code at `:588-614` actually runs. If the same tensor is
also copied through the fast path, record both and compare — that pair is what
distinguishes "the converter widens" from "the two paths size differently".

---

## 5. Box / rebuild protocol (the operational gate)

This is a **csrc change**, so it needs a rebuild, not a source sync:

1. **Claim the box** before any device run; release after (`hw-probe-coordination`).
   Never relaunch a deterministic failure.
2. Sync the lane tree to the box (`hw-box-sync`), `env.sh` in every SSH command.
3. **Rebuild the extension on the box** with the canonical interpreter (pyenv 3.11.4;
   the triton C extension refuses under 3.12). Record the exact build command and
   its full output (never a piped tail).
4. Run the device tests (Slices 3 and 4) under the claim; capture full output, exit
   code, timestamps.
5. Release the claim; commit the receipt under
   `spyre-workspaces/receipts/torchspyre-int64-graph-policy/stage1/` with
   `SHA256SUMS`.
6. **Clear `~/.sentient_cache` and restart Python** before any binary comparison —
   a stale cache masks a fix.

If the box is unavailable, Slices 1 and 2 still land (pure host/codegen, no rebuild);
Slices 3 and 4 hold at "implemented, unwitnessed" and are **not** called done.

---

## 6. E entry conditions (plan only — no Stage-2 code)

Ruling 3 makes E the destination but gates it behind two entry conditions. Both are
planning artefacts, not code:

1. **Where an `i32→rfp32` (`mode_bit=1`) lowering would attach.** Facts already
   established: the emitter and lowering accept the bit
   (`scheduler/emitter.py:840`, `mlir/lowering.py:1693,1812,2412`, default 0); **no
   production lowering sets `mode_bit=1`** — the only assignments are the emulator
   model (`emulator/threefry_recipe_model.py:1123,1131,1136`). The entry artefact is
   the named `InstructionTemplate` and the pass/route point where an int32-arith TTIR
   op becomes an rfp32 SFP sequence. If no point exists without new codegen
   structure, say so — that is an architecture cost, and it is the decision input.
   **Stop rule:** the artefact must contain either a named `InstructionTemplate`
   plus the pass/route point, or an explicit "no attachment point exists; new
   structure required" with that structure named. A verb ("identify where it
   attaches") with neither output is an undischarged condition — it pushes the hard
   question into code, which is what the entry condition exists to prevent.
2. **The per-tensor-class 2^24 bound proof**, written separately for:
   - **accumulation** (`cumsum`, `sum`, `prod`): `N · max|term| < 2^24`;
   - **`i·stride + j`**: the *product* and the *sum* both `< 2^24`, distinguishing
     element values from address offsets (address math is a different capability —
     `L3Opcode.ADDLARIMM/SUBLARIMM/ADDEARIMM`, `gen/rcudd1a/opcodes.py:37-40`).

   "RoPE ≥128K, embedding ~200K ⇒ ≤2^21" is upstream #1547's table, **not** a proof;
   the proof is per class and is the gate.

---

## 7. Non-goals

- No Branch D work (parked by Ruling 3).
- No Stage-2/E code before Stage 1 lands and §6 is discharged.
- Do not widen codegen: `I64PointerDataError` is the law Stage 2 respects.
- Do not change the logical dtype at the eager boundary (Ruling 2).
- **Do not size the staging buffer to `get_device_size_in_bytes`.** The host buffer is
  the *output* of an int32→int64 widening copy (`generate_dci` `:362-372`,
  `:418-419`); sizing it to device bytes (2048) under-sizes the 4096-byte destination
  and manufactures the overrun. The earlier "odd one out" framing is withdrawn (§3.0).
- **Do not carry a T-DC-121 mechanism that the code contradicts.** If 3a cannot
  produce a receipt pinning the failure, the slice reports "not established" and
  writes no code.
- Do not "fix" a size by clamping or by a try/except.
