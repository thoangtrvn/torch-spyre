# PLAN — Stage 1 implementation (int64 three-dtype, AIU 1.0)

Status: **PLAN ONLY. No code until reviewed.** Companion to
`plans/PLAN_int64_three_dtype.md` (the design plan); this file is the work plan for
Stage 1 under **Ruling 3 (USER, 2026-11-16): Stage 1 first, then Branch E.**

Lane `torchspyre-int64-graph-policy` (repo `torch-spyre-thoangtrvn`, base `b413c3e3`).
CSRC change → **rebuild on the box under the hardware claim protocol**.

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

## 1. Slice 1 — one owner for `get_device_dtype`

**Defect (verified):**

| side | site | int64 answer | elems/stick |
|---|---|---|---|
| C++ runtime | `torch_spyre/csrc/module.cpp:190-195` (`get_elem_in_stick`), `:198-202` (`get_device_dtype`) — both `stringToDTDataFormatPair(str_type).second` | `IEEE_INT32` (`csrc/types_mapping.h:71`, `{"int64", {IEEE_INT64, IEEE_INT32}}`) | **32** |
| codegen-only Python | `torch_spyre/_C.py:137` `get_device_dtype` → `_get_dtype_map()[dtype]`, map built at `:116-121` with `torch.int64: DataFormats.IEEE_INT64` | `IEEE_INT64` | **16** (`_C.py:99-100`) |

**Change:** make the C++ pair's `.second` the single owner of the physical/compute
format. The Python codegen-only accessor must **delegate to the same truth or be
deleted**; it must not carry its own logical→device map (`_C.py:119`).

Two acceptable implementations, in preference order:

1. **Delete the Python map and route the codegen-only accessor through the C++
   definition** (generate or bind it), so there is literally one table. Preferred —
   it makes the two-answer class unrepresentable rather than merely equal today.
2. If binding is impractical in codegen-only mode (no extension loaded), keep
   `_C.py` but derive its int64 entry from the pair (int32/32) with a comment naming
   `types_mapping.h:71` as the owner, and add the equality test below as the guard.

**Do not** change the *logical* dtype: `torch.int64` stays int64 at the eager
`.to("spyre")` boundary (Ruling 2; `tests/test_spyre.py:236`).

**Blast radius before the change:** enumerate every caller of `get_device_dtype` and
`get_elem_in_stick` (host Python and C++) and list them in the commit. This is a
behaviour change to a shared accessor: each caller that sizes an allocation from it
is a latent half-size bug today and a changed number tomorrow. The list is produced
by search, not by memory, and an empty list means the accessor is dead (say so).

**Tests (red-first):**
- `test_device_dtype_one_answer`: `get_device_dtype(torch.int64) == IEEE_INT32` and
  `get_elem_in_stick(torch.int64) == 32` in **both** build modes. Fails today in
  codegen-only (returns `IEEE_INT64` / 16) — that failure is the proof of C1.
- `test_device_dtype_matches_cpp_pair`: for every dtype in
  `torchScalarToString`/`stringToDTDataFormatPair`, the Python answer equals the
  C++ `.second`. This is the anti-drift guard, not just an int64 special case.

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

## 3. Slice 3 — T-DC-121: derive the staging buffer from the device size

### 3a. Establish the failing site FIRST — do not presume the branch

The staging branch at `spyre_mem.cpp:575-601` is entered only when
`!is_non_overlapping_and_dense_default() || dma_numel > self.numel()` (`:594-595`).
For a **dense** `[1,512]` int64 tensor `dma_sizes == [1,512]` (set at allocation,
`spyre_mem.cpp:462,512,543`; `codegen_backend.cpp:559`), so `dma_numel == 512 ==
numel`: `physical_exceeds_logical` is **false** and the tensor **is** dense — so the
staging branch is **not taken**, and the `[1,512]` case goes through the **fast dense
path** (`copy_from = &self; copy_to = &dst`; `copyAsync` at `:608`).

**Consequence: the overrun's branch is not established — and neither is the dense
path's safety.** Two candidate overrun sites remain:
- **(a) the fast dense path**, where host `dst` is int64-sized (4096 B) against
  2048 B of device storage — a real 2× mismatch if the DMA does not widen;
- **(b) the staging branch**, for a non-dense/sliced view.

Knowing a dense tensor *takes* the fast path says nothing about whether that path is
*safe*. **3a decides which of (i)-(ii) holds**, and the plan does not assert either:
- **(i)** the dense case passes today → it is a valid **passing control** (see the
  test list), and the defect is confined to (b);
- **(ii)** the dense case **also overruns** → it is a **second defect**, the
  "control" becomes a **failing test** (not a control), and the fix must cover both
  paths.

**The measured `[1,512]` trigger is a sliced view** (a dense one cannot enter the
staging branch), which is consistent with (i) or (ii) — it does not choose between
them.

**The trigger is the density clause, not the size clause.** The defect is reachable
through `!is_non_overlapping_and_dense_default()`. The size clause
(`physical_exceeds_logical`, `512 > 512` = false) is genuinely blind to this case, but
it is **not** the clause that fires here. **If a fix touches `physical_exceeds_logical`,
it is fixing the wrong predicate** and must be rejected in review.

**Slice 3 opens with a site-establishment step whose output NAMES THE BRANCH** — not
"reproduced". Capture, for **both** a dense `[1,512]` int64 and its sliced
counterpart, a logged evaluation of `is_non_overlapping_and_dense_default()`,
`dma_numel`, `self.numel()` and `physical_exceeds_logical`, and the branch taken.
The fix target is then chosen by observation. Confinement to `:575-601` is
**conditional** on that; if the site is the fast path, the slice moves there and the
"other dtypes unchanged" tests still apply.

**Site as previously assumed (confirm in 3a):** `spyre_mem.cpp:575-601` (D2H staging
branch), copy at `:608` (`stream.copyAsync(*copy_from, *copy_to)`), logical view applied
at `:611-614` (`cpu_alloc.as_strided(...)` → `dst.copy_(cpu_view)`).

**Framing: this site is the odd one out, not a new capability.** Within the same
file, the staging path is the **one allocation site that does not derive its byte
size from `get_device_size_in_bytes`**. That helper is declared at
`spyre_tensor_impl.h:149`, defined at `spyre_tensor_impl.cpp:286`, and already used
at `spyre_mem.cpp:449`, `:493`, `:526` and `:652`. So the change is **"join the
convention that already exists four times in this file"**, not "introduce a sizing
rule": the implementer should mimic those four call sites and the reviewer can check
conformance by comparison rather than by argument.

The staging branch builds `alloc_view = at::as_strided(self, dma_sizes, dma_strides)`
and then

```cpp
cpu_alloc = at::empty(alloc_sizes, dst.options());   // alloc_sizes = dma_sizes
```

`dst.options()` carries the **logical** dtype (int64 = 8B), while the device
allocation is `get_device_size_in_bytes(spyre_layout)` bytes
(`spyre_tensor_impl.cpp:286`; declared `spyre_tensor_impl.h:149`; same helper already
used at `spyre_mem.cpp:449,493,526,652`). For an int64 tensor the two disagree by 2×:
`[1,512]` logical = 4096 B of host staging against 2048 B of device storage.

**Change:** size the staging allocation from the **device** geometry, not from the
logical element count × logical itemsize. Concretely, allocate the staging buffer
with the **device element width** and let the CPU-side view/copy perform the widening,
i.e. the allocation's byte size must equal `get_device_size_in_bytes(spyre_layout)`.
Scope follows 3a: if the dense case passes (case i), confine the change to the
staging branch of `spyre_mem.cpp:575-601` and do not touch the fast path; if the
dense case also overruns (case ii), the fast path is **in scope** as a second defect
and cannot be excluded to keep the diff small.

**Tests:**
- **`test_d2h_dense_int64_1x512_control` — required, but its STATUS IS AN OUTPUT OF
  3a, not an input to this plan.** A **dense** `[1,512]` int64 D2H round-trip. The
  test exists either way; 3a decides which it is:
  - if the dense case passes today (3a case i) it is a **passing control** — without
    it, a fix that also changes the dense path cannot be told apart from one that
    repairs only the sliced path;
  - if the dense case overruns (3a case ii) it is a **failing test = second defect**,
    and the plan's premise that the control passes is false.
  Writing "works today" into the plan before 3a measures it is the
  assert-the-green-without-measuring class; it is deliberately not asserted here.
- `test_d2h_sliced_int64_1x512`: the **sliced** `[1,512]` view (the actual T-DC-121
  trigger per 3a) allocates 2048 B of staging, not 4096, and round-trips values
  correctly. Fails today. *(Site per 3a; if 3a finds a different path, the assertion
  moves with it and the control above stays.)*
- `test_d2h_staging_float16_unchanged` / `test_d2h_staging_int32_unchanged`: for a
  DL16 / int32 tensor the staging byte size is **unchanged**. **What this pair cannot
  catch, stated so no reader over-reads a green:** a change that alters **only the
  int64 staging bytes** passes both by construction. Their green covers fp16 and
  int32 only; the int64 path's coverage comes from the `[1,512]` test and the
  regression enumeration, never from these two.
- **Regression bar (non-negotiable):** every currently-passing int64 path stays
  byte-identical. **The report must include the enumerated list of those paths.** An
  empty enumeration is a **STOP**, not a pass: *"byte-identical over an empty set is
  not evidence"* — the bar is met only by a non-empty list of named paths and their
  before/after sizes.

---

## 4. Slice 4 — witness the int32→int64 D2H conversion (owed before Slice 3 is "done")

**The claim:** `generate_dci` sets
`dci.dataformat_src_ = host2device ? dtype_cpu : dtype_dev`
(`spyre_mem.cpp:362-373`), so a D2H copy converts `dtype_dev` (INT32 for an int64
tensor) → `dtype_cpu` (IEEE_INT64) inside the transfer. **That conversion is
runtime-claimed with no committed witness.**

**Why it blocks Slice 3 — a hard dependency, not a hedge (non-negotiable ordering):**
whether the widening happens in the DMA engine or must happen on the host decides
which side of the staging buffer is mis-sized. If the DMA widens, the host destination
must be int64-sized and the *device read* is what needs the device-size bound; if it
does not, the host buffer must be int32 and the widening must be a host `copy_`.
Guessing produces a fix that passes one shape and overruns another — and the failure
would surface only after the fix was written. **Slice 4's witness lands before Slice 3
is called done.**

**Witness required (device, claim protocol):** an int64 tensor with values that
distinguish int32 from int64 width and sign (e.g. `[-3, -1, 0, 1, 2^20]` on device),
copied D2H through the **non-staging** path, asserting exact values. It must be a
committed receipt, not a verbal result. This is the "runtime-claimed, no committed
witness" the reviewer flagged.

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
- Do not "fix" the staging size by clamping or by a try/except — the size derives
  from the device geometry or the bug stands.
