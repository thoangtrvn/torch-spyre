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

Candidate mechanisms, none asserted:
- **(a) Converter ignores `dtype_dev` and copies raw bytes.** Then a D2H writes 2048 B
  into a 4096 B host buffer — under-filled, not an overrun, wrong values. A
  *correctness* bug, not a heap bug.
- **(b) Over-READ from a device buffer sized in int32 units.** The device storage is
  `get_device_size_in_bytes` = 2048 B; if anything asks the converter for 4096 B of
  input, that is a read past the allocation. This is the only overrun-shaped candidate
  consistent with a widening converter.
- **(c) The defect is not on this path at all** — the failing site is elsewhere
  (allocation, view, or `dst.copy_(cpu_view)` at `:611-614`).

**3a deliverable:** reproduce the exact reported failure and record — command, tensor
(shape, dtype, whether dense or a slice, `storage_offset`), exception text, and a
logged evaluation of `is_non_overlapping_and_dense_default()`, `dma_numel`,
`self.numel()`, `physical_exceeds_logical`, the branch taken, and the byte size of
each buffer at the point of failure. **If no such receipt can be produced, the slice
reports "T-DC-121 not established" and no code is written.** A plausible mechanism
that the code contradicts is worse than an open question.

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
- **Do not size the staging buffer to `get_device_size_in_bytes`.** The host buffer is
  the *output* of an int32→int64 widening copy (`generate_dci` `:362-372`,
  `:418-419`); sizing it to device bytes (2048) under-sizes the 4096-byte destination
  and manufactures the overrun. The earlier "odd one out" framing is withdrawn (§3.0).
- **Do not carry a T-DC-121 mechanism that the code contradicts.** If 3a cannot
  produce a receipt pinning the failure, the slice reports "not established" and
  writes no code.
- Do not "fix" a size by clamping or by a try/except.
