# PLAN — int64 three-dtype model, normalization, and fallback (AIU 1.0)

Status: **PLAN ONLY. No code until the plan is reviewed.**
Lane: `torchspyre-int64-graph-policy` (repo `torch-spyre-thoangtrvn`, base `b413c3e3`).
Owner of the Stage-1 defect below: this lane.

Binding input: USER three-stage spec (logical/storage/compute separation; selective
INT32 normalization with per-intermediate safety; explicit fallback only; no silent
cast). This plan designs to it and does not implement it.

Applies to AIU 1.0 / RCUDD1A. 1.5 (`SEN1P5`) is out of scope (encoding stubs).

---

## 0. Parked artifact

Commit `275fde3e` on branch `torchspyre-int64-graph-policy` (a graph-level metadata
reinterpret of int64 example inputs) **exists and is superseded by this plan.**
Do not merge it. It performs a silent-ish reinterpret that the Stage-3 hard
constraint ("do not silently cast to INT32 to make compilation succeed") would
require justification it does not carry. It is left in place, unreverted
(append-only). It is useful only as a probe artifact: it demonstrated that the
emitted Triton signature turns `*i64,*i64` -> `*i32,*i32` under that approach.

---

## 0.5 Rulings applied (2026-11-16)

- **Ruling 2 — logical dtype is preserved at every user-visible boundary.**
  `logical_dtype` stays `torch.int64` at the eager `.to("spyre")` boundary
  (`tests/test_spyre.py:236` dtype-preserved). **Normalization is an in-graph
  transformation** (torch-spyre Inductor boundary, USER Stage-2). **Storage width
  is the runtime's business and must be consistent** — that third clause is C1 and
  T-DC-121. Stage 2 therefore never changes the eager tensor dtype; it changes only
  what the *compiled graph* declares/executes.

- **Ruling 1 — P0 (exactness bound) answered from the lowering, not hardware
  first.** See §4. Provisional result: **no native int32 *data*-arithmetic path; the
  exact native integer bound is int16**, from E1 (ISA doc) + E2 (codegen static
  classification) — **documentary/static only**. This is a conflict with the USER
  Stage-2 premise ("INT32 execution preferred provided range and arithmetic are
  safe") and is **not settled**: the USER directed an RTL check as the class that can
  refute a document, and §4 is written as a three-branch fork on that answer. No
  Stage-2 code until the branch is chosen.

---

## 1. Conflicts between the spec / external cites and this tree (verified)

These are the highest-value findings: places where the framing did not match the code.

### C1 (Stage-1 defect, live): `get_device_dtype` answers two ways by build mode

- **C++ runtime** — `torch_spyre/csrc/module.cpp:190-195 get_elem_in_stick` and
  `:198-202 get_device_dtype` **both** return
  `stringToDTDataFormatPair(str_type).second` (the Spyre side). For int64 the pair is
  `{"int64", {IEEE_INT64, IEEE_INT32}}` (`csrc/types_mapping.h:71`), so the Spyre side
  is **IEEE_INT32**, `elems_per_stick == 32`.
- **Codegen-only Python** — `torch_spyre/_C.py:137 get_device_dtype` returns
  `_TORCH_DTYPE_TO_DEVICE_DTYPE[dtype]`, whose map is
  `torch.int64 -> DataFormats.IEEE_INT64` (`_C.py:119`); `elems_per_stick()` for
  `IEEE_INT64` is **16** (`_C.py:99-100`). `_C.py:145 get_elem_in_stick` inherits the
  same wrong answer.

**Same name, same question, two answers, selected by build mode.** This is a
single-owner violation whose blast radius is every host readback sizing decision, and
it is the mechanism behind T-DC-121's half-size geometry (allocator int32-width vs
copy logical int64-width). T-DC-121 is not yet filed in `tracking/`; it is being
filed at window formation with owner = this lane. The mechanism is verified even
though the ticket row does not exist yet.

### C2: `codegen_ops.py` is a generated, gitignored artifact — not source

`torch_spyre/codegen_ops.py` is gitignored (`.gitignore:388`) and untracked
(`git ls-files torch_spyre/codegen_ops.py` empty); it is emitted at
`torch_spyre/ops/eager.py:232 # INSERT_CODEGEN_HERE`. Do not cite it as source.
The tracked cumsum CPU path is `torch_spyre/ops/fallbacks.py:304`
(`register_fallback_default([... aten.cumsum ...])`).

### C3: the I64 refusal is the downstream elementwise kernel, not cumsum
cumsum has no device lowering; it runs on the CPU fallback (`fallbacks.py:304`) and
the host int64 result is onboarded. The device kernel that raises
`I64PointerDataError` (`codegen/.../ttir_parser/_descriptor.py:502-522`) is the
downstream elementwise int64 arithmetic: `triton_unk_fused_add_embedding_mul_sub_0`
executes `arith.muli/subi/addi : tensor<32xi64>`
(`receipts/embedding-hw-sweep/.../facebook__opt-125m.strict.ttir1.mlir`). The wall is
that elementwise i64 data, not an absent cumsum lowering.

### C4: `_TYPECAST_OPS_TABLE` / `torch_spyre/_inductor/dtype_ops.py` do not exist here
`grep -rn "_TYPECAST_OPS_TABLE\|dtype_ops" torch_spyre/` -> no match. That table is
upstream-main-only (per issue #4463) and post-dates this fork. A typecast-key fix
("A-narrow") has **no target** in this tree. `get_device_dtype` exists but is the
broken accessor of C1, not a fix.

### C5: i64 already has a codegen metadata-rewrite notion
`codegen/sentient_codegen/backend/admission/generated_inventory.py:172`:
`"i64": BoundedInventoryRule(..., "rewritten_and_verified", "metadata_i64")`
(`"i32"` at `:171` is `consumed_by_route` / `integer_type`). Stage 2 should **reuse**
this rewrite surface, not build a parallel one.

### C6 (load-bearing unknown): integer-exactness bound is unverified in this fork
Our TTIR uses `arith.muli : tensor<32xi32>` — a true integer opcode — but it is
**not verified** whether this backend executes it as integer (exact to 2^31) or via
an fp32 pivot (exact only to 2^24; upstream #5852 describes an fp32 pivot, but that
is their backend). This determines the exactness bound in §4 and is the **first
probe of Stage 2** (§7, P0). The bound must be a single named constant derived from
that probe, never a baked-in number.

---

## 2. Three-dtype model (Stage 1)

Three named dtypes, **one owner each**:

| dtype | definition | single owner | invariant |
|---|---|---|---|
| **`logical_dtype`** | what torch exposes (`torch.int64`); user ABI | torch tensor metadata / eager `.to("spyre")` contract (`_monkey_patch.spyre_to`; `tests/test_spyre.py:236`) | stays int64; not a physical width |
| **`storage_dtype`** | physical element width in device memory; unit of storage size / byte-stride | the storage + `SpyreTensorLayout` (`csrc/spyre_storage_impl.*`, `csrc/codegen_backend.cpp`) | **4 bytes (INT32)** for an int64-logical tensor |
| **`compute_dtype`** | format the kernel executes in; kernel-arg element type | the descriptor/codegen device side; **the C++ pair `.second`** (`module.cpp:198`) | `IEEE_INT32` for int64 |

**Single owner of `compute_dtype`/physical format:** the C++ `stringToDTDataFormatPair(..).second`.
The Python accessor `_C.py:137` must **delegate to the same truth (or be deleted)** —
it must not carry its own logical->device map (`_C.py:119`). Host readback sizing
must call a **distinct, unambiguously-named** accessor (e.g. `physical_storage_dtype`),
never the overloaded `get_device_dtype`, precisely because of C1.

### Which accessor each operation consults

| operation | must consult | correct value | defect if it consults the wrong one |
|---|---|---|---|
| stride (torch API) | `logical_dtype` (in elements) | int64 elements | — |
| stride -> bytes (DMA) | `storage_dtype` | x4 | x8 (logical itemsize) => over-stride / OOB |
| storage size | `storage_dtype` | `numel*4` | logical-sized => `setStorage` / T-DC-121 overrun |
| copy H2D / D2H | the CPU/Spyre **pair** (`types_mapping.h:59-61,71`) via `generate_dci` | int64 <-> INT32 | width mismatch if storage is not INT32-width |
| view / `as_strided(dtype)` | `storage_dtype` | reinterpret at 4B | logical-int64 view mis-strides 4B storage (the #4329 `storage_offset=1` clone class) |
| host transfer / readback (`ops/embedding.py:94 .cpu().tolist()`) | `storage_dtype`/`compute_dtype` | size from INT32 (32/stick) | sizing from `get_elem_in_stick(int64)` => 16 (codegen-only) vs 32 (device) |

**Stage-1 acceptance:** C1 fixed (one answer across build modes), and for each of
{stride, storage size, copy, view, host transfer} the row above holds with a test.
A type-routing key fix alone is **not** Stage-1 complete (USER).

---

## 3. Four-origin taxonomy (Stage 2)

Selective int64->int32 normalization at the torch-spyre Inductor boundary,
propagated consistently through typecasts, kernel inputs, and fused ops. Each
carrier is classified by origin; **tensor shape is never used as a proof that
elements fit INT32.**

| origin | example | proof obligation |
|---|---|---|
| **Host-provided int64** | position_ids / input_ids buffers registered by HF and moved to spyre | provenance bound at the host producer (arange/mask => vocab/seq bound), attached as an explicit contract guard at the boundary |
| **Compile-time constant** | `dense<1>`, `dense<2>`, `258880` | literal `<= INT32_MAX` checked at lowering; refuse otherwise |
| **In-graph index** | derived index expression `i*stride+j` | symbolic max = product/sum of tensor extents; unknown/dynamic => cannot normalize (Stage 3) |
| **Dynamically calculated int64** | `cumsum` accumulation, runtime positions | intermediate-overflow proof (§4); the hardest class; unprovable => typed refusal or explicit partition |

Normalization is applied **only** to an origin whose proof obligation is discharged.
An unproven origin refuses (never a silent cast).

---

## 4. Value-range and intermediate-overflow contract (Stage 2)

### 4.0 P0 result and its **evidence class**

Finding (provisional): **the AIU appears to have no native int32 *data* arithmetic,
so `E = 2^31` does not hold**; the widest *exact* native integer path is int16.

**Evidence class: documentary/static — two artifact classes, zero silicon, zero RTL.
This is deliberately not yet a settled fact.**

- **E1 — ISA document** (`wiki/specs/isa/sentient-isa-1p0.md:2231-2262`): every
  multiply is floating — `SFP_FMUL` (0x00, `:2235`), `SFP_FMA` (0x04),
  `SFP_FNMS` (0x06, `:2237`); the enumerated table has **no integer multiply**.
  Integer handling is `SFP_ICVT` (convert), `SFP_IME`, `SFP_SHR`, `SFP_LOGICAL`,
  `SFP_SELECT`. The only 32-bit rows are **conversions** through the rfp32
  container — `F2I32`/`R2I32`/`I2F32` (`codegen/sentient_codegen/gen/rcudd1a/opcodes.py:745-749`,
  `ICVTMode`) — not arithmetic. PE "integer arithmetic" is the **int16 FMA lane**
  (`wiki/concepts/core-functional-units.md:112` lists int16/int8/int4/int2, all
  sub-32-bit); PT integer is int8/int4 -> **int24 accumulate** via ARF
  (`sentient-isa-1p0.md:80,1635`).
- **E2 — codegen static classification** (`codegen/sentient_codegen/backend/ttir_to_schedule.py:4222`):
  `_AUXILIARY_DTYPES = frozenset({"i1","i32","i64"})` — i32/i64 are **auxiliary
  structural** dtypes; "a kernel whose dtypes are ALL auxiliary has no compute dtype
  to resolve" (`:4219-4221`). `codegen/sentient_codegen/backend/ttir_analyzer.py:90`
  calls `addi/subi/muli/divsi/remsi` **"INFRASTRUCTURE: integer index/pointer
  math"**, not compute; the pointwise compute set is float-only
  (`_FLOAT_COMPUTE_ARITH_OPS`, `ttir_analyzer.py:98`). A materialized int32 *data*
  result of arithmetic therefore has no supported exact device lowering.

**E1 and E2 agree, but both are documents/static tables. Two documentary classes
agreeing is not the standard the USER set — silicon is, and RTL is the class that can
refute a document. This is why the decision below forks instead of settling.**

The RTL check is load-bearing precisely because **the ISA document is demonstrably
unreliable adjacent to this exact question**: `sentient-isa-1p0.md:2239` labels
`SFP_IME` **"Integer multiply-extend"**, while
`codegen/sentient_codegen/gen/rcudd1a/opcodes.py:759` (`class IMEMode`) defines it
as **Insert / Modify Exponent** (INSERT/ADD/SUBTRACT on the S2 exponent), and the
same document itself says **"Insert/merge element"** at `:1427`. A document that
mislabels one row of this table is not a safe enumeration of arithmetic capability;
E1 is a hint, not proof. A read-only RTL investigation against `../RTL/hw1p5/logic/`
is dispatched. **There is no 1p0 RTL source in the tree, so any RTL conclusion is an
inference about 1p0 from 1p5 and must be labelled as such.**

### 4.1 Decision fork on the RTL answer (all branches written now)

The fork decides §4, not this plan's section numbering. Each branch carries its
**falsifier** — the observation that would move the decision to another branch.

**Branch A — RTL confirms no exact integer *data* datapath (current expectation).**
- Condition: no exact integer multiply/ALU feeding the vector data path; integer only
  as address-generation (LX/DIU) and as int16/int24 sub-word lanes.
- Consequence: carriers split, exactly as proposed in §4.2:
  1. position/index values **stay in the address-generation path** — consumed only as
     gather indices / addresses (embedding `weight[pos]`), executed natively as
     "infrastructure" math, never materialized as data;
  2. materialized position **data** (OPT's stored `(cumsum*mask-1).long()+2`) runs on
     the **CPU fallback** (Stage 3 explicit — today's #4317 behaviour: correct,
     slow). `cumsum*mask` with mask in {0,1} is a **select** `SFP_SELECT` (0x0E),
     exact at any width, so the mask-multiply is not the problem — the accumulation
     range is.
- **Cost (must be stated, not hidden):** a host round trip **per call** for every
  encoder that materializes position embeddings — a D2H sync on the decode hot path,
  plus a CPU-side cumsum over `seq_len`, plus an H2D back. For a batch-1 decode step
  that is a fixed per-step latency add on top of the whole graph, and it defeats
  graph capture. This is the price of exactness; it is not free.
- **Falsifier:** any exact integer multiply/ALU ≥32 bits on the vector data path in
  the RTL (moves to Branch B), or evidence that materialized positions never reach
  the device as data in any in-scope model (weakens to "branch A unnecessary").

**Branch B — RTL finds an exact integer datapath of width W (unlikely; the point of
checking).**
- Condition: an integer multiply/ALU of width W >= 16 on the data path (not just the
  address path).
- Consequence: USER Stage 2 becomes performable as originally specified, and the
  exactness bound comes from that datapath: `E = 2^(W-1)` (signed) **for that
  datapath's op set**. If W = 32, `E = 2^31` and RoPE(128K)/embedding(200K) fit; if
  W = 16, it lands back on Branch A for those carriers.
- **Falsifier:** the datapath is conversion-only (FP↔int via rfp32), or present but
  not reachable from the pointwise/elementwise route this backend emits.

**Branch C — no exact path AND no acceptable fallback for performance reasons.**
This is the branch nobody wants written down; record it so the choice is explicit,
**as an option with its consequences, not a recommendation.**
- Condition: Branch A holds, and the per-call host round trip is judged unacceptable
  on the decode hot path (or the model cannot afford a CPU partition).
- Consequence: an **explicit, typed narrow contract** — "int64 data arithmetic is
  supported only within ±2^15 (int16); values above are refused typed at the
  Inductor boundary (`UnsupportedInt64RangeError`)". This **satisfies** the USER's
  "do not silently cast" rule (it refuses, it does not truncate) **while conceding
  the capability** — RoPE/embedding-scale position arithmetic is simply unsupported
  on device under this contract. It is honest, it is fast, and it does not work for
  the models in upstream #1547.
- **Falsifier:** a carrier whose bound is provably ≤2^15 (e.g. a model with
  seq_len ≤ 32768 and no embedding-table accumulation) — for that carrier the
  contract is sufficient; a u16/u15 interpretation extends it if RTL shows unsigned
  lanes.

**E3 (RTL) and E4 (device KAT) are the two classes that can settle this.** Until E3
lands, §4.2's per-op bound is written per branch; the branch is chosen when E3
reports, and E4 (§7 P0-confirm) confirms on silicon.

### 4.2 Per-op bound under Branch A (and the bound law under Branch B)

`E` is a single named constant **per op class**, derived from the chosen branch,
never a baked-in 2^31. Check the running intermediate, not just the final tensor
(USER condition 2):

- **accumulation** (`cumsum`, `sum`, `prod`): bound `N * max|term| <= E_op`. Attention
  -mask cumsum (terms in {0,1}) <= seq_len: safe **only** if seq_len <= `E_op`; at
  RoPE 128K it is NOT safe under int16/DL16 -> CPU fallback.
- **product** (`mul`; fused address expressions): `a*b <= E_op` even when a,b fit.
  The affine-index form `i*stride+j` overflows the product with in-range `i,j,result`
  — but this is **address math** (§5), executed natively in the address path; bound
  the product against the *address* width, not the element width.
- **elementwise** (`add`, `sub`, `eq`, `lt`, `clamp`): operand bounds; int16/DL16
  range for data, address width for index.
- **op classes where mid-chain overflow can occur with in-range endpoints:**
  accumulation, product, and affine index computation (`i*stride+j`) — a
  final-tensor-only check passes them incorrectly (condition 2).

Under **Branch B** this list is unchanged in form; only `E_op` is re-derived from the
RTL datapath width `W` (`E = 2^(W-1)` for signed, `2^W - 1` if unsigned) instead of
from the int16/DL16 lanes. Under **Branch C** there is no per-op derivation: a single
constant `E = 2^15` gates every carrier and the rest refuse typed.

**Placement:** the bound is a symbolic claim discharged at the Inductor boundary from
the §3 origin taxonomy; undischarged -> Stage-3 refusal. Never a runtime clamp.

---

## 5. Addressing capability (USER condition 3) — separate from element width

Element width <=32b does **not** imply an index/address fits a 32-bit element, nor
the reverse. Strides and fused address expressions may need wider addressing even when
elements fit INT32. Two distinct capabilities:

- **element width** (§4): does a *value* fit.
- **addressing**: does a *byte/element offset* fit the address registers and fused
  expressions. KV-cache "range = addressable space" is this category.

The embedding-table span wall (#6277: 384 MB table > 256 MB per-core span; #6322:
704 MB) is an **addressing / work-division** limit, not a dtype limit. It must be
designed as table partitioning / restickify splitting, **not** "make the table fit
int32." A design that fixes int64 routing and calls embedding solved is refuted by
#6277/#6322. This section stays separate from §4.

---

## 6. Refusal taxonomy (Stage 3) — layer names

Explicit fallback **only** when safe lowering is impossible, on a supported graph
partition with valid transfers; otherwise raise a typed unsupported-dtype error. No
silent INT32 cast.

| condition | layer | error |
|---|---|---|
| value/expression exceeds, or cannot be proven within, `E` (§4) | torch-spyre Inductor boundary (`_inductor/`) | `UnsupportedInt64RangeError` (this lane) |
| int64 data reaches a kernel arg after Stage 2 (defense in depth) | codegen descriptor | `I64PointerDataError` (`backend/ttir_parser/_descriptor.py:502`) |
| int64 tensor-descriptor element type | codegen descriptor | `I64DescriptorDataError` (`backend/ttir_parser/_descriptor.py:444`) |
| unsupported int64 op, fallback chosen | supported partition, CPU fallback + `FallbackWarning` (today's #4317 path, `fallbacks.py:304`) | warning + CPU partition |
| fallback attempted with a CPU-resident intermediate | `spyre::to_dtype_cpu` (PrivateUse1-only, #5070) | `NotImplementedError` |
| no host data channel (codegen-only) | runtime | `SpyreRuntimeUnavailableError` (`torch_spyre/_runtime.py:27`) |

After Stage 2, `I64PointerDataError`/`I64DescriptorDataError` must be
defense-in-depth only, never the primary user-facing refusal.

---

## 7. Falsification plan (plan-first; no code)

- **P0 (provisional, from lowering; RTL + device KAT owed):** §4.0 records E1 (ISA
  doc) + E2 (codegen classification) agreeing that there is **no native int32 data
  arithmetic** — but both are documentary/static. **E3 (RTL, dispatched against
  `../RTL/hw1p5/logic/`, labelled as a 1p5->1p0 inference) decides the §4.1 branch;
  E4 (device KAT) confirms on silicon.** E4: run an i32 data `muli`/`addi` and record
  whether it refuses or lowers via a float pivot — that is the class that bounds `E`
  for real. Do not write Stage-2 code on E1+E2 alone.
- **P1 (Stage 1 geometry):** assert one `get_device_dtype(int64)` answer across
  codegen-only and runtime, and `elems_per_stick(int64)` identical in both build modes
  (currently **16 vs 32** — fails today, proving C1). Add: storage size = `numel*4`;
  `view(dtype)` + `storage_offset=1` clone (#4329 class); D2H readback width. These
  catch the half-storage/T-DC-121 geometry.
- **P2 (before arm):** `receipts/codegen-constant-nonmeta/repro_03_layer_behind.py` —
  `I64PointerDataError` independent of the constant axis. Do **not** reopen the
  constant axis (admitting the constant only relocates the refusal — a known
  false-green).
- **P3 (Stage 2):** per-origin probes asserting (a) normalization applied only with a
  discharged proof; (b) a deliberately-overflowing `cumsum` and `i*stride+j` are
  **refused typed, not truncated**; (c) no `!tt.ptr<i64>` survives for admitted graphs
  (offline signature check: base `*i64,*i64` -> want `*i32,*i32`).
- **P4 (Stage 3):** an unsupported int64 op yields a typed refusal, or an explicit CPU
  partition with `FallbackWarning`; codegen-only CPU-resident intermediate raises
  (#5070 seam).
- **P5 (HW KAT, last):** fp16 output vs CPU for the OPT position-id shape once the
  unrelated structured-production-subset wall (`constant_non_metadata_user`) clears.
  That wall is separate and out of scope here.

---

## 8. Explicit non-goals

- Do not reopen the constant axis (P2).
- Do not merge parked `275fde3e`.
- Do not widen codegen: `I64PointerDataError` is the law Stage 2 respects.
- Do not conflate the embedding-table span wall (§5) with the element-width problem.
- 1.5 (`SEN1P5`) encoding is out of scope.

---

## 9. Trial-merge vs staging `7fff29f42` (f32-pointer refusal)

`7fff29f42` merges `f32-pointer-element-grain-refusal` and puts the f32 refusal
**above every route**: `ttir_to_schedule._refuse_compute_only_pointer_pointees`
(`codegen/sentient_codegen/backend/ttir_to_schedule.py:4283`, called at `:5571`) is
declared "the ONE owner of the pointer-grain law", keyed on pointer **pointee
spellings** in `kernel_info.args`; `_resolve_element_bits` skips `_AUXILIARY_DTYPES`
(`:4222`).

**Textual conflict: none.** The f32 law edits the codegen repo
(`codegen/sentient_codegen/backend/ttir_to_schedule.py`); this plan's Stage-1/2/3
edits live in the torch-spyre repo (`torch_spyre/_C.py`, `csrc/module.cpp`,
`csrc/types_mapping.h`, `torch_spyre/_inductor/`, `ops/`). C5 *reads*
`generated_inventory.py:172`, it does not edit it.

**Semantic conflict: none, provided the boundary is respected — and it constrains us.**
- int64 normalization is **upstream in torch-spyre** (USER Stage-2), so codegen sees
  i32 pointees and never interacts with the f32 grain law (different dtype; i32 is
  auxiliary, skipped by `_resolve_element_bits`). The i64 and f32 laws are parallel
  single-owner guards at different points (i64 at descriptor parse
  `_descriptor.py:502-522`; f32 at `compile_ttir_to_spyre` route entry `:5571`).
- **Hard constraint (the seam the lead flagged):** any int64 normalization that tried
  to rewrite `!tt.ptr<i64>` -> `!tt.ptr<i32>` *inside codegen* would contend with
  `_refuse_compute_only_pointer_pointees`'s "ONE owner of the pointer-grain law" on
  the same `kernel_info.args` pointees. This is forbidden by the design: int64
  normalization stays in torch-spyre; codegen keeps both guards untouched.
- Watch item: the f32 law keys on `_COMPUTE_ONLY_16_BIT_DTYPES` and does not consider
  i64; `_AUXILIARY_DTYPES` includes i32/i64. Confirm at merge time that no codegen
  change reclassifies i32/i64 pointees.

**Result: trial-merge clean (no textual, no semantic conflict) under the
"normalization stays upstream" rule.**
