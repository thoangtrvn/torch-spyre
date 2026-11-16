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

- **Ruling 1 — P0 (exactness bound) answered from the lowering, then settled by
  RTL.** See §4. **No native int32 *data*-arithmetic path** (E1 ISA doc + E2 codegen
  static classification + **E3 RTL** `../RTL/hw1p5` @ `9c15c33d81`: no PE/SFP integer datapath,
  int16 tied off, PT int8→int24 the only integer arithmetic). E3 **selects Branch
  A**; **E4 (device KAT) owed** for the 1p0 silicon witness. The earlier int16 premise
  is refuted and recorded in §4.0's corrections table. This is a conflict with the
  USER Stage-2 premise ("INT32 execution preferred provided range and arithmetic are
  safe"); §4 now carries five branches (A selected, B kept for E4, C written option,
  **D = PT limb decomposition**, and **E = rfp32 direct (2^24 exactness), whose M=1
  premise is a spec self-conflict pending the `m1-arith-width` probe**). D and E both
  need new lowerings and neither is recommended. No Stage-2 code until the branch is
  chosen.

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

Finding (settled at E3): **the AIU has no native int32 *data* arithmetic, so
`E = 2^31` does not hold**; the widest *exact* integer on the chip is PT's int24
accumulator, fed by int8 (int4 in 1p0) operands. **There is no int16 integer lane to
anchor a bound on.**

**Evidence classes, in the order they landed:**

- **E1 — ISA document** (`wiki/specs/isa/sentient-isa-1p0.md:2231-2262`): every
  multiply is floating — `SFP_FMUL` (0x00, `:2235`), `SFP_FMA` (0x04),
  `SFP_FNMS` (0x06, `:2237`); the enumerated table has **no integer multiply**.
  Integer handling is `SFP_ICVT` (convert), `SFP_IME`, `SFP_SHR`, `SFP_LOGICAL`,
  `SFP_SELECT`. The only 32-bit rows are **conversions** through the rfp32
  container — `F2I32`/`R2I32`/`I2F32` (`codegen/sentient_codegen/gen/rcudd1a/opcodes.py:745-749`,
  `ICVTMode`) — not arithmetic. PT integer is int8/int4 -> **int24 accumulate** via
  ARF (`sentient-isa-1p0.md:80,1635`).
- **E2 — codegen static classification** (`codegen/sentient_codegen/backend/ttir_to_schedule.py:4222`):
  `_AUXILIARY_DTYPES = frozenset({"i1","i32","i64"})` — i32/i64 are **auxiliary
  structural** dtypes; "a kernel whose dtypes are ALL auxiliary has no compute dtype
  to resolve" (`:4219-4221`). `codegen/sentient_codegen/backend/ttir_analyzer.py:90`
  calls `addi/subi/muli/divsi/remsi` **"INFRASTRUCTURE: integer index/pointer
  math"**, not compute; the pointwise compute set is float-only
  (`_FLOAT_COMPUTE_ARITH_OPS`, `ttir_analyzer.py:98`). A materialized int32 *data*
  result of arithmetic therefore has no supported exact device lowering.
- **E3 — RTL** (`../RTL/hw1p5` @ `9c15c33d81`; **1p5 source, so a labelled
  inference about 1p0**): **no integer arithmetic datapath in PE or SFP.** Proof,
  verified directly:
  - `rpd/vhdl/rpd_pesfp_pmu.vhdl:40-62` — the PMU decode list is
    shift/splat/fest/gcvt/logical/compress/fcvt/permute/copy_lfsr/load_lfsr; **no
    integer add, no integer multiply**.
  - `rpd/vhdl/rpd_pesfp_fp16_top.vhdl:247` — `in_ctrl_x_int16 => '0'`: the PE x-int16
    control is **tied off**, not a compute lane.
  - `rpd/vhdl/rpd_pt_fx_engine.vhdl:111-140` — PT is the only integer arithmetic:
    24-bit result per lane (`e2_fpu_result_NORM_EXT1(24*i+24-1 downto 24*i)`), int8
    mode live (`gbl_pt_int8`) — in the **1p5** RTL the int4 mode is commented out
    (`:31,45,76,119-121`), so "int4 operands" is a 1p0-only claim.
  - `rpd/vhdl/rpd_pesfp_fp32_aln.vhdl:393-441` — `ime insert` / `ime subt` / `ime
    add` confirmed: SFP_IME manipulates the **exponent**, it does not multiply.
- **E4 — device KAT (still owed):** confirm on silicon that an i32 data `muli`/`addi`
  on the emitted route refuses or lowers inexactly. E3 is 1p5 RTL; E4 is the 1p0
  silicon witness.

**Why E1 alone was not enough (kept as recorded history):** the ISA doc labels
`SFP_IME` **"Integer multiply-extend"** at `sentient-isa-1p0.md:2239`, while
`codegen/sentient_codegen/gen/rcudd1a/opcodes.py:759` (`class IMEMode`) defines it as
**Insert / Modify Exponent**, and the same document says **"Insert/merge element"**
at `:1427`. E3 confirms the RTL side (`fp32_aln.vhdl`): the doc row is an erratum.
The E1-hint reasoning was right for the right reason — a document that mislabels one
row of this table is not a safe enumeration of arithmetic capability.

**Corrections from E3 — refuted premises, recorded rather than silently edited**
(a plan that drops a refuted premise teaches nobody):

| premise (as previously written) | status | E3 cite |
|---|---|---|
| "PE integer arithmetic is the int16 FMA lane, exact to 2^15" | **REFUTED.** `in_ctrl_x_int16` forces a fixed exponent into the **floating** FMA and is tied off in the PE top; it is not an integer multiplier. Exact only to ~2^10 — the ISA doc's own "first inexact integer is 1025". | `rpd_pesfp_fp16_top.vhdl:247` |
| "the widest exact native integer path is int16 (2^15)" | **REFUTED.** The widest exact integer is PT's **int24 accumulator** (`\|acc\| ≤ 2^23`), fed by int8 (int4 in 1p0) operands. No 2^15 lane exists. | `rpd_pt_fx_engine.vhdl:111-140` |
| "Branch C anchors on ±2^15 (int16)" | **RESTATED.** There is no 2^15 lane. Any narrow contract must be expressed against int8/int4 operands and the int24 accumulator, not int16. | as above |
| "there is no evidenced rfp32 multiply (FMA is 16-bit only)" (my §4.2 wording) | **REFINED.** rfp32 FMA/FMUL at M=1 is *prose-supported* and the format is 1-8-23, but **M=1 arithmetic semantics is a spec self-conflict** (Alg. 146 = two DL16 half-products) and the deciding probe is designed, not run. So "no rfp32 multiply" overstates what the tree says; "rfp32 multiply exists in prose, unverified in silicon" is the accurate statement. | `threefry_recipe_model.py:339-355`, `rng_probe_readings.py:136-148`, `precision-formats.md:15` |

### 4.1 Decision fork — E3 selects Branch A; D is the open alternative

**E3 selects Branch A** (no exact integer *data* datapath). Branches A–E are all
written; each carries its **falsifier**. A is the current expectation; D is the one
branch that could preserve on-device exactness and is written with its cost, **not
recommended**; C stays a written option, **not a recommendation**.

**Branch A — SELECTED by E3: no exact integer *data* datapath.**
- Condition (met): no exact integer multiply/ALU on the vector data path; integer
  only as address-generation (LX/DIU) and as PT int8/int4 -> int24.
- Consequence: carriers split:
  1. position/index values **stay in the address-generation path** — consumed only as
     gather indices / addresses (embedding `weight[pos]`), executed natively as
     "infrastructure" math, never materialized as data;
  2. materialized position **data** (OPT's stored `(cumsum*mask-1).long()+2`) runs on
     the **CPU fallback** (Stage 3 explicit — today's #4317 behaviour: correct,
     slow). `cumsum*mask` with mask in {0,1} is a **select** `SFP_SELECT` (0x0E),
     exact at any width, so the mask-multiply is not the problem — the accumulation
     range is.
- **Cost (stated, not hidden):** a host round trip **per call** for every encoder that
  materializes position embeddings — D2H sync + CPU cumsum over `seq_len` + H2D — on
  the decode hot path, defeating graph capture. Exactness is not free.
- **Falsifier:** an exact integer multiply/ALU on the PE/SFP vector data path (moves
  to B), or evidence that materialized positions never reach the device as data in
  any in-scope model (A becomes unnecessary).

**Branch B — no longer reachable on the E3 evidence.**
- Condition: an integer multiply/ALU of width W ≥ 16 on the data path. E3 did not
  find one (PMU list, tied-off int16, PT-only integer).
- Consequence if it existed: `E = 2^(W-1)`; W = 32 would fit RoPE/embedding.
- **Kept because it is the branch E4 could resurrect:** if the 1p0 silicon (E4) shows
  an integer datapath the 1p5 RTL lacks, B reopens. Falsifier: datapath is
  conversion-only (rfp32) or unreachable from the emitted pointwise route.

**Branch C — no exact path AND no acceptable fallback. Written option, not a
recommendation.**
- Condition: A holds, and the per-call host round trip (or a CPU partition) is judged
  unacceptable on the decode hot path.
- Consequence: an **explicit, typed narrow contract** — "int64 data arithmetic is
  supported only within the exact range of the only integer path that exists (PT
  int8 operands / int24 accumulator); values above are refused typed at the Inductor
  boundary (`UnsupportedInt64RangeError`)". **Restated from ±2^15** because there is
  no int16 lane (§4.0 corrections). It satisfies "do not silently cast" by refusing,
  while **conceding the capability** — it is honest and fast and does not work for
  the #1547 models.
- **Falsifier:** a carrier whose bound is provably inside the PT int8/int24 exact
  range — for that carrier the narrow contract is sufficient.

**Branch D — PT limb decomposition: exact on device, no host round trip. Not
recommended; recorded with its cost.**
- **Idea.** A direct PT encoding caps operands at int8 magnitude, which is why this
  plan previously dismissed PT for these workloads. Limb decomposition is the
  counter-argument: represent a ~2^21 integer as **k int8 limbs** and compute the
  product as a limb convolution. Each int8×int8 product is ≤ 2^14 and lands exactly
  in the 24-bit accumulator (`|acc| ≤ 2^23`); accumulating ≤ k² of them plus carries
  stays far inside 2^23. This gives **exact RoPE/embedding-range arithmetic on
  device, with no host round trip** — the cost A pays and D avoids.
- **Limb count vs range (general rule).** For an unsigned magnitude with signed
  range ±R, `k = ceil(log2(R+1) / 8)`: ±2^21 needs **3 limbs**; ±2^23 still 3; ±2^31
  → 4. A 3×3 convolution gives **9 products + carry chain** per output element
  (2-limb → 4 products; the general rule is k² products + O(k²) carry adds).
- **Signed treatment.** int8 limbs must be either **offset-binary** (add 128, correct
  the cross-terms) or **sign-extended with a signed×signed limb multiply**; the plan
  must pick one and prove the carry/borrow chain exact, because a wrong signed limb
  convention is a silent-wrong-answer class, not a refusal.
- **Throughput cost.** ~9 int8 MACs + carry per logical multiply vs 1 floating FMA.
  PT is the MAC engine, so the arithmetic is in the right place, but the **limb
  convolution multiplies instruction count by ~9** and adds a carry chain that the
  FP path does not have. Whether that beats a host round trip is workload-dependent
  and **must be measured** (E4-class device probe), not assumed.
- **Where the arms land (architecture cost, not a code change).** This requires a PT
  route reachable from the emitted pointwise path. Today PT carries matmul MACs; the
  pointwise/elementwise route emits SFP ops. So D needs a **new lowering** from the
  pointwise path to a PT limb-convolution sequence — an **architecture cost**, stated
  as such, not a small patch. If no PT route is reachable from pointwise, D is not
  implementable without new codegen structure.
- **Falsifier:** PT unreachable from the pointwise route (⇒ D needs new lowering, may
  be judged out of scope), or the carry chain + 9× instruction count dominating to
  the point where the Branch-A CPU fallback is cheaper (⇒ D loses on its own merits).

**Branch E — rfp32 direct; 16-bit halves only above the 2^24 bound. Newly derived
from the RNG design record; premise partially unverified. Not recommended yet.**
- **Source (re-derived from the tree, not taken on trust):**
  `spyre-workspaces/rng-design-handoff/rng-program-problem-and-hurdles.md:46`
  (mirror `wiki/design/45-device-rng-problem-and-hurdles.md:51`) records that SFP
  FMA/FMUL/FNMS run "in DL16 (M=0) or **rfp32** (M=1: 24-bit significand, no
  denormals)", and that "32-bit integer arithmetic is done as exact float arithmetic
  on 16-bit halves, because every half-word sum stays below 2^24". The format fact is
  independently confirmed: `wiki/foundations/hardware/precision-formats.md:15` gives
  rfp32 as **1-8-23** (24-bit significand) and `:126` "no denormal representations";
  the SFP M bit is documented at `sentient-isa-1p0.md:2359` ("0=16-bit (DL16),
  1=32-bit (rfp32)").
- **Exactness law.** A rfp32 word holds an integer exactly iff it needs ≤24
  significant bits ⇒ exact on `[0, 2^24]`. So **RoPE (≥2^17) and embedding (~2^18)
  values fit a single rfp32 word exactly.** `a + b` is exact iff the *sum* stays
  below 2^24; `i·stride + j` is exact iff the *product and sum* stay below 2^24 —
  **that is USER Stage-2 condition 2 written as a checkable bound**, and it is a much
  stronger regime than DL16's 2^10 (`sentient-isa-1p0.md:75`, first inexact integer
  1025). 16-bit halves are needed only where an intermediate can exceed 2^24 or a
  32-bit bitwise op is required (the RNG case), not for the carriers themselves.
- **The premise that is NOT verified — and the repo already says so.** "M=1 executes
  as a true 32-bit rfp32 FMA" is **a spec self-conflict, not a settled fact**:
  `codegen/sentient_codegen/emulator/threefry_recipe_model.py:339-355` records
  `FmulM1`: "the prose says M=1 selects rfp32; Alg. 146 and the mnemonic are 16-bit
  only, which on a 32-bit word is two DL16 products. A spec self-conflict.
  `DL16_HALVES` is not modelled: it refuses", and
  `emulator/rng_probe_readings.py:136-148` records `M1ArithWidth =
  M1_EXECUTES_AS_DL16` — "silicon cannot tell 'bit 44 ignored' from 'bit 44 honoured
  as half-wise DL16 math'" — with **whether bit 44 is read at all** answered by the
  `p1-icvt-m0` / `p1-icvt-m1` probe pair (`emulator/rng_probe_catalog.py:115-116`)
  and the `m1-arith-width` probe (`:1277`). Those probes are **designed and not yet
  run** (`receipts/box-runner-sync-fix/migration-note-rng-silicon-probes.md` says the
  run sheet is prepared, the fix must land first). **So the RNG recipe's 120,900-
  vector bit-identity is bit-exact under the *model's* RFP32 assumption — it
  validates the algebra, not the silicon's M=1 semantics.** Do not cite it as proof
  that rfp32 FMA is real.
- **Route cost.** No route emits `mode_bit=1` today: the emitter and lowering accept
  it (`scheduler/emitter.py:840`, `mlir/lowering.py:1693,1812,2412` default 0), but
  no `InstructionTemplate` sets it (grep: no `mode_bit=1` call site outside the
  emitter's validation), and `_AUXILIARY_DTYPES` (§4.0 E2) means i32 *data* has no
  compute dtype today. So E, like D, needs a **new lowering** from i32 data to an
  rfp32 SFP op sequence — architecture cost, not a patch. It is **cheaper per element
  than D** (1 FMA vs 9 int8 MACs + carry) *if* the premise holds; the M=1 throughput
  penalty versus DL16 is undocumented and must be measured.
- **Falsifier (cheap, and already designed):** if `m1-arith-width` / `p1-icvt-m1`
  shows M=1 arithmetic is half-wise DL16 (or bit 44 ignored), E is refuted and D (or
  A) stands. Secondary falsifier: the new i32→rfp32 lowering cannot be emitted on the
  pointwise route, or M=1 halves throughput enough that A's fallback wins.

**Branch ranking, stated plainly:** E **dominates D** on op count *if* the M=1
premise holds (1 FMA vs 9 MACs); D does not depend on an unverified mode bit. A is
the current position. **E and D both require new lowerings; neither is recommended
until the `m1-arith-width` probe and the USER's decision.**

**E3 (RTL) selected Branch A; E4 (device KAT) is owed** to confirm on 1p0 silicon
that an i32 data `muli`/`addi` on the emitted route refuses or is inexact — E3 is
1p5 source, E4 is the 1p0 witness. **Branches D and E are the open alternatives** and
are not decided by E3: D by a device probe of the limb-convolution cost plus PT
reachability; **E by the already-designed `m1-arith-width` / `p1-icvt-m1` probe
(whether M=1 FMA executes as rfp32 or as half-wise DL16) plus i32→rfp32 lowering
reachability.**

### 4.2 Per-op bound under Branch A (and the bound law under B / C / D)

`E` is a single named constant **per op class**, derived from the selected branch,
never a baked-in 2^31. Check the running intermediate, not just the final tensor
(USER condition 2):

- **accumulation** (`cumsum`, `sum`, `prod`): bound `N * max|term| <= E_op`. Attention
  -mask cumsum (terms in {0,1}) <= seq_len: safe **only** if seq_len <= `E_op`; at
  RoPE 128K it is NOT safe under the DL16 FMA (exact ~2^10) -> CPU fallback.
- **product** (`mul`; fused address expressions): `a*b <= E_op` even when a,b fit.
  The affine-index form `i*stride+j` overflows the product with in-range `i,j,result`
  — but this is **address math** (§5), executed natively in the address path; bound
  the product against the *address* width, not the element width.
- **elementwise** (`add`, `sub`, `eq`, `lt`, `clamp`): operand bounds; DL16 FMA range
  (~2^10 exact) for data, address width for index.
- **op classes where mid-chain overflow can occur with in-range endpoints:**
  accumulation, product, and affine index computation (`i*stride+j`) — a
  final-tensor-only check passes them incorrectly (condition 2).

Per branch: **A** — `E_op` ~2^10 exact for data (DL16 FMA), unlimited in the address
path; values above → CPU fallback. **B** (not reachable on E3, kept for E4) —
`E_op = 2^(W-1)`. **C** — one constant from the PT int8/int24 exact range gates
every carrier (not 2^15 — no int16 lane); rest refuse typed. **D** — `E_op` is the
limb-decomposition bound: exact for any `R` with `k = ceil(log2(R+1)/8)` limbs, at
`k²` int8 products + carry chain per element. **E** — `E_op = 2^24` per rfp32 word
(24-bit significand), exact iff every intermediate stays below it; 16-bit halves only
where the bound fails or a 32-bit bitwise op is needed — **but E's premise (M=1
executes as rfp32) is unverified pending `m1-arith-width`.**

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

- **P0 (E3 SELECTS Branch A; E4 device KAT owed):** §4.0 records E1 (ISA doc) + E2
  (codegen classification) + **E3 (RTL `../RTL/hw1p5` @ `9c15c33d81`, a labelled
  1p5→1p0 inference)** establishing **no native int32 data arithmetic and no PE/SFP
  integer datapath**. E3 selected Branch A. **E4 still owed:** run an i32 data
  `muli`/`addi` on the emitted route and record whether it refuses or lowers
  inexactly — the 1p0 silicon witness E3 cannot be. **Branch D is decided
  separately:** a device probe of the PT limb-convolution (9 int8 MACs + carry per
  logical multiply) versus the Branch-A host round trip, plus a reachability check
  of PT from the pointwise route. **Branch E is decided by the already-designed
  `m1-arith-width` / `p1-icvt-m1` probe** (`emulator/rng_probe_catalog.py:115-116,1277`;
  run sheet prepared, not yet run) — does M=1 FMA execute as rfp32 or as half-wise
  DL16? If rfp32, E dominates D (1 FMA vs 9 MACs) and becomes the candidate; if
  half-wise DL16, E is refuted. Do not write Stage-2 code until the USER rules on
  A vs D vs E.
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
- Do not reintroduce the refuted int16 premise (§4.0 corrections); int8/int4→int24
  is the only exact integer arithmetic.
- Do not recommend Branch D — it is written with its cost so the A-vs-D choice is
  explicit; the USER rules.
- Do not cite the RNG recipe's 120,900-vector bit-identity as proof that rfp32 M=1
  FMA is real on silicon. It is bit-exact under the *model's* RFP32 assumption; the
  M=1 semantics is a spec self-conflict settled only by the `m1-arith-width` probe
  (Branch E's premise).
- 1.5 (`SEN1P5`) encoding is out of scope (E3 RTL facts are used only as a labelled
  inference about 1p0).

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
