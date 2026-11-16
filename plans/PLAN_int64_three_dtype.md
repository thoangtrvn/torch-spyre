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

**Exactness bound `E`** = the largest magnitude the backend preserves exactly for
integer add/mul:
`E = INT32_INT_EXACT` (2^31) if the §7-P0 probe shows true-integer execution, else
`E = FP32_MANTISSA_EXACT` (2^24) if it shows an fp32 pivot. **`E` is a single named
constant derived from the probe, not baked in.** Condition 2 is checked against the
running intermediate, not only endpoints.

Value evidence (upstream #1547): embedding ~200K, RoPE position >=128K context,
KV-cache = addressable space (an address, not a value — §5), MoE ~16 experts. Every
**value** carrier is ~4 orders below 2^31 (and below 2^24), so the cited carriers are
safe under either `E`; the contract must still prove it per origin.

Per op class (check the running intermediate):

- **accumulation** (`cumsum`, `sum`, `prod`): bound `N * max|term| <= E`. Attention
  -mask cumsum (terms in {0,1}) <= seq_len: safe. A general int32 accumulation can
  exceed `E` mid-chain with in-range inputs and outputs -> **refuse typed unless
  `N * max|term| <= E` is proven.**
- **product** (`mul`; fused address expressions): `a*b <= E` even when a,b fit. The
  affine-index form `i*stride+j` is the canonical case — a large `stride` overflows
  the product with in-range `i,j,result`. Bound the **product**, not the factors.
- **elementwise** (`add`, `sub`, `eq`, `lt`, `clamp`): operand bounds; safe for the
  cited carriers, but must be proven per origin.
- **op classes that can overflow mid-chain while inputs/outputs fit, and therefore
  need explicit rules:** accumulation, product, and affine index computation
  (`i*stride+j`). A final-tensor-only check passes these incorrectly (USER condition 2).

**Placement:** the bound is a symbolic claim discharged at the Inductor boundary from
the §3 origin taxonomy; undischarged -> Stage-3 refusal. It is **not** a runtime
clamp.

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
| int64 data reaches a kernel arg after Stage 2 (defense in depth) | codegen descriptor | `I64PointerDataError` (`_descriptor.py:502-522`) |
| int64 tensor-descriptor element type | codegen descriptor | `I64DescriptorDataError` (`_descriptor.py:444-496`) |
| unsupported int64 op, fallback chosen | supported partition, CPU fallback + `FallbackWarning` (today's #4317 path, `fallbacks.py:304`) | warning + CPU partition |
| fallback attempted with a CPU-resident intermediate | `spyre::to_dtype_cpu` (PrivateUse1-only, #5070) | `NotImplementedError` |
| no host data channel (codegen-only) | runtime | `SpyreRuntimeUnavailableError` (`torch_spyre/_runtime.py:27`) |

After Stage 2, `I64PointerDataError`/`I64DescriptorDataError` must be
defense-in-depth only, never the primary user-facing refusal.

---

## 7. Falsification plan (plan-first; no code)

- **P0 (first, Stage 2 gate): exactness probe.** Determine whether
  `arith.muli : tensor<32xi32>` and i32 add execute as true integer or via an fp32
  pivot in this backend; fix `E` (§4) from the result. Everything in §4 depends on it.
  Until P0 answers, `E` is treated as the conservative `FP32_MANTISSA_EXACT`.
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
