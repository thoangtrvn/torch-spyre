# RNG host↔device contract — FROZEN 2026-11-19 (T=1)

**Purpose.** The single interface the torch-spyre host lane (A) and the codegen
backend lane (B) BOTH build to, so their halves meet deterministically instead of
diverging into a silent params-format mismatch. Freeze this; do not re-invent
either half. Grounded in the IMPLEMENTED reference (the silicon-witnessed KAT),
not doc prose.

**Scope.** T=1, C=1, 32 lanes, rounds=20, **Threefry2x32-20**, single core, RAW
state egress. T≥2 / the uniform fp32-grid user-facing output are OUT (T-DV-34 /
P5 unbuilt). This CONNECTS the already-silicon-witnessed raw-state KAT — it does
not re-derive or re-witness it.

## Reference (the real implementation — cite, don't paraphrase)
- Host stream builder: `sentient_codegen.emulator.threefry_recipe_model.build_band_lxlu_stream(blocks, *, x0_base, x1, rotations, injections, rounds, setup_staging=SetupStaging.LXLU, egress=EgressMode.RAW_STATE) -> tuple[StreamWord, ...]` (`threefry_recipe_model.py:1935`). `blocks` = per-lane counter indices, **uint32, all < 2^23** (rfp32 exact; split the launch otherwise).
- Counts (rounds=20): `band_lxlu_pop_count(20, RAW_STATE)=251`, `band_egress_stick_count(20, RAW_STATE)=4`. (UNIFORM: 257 / 2.)
- Kernel factories: `scheduler/rng.py` — `rng_kat_pattern_raw_egress_offline(tiling)` (raw) / `rng_kat_pattern_offline` (uniform); production `rng_kat_pattern` (refuses until admitted). `RNG_KAT_ROUNDS=20` (`scheduler/ir.py:594`), `_RNG_KAT_LANES=32`.
- Silicon receipt of record: raw binary sha256 **`e697b657d14269f219626f9e7418b29c223332e2e8eb03a61c3e091f4d263d9d`** (4352 B, 4 raw egress sticks, 251 pops), receipt set `silicon-71dcf6e2ef41`, three published Random123 2x32-20 anchors bit-exact.

## Host → params stream (lane A computes this)
- Derive host-side from `(seed, offset/block)` per doc 43 §3.1 (host-only k′): `x0_base`, `x1`, `rotations[r] = ROTATIONS_2X32[r % 8]`, `injections = injection_words_2x32((k0,k1), rounds)` (length `rounds//4 = 5`).
- The params stream = the **251 `StreamWord`** tuple (each = 32 uint32 lanes) in recipe order: 4 setup-constant sticks (`0x4B00FFFF, 0x4B7F0000, 0x37800000, 0x4B000000`), then COUNTER, then the per-round body. **Lane A must port `build_band_lxlu_stream` byte-for-byte** (the kernel consumes exactly this stream); a divergence here is the failure mode this contract exists to prevent.

## Launch ABI
- **Input** params tensor: the 251-stick LXLU stream, uint32 × 32 lanes, at HBM **segment 0** (`declared_segment_count=2`, `xlat_segments {input:0, output:1}`).
- **Output**: RAW_STATE = **4 sticks × 32 lanes**; each word = `0x4B000000 | half`, `half ∈ [0, 0x10000)`. Host reconstructs `x0=(x0h<<16)|x0l`, `x1=(x1h<<16)|x1l`. (UNIFORM = 2 sticks, `u = hi·2⁻¹⁶ + floor(lo/256)·2⁻²⁴` — **the user-facing grid is UNWITNESSED; do NOT claim it**.)

## Honesty bounds (both lanes)
- T=1 only; refuse T≥2 typed (DV-34 unbuilt). Do NOT claim the uniform grid (P5 owed) or shape generality.
- Registration exposes a **T=1 reduce route**; it does NOT change the production uniform epilogue (`ed41ff0a`) or admit T≥2.
- The grade/record work (u=2 SILICON_STRICT) is separate and already lands as its own member.

## How the two lanes meet
- **A** (`torch-spyre-thoangtrvn`): real `SpyreGeneratorImpl` (seed/offset) + honest `manual_seed`/`get`/`set_rng_state` + a lowering scaffold that produces THIS params stream, routes to `rng_kat`, and **refuses typed until B registers it** (and refuses T≥2 / uniform grid).
- **B** (`spyre-knowledgebase` codegen): register `rng_kat` (`PATTERN_REGISTRY` + factory + `resolve_pattern`), emitting the T=1 raw/band kernel, **refusing T≥2**, KAT-tested against `e697b657`.
- **Integration (separate):** A↔B connected + end-to-end T=1 KAT. Because A refuses until B lands, nothing half-connects — they meet only at this contract.
