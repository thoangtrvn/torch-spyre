"""Torch-free embedding launch planner.

Turns embedding shape + flat token indices into per-launch init-packet binaries
by calling the hardware-verified sentient_codegen embedding path. Imports ONLY
sentient_codegen + stdlib (NO torch / torch_spyre) so it is unit-testable on a
host without the spyre C extension. The torch/device wrapper lives in
torch_spyre/ops/embedding.py.

Supports multi-stick d_model:
  - d_model <= 512 (spt <= 8): EBR path (plane-major, hardware-verified PM-T6).
    tokens_per_core = floor(EBR / spt), EBR=8 is the binding limit (tighter
    than LAR 16). Max d_model per EBR launch = EBR * elements_per_stick = 512.
  - d_model > 512 (spt > 8): IBR path (row-contiguous weight, LDIM indirect
    gather). Per-launch token ceiling = 32 (IBR file size), not EBR-8.
    Tokens > 32 loop/multi-launch identical to EBR-path token chunking.

IBR PATH WEIGHT CONVENTION:
  The EBR path passes (vocab, d_model) weight directly — stickified plane-major.
  The IBR path requires row-contiguous layout: token t's spt sticks at positions
  [t*spt, (t+1)*spt). The host achieves this by reshaping (vocab, d_model) →
  (vocab*spt, 64) before the H2D copy.  A (M, 64) tensor stickifies one-stick-
  per-row (SpyreTensorLayout([M, 64], float16) → stride_map=[64, 1]), giving the
  row-contiguous layout LDIM needs.  d_model must be a multiple of 64 (elements
  per stick at 16-bit); non-multiples must be padded to the next multiple before
  calling (most LLM d_model values are multiples of 64 or 128).
"""
from __future__ import annotations

import math


class EmbeddingHostError(RuntimeError):
    """Raised when sentient_codegen is unreachable or embedding cannot be built."""


# IBR file size (hardware constant: LDIM populate reads 32 × 32-bit entries).
# This is the per-launch token ceiling for the IBR path.
_IBR_FILE_SIZE = 32

# Build-once cache for the cacheable looped IBR binary + its token-count patch
# model, keyed by (target, d_model, element_bits, out_segment).  The binary depends
# ONLY on geometry (token indices live in the separate IBR-table tensor, NOT in the
# binary — verified: plan_and_build_embedding_ibr_binaries never passes slice_indices
# to schedule/compile), so one cached entry serves every batch and token count at
# that geometry.  Per call, the launch binary is produced by patching the loop count
# (~microseconds) instead of re-scheduling + re-compiling (~2.6 ms/call) — the win
# that made nn.Embedding(d=2048) timeouts go away.  Module-level (process-lifetime)
# cache: entries are immutable bytes + a small patch model; no eviction needed (one
# entry per distinct (d_model, element_bits) the model uses, typically 1-2).
_IBR_LOOPED_CACHE: dict = {}

# EBR path d_model ceiling (inclusive): tokens with d_model <= this value use
# the EBR plane-major gather path (hardware-verified PM-T6).  d_model > this
# value uses the IBR indirect gather path.
_EBR_D_MODEL_MAX = 512

# IBR single-chunk d_model boundary (inclusive): the IBR gather reads a token row
# in ceil(spt/32) burst chunks, where spt = d_model / elements_per_stick.  A
# single LDIM burst covers at most 32 sticks (BURST field max), so d_model up to
# 32 * elements_per_stick = 2048 (at 16-bit) needs only ONE chunk per token (C=1);
# d_model > 2048 needs C>1 chunks.  This constant marks that C=1/C>1 boundary — it
# is INFORMATIONAL, not a raise threshold (no code path gates on it; d_model>2048
# routes to the multi-chunk builder in _build_embedding_launches_ibr, it does NOT
# raise).
#
# BOTH paths are HARDWARE-VERIFIED on AIU 1.0 (max_diff=0):
#   - C=1  (d_model <= 2048): IBR T7 — real aten.embedding d=768, d=2048.
#   - C>1  (d_model  > 2048): 2026-07-11 — real aten.embedding d=4096 (C=2, N=2/4),
#     d=5120 (C=3, N=2), d=8192 (C=4, N=2/3), all max_diff=0 across every token.
#     The OLD C>1 emission (distinct EAR[j]=32*j + addlarimm between chunks) HUNG
#     the device; the fix walks the C 32-stick chunks with a single-EAR0/LAR0 `.u`
#     gather each (deeptools single-gather model), earimm EAR0,0 resetting the
#     intra-row base per token while LAR0 walks the deposit continuously.  See
#     scheduler/embedding.py::embedding_ibr_pattern (the UNROLLED, dispatched form).
_IBR_SINGLE_CHUNK_D_MODEL_MAX = 2048


def build_ibr_address_table(
    indices: list[int],
    spt: int,
    segment: int,
    seg_bits: int,
) -> "object":  # returns numpy.ndarray[int32] or a list of int (if numpy absent)
    """Build a (1, 32) int32 IBR address table for an IBR-path launch.

    The LDIM populate instruction reads one 128-byte HBM flit (32 × 32-bit
    integers) into the hardware IBR register file.  Entry t is the absolute
    HBM stick address of token indices[t]'s base row in the source tensor:

        IBR[t] = (segment << seg_bits) + indices[t] * spt

    where:
      - ``segment`` is the XLAT segment index of the weight tensor (1 in the
        standard 3-tensor layout: ibr_table=0, weight=1, output=2).
      - ``seg_bits`` is the segment-size-bits constant from the generation
        (27 for rcudd1a; from ``gen.hw.segment_size_bits``).
      - ``spt`` = sticks_per_token = ceil(d_model / elements_per_stick).
      - Unused entries (len(indices) .. 31) are 0.

    Returns a numpy.ndarray of shape (1, 32) and dtype int32.  Importable
    from here (torch_spyre.ops._embedding_host) so tests can use it without
    needing embedding.py's torch import.

    Reuse for KV block tables: a KV block-table lookup uses the same IBR
    construction — given block indices and sticks-per-block (spt), the same
    function returns absolute stick addresses for the IBR populate.

    Raises:
        ValueError: if ``len(indices) > _IBR_FILE_SIZE``, ``spt < 1``, or
                    ``segment < 0``.
    """
    try:
        from sentient_codegen.encoder.embedding_dispatch import (
            build_ibr_address_table as _cg_build,
        )
    except ImportError as e:
        raise EmbeddingHostError(
            "sentient_codegen not importable — source env.sh so codegen is "
            "on PYTHONPATH (required for build_ibr_address_table)."
        ) from e

    return _cg_build(indices, spt, segment=segment, seg_bits=seg_bits)


def _tokens_per_core(ebr_count: int, lar_count: int, sticks_per_token: int) -> int:
    """Plane-major register-budget ceiling for tokens per core.

    Returns floor(min(ebr_count, lar_count) / sticks_per_token).  EBR (8) is
    the binding limit on AIU 1.0 (tighter than LAR 16), so for standard
    hardware ebr_count=8, lar_count=16:

      spt=1  → 8 tokens/core  (single-stick, unchanged from old model)
      spt=2  → 4 tokens/core  (d=128 at 16-bit)
      spt=4  → 2 tokens/core  (d=256 at 16-bit)
      spt=8  → 1 token/core   (d=512 at 16-bit, maximum spt under this model)
      spt=16 → 0              (d=1024 exceeds EBR file; caller raises)

    This is the canonical formula for the PM-T3 plane-major model.  It is
    called by build_embedding_launches and tested directly in
    test_embedding_host.py to keep the coverage non-tautological.
    """
    return min(ebr_count // sticks_per_token, lar_count // sticks_per_token)


def _build_embedding_launches_ibr(
    d_model: int,
    element_bits: int,
    flat_idx: list[int],
    gen,
) -> tuple[list[tuple[tuple[int, int], bytes]], int]:
    """Build IBR-path launches for d_model > _EBR_D_MODEL_MAX.

    Per-launch token ceiling = _IBR_FILE_SIZE=32 (IBR file size).
    Tokens > 32 are split into multiple launches (identical to EBR-path chunking).

    Returns (launches, tokens_per_launch) where tokens_per_launch <= 32.

    This is a private helper called by build_embedding_launches.  It is factored
    here so plan_and_build_embedding_ibr_binaries can be called directly for
    advanced use cases (e.g. pre-built IBR table reuse).
    """
    try:
        from sentient_codegen.encoder.embedding_dispatch import (
            build_looped_ibr_binary_cached,
            plan_and_build_embedding_ibr_binaries_cached,
            plan_and_build_embedding_ibr_binaries,
        )
    except ImportError as e:
        raise EmbeddingHostError(
            "sentient_codegen not importable — source env.sh so codegen is "
            "on PYTHONPATH (required for IBR embedding dispatch)."
        ) from e

    # Chunks-per-token: the IBR gather reads a token row in ceil(spt/32) burst
    # chunks (L3 BURST field max = 32 sticks).  This selects the emission path.
    elements_per_stick = gen.hw.stick_bytes * 8 // element_bits   # 64 at 16-bit
    spt = math.ceil(d_model / elements_per_stick)
    chunks_per_token = math.ceil(spt / _IBR_FILE_SIZE)            # C = ceil(spt/32)

    tokens_per_launch = min(_IBR_FILE_SIZE, len(flat_idx))

    if chunks_per_token > 1:
        # MULTI-CHUNK PATH (C>1, d_model>2048): a token row spans C 32-stick chunks.
        # The single-EAR0/LAR0 `.u`-walk UNROLLED gather (embedding_ibr_pattern in
        # codegen) covers the whole row per token — the deeptools single-gather
        # model (dcgbeCodegen.cpp:1517-1526).  The OLD distinct-EAR[j] emission and
        # the still-unimplemented looped C>1 form both HANG AIU 1.0, so the build-
        # once+patch cached-looped fast path (below) is C=1 ONLY — route C>1 to the
        # unrolled builder, which re-schedules per launch (fine: C>1 launches are
        # few and overhead-bound).  HARDWARE-VERIFIED on AIU 1.0 (2026-07-11):
        # real aten.embedding d_model=4096 (C=2), 5120 (C=3), 8192 (C=4), tokens=2..4,
        # all max_diff=0 across every token.
        launches = plan_and_build_embedding_ibr_binaries(
            flat_idx, d_model, gen,
            element_bits=element_bits,
            tokens_per_launch=tokens_per_launch,
            out_segment=2,  # 3-tensor layout: weight=XLAT[0], ibr=XLAT[1], output=XLAT[2]
        )
        return launches, tokens_per_launch

    # Build-ONCE + patch: the cacheable looped binary depends only on geometry, so
    # build it once per (target, d_model, element_bits, out_segment) and patch the
    # loop count per launch (~µs) instead of re-compiling per call (~2.6 ms).  Each
    # patched launch binary is byte-identical to a native looped build at that
    # launch's token count (hardware-verified: build-once+patch probe + the
    # token1-empty .u-walk fix).  Indices are NOT in the binary (they ride the
    # separate IBR-table tensor), so the cache is correct across batches/counts.
    out_segment = 2  # 3-tensor layout: weight=XLAT[0], ibr=XLAT[1], output=XLAT[2]
    cache_key = (gen.senarch, d_model, element_bits, out_segment)
    entry = _IBR_LOOPED_CACHE.get(cache_key)
    if entry is None:
        entry = build_looped_ibr_binary_cached(
            d_model, gen, element_bits=element_bits, out_segment=out_segment,
        )
        _IBR_LOOPED_CACHE[cache_key] = entry
    cached_binary, patch_model = entry

    launches = plan_and_build_embedding_ibr_binaries_cached(
        flat_idx, d_model, gen,
        element_bits=element_bits,
        tokens_per_launch=tokens_per_launch,
        out_segment=out_segment,
        cached_binary=cached_binary,
        patch_model=patch_model,
    )
    return launches, tokens_per_launch


def build_embedding_launches(
    vocab: int,
    d_model: int,
    element_bits: int,
    flat_idx: list[int],
    target: str = "rcudd1a",
    max_cores: int = 32,
) -> tuple[list[tuple[tuple[int, int], bytes]], int]:
    """Plan + build per-launch embedding binaries for a flat list of token indices.

    Returns (launches, tokens_per_launch) where:
      - launches is list[((start, end), binary_bytes)] covering [0, len(flat_idx)).
      - tokens_per_launch is num_cores * K — the number of output sticks each
        binary writes regardless of how many real tokens are in the final launch.
        The caller must allocate a tokens_per_launch-sized buffer for each launch
        and copy only the real rows (buf[:end-start]) into the output tensor.

    Each binary is built for exactly tokens_per_launch output slots; passing a
    shorter slice (as the old API did for the final partial launch) causes the
    binary to overrun the slice and corrupt adjacent memory.

    Raises EmbeddingHostError if sentient_codegen is not importable (env.sh not
    sourced / codegen not on PYTHONPATH). Raises ValueError for invalid indices
    (out of vocab, negative) — propagated from the codegen address math.
    """
    try:
        from sentient_codegen.scheduler.engine import schedule, TilingConfig
        from sentient_codegen.encoder.embedding_dispatch import (
            plan_and_build_embedding_binaries,
        )
        from sentient_codegen.gen.registry import get_generation
    except ImportError as e:
        raise EmbeddingHostError(
            "sentient_codegen not importable — source env.sh so $DTI_PROJECT_ROOT/"
            "codegen is on PYTHONPATH (the embedding op needs the codegen package)."
        ) from e

    if not flat_idx:
        return [], 0

    # Bounds check: indices must be in [0, vocab).
    for idx in flat_idx:
        if idx < 0 or idx >= vocab:
            raise ValueError(f"token index {idx} out of range [0, vocab={vocab})")

    gen = get_generation(target)
    ebr_count = gen.get_unit_spec("L3LU").registers["EBR"].count  # 8 on 1p0
    lar_count = gen.get_unit_spec("L3LU").registers["LAR"].count  # 16 on 1p0
    elements_per_stick = gen.hw.stick_bytes * 8 // element_bits   # 64 at 16-bit (DL16)

    sticks_per_token = math.ceil(d_model * element_bits / 8 / gen.hw.stick_bytes)
    N = len(flat_idx)

    # PATH SELECTION: d_model <= _EBR_D_MODEL_MAX (512) → EBR path (PM-T6, proven).
    #                 d_model >  _EBR_D_MODEL_MAX (512) → IBR path (Task 5).
    #
    # EBR path (d_model <= 512, spt <= 8):
    #   Multi-stick embedding is HARDWARE-VERIFIED on AIU 1.0 as of PM-T6 (2026-06-21).
    #   All four plane-major bugs fixed and confirmed on silicon (max_diff=0):
    #   (1) plane-major SOURCE addressing, (2) per-(token,plane) distinct-EBR gather,
    #   (3) plane-major OUTPUT scatter, (4) wide-row pointwise N-tiling.
    #   Hardware ladder: 2-tok spt=2, 4-tok spt=2, 2-tok spt=4 (d=256), multi-core
    #   2c×2tok all PASS max_diff=0.
    #   tokens_per_core = floor(ebr_count / spt). EBR (8) binds tighter than LAR (16).
    #   For spt=1: tokens_per_core = min(8, 16) = 8 (unchanged).
    #
    # IBR path (d_model > 512, spt > 8):
    #   Replaces the EBR-8 ceiling with the IBR-file ceiling (32 tokens/launch).
    #   Single-core launch; multi-launch for >32 tokens (same as EBR chunking).
    #   Weight must be passed row-contiguous (reshaped in embedding.py).
    #   Offline-validated (ibr_probe_emulator_check.py, Task 4).
    #   Device verification: ibr_t5_device.py (pending controller run, Task 7).
    if d_model > _EBR_D_MODEL_MAX:
        # IBR path: dispatch to indirect gather.
        return _build_embedding_launches_ibr(d_model, element_bits, flat_idx, gen)

    # EBR path: plane-major gather (d_model <= 512).
    tokens_per_core = _tokens_per_core(ebr_count, lar_count, sticks_per_token)
    if tokens_per_core < 1:
        # This branch is unreachable for d_model <= 512 (spt <= 8 <= ebr_count=8),
        # but kept as a safety net for non-standard hardware with smaller EBR files.
        raise NotImplementedError(
            f"embedding: d_model={d_model} → sticks_per_token={sticks_per_token}; "
            f"one token's planes ({sticks_per_token}) exceed the L3LU EBR file "
            f"({ebr_count}). Max d_model per EBR launch = "
            f"{ebr_count * elements_per_stick} at {element_bits}-bit. "
            f"Use d_model > {_EBR_D_MODEL_MAX} to trigger the IBR path."
        )

    # sticks_per_token is forwarded to the scheduler via tile_n=d_model;
    # here we only need the launch geometry.
    K = min(tokens_per_core, N)
    num_cores = min(max_cores, math.ceil(N / K))
    tokens_per_launch = num_cores * K  # tile_m = total tokens (load-bearing tiling rule)

    scheduled = schedule(
        "embedding",
        TilingConfig(M=tokens_per_launch, K=1, N=d_model, tile_m=tokens_per_launch,
                     tile_k=1, tile_n=d_model, num_cores=num_cores,
                     element_bits=element_bits),
        target,
    )
    if scheduled.address_spec is None:
        raise EmbeddingHostError("schedule('embedding') produced no address_spec")

    launches = plan_and_build_embedding_binaries(
        scheduled, flat_idx, gen, num_cores=num_cores, tokens_per_core=K,
        vocab=vocab,
    )
    return launches, tokens_per_launch
