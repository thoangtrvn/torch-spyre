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

# Module-level cache for looped IBR binaries.
#
# Key: (d_model: int, element_bits: int)
# Value: (cached_binary: bytes, patch_info: tuple[int, int, int])
#   where patch_info = (byte_offset, bit_lo, bit_hi) for patch_loop_count.
#
# The cached binary is built ONCE per (d_model, element_bits) at
# CACHED_TOKENS=32 (the full IBR file ceiling).  Per call, only the
# JCR_cnt jimmcopy IMM field is patched (4-byte surgery, ~microseconds)
# instead of rebuilding the entire init packet (seconds).  The weight
# reshape is also hoisted: see _ibr_weight_cache in embedding.py.
#
# Tests can clear this cache (``_ibr_binary_cache.clear()``) before
# asserting build-once behavior.
_ibr_binary_cache: dict = {}

# EBR path d_model ceiling (inclusive): tokens with d_model <= this value use
# the EBR plane-major gather path (hardware-verified PM-T6).  d_model > this
# value uses the IBR indirect gather path.
_EBR_D_MODEL_MAX = 512

# IBR single-chunk d_model ceiling (inclusive): the IBR gather reads a token row
# in ceil(spt/32) burst chunks, where spt = d_model / elements_per_stick.  A
# single LDIM burst covers at most 32 sticks (BURST field max), so d_model up to
# 32 * elements_per_stick = 2048 (at 16-bit) needs only ONE chunk per token (C=1).
# The C=1 path is HARDWARE-VERIFIED (IBR T7: real aten.embedding d=768 and d=2048,
# max_diff=0 on silicon).  The MULTI-chunk path (C>1, d_model > 2048) — which emits
# multiple LDIM gathers per token with addlarimm advances between chunks — is NOT
# yet hardware-verified: it HANGS the device on AIU 1.0 (IBR T7: d=4096 spt=64 C=2
# timed out while d=2048 returned in 6.3s).  Until the chunked-read path is fixed
# and verified, d_model > this value raises rather than hanging the device.
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

    Returns (launches, tokens_per_launch) where tokens_per_launch == _IBR_FILE_SIZE=32.

    CACHING DESIGN:
      The looped embedding_ibr binary is built ONCE per (d_model, element_bits)
      and stored in the module-level ``_ibr_binary_cache``.  Per call, only the
      JCR_cnt jimmcopy IMM field is patched (4-byte surgery, microseconds) to set
      the actual loop count (real tokens in that launch).  The binary is built at
      CACHED_TOKENS=32; the non-L3LU units (LXLU/SFP/LXSU/L3SU) always operate on
      the 32-token geometry; the caller reads only the first real_tokens rows of the
      output buffer.

    This is a private helper called by build_embedding_launches.
    """
    try:
        from sentient_codegen.encoder.embedding_dispatch import (
            build_looped_ibr_binary_cached,
            patch_loop_count,
            plan_embedding_passes,
            CACHED_TOKENS,
        )
    except ImportError as e:
        raise EmbeddingHostError(
            "sentient_codegen not importable — source env.sh so codegen is "
            "on PYTHONPATH (required for IBR embedding dispatch)."
        ) from e

    # Multi-chunk guard: the IBR gather reads a token row in ceil(spt/32) burst
    # chunks (BURST field max = 32 sticks).  The single-chunk path (C=1) is
    # hardware-verified; the multi-chunk path (C>1) HANGS the device on AIU 1.0
    # (IBR T7) and is not yet fixed.  Fail loud rather than hang.  Derive the
    # ceiling from elements_per_stick so it stays generation-correct.
    elements_per_stick = gen.hw.stick_bytes * 8 // element_bits   # 64 at 16-bit
    spt = math.ceil(d_model / elements_per_stick)
    chunks_per_token = math.ceil(spt / _IBR_FILE_SIZE)            # C = ceil(spt/32)
    if chunks_per_token > 1:
        max_d = _IBR_FILE_SIZE * elements_per_stick               # 2048 at 16-bit
        raise NotImplementedError(
            f"embedding: d_model={d_model} needs sticks_per_token={spt} → "
            f"chunks_per_token={chunks_per_token} (>1).  The multi-chunk IBR gather "
            f"path is not yet hardware-verified (it hangs the device on AIU 1.0). "
            f"Single-chunk IBR (d_model <= {max_d} at {element_bits}-bit) is "
            f"hardware-verified; the chunked-read path is a tracked follow-on."
        )

    # Build-once: retrieve the cached binary or build it on first call.
    cache_key = (d_model, element_bits)
    if cache_key not in _ibr_binary_cache:
        # Build the looped binary at CACHED_TOKENS=32 (one-time cost, seconds).
        # Subsequent calls for the same (d_model, element_bits) reuse this binary.
        cached_binary, patch_info = build_looped_ibr_binary_cached(
            d_model, gen, element_bits=element_bits, out_segment=2,
        )
        _ibr_binary_cache[cache_key] = (cached_binary, patch_info)

    cached_binary, patch_info = _ibr_binary_cache[cache_key]

    # tokens_per_launch is always CACHED_TOKENS=32 (the non-L3LU units encode
    # the 32-token geometry; partial final launches are padded by IBR-table zeros).
    tokens_per_launch = CACHED_TOKENS

    # Plan the launch slices (each slice is <= tokens_per_launch tokens).
    passes = plan_embedding_passes(
        len(flat_idx), num_cores=1, tokens_per_launch=tokens_per_launch,
    )

    launches: list[tuple[tuple[int, int], bytes]] = []
    for start, end in passes:
        n_real = end - start
        # Patch the cached binary with the actual token count for this launch.
        # This is a 4-byte surgery on the JCR_cnt jimmcopy IMM field (~microseconds).
        patched = patch_loop_count(cached_binary, n_real)
        launches.append(((start, end), patched))

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
