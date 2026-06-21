"""Torch-free embedding launch planner.

Turns embedding shape + flat token indices into per-launch init-packet binaries
by calling the hardware-verified sentient_codegen embedding path. Imports ONLY
sentient_codegen + stdlib (NO torch / torch_spyre) so it is unit-testable on a
host without the spyre C extension. The torch/device wrapper lives in
torch_spyre/ops/embedding.py.

Supports multi-stick d_model: the per-launch token capacity is derived from the
L3LU register file (EBR count and LAR count) and the number of load chunks C
required per token row (ceil(sticks_per_token / L3_BURST_MAX)).
"""
from __future__ import annotations

import math


class EmbeddingHostError(RuntimeError):
    """Raised when sentient_codegen is unreachable or embedding cannot be built."""


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

    gen = get_generation(target)
    ebr_count = gen.get_unit_spec("L3LU").registers["EBR"].count  # 8 on 1p0
    lar_count = gen.get_unit_spec("L3LU").registers["LAR"].count  # 16 on 1p0
    elements_per_stick = gen.hw.stick_bytes * 8 // element_bits   # 64 at 16-bit (DL16)
    L3_BURST_MAX = 32  # L3.BURST_MAX — sticks per ld.hbm burst

    # Per-launch capacity (single source of truth, matches the codegen scheduler):
    #   spt = sticks per token row; C = load instructions per token row.
    #   A token consumes 1 EBR (base) + C EAR (chunk offsets) + C LAR (deposits),
    #   so tokens_per_core is bounded by EBR (8) and by LAR // C.
    sticks_per_token = math.ceil(d_model * element_bits / 8 / gen.hw.stick_bytes)
    C = math.ceil(sticks_per_token / L3_BURST_MAX)
    N = len(flat_idx)

    # HARDWARE-BLOCKED (2026-06-21): multi-stick embedding (sticks_per_token > 1)
    # produces SCRAMBLED output on AIU 1.0 silicon (hardware-confirmed: d=128
    # 2-stick token → values from wrong (token,stick) positions; d=2048 → device
    # hang). Root cause (diagnosed): embedding's copy/output pipeline
    # (LXLU→SFP→LXSU→L3SU) is the FLAT pointwise dataflow, which walks LX as one
    # contiguous sticks_per_core run with a single stride — but the L3LU gather
    # DEPOSITS per-token with the source stride baked into EBR/EAR, so the two LX
    # layout views disagree for spt>1. spt=1 (d_model==elements_per_stick) works
    # only because 1 stick == 1 token makes both views coincide. The real fix is a
    # per-token / spt-stick-granular pipeline matching the gather deposit layout
    # (its own codegen effort). Scratchpad-overlap (separate input/output regions)
    # and wide-row pointwise N-tiling are FIXED + verified, but are not sufficient.
    # Reject multi-stick loudly rather than ship scrambled/hanging output.
    if sticks_per_token > 1:
        raise NotImplementedError(
            f"embedding: d_model={d_model} needs sticks_per_token={sticks_per_token} "
            f"(>1). Multi-stick embedding is HARDWARE-BLOCKED on AIU 1.0: the "
            f"copy/output pipeline walks LX contiguously while the gather deposits "
            f"per-token, scrambling output for spt>1 (d=128 hardware-confirmed). "
            f"Only single-stick d_model <= {elements_per_stick} ({element_bits}-bit) "
            f"is hardware-verified. The per-token pipeline-layout fix is the "
            f"remaining blocker (separate codegen effort)."
        )
    tokens_per_core = min(ebr_count, lar_count // C)
    if tokens_per_core < 1:
        raise NotImplementedError(
            f"embedding: d_model={d_model} → sticks_per_token={sticks_per_token}, "
            f"C={C} chunks/token; one token's row exceeds the L3LU LAR file "
            f"({lar_count}). Max supported sticks_per_token={lar_count * L3_BURST_MAX} "
            f"(d_model≈{lar_count * L3_BURST_MAX * elements_per_stick})."
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
    # Bounds: indices must be in [0, vocab). compute_embedding_addresses also checks
    # against spec.vocab_size, but the spec is built without vocab; enforce here.
    for idx in flat_idx:
        if idx < 0 or idx >= vocab:
            raise ValueError(f"token index {idx} out of range [0, vocab={vocab})")

    launches = plan_and_build_embedding_binaries(
        scheduled, flat_idx, gen, num_cores=num_cores, tokens_per_core=K,
    )
    return launches, tokens_per_launch
