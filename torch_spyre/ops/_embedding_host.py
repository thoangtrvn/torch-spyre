"""Torch-free embedding launch planner.

Turns embedding shape + flat token indices into per-launch init-packet binaries
by calling the hardware-verified sentient_codegen embedding path. Imports ONLY
sentient_codegen + stdlib (NO torch / torch_spyre) so it is unit-testable on a
host without the spyre C extension. The torch/device wrapper lives in
torch_spyre/ops/embedding.py.

Supports multi-stick d_model: the per-launch token capacity is derived from the
L3LU register file. Plane-major model (PM-T3): tokens_per_core = floor(EBR / spt),
where spt = sticks_per_token. EBR (8) is the binding limit (tighter than LAR 16).
Max d_model per launch = EBR * elements_per_stick = 512 at 16-bit.
"""
from __future__ import annotations

import math


class EmbeddingHostError(RuntimeError):
    """Raised when sentient_codegen is unreachable or embedding cannot be built."""


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

    # Per-launch capacity (plane-major model, matches the codegen scheduler PM-T3):
    #   spt = sticks per token row (= planes per token).
    #   Each (token, plane) = 1 EBR deposit + 1 LAR deposit.
    #   tokens_per_core = floor(ebr_count / spt) — EBR (8) binds tighter than LAR (16).
    #   Max d_model per launch: ebr_count * elements_per_stick = 8 * 64 = 512 at 16-bit.
    sticks_per_token = math.ceil(d_model * element_bits / 8 / gen.hw.stick_bytes)
    N = len(flat_idx)

    # Multi-stick embedding (sticks_per_token > 1) is HARDWARE-VERIFIED on AIU 1.0
    # as of PM-T6 (2026-06-21). The bug had four facets, all fixed and confirmed
    # on silicon (max_diff=0): (1) plane-major SOURCE addressing — token idx,
    # plane s reads HBM stick idx + s*vocab (was idx*spt); (2) per-(token,plane)
    # distinct-EBR ld.hbm gather (no .u — the patched EBR carries the full source
    # stick, so auto-increment on a shared EAR0 added a spurious +s drift);
    # (3) plane-major OUTPUT scatter — the (N, d_model) output is itself stickified
    # plane-major, so token t plane s is stored to output stick s*N + t (the L3SU
    # output EAR is seeded plane-major; was a contiguous store that transposed
    # tokens<->planes for multi-token launches); (4) wide-row pointwise N-tiling.
    # Hardware ladder (pm_t6_hw_verify.py): 2-tok spt=2, 4-tok spt=2, 2-tok spt=4
    # (d=256), and multi-core 2c×2tok all PASS max_diff=0. The remaining limit is
    # purely the EBR-file ceiling below (d_model > ebr_count*elements_per_stick),
    # not a correctness defect.
    #
    # Plane-major ceiling: floor(ebr_count / spt). EBR (8) binds tighter than LAR (16).
    # For spt=1: tokens_per_core = min(8, 16) = 8 (unchanged from old model).
    # For spt > ebr_count: ceiling < 1 → one token's planes exceed EBR file; needs a
    # different gather model (multi-launch over planes, not yet implemented).
    tokens_per_core = _tokens_per_core(ebr_count, lar_count, sticks_per_token)
    if tokens_per_core < 1:
        raise NotImplementedError(
            f"embedding: d_model={d_model} → sticks_per_token={sticks_per_token}; "
            f"one token's planes ({sticks_per_token}) exceed the L3LU EBR file "
            f"({ebr_count}). Max d_model per launch = "
            f"{ebr_count * elements_per_stick} at {element_bits}-bit. "
            f"Multi-plane launch (d_model > {ebr_count * elements_per_stick}) "
            f"needs a different gather model — not yet implemented."
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
        vocab=vocab,
    )
    return launches, tokens_per_launch
