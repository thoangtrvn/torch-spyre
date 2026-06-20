"""Torch-free embedding launch planner.

Turns embedding shape + flat token indices into per-launch init-packet binaries
by calling the hardware-verified sentient_codegen embedding path. Imports ONLY
sentient_codegen + stdlib (NO torch / torch_spyre) so it is unit-testable on a
host without the spyre C extension. The torch/device wrapper lives in
torch_spyre/ops/embedding.py.
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
) -> list[tuple[tuple[int, int], bytes]]:
    """Plan + build per-launch embedding binaries for a flat list of token indices.

    Returns list[((start, end), binary_bytes)] covering [0, len(flat_idx)).
    Each binary gathers flat_idx[start:end] on (end-start)-capped cores; the caller
    writes its output rows [start:end].

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
        return []

    gen = get_generation(target)
    ebr_count = gen.get_unit_spec("L3LU").registers["EBR"].count  # 8 on 1p0
    elements_per_stick = gen.hw.stick_bytes * 8 // element_bits   # 64 for DL16
    # sticks_per_token is folded into the schedule via tile_n=d_model; here we only
    # need the launch geometry. K = tokens/core, capped by ebr_count.
    N = len(flat_idx)
    K = min(ebr_count, N)
    num_cores = min(max_cores, math.ceil(N / K))
    total = num_cores * K  # tile_m = total tokens (the load-bearing tiling rule)

    scheduled = schedule(
        "embedding",
        TilingConfig(M=total, K=1, N=d_model, tile_m=total,
                     tile_k=1, tile_n=d_model, num_cores=num_cores),
        target,
    )
    if scheduled.address_spec is None:
        raise EmbeddingHostError("schedule('embedding') produced no address_spec")
    if scheduled.address_spec.vocab_size and vocab > 0:
        # vocab bound check happens in compute_embedding_addresses; this is belt-and-suspenders
        pass
    # Bounds: indices must be in [0, vocab). compute_embedding_addresses also checks
    # against spec.vocab_size, but the spec is built without vocab; enforce here.
    for idx in flat_idx:
        if idx < 0 or idx >= vocab:
            raise ValueError(f"token index {idx} out of range [0, vocab={vocab})")

    return plan_and_build_embedding_binaries(
        scheduled, flat_idx, gen, num_cores=num_cores, tokens_per_core=K,
    )
