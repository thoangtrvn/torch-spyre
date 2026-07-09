"""Dedicated HW test for the embedding op (aten.embedding / torch.nn.Embedding on spyre).

PURPOSE
-------
One file per supported op, enumerating EVERY shape the op should handle — driven by
the REAL model configs we target — with a DL16-exact known-answer check on device
vs CPU. Shapes that are not yet hardware-verified are marked xfail/skip with the
exact reason and tracking note, so this file is the living contract for embedding
coverage: as each codegen fix lands, its xfail flips to a pass and the gap closes
visibly.

WHY known-answer (not random): AIU 1.0 uses DL16 (1-6-9) — integers > 1024 are not
represented exactly. So every expected value here is kept <= 1024 (see _dl16_exact
table build) to avoid spurious max_diff. See the project DL16 note.

SHAPE COVERAGE (the contract)
-----------------------------
The embedding row width d_model determines the codegen path:
  spt = ceil(d_model / 64)              (sticks per token, 64 elems/stick at 16-bit)
  C   = ceil(spt / 32)                  (L3 IBR gather chunks/token; BURST max 32)
  d_model <= 2048  -> spt<=32, C=1  -> single-chunk IBR gather  (HW-verified: IBR-T7)
  d_model  > 2048  -> C>=2          -> multi-chunk IBR gather    (BLOCKED: L3 hang)
Additionally the LX burst chunking (<=64 sticks/beat) matters for spt>64 (d>=4160).

REAL MODEL EMBEDDING SHAPES (vocab x d_model) — the shapes we must support:
  GPT-2                  50257 x 768    (spt=12,  C=1)  supported today
  d<=2048 sanity          *    x 2048   (spt=32,  C=1)  supported today
  GPT-OSS 20B           201088 x 2880   (spt=45,  C=2)  BLOCKED (L3 multi-chunk)
  Qwen2.5-VL 7B         152064 x 3584   (spt=56,  C=2)  BLOCKED
  Llama-3.1-8B          128256 x 4096   (spt=64,  C=2)  BLOCKED
  Mistral-Small          32768 x 4096   (spt=64,  C=2)  BLOCKED
  Ministral-8B          131072 x 4096   (spt=64,  C=2)  BLOCKED
  Granite-4-Hybrid       49152 x 4096   (spt=64,  C=2)  BLOCKED
  Ministral-14B         131072 x 5120   (spt=80,  C=3)  BLOCKED (also LX burst>64)

Once the L3 multi-chunk IBR gather fix lands (see codegen
.git/sdd/EMBEDDING_HW_REALSCALE_2026-07-09.md), remove the xfail marks on the
C>=2 cases and they must pass with max_diff == 0.
"""
import math

import pytest
import torch

# Skip the whole file if the spyre device isn't available (CI without hardware).
_SPYRE_AVAILABLE = False
try:
    import torch_spyre  # noqa: F401 — autoloads the device
    _SPYRE_AVAILABLE = torch.zeros(1).to("spyre") is not None
except Exception:
    _SPYRE_AVAILABLE = False

# NOTE: only the on-device tests are gated on hardware (via requires_hw below).
# The pure-arithmetic path-classification test runs everywhere (incl. CI without a
# board) so the shape contract is always validated.
requires_hw = pytest.mark.skipif(
    not _SPYRE_AVAILABLE, reason="spyre device not available (no hardware)"
)

_EPS = 64  # elements per stick at 16-bit (128 bytes / 2 bytes)


def _spt(d_model: int) -> int:
    return math.ceil(d_model / _EPS)


def _chunks(d_model: int) -> int:
    return math.ceil(_spt(d_model) / 32)


# (id, vocab, d_model, n_tokens, needs_multichunk_fix)
# n_tokens kept small so products stay DL16-exact; vocab downsized for the large-d
# cases so the test table fits memory while preserving the spt/C codegen path
# (the path depends on d_model, not vocab).
# needs_multichunk_fix flips to False across the board: the C>=2 multi-chunk IBR
# gather is HW-verified as of 2026-07-09 (d=4096 C=2 and d=5120 C=3 both max_diff=0.0;
# see codegen commit "embedding: HW-verified multi-chunk (C>1) IBR gather"). All target
# LLM d_models now supported.
_EMBED_SHAPES = [
    # --- C=1, d_model<=2048 ---
    ("gpt2_d768",          4096,  768,   8,  False),
    ("d1024",              4096,  1024,  8,  False),
    ("d2048_boundary",     4096,  2048,  4,  False),
    # --- C>=2, d_model>2048 (HW-verified 2026-07-09) ---
    ("gptoss_d2880",       8192,  2880,  4,  False),
    ("qwen_d3584",         8192,  3584,  4,  False),
    ("llama31_d4096",      8192,  4096,  2,  False),
    ("ministral14b_d5120", 8192,  5120,  2,  False),   # C=3 (spt%32=16 epilogue)
]


def _known_answer_table(vocab: int, d_model: int) -> torch.Tensor:
    """DL16-exact embedding table: row r, col c holds ((r % 8) + 1) * ((c % 4) + 1).

    Max value = 8 * 4 = 32 <= 1024 -> exact in DL16. Distinct per (row-group, col-lane)
    so a wrong gather (dropped chunk / wrong token) is detectable, not masked.
    """
    t = torch.empty(vocab, d_model, dtype=torch.float16)
    rgrp = (torch.arange(vocab) % 8 + 1).to(torch.float16).unsqueeze(1)
    clane = (torch.arange(d_model) % 4 + 1).to(torch.float16).unsqueeze(0)
    t.copy_(rgrp * clane)
    return t


def _run_embedding(vocab: int, d_model: int, n_tokens: int):
    """torch.nn.Embedding forward on spyre vs CPU reference. Returns max_diff."""
    table = _known_answer_table(vocab, d_model)
    # Spread token ids across the vocab so per-token addressing is exercised.
    idx = torch.tensor([(vocab - 1 - (k * 7) % vocab) for k in range(n_tokens)],
                       dtype=torch.int64)
    out_dev = torch.ops.aten.embedding(table.to("spyre"), idx.to("spyre")).cpu()
    out_ref = table[idx]  # CPU reference gather
    return (out_dev - out_ref).abs().max().item()


@requires_hw
@pytest.mark.parametrize(
    "shape_id,vocab,d_model,n_tokens,needs_multichunk",
    _EMBED_SHAPES,
    ids=[s[0] for s in _EMBED_SHAPES],
)
def test_embedding_shape(shape_id, vocab, d_model, n_tokens, needs_multichunk):
    """aten.embedding on device matches CPU (max_diff==0) for each model d_model.

    C>=2 (d_model>2048) shapes are xfail until the L3 multi-chunk IBR gather fix
    lands (they currently NotImplementedError-guard or hang). Remove the xfail
    below when the fix is HW-verified.
    """
    if needs_multichunk:
        pytest.xfail(
            f"{shape_id}: d_model={d_model} spt={_spt(d_model)} C={_chunks(d_model)} "
            f">1 — L3 multi-chunk IBR gather not yet HW-verified (hangs AIU 1.0). "
            f"Tracked: codegen .git/sdd/EMBEDDING_HW_REALSCALE_2026-07-09.md."
        )
    max_diff = _run_embedding(vocab, d_model, n_tokens)
    assert max_diff == 0.0, (
        f"{shape_id} (vocab={vocab} d_model={d_model} n={n_tokens}, "
        f"spt={_spt(d_model)} C={_chunks(d_model)}): max_diff={max_diff} != 0"
    )


def test_embedding_path_classification():
    """Guard the spt/C derivation the codegen path selection depends on — a pure
    arithmetic check that runs without hardware, so the shape contract is validated
    even in CI. If these change, the codegen dispatch thresholds must be revisited.
    """
    assert _spt(768) == 12 and _chunks(768) == 1
    assert _spt(2048) == 32 and _chunks(2048) == 1     # C=1 ceiling
    assert _spt(2880) == 45 and _chunks(2880) == 2     # first C=2
    assert _spt(4096) == 64 and _chunks(4096) == 2
    assert _spt(5120) == 80 and _chunks(5120) == 3     # C=3
