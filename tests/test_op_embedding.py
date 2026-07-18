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
# The codegen gather path depends ONLY on d_model (spt=ceil(d/64), C=ceil(spt/32)),
# NOT on vocab — so one shape per distinct d_model covers every model with that hidden
# size. The comments list the real embedding/LLM models each d_model represents.
_EMBED_SHAPES = [
    # --- C=1, d_model<=2048 (single-chunk gather) ---
    ("d384_spt6",    4096,  384,  8, False),  # all-MiniLM-L6-v2; Granite-Embedding-*-multilingual (EBR path, d<=512)
    ("d512_ebr_ceil", 4096, 512,  8, False),  # Pythia 70M; EBR/IBR boundary (_EBR_D_MODEL_MAX=512)
    ("d768_spt12",   4096,  768,  8, False),  # gpt2; GPT-Neo 125M; BGE-base-en-v1.5; all-mpnet-base-v2;
                                              # ModernBERT-embed-base; GTE-ModernBERT-base; (EmbeddingGemma 300M)
    ("d1024_spt16",  4096, 1024,  8, False),  # Qwen3-Embedding-0.6B; BGE-M3
    ("d1536_spt24",  4096, 1536,  4, False),  # GTE-Qwen2-1.5B; Qwen2.5-1.5B
    ("d2048_boundary", 4096, 2048, 4, False), # C=1 ceiling. Granite 3.3 2B / 4.0 1B; SmolLM3 3B;
                                              # TinyLlama; OLMo(2) 1B; Falcon3 1B
    # --- C>=2, d_model>2048 (multi-chunk gather, HW-verified 2026-07-09) ---
    ("gptoss_d2880",       8192, 2880, 4, False),   # GPT-OSS 20B (spt=45, epilogue 13)
    ("phi_d3072_C2",       8192, 3072, 4, False),   # Phi-4-mini; Phi-3.5-mini; Llama-3.2-3B (spt=48, epilogue 16)
    ("qwen_d3584",         8192, 3584, 4, False),    # Qwen2.5-VL 7B; Qwen2.5 7B
    ("d4096_spt64_C2",     8192, 4096, 2, False),    # Llama-3.1; Mistral 7B; Granite 3.3 8B; Yi 1.5 6B;
                                                     # Ministral-8B; E5/SFR/Linq-Embed-Mistral
    ("mistral_small_d5120", 8192, 5120, 2, False),   # Mistral Small 3 24B; Ministral-3 14B (C=3, spt%32=16 epilogue)
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


# Three entry points a real model can reach the embedding gather through. They take
# different dispatch routes, so each must be exercised:
#   aten        — torch.ops.aten.embedding directly (the op the fork registers a
#                 spyre kernel on).
#   nn_eager    — torch.nn.Embedding.forward with no compile (eager dispatch → the
#                 same registered aten kernel, but via the module + F.embedding).
#   nn_compiled — torch.compile(nn.Embedding) (Inductor graph capture + lowering;
#                 the path a compiled model actually uses).
_ENTRY_POINTS = ("aten", "nn_eager", "nn_compiled")


def _make_idx(vocab: int, n_tokens: int) -> torch.Tensor:
    # Spread token ids across the vocab so per-token addressing is exercised.
    return torch.tensor([(vocab - 1 - (k * 7) % vocab) for k in range(n_tokens)],
                        dtype=torch.int64)


def _make_nn_embedding(table: torch.Tensor) -> torch.nn.Embedding:
    """nn.Embedding whose weight IS the known-answer table (fp16)."""
    vocab, d_model = table.shape
    emb = torch.nn.Embedding(vocab, d_model, dtype=torch.float16)
    with torch.no_grad():
        emb.weight.copy_(table)
    return emb


def _run_embedding(vocab: int, d_model: int, n_tokens: int, entry: str = "aten"):
    """Embedding forward on spyre via `entry` vs CPU reference. Returns max_diff."""
    table = _known_answer_table(vocab, d_model)
    idx = _make_idx(vocab, n_tokens)
    out_ref = table[idx]  # CPU reference gather

    if entry == "aten":
        out_dev = torch.ops.aten.embedding(table.to("spyre"), idx.to("spyre")).cpu()
    elif entry == "nn_eager":
        emb = _make_nn_embedding(table).to("spyre")
        out_dev = emb(idx.to("spyre")).cpu()
    elif entry == "nn_compiled":
        emb = _make_nn_embedding(table).to("spyre")
        torch._dynamo.reset()
        compiled = torch.compile(emb)
        out_dev = compiled(idx.to("spyre")).cpu()
    else:
        raise ValueError(f"unknown entry point: {entry!r}")

    return (out_dev - out_ref).abs().max().item()


@requires_hw
@pytest.mark.parametrize("entry", _ENTRY_POINTS)
@pytest.mark.parametrize(
    "shape_id,vocab,d_model,n_tokens,needs_multichunk",
    _EMBED_SHAPES,
    ids=[s[0] for s in _EMBED_SHAPES],
)
def test_embedding_shape(shape_id, vocab, d_model, n_tokens, needs_multichunk, entry):
    """Embedding on device matches CPU (max_diff==0) for each model d_model, via
    all three entry points (aten op, nn.Embedding eager, nn.Embedding compiled).

    Entry-point routing (HW-observed 2026-07-09):
      aten / nn_eager  -> torch.ops.aten.embedding -> the fork's registered IBR
                          gather kernel.
      nn_compiled      -> a spyre decomposition (torch_spyre/_inductor/decompositions.py
                          spyre_embedding) keeps aten.embedding OPAQUE under
                          torch.compile and routes it to the same custom spyre::embedding
                          IBR op, instead of letting Inductor decompose it to a generic
                          gather Triton kernel (which codegen fails loud on). All three
                          entry points converge on the one HW-verified IBR gather.
    All HW-verified across C=1 and C>=2 (multi-chunk) for every target LLM d_model.

    needs_multichunk is retained for documentation of the C>=2 shapes; the multi-chunk
    gather is HW-verified so no shape is xfail.
    """
    max_diff = _run_embedding(vocab, d_model, n_tokens, entry=entry)
    assert max_diff == 0.0, (
        f"{shape_id}/{entry} (vocab={vocab} d_model={d_model} n={n_tokens}, "
        f"spt={_spt(d_model)} C={_chunks(d_model)}): max_diff={max_diff} != 0"
    )


# ---------------------------------------------------------------------------
# TOKEN-COUNT axis — the resolved C>1 multi-chunk 32-token register-wrap regression.
#
# The main test_embedding_shape sweep varies d_model (spt/C path) but pins n_tokens per
# shape (mostly 2-8). It does NOT sweep the TOKEN COUNT at a fixed C>1 d_model — which is
# exactly the axis of the resolved 32-token bug: C>1 (d_model>2048) at a 32-token prefill
# was max_diff=511 (L3SU register-index wrap — 4-bit S0/S2 index over 16 EAR/16 LAR regs,
# output sticks >512 wraps mod-16 and chunk 16 collides on register 0), fixed 2026 (commit
# 18df559 / [[embedding-multichunk-32token-bug]]) → now HW-verified max_diff=0. Sweeping
# n_tokens ∈ {1 (decode), 8, 16, 32 (the prior fault point)} at a C=2 d_model is the
# regression-lock for that fix — it must stay max_diff=0 as tokens cross the wrap boundary.
# ---------------------------------------------------------------------------
_TOKEN_SWEEP = [1, 8, 16, 32]     # decode → the 32-token prefill that was max_diff=511


@requires_hw
@pytest.mark.parametrize("entry", _ENTRY_POINTS)
@pytest.mark.parametrize("n_tokens", _TOKEN_SWEEP, ids=[f"tok{n}" for n in _TOKEN_SWEEP])
@pytest.mark.parametrize(
    "shape_id,d_model",
    [("d4096_C2", 4096), ("qwen_d3584_C2", 3584)],
    ids=["d4096_C2", "qwen_d3584_C2"],
)
def test_embedding_multichunk_token_sweep(shape_id, d_model, n_tokens, entry):
    """C>1 (multi-chunk) embedding gather across a token-count sweep (1→32) via all three
    entry points — the regression-lock for the resolved 32-token L3SU register-index wrap
    (was max_diff=511 at 32 tokens; must stay max_diff=0). Vocab downsized (path depends on
    d_model, not vocab)."""
    vocab = 8192
    max_diff = _run_embedding(vocab, d_model, n_tokens, entry=entry)
    assert max_diff == 0.0, (
        f"{shape_id}/tok{n_tokens}/{entry} (d_model={d_model}, spt={_spt(d_model)} "
        f"C={_chunks(d_model)}): max_diff={max_diff} != 0 — the 32-token multi-chunk "
        "register-wrap regression (embedding-multichunk-32token-bug) may have returned."
    )


def test_embedding_path_classification():
    """Guard the spt/C derivation the codegen path selection depends on — a pure
    arithmetic check that runs without hardware, so the shape contract is validated
    even in CI. If these change, the codegen dispatch thresholds must be revisited.
    """
    assert _spt(384) == 6 and _chunks(384) == 1        # all-MiniLM-L6-v2, Granite-Embedding
    assert _spt(512) == 8 and _chunks(512) == 1        # Pythia 70M; EBR/IBR boundary
    assert _spt(768) == 12 and _chunks(768) == 1       # BGE-base, mpnet, ModernBERT-embed, gpt2, gpt-neo
    assert _spt(1024) == 16 and _chunks(1024) == 1     # Qwen3-Embedding-0.6B, BGE-M3
    assert _spt(1536) == 24 and _chunks(1536) == 1     # GTE-Qwen2-1.5B, Qwen2.5-1.5B
    assert _spt(2048) == 32 and _chunks(2048) == 1     # C=1 ceiling (many 1-3B LLMs)
    assert _spt(2880) == 45 and _chunks(2880) == 2     # first C=2
    assert _spt(3072) == 48 and _chunks(3072) == 2     # Phi-4-mini, Phi-3.5-mini, Llama-3.2-3B
    assert _spt(3584) == 56 and _chunks(3584) == 2     # Qwen 7B
    assert _spt(4096) == 64 and _chunks(4096) == 2     # Mistral/Llama-8B family
    assert _spt(5120) == 80 and _chunks(5120) == 3     # C=3 (Mistral Small 24B)
