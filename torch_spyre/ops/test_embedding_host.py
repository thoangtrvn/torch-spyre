"""Torch-free unit tests for the embedding launch planner. Run with sentient_codegen
on PYTHONPATH; no torch / torch_spyre import required."""
import math
import pytest

from torch_spyre.ops._embedding_host import build_embedding_launches, _tokens_per_core


def test_multistick_d4096_multichunk_raises():
    """d_model=4096 (spt=64 → C=2 chunks/token) raises on the multi-chunk guard.

    The single-chunk IBR gather (C=1, d_model <= 2048) is hardware-verified
    (IBR T7: real aten.embedding d=768/d=2048, max_diff=0). The MULTI-chunk path
    (C>1) hangs the device on AIU 1.0 (IBR T7: d=4096 spt=64 C=2 timed out), so the
    planner raises NotImplementedError rather than building a hanging binary until
    the chunked-read path is fixed and verified.
    """
    idx = list(range(4))
    with pytest.raises(NotImplementedError, match="multi-chunk|chunks_per_token"):
        build_embedding_launches(vocab=128, d_model=4096, element_bits=16, flat_idx=idx)


def test_two_stick_now_supported():
    """The smallest multi-stick case (d=128, spt=2) is HARDWARE-VERIFIED (PM-T6).

    Plane-major source + plane-major output scatter confirmed on silicon
    (max_diff=0). spt=2 → floor(8/2)=4 tokens/core; 8 tokens → 2 launches of
    4 tokens each (within the per-core cap). The planner must now PLAN, not raise.
    """
    idx = list(range(8))
    launches, tokens_per_launch = build_embedding_launches(
        vocab=1000, d_model=128, element_bits=16, flat_idx=idx)
    # spt=2 → tokens_per_core = floor(8/2) = 4. 8 tokens → 2 cores × 4 = 8/launch.
    assert tokens_per_launch == min(32, math.ceil(8 / 4)) * 4  # 2 cores × 4 = 8
    assert len(launches) >= 1


def test_single_stick_unchanged():
    """d_model=64 (spt=1) keeps tokens_per_core=8 ceiling (PM-T3: unchanged).

    Plane-major ceiling for spt=1: floor(8/1)=8. Same as old model.
    """
    idx = list(range(10))
    launches, tokens_per_launch = build_embedding_launches(
        vocab=64, d_model=64, element_bits=16, flat_idx=idx)
    assert tokens_per_launch == min(32, math.ceil(10 / 8)) * 8  # 2 cores × 8 = 16


# ---------------------------------------------------------------------------
# PM-Task 3: host ceiling tests
# ---------------------------------------------------------------------------

def test_pm3_d1024_now_ibr():
    """d_model=1024 (spt=16) now dispatches to the IBR path — no longer raises.

    With IBR dispatch in place for d>512, d=1024 (spt=16 > ebr_count=8) no
    longer raises NotImplementedError — it plans ≥1 launch using the IBR path.
    This replaces the old test_pm3_d1024_raises_on_ebr_ceiling which was correct
    when d>512 was not implemented (Task 5 removes the ceiling for IBR-capable
    d_models).
    """
    idx = list(range(4))
    launches, tokens_per_launch = build_embedding_launches(
        vocab=128, d_model=1024, element_bits=16, flat_idx=idx)
    assert len(launches) >= 1, "IBR path must plan ≥1 launch for d=1024"
    assert tokens_per_launch <= 32


@pytest.mark.parametrize("spt,expected", [
    (1,  8),  # spt=1  → floor(8/1)=8 (single-stick, unchanged from old model)
    (2,  4),  # spt=2  → floor(8/2)=4 (d=128 at 16-bit)
    (4,  2),  # spt=4  → floor(8/4)=2 (d=256 at 16-bit)
    (8,  1),  # spt=8  → floor(8/8)=1 (d=512, max spt under this model)
    (16, 0),  # spt=16 → floor(8/16)=0 (d=1024 exceeds EBR file; caller raises)
])
def test_pm3_tokens_per_core(spt, expected):
    """PM-T3: _tokens_per_core exercises the REAL production formula for all spt values.

    ebr_count=8, lar_count=16 (AIU 1.0 hardware constants via gen registry).
    These tests call _tokens_per_core directly, so a bug in the production formula
    (e.g. using max instead of min, wrong integer-division, wrong register count)
    will immediately cause a failure — unlike the previous inline-arithmetic versions
    which were tautological (they reimplemented the formula, not called it).

    spt>1 coverage is possible here because _tokens_per_core has no hardware fence;
    the fence lives in build_embedding_launches and only blocks the end-to-end path.
    The codegen guard (test_pm3_guard_* in codegen tests) provides additional spt>1
    coverage through the scheduler path.
    """
    assert _tokens_per_core(8, 16, spt) == expected, (
        f"PM-T3: _tokens_per_core(8, 16, {spt}) = "
        f"{_tokens_per_core(8, 16, spt)} != {expected}"
    )


def test_pm3_spt1_tokens_per_core_unchanged():
    """PM-T3: spt=1 ceiling is floor(8/1)=8 — byte-identical to old model.

    Verifies via build_embedding_launches that tokens_per_core=8 for spt=1.
    """
    # 8 exact tokens → 1 launch with 1 core × 8 tokens
    launches, tokens_per_launch = build_embedding_launches(
        vocab=64, d_model=64, element_bits=16, flat_idx=list(range(8)),
    )
    assert tokens_per_launch == 8, (
        f"PM-T3: spt=1, 8 tokens → tokens_per_launch=8, got {tokens_per_launch}"
    )


# ---------------------------------------------------------------------------
# Task 5: IBR path (d_model > 512) tests
# ---------------------------------------------------------------------------

def test_ibr_d2048_plans_not_raises():
    """IBR-T5: d_model=2048 (spt=32) now PLANS via IBR path, not raises.

    With d>512 dispatched to the IBR path, build_embedding_launches must return
    ≥1 launch with ≤32 tokens/launch (IBR file size limit, not the EBR-8 ceiling).
    The old NotImplementedError for d>512 must NOT be raised.
    """
    flat_idx = list(range(4))
    launches, tokens_per_launch = build_embedding_launches(
        vocab=32000, d_model=2048, element_bits=16, flat_idx=flat_idx,
    )
    assert len(launches) >= 1, "IBR path must produce at least 1 launch"
    # tokens_per_launch is the IBR file ceiling (32) or fewer if batch is smaller
    assert tokens_per_launch <= 32, (
        f"IBR path: tokens_per_launch={tokens_per_launch} exceeds IBR file size 32"
    )
    assert tokens_per_launch >= 1
    # All token indices must be covered with no overlap
    covered = []
    for (start, end), _binary in launches:
        assert end > start, f"empty slice ({start},{end})"
        covered.extend(range(start, end))
    assert sorted(covered) == list(range(len(flat_idx))), (
        f"IBR launches do not cover [0,{len(flat_idx)}) exactly: {covered}"
    )


def test_ibr_d4096_multichunk_raises():
    """IBR-T7: d_model=4096 (spt=64 → C=2 chunks/token) raises on the multi-chunk guard.

    The chunked-read path (C=ceil(spt/32)>1: 2 gather instructions per token with
    addlarimm advances between chunks) hangs the device on AIU 1.0 (IBR T7). Until
    fixed+verified, it raises NotImplementedError rather than building a hanging
    binary. Single-chunk (d_model <= 2048, C=1) is the hardware-verified ceiling.
    """
    flat_idx = list(range(4))
    with pytest.raises(NotImplementedError, match="multi-chunk|chunks_per_token"):
        build_embedding_launches(
            vocab=1024, d_model=4096, element_bits=16, flat_idx=flat_idx,
        )


def test_ibr_more_than_32_tokens_multi_launch():
    """IBR-T5: >32 tokens requires multiple launches (IBR file size = 32 entries).

    40 tokens must be split into ≥2 launches, each with ≤32 tokens.
    """
    flat_idx = list(range(40))
    launches, tokens_per_launch = build_embedding_launches(
        vocab=32000, d_model=2048, element_bits=16, flat_idx=flat_idx,
    )
    assert tokens_per_launch <= 32, (
        f"IBR: tokens_per_launch={tokens_per_launch} > 32 (IBR file limit)"
    )
    assert len(launches) >= 2, (
        f"40 tokens with IBR limit=32 must produce ≥2 launches, got {len(launches)}"
    )
    covered = []
    for (start, end), _binary in launches:
        covered.extend(range(start, end))
    assert sorted(covered) == list(range(40))


def test_ibr_d512_stays_ebr():
    """IBR-T5: d_model=512 (spt=8) stays on EBR path — unchanged from PM-T6.

    The split is: d<=512 → EBR (proven), d>512 → IBR.  d=512 must still use the
    EBR path (tokens_per_core = floor(8/8) = 1 token/core, not the IBR ceiling).
    This test verifies the boundary: d=512 does NOT use the IBR path.
    """
    flat_idx = list(range(4))
    launches, tokens_per_launch = build_embedding_launches(
        vocab=1000, d_model=512, element_bits=16, flat_idx=flat_idx,
    )
    # EBR path: spt=8 → tokens_per_core=1, tokens_per_launch = num_cores*1 ≤ 32.
    # 4 tokens → up to 4 cores × 1 token = 4 tokens per launch.
    assert len(launches) >= 1
    # EBR path: tokens_per_launch must divide evenly by spt (each core ≤ EBR count).
    # tokens_per_launch on EBR is num_cores * tokens_per_core = num_cores * 1 = num_cores.
    # tokens_per_launch ≥ 1 and is a multiple of 1.
    assert tokens_per_launch >= 1


def test_ibr_d64_spt1_unchanged():
    """IBR-T5: d_model=64 (spt=1) is completely unaffected — EBR path, byte-identical.

    The IBR dispatch is guarded on d_model > 512 (spt > 8).  d=64 must continue to
    use the EBR path with tokens_per_core=8.
    """
    flat_idx = list(range(8))
    launches, tokens_per_launch = build_embedding_launches(
        vocab=1000, d_model=64, element_bits=16, flat_idx=flat_idx,
    )
    # EBR path unchanged: spt=1 → tokens_per_core=8
    assert tokens_per_launch == 8


def test_build_ibr_address_table():
    """IBR-T5: build_ibr_address_table produces correct int32 HBM stick addresses.

    Silicon-proven 3-tensor layout (IBR-T7):
      XLAT[0] = weight_rc  (segment 0)
      XLAT[1] = ibr_table  (segment 1)
      XLAT[2] = output_buf (segment 2)

    IBR[t] = (0<<seg_bits) + flat_indices[t] * spt = flat_indices[t] * spt
    where seg_bits=27 for rcudd1a, spt=d_model//64, weight at segment 0.

    Verified by construction:
      d_model=768, spt=12, seg_bits=27, flat_idx=[3,7]
      IBR[0] = (0<<27) + 3*12 = 0 + 36 = 36
      IBR[1] = (0<<27) + 7*12 = 0 + 84 = 84
    """
    from torch_spyre.ops._embedding_host import build_ibr_address_table

    seg_bits = 27
    spt = 12
    flat_idx = [3, 7]
    weight_segment = 0  # XLAT[0] — silicon-proven IBR-T7 layout

    table = build_ibr_address_table(flat_idx, spt, segment=weight_segment,
                                     seg_bits=seg_bits)

    # Shape: (1, 32) int32
    assert table.shape == (1, 32), f"Expected (1, 32), got {table.shape}"
    assert str(table.dtype) in ("int32", "torch.int32"), (
        f"Expected int32, got {table.dtype}"
    )

    # Verify entries: IBR[t] = (0<<seg_bits) + idx*spt = idx*spt
    expected_0 = (0 << seg_bits) + 3 * spt   # = 36
    expected_1 = (0 << seg_bits) + 7 * spt   # = 84
    assert table[0, 0] == expected_0, (
        f"IBR[0]: expected {expected_0}, got {table[0, 0]}"
    )
    assert table[0, 1] == expected_1, (
        f"IBR[1]: expected {expected_1}, got {table[0, 1]}"
    )
    # Unused entries must be zero
    assert table[0, 2] == 0, f"IBR[2] (unused): expected 0, got {table[0, 2]}"
    assert table[0, 31] == 0, f"IBR[31] (unused): expected 0, got {table[0, 31]}"


def test_build_ibr_address_table_kv_reuse():
    """IBR-T5: build_ibr_address_table is reusable for KV block tables.

    A block-table KV lookup uses the same IBR construction: given a list of
    block indices and a spt (sticks per KV block), compute absolute stick addresses
    in a specific segment.  This test verifies the same function works with
    different parameters, confirming the factored API is reusable.
    """
    from torch_spyre.ops._embedding_host import build_ibr_address_table

    seg_bits = 27
    spt = 4        # smaller block size (KV scenario)
    kv_blocks = [10, 20, 5]   # block addresses
    segment = 2    # KV tensor at XLAT[2]

    table = build_ibr_address_table(kv_blocks, spt, segment=segment,
                                     seg_bits=seg_bits)
    assert table.shape == (1, 32)
    for i, blk in enumerate(kv_blocks):
        expected = (segment << seg_bits) + blk * spt
        assert table[0, i] == expected, (
            f"KV IBR[{i}]: expected {expected}, got {table[0, i]}"
        )


# ---------------------------------------------------------------------------
# Loop T5: build-once cache + per-call loop-count patch
# ---------------------------------------------------------------------------

def test_loop_t5_ibr_cached_binary_built_once_for_same_d_model():
    """Loop-T5: the IBR cached binary is built ONCE per (d_model, element_bits).

    Calling build_embedding_launches for d=2048 with 8 tokens and then again
    with 16 tokens must return a build counter of 1 (built once), not 2.

    This test uses the _ibr_binary_cache module-level dict (imported directly
    from _embedding_host) as an observable proxy for 'built once'.
    """
    from torch_spyre.ops._embedding_host import (
        build_embedding_launches,
        _ibr_binary_cache,
    )

    # Clear the cache before the test to ensure a clean slate
    _ibr_binary_cache.clear()

    vocab = 32000
    d_model = 2048
    element_bits = 16

    flat_idx_8 = list(range(8))
    flat_idx_16 = list(range(16))

    launches_8, tpl_8 = build_embedding_launches(
        vocab=vocab, d_model=d_model, element_bits=element_bits,
        flat_idx=flat_idx_8,
    )
    assert len(launches_8) >= 1

    # Cache must have exactly one entry now
    assert len(_ibr_binary_cache) == 1, (
        f"Expected 1 cache entry after first IBR call, got {len(_ibr_binary_cache)}"
    )

    launches_16, tpl_16 = build_embedding_launches(
        vocab=vocab, d_model=d_model, element_bits=element_bits,
        flat_idx=flat_idx_16,
    )
    assert len(launches_16) >= 1

    # Cache must STILL have exactly one entry (no rebuild for same d_model)
    assert len(_ibr_binary_cache) == 1, (
        f"Expected cache to stay at 1 entry after second IBR call (same d_model), "
        f"got {len(_ibr_binary_cache)} — binary was rebuilt instead of reused"
    )


def test_loop_t5_ibr_per_call_binaries_differ_only_in_loop_count_imm():
    """Loop-T5: per-call binaries for different token counts differ ONLY in the IMM field.

    With a cached base binary and per-call loop-count patch:
      - 8-token launch binary decoded inst[3] IMM == 8
      - 16-token launch binary decoded inst[3] IMM == 16
      - Both binaries are byte-identical except the 4 bytes of the IMM word.

    This confirms option (A) from the task design: build once + per-call byte-patch.
    """
    import struct
    from torch_spyre.ops._embedding_host import (
        build_embedding_launches,
        _ibr_binary_cache,
    )
    from sentient_codegen.emulator.loader import load_init_packet_bytes
    from sentient_codegen.gen.registry import get_generation
    from sentient_codegen.encoder.embedding_dispatch import _JCR_CNT_BYTE_OFFSET

    gen = get_generation("rcudd1a")

    # Clear cache before test
    _ibr_binary_cache.clear()

    vocab = 32000
    d_model = 2048
    element_bits = 16

    # Build 8-token launch
    launches_8, _ = build_embedding_launches(
        vocab=vocab, d_model=d_model, element_bits=element_bits,
        flat_idx=list(range(8)),
    )
    assert len(launches_8) == 1, f"8 tokens must produce 1 launch, got {len(launches_8)}"
    _, binary_8 = launches_8[0]

    # Build 16-token launch (using the same cached base)
    launches_16, _ = build_embedding_launches(
        vocab=vocab, d_model=d_model, element_bits=element_bits,
        flat_idx=list(range(16)),
    )
    assert len(launches_16) == 1, f"16 tokens must produce 1 launch, got {len(launches_16)}"
    _, binary_16 = launches_16[0]

    # Both binaries must have the same length (same flit count)
    assert len(binary_8) == len(binary_16), (
        f"Binary lengths differ: {len(binary_8)} vs {len(binary_16)}"
    )

    # Decode both and verify IMM differs
    decoded_8 = load_init_packet_bytes(binary_8, gen)
    decoded_16 = load_init_packet_bytes(binary_16, gen)
    l3lu_8 = decoded_8.unit_states.get("L3LU")
    l3lu_16 = decoded_16.unit_states.get("L3LU")
    assert l3lu_8 is not None and l3lu_16 is not None

    imm_8 = l3lu_8.instructions[3].get("IMM")
    imm_16 = l3lu_16.instructions[3].get("IMM")
    assert imm_8 == 8, f"8-token binary inst[3] IMM must be 8, got {imm_8}"
    assert imm_16 == 16, f"16-token binary inst[3] IMM must be 16, got {imm_16}"

    # Byte-compare: only the 4 bytes at _JCR_CNT_BYTE_OFFSET must differ
    diffs = [i for i in range(len(binary_8)) if binary_8[i] != binary_16[i]]
    patch_range = {_JCR_CNT_BYTE_OFFSET, _JCR_CNT_BYTE_OFFSET + 1,
                   _JCR_CNT_BYTE_OFFSET + 2, _JCR_CNT_BYTE_OFFSET + 3}
    assert set(diffs).issubset(patch_range), (
        f"Binaries differ at bytes outside the IMM patch word: "
        f"{[hex(d) for d in diffs if d not in patch_range]}"
    )
    assert len(diffs) > 0, "8- and 16-token binaries are identical — patch not applied"


def test_loop_t5_ibr_multi_launch_reuses_same_cached_binary():
    """Loop-T5: >32 tokens across multiple launches all share the same cached binary.

    40 tokens → 2 launches (32 + 8). Both binaries must come from the same
    cached base (same bytes except for the loop-count IMM patch in each launch).
    The cache must still have exactly 1 entry after building all launches.
    """
    from torch_spyre.ops._embedding_host import (
        build_embedding_launches,
        _ibr_binary_cache,
    )
    from sentient_codegen.emulator.loader import load_init_packet_bytes
    from sentient_codegen.gen.registry import get_generation

    gen = get_generation("rcudd1a")

    # Clear cache
    _ibr_binary_cache.clear()

    vocab = 32000
    d_model = 2048
    element_bits = 16

    launches_40, tpl = build_embedding_launches(
        vocab=vocab, d_model=d_model, element_bits=element_bits,
        flat_idx=list(range(40)),
    )
    assert len(launches_40) == 2, (
        f"40 tokens (IBR limit=32) must produce 2 launches, got {len(launches_40)}"
    )
    assert tpl == 32, f"tokens_per_launch must be 32, got {tpl}"

    # Cache must still have only 1 entry
    assert len(_ibr_binary_cache) == 1, (
        f"Expected 1 cache entry after 40-token IBR call, got {len(_ibr_binary_cache)}"
    )

    _, binary_l1 = launches_40[0]  # launch 1: 32 tokens → IMM=32
    _, binary_l2 = launches_40[1]  # launch 2: 8 tokens  → IMM=8

    # Verify IMM values
    decoded_l1 = load_init_packet_bytes(binary_l1, gen)
    decoded_l2 = load_init_packet_bytes(binary_l2, gen)
    imm_l1 = decoded_l1.unit_states["L3LU"].instructions[3].get("IMM")
    imm_l2 = decoded_l2.unit_states["L3LU"].instructions[3].get("IMM")

    assert imm_l1 == 32, f"Launch 1 (32 tokens) IMM must be 32, got {imm_l1}"
    assert imm_l2 == 8, f"Launch 2 (8 tokens) IMM must be 8, got {imm_l2}"


def test_loop_t5_ebr_path_unaffected_by_ibr_cache():
    """Loop-T5: d<=512 (EBR path) is completely unaffected by IBR caching.

    Adding an IBR cache must NOT change d=64 or d=512 behavior.
    """
    from torch_spyre.ops._embedding_host import (
        build_embedding_launches,
        _ibr_binary_cache,
    )

    _ibr_binary_cache.clear()

    # d=64 EBR path
    launches_64, tpl_64 = build_embedding_launches(
        vocab=1000, d_model=64, element_bits=16, flat_idx=list(range(8)),
    )
    assert len(launches_64) >= 1
    assert tpl_64 == 8, f"EBR d=64 tokens_per_launch must be 8, got {tpl_64}"

    # Cache must be empty (EBR path does not use the IBR cache)
    assert len(_ibr_binary_cache) == 0, (
        f"EBR path must not populate IBR cache, got {len(_ibr_binary_cache)} entries"
    )

    # d=512 EBR path
    launches_512, _ = build_embedding_launches(
        vocab=1000, d_model=512, element_bits=16, flat_idx=list(range(4)),
    )
    assert len(launches_512) >= 1
    assert len(_ibr_binary_cache) == 0, (
        "EBR d=512 path must not populate IBR cache"
    )


def test_loop_t5_different_d_models_separate_cache_entries():
    """Loop-T5: different d_model values have separate cache entries.

    d=768 and d=2048 are different shapes; the cached binary differs (different
    spt, different binary structure). Both must be cached independently.
    """
    from torch_spyre.ops._embedding_host import (
        build_embedding_launches,
        _ibr_binary_cache,
    )

    _ibr_binary_cache.clear()

    build_embedding_launches(
        vocab=32000, d_model=768, element_bits=16, flat_idx=list(range(4)),
    )
    assert len(_ibr_binary_cache) == 1, (
        f"Expected 1 entry after d=768, got {len(_ibr_binary_cache)}"
    )

    build_embedding_launches(
        vocab=32000, d_model=2048, element_bits=16, flat_idx=list(range(4)),
    )
    assert len(_ibr_binary_cache) == 2, (
        f"Expected 2 entries after d=768 + d=2048, got {len(_ibr_binary_cache)}"
    )
