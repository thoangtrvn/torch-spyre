"""Unit tests for the row-contiguous embedding-weight cache.

Torch-only (no torch_spyre._C / no device): imports _embedding_weight_cache
directly so the cache-key / staleness logic is testable on a plain host. These
tests are the guard against the #1 correctness risk of the reshape cache -- a
STALE WRONG-DATA hit. A slow reshape is merely slow; a wrong-data hit silently
returns another table's rows.

Run: python3 -m pytest torch_spyre/ops/test_embedding_weight_cache.py -q
"""
import pytest

torch = pytest.importorskip("torch")

from torch_spyre.ops._embedding_weight_cache import (  # noqa: E402
    get_row_contiguous_weight,
    clear_row_contiguous_weight_cache,
    _ROW_CONTIGUOUS_WEIGHT_CACHE,
)


# On a host with no spyre device, weight.cpu()/.to(device) are no-ops (CPU->CPU),
# so get_row_contiguous_weight just exercises the reshape + cache logic — exactly
# what we want to test. eps=64 (16-bit elems/stick); d_model = spt*eps.
_EPS = 64


def _make_weight(vocab, d_model, fill):
    """A (vocab, d_model) fp16 table whose rows are distinct per (row, position)."""
    w = torch.empty((vocab, d_model), dtype=torch.float16)
    for r in range(vocab):
        # distinct-per-(row,col) content so a wrong table cannot alias by coincidence
        w[r] = torch.arange(d_model, dtype=torch.float16) + (r * 1000 + fill)
    return w


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_row_contiguous_weight_cache()
    yield
    clear_row_contiguous_weight_cache()


def test_reshape_is_correct():
    """The cached result equals the direct reshape (functional correctness)."""
    vocab, spt = 8, 4
    d_model = spt * _EPS
    w = _make_weight(vocab, d_model, fill=0)
    got = get_row_contiguous_weight(w, vocab, spt, _EPS)
    expected = w.cpu().reshape(vocab * spt, _EPS)
    assert got.shape == (vocab * spt, _EPS)
    assert torch.equal(got, expected)


def test_same_weight_hits_cache():
    """Second call with the SAME tensor returns the cached object (fast path)."""
    vocab, spt = 8, 4
    d_model = spt * _EPS
    w = _make_weight(vocab, d_model, fill=0)
    first = get_row_contiguous_weight(w, vocab, spt, _EPS)
    second = get_row_contiguous_weight(w, vocab, spt, _EPS)
    assert second is first, "same weight must reuse the cached row-contiguous tensor"
    assert len(_ROW_CONTIGUOUS_WEIGHT_CACHE) == 1


def test_different_weight_same_shape_no_stale_hit():
    """THE #1 RISK: two DIFFERENT tables with identical (vocab, d_model, dtype)
    must NOT collide. The second must get ITS OWN data, never the first's."""
    vocab, spt = 8, 4
    d_model = spt * _EPS
    w1 = _make_weight(vocab, d_model, fill=0)
    w2 = _make_weight(vocab, d_model, fill=500)  # same shape/dtype, different data
    assert not torch.equal(w1, w2)

    rc1 = get_row_contiguous_weight(w1, vocab, spt, _EPS)
    rc2 = get_row_contiguous_weight(w2, vocab, spt, _EPS)

    # w2 must map to its OWN row-contiguous data, not w1's stale copy.
    assert torch.equal(rc2, w2.cpu().reshape(vocab * spt, _EPS))
    assert not torch.equal(rc2, rc1), "STALE HIT: w2 returned w1's cached table!"
    # And w1 is unaffected / still correct.
    assert torch.equal(rc1, w1.cpu().reshape(vocab * spt, _EPS))


def test_inplace_mutation_invalidates():
    """Same tensor object, mutated in place (weight update): must rebuild, not
    return the pre-mutation cached copy."""
    vocab, spt = 8, 4
    d_model = spt * _EPS
    w = _make_weight(vocab, d_model, fill=0)
    rc_before = get_row_contiguous_weight(w, vocab, spt, _EPS).clone()

    w.add_(7.0)  # version-tracked in-place update (bumps w._version)
    rc_after = get_row_contiguous_weight(w, vocab, spt, _EPS)

    assert torch.equal(rc_after, w.cpu().reshape(vocab * spt, _EPS)), \
        "mutated weight must yield freshly-reshaped data"
    assert not torch.equal(rc_after, rc_before), \
        "STALE HIT: returned pre-mutation table after in-place update!"


def test_copy_update_invalidates():
    """Optimizer/weight-load path uses copy_ (also version-tracked): must rebuild."""
    vocab, spt = 8, 4
    d_model = spt * _EPS
    w = _make_weight(vocab, d_model, fill=0)
    _ = get_row_contiguous_weight(w, vocab, spt, _EPS)

    w.copy_(_make_weight(vocab, d_model, fill=900))
    rc_after = get_row_contiguous_weight(w, vocab, spt, _EPS)
    assert torch.equal(rc_after, w.cpu().reshape(vocab * spt, _EPS))


def test_address_reuse_aba_no_stale_hit():
    """ABA: a freed table's data_ptr can be reused by a NEW table. The weakref
    identity guard must reject the recycled-address collision (rebuild, not hit).

    We can't force the allocator to reuse an address deterministically, so we
    simulate the exact failure the guard defends against: an entry keyed at some
    (ptr, shape, dtype) whose source tensor has been freed (dead weakref), then a
    new tensor that lands on the SAME key. A key-only cache would hand back the
    dead entry's data; the weakref check must force a rebuild."""
    vocab, spt = 8, 4
    d_model = spt * _EPS

    w_old = _make_weight(vocab, d_model, fill=0)
    rc_old = get_row_contiguous_weight(w_old, vocab, spt, _EPS)
    old_key = (w_old.data_ptr(), tuple(w_old.shape), w_old.dtype)
    assert old_key in _ROW_CONTIGUOUS_WEIGHT_CACHE

    # Forge the ABA collision: a NEW weight forced onto w_old's cache key with the
    # old (now-stale) entry still present but its weakref pointing at a dead object.
    w_new = _make_weight(vocab, d_model, fill=777)
    entry = _ROW_CONTIGUOUS_WEIGHT_CACHE[old_key]
    import weakref
    dead = _make_weight(1, _EPS, fill=0)
    dead_ref = weakref.ref(dead)
    del dead  # ref() now returns None -> simulates the freed original table
    assert dead_ref() is None
    entry.ref = dead_ref  # entry now looks like a recycled address

    # Re-key w_new onto the stale entry's slot to model data_ptr reuse.
    new_key = (w_new.data_ptr(), tuple(w_new.shape), w_new.dtype)
    _ROW_CONTIGUOUS_WEIGHT_CACHE.pop(new_key, None)
    _ROW_CONTIGUOUS_WEIGHT_CACHE[new_key] = entry  # dead-weakref entry at w_new's key

    rc_new = get_row_contiguous_weight(w_new, vocab, spt, _EPS)
    assert torch.equal(rc_new, w_new.cpu().reshape(vocab * spt, _EPS)), \
        "STALE HIT: dead-weakref (ABA) entry served instead of rebuilding!"
    assert not torch.equal(rc_new, rc_old)


def test_two_models_coexist():
    """Two distinct embedding tables (different shapes) can be cached at once and
    each returns its own data across interleaved calls."""
    wa = _make_weight(8, 4 * _EPS, fill=0)     # d=256
    wb = _make_weight(6, 3 * _EPS, fill=100)   # d=192
    rca1 = get_row_contiguous_weight(wa, 8, 4, _EPS)
    rcb1 = get_row_contiguous_weight(wb, 6, 3, _EPS)
    rca2 = get_row_contiguous_weight(wa, 8, 4, _EPS)  # interleaved re-hit
    rcb2 = get_row_contiguous_weight(wb, 6, 3, _EPS)
    assert rca2 is rca1 and rcb2 is rcb1
    assert torch.equal(rca1, wa.cpu().reshape(8 * 4, _EPS))
    assert torch.equal(rcb1, wb.cpu().reshape(6 * 3, _EPS))
    assert len(_ROW_CONTIGUOUS_WEIGHT_CACHE) == 2


def test_dead_entry_pruned_on_rebuild():
    """A freed table's cache entry is pruned when a later rebuild occurs, so the
    cache does not grow without bound as weights come and go.

    NOTE: we forge the dead weakref rather than rely on `del w1`. On a real spyre
    device the cached weight_rc is an independent H2D copy, so freeing the source
    weight kills its weakref and the entry becomes prunable. On this CPU-only host
    `.to(device)` is a no-op and weight_rc is a VIEW whose `._base` pins the source
    tensor alive, so `del w1` would NOT free it -- a host-simulation artifact, not
    a cache bug. Forging the dead weakref tests the prune loop directly and
    portably (same technique as the ABA test)."""
    w1 = _make_weight(8, 4 * _EPS, fill=0)
    _ = get_row_contiguous_weight(w1, 8, 4, _EPS)
    assert len(_ROW_CONTIGUOUS_WEIGHT_CACHE) == 1

    # Simulate w1 having been freed: make its entry's weakref dead.
    import weakref
    dead = _make_weight(1, _EPS, fill=0)
    dead_ref = weakref.ref(dead)
    del dead
    assert dead_ref() is None
    ((_only_key, _only_entry),) = list(_ROW_CONTIGUOUS_WEIGHT_CACHE.items())
    _only_entry.ref = dead_ref

    w2 = _make_weight(6, 3 * _EPS, fill=1)
    _ = get_row_contiguous_weight(w2, 6, 3, _EPS)  # rebuild -> prunes the dead entry
    assert _only_key not in _ROW_CONTIGUOUS_WEIGHT_CACHE, \
        "dead (freed-weight) entry should have been pruned on rebuild"
    assert len(_ROW_CONTIGUOUS_WEIGHT_CACHE) == 1
