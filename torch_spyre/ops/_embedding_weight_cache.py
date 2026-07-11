# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Row-contiguous embedding-weight cache (torch-aware, IBR path).

The IBR embedding gather needs the (vocab, d_model) table relaid out
plane-major -> row-contiguous as (vocab*spt, eps): token t's spt sticks at
device positions [t*spt, (t+1)*spt). That relayout is a full D2H+H2D of the
ENTIRE table and costs ~3.3 ms/MB (Granite-8B d=4096 ~= 1.3 s; Mistral-Small-24B
d=5120 ~= 4.4 s). The table is STATIC across forward passes and the relayout
depends ONLY on (weight data, vocab, spt) -- never on the token indices -- so we
build it ONCE and reuse the cached copy on every subsequent forward, rebuilding
only when the underlying weight tensor actually changes.

Why this module is separate from embedding.py: embedding.py imports
torch_spyre._C (the device C extension), which is absent on a plain host. This
module imports ONLY torch + stdlib, so the cache-key / staleness logic is
unit-testable offline (the #1 correctness risk -- a stale wrong-data hit -- is
guarded by a host test against this module directly). It mirrors the build-once,
identity-keyed spirit of _embedding_host._IBR_LOOPED_CACHE (which caches the
geometry-only binary), but lives here because it stores a torch device tensor.

STALE-CACHE CORRECTNESS (the load-bearing argument)
---------------------------------------------------
A wrong-data cache hit is far worse than a slow reshape, so the key must reject:
  (1) a DIFFERENT weight tensor that happens to share (vocab, d_model, dtype), and
  (2) the SAME tensor after its data was mutated (weight update / reload).

Key = (data_ptr, shape, dtype). The entry additionally holds a weakref to the
exact weight object and its _version, both re-checked on every candidate hit:

  * weakref-identity (`entry.ref() is weight`) defeats ADDRESS REUSE (ABA). If the
    cached tensor is freed and a new tensor reuses its data_ptr (torch's caching
    allocator does exactly this), the weakref is dead -> `None is weight` is False
    -> MISS -> rebuild. Two LIVE tensors cannot share a data_ptr, so a distinct
    table always either keys differently OR fails this identity check. weakref
    identity is a STRONGER proof than data_ptr alone or a content hash (which can
    collide): it is true iff `weight` is literally the object we cached.
  * _version defeats IN-PLACE MUTATION. torch bumps tensor._version on every
    version-tracked in-place op (add_, mul_, copy_, and no_grad optimizer updates
    all bump it). A changed table -> version mismatch -> MISS -> rebuild.

Residual hole (documented, NOT hidden): mutating through `weight.data.add_(...)`
bypasses torch's version counter, so it would not be detected. That path is not
used by normal weight loading or optimizers (which use copy_/no_grad add_, both
version-tracked), and inference embedding tables are frozen. We deliberately do
NOT close it with an O(N) content hash -- that would reintroduce the full-table
touch this cache exists to remove -- nor with an O(1) sampled hash, which gives
false confidence (misses unsampled mutations) and needs a fragile per-element
device D2H. If a version-untracked mutation path ever becomes real for embedding
tables, clear the cache explicitly (see clear_row_contiguous_weight_cache).
"""
from __future__ import annotations

import weakref

import torch


class _RCEntry:
    """A cached row-contiguous weight plus the identity/version it was built from."""

    __slots__ = ("weight_rc", "ref", "version")

    def __init__(self, weight_rc: torch.Tensor, ref, version: int):
        self.weight_rc = weight_rc
        self.ref = ref            # weakref.ref to the source weight tensor (ABA guard)
        self.version = version    # source weight._version at build time (mutation guard)


# Module-level (process-lifetime) cache. One entry per distinct embedding table the
# process uses (typically 1-2 for an LLM), keyed by (data_ptr, shape, dtype) and
# validated by weakref-identity + _version. No eviction policy needed; dead-weakref
# entries are pruned opportunistically on rebuild so a freed table's cached copy does
# not linger indefinitely.
_ROW_CONTIGUOUS_WEIGHT_CACHE: dict = {}


def get_row_contiguous_weight(
    weight: torch.Tensor,
    vocab: int,
    spt: int,
    eps: int,
) -> torch.Tensor:
    """Return the row-contiguous (vocab*spt, eps) copy of `weight` on its device.

    Builds it on first use (cold) and reuses the cached copy on subsequent calls
    with the SAME (identity- and version-checked) weight tensor (warm). Equivalent
    per call to::

        weight.cpu().reshape(vocab * spt, eps).to(weight.device)

    which depends only on (weight data, vocab, spt) -- never on the token indices --
    so it is safe to cache across forward passes. See the module docstring for the
    stale-cache correctness argument (the weakref-identity + _version guard).
    """
    key = (weight.data_ptr(), tuple(weight.shape), weight.dtype)
    entry = _ROW_CONTIGUOUS_WEIGHT_CACHE.get(key)
    if entry is not None:
        cached = entry.ref()
        # HIT only if it is literally the same, unmutated tensor. Otherwise fall
        # through to rebuild (defeats ABA address-reuse and in-place mutation).
        if cached is weight and entry.version == weight._version:
            return entry.weight_rc

    # Cold / stale: rebuild the row-contiguous copy (the expensive D2H+H2D relayout).
    weight_rc = weight.cpu().reshape(vocab * spt, eps).to(weight.device)

    # Prune entries whose source tensor has been freed, so cached device copies of
    # dead tables do not accumulate (cache is tiny; this scan is cheap).
    for dead_key in [k for k, e in _ROW_CONTIGUOUS_WEIGHT_CACHE.items()
                     if e.ref() is None]:
        del _ROW_CONTIGUOUS_WEIGHT_CACHE[dead_key]

    try:
        ref = weakref.ref(weight)
    except TypeError:
        # Not weakref-able (should not happen for torch.Tensor / nn.Parameter):
        # skip caching entirely -- always correct, just pays the reshape each call.
        return weight_rc

    _ROW_CONTIGUOUS_WEIGHT_CACHE[key] = _RCEntry(weight_rc, ref, weight._version)
    return weight_rc


def clear_row_contiguous_weight_cache() -> None:
    """Drop all cached row-contiguous weights (test hygiene / explicit invalidation)."""
    _ROW_CONTIGUOUS_WEIGHT_CACHE.clear()
