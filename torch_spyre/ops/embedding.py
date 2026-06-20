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

"""spyre::embedding custom op — on-device token gather via sentient_codegen.

Reads indices on the host, builds per-launch init-packet binaries with the
torch-free _embedding_host helper, allocates the output on device, launches each
binary, and reshapes to PyTorch's (*idx_shape, d_model). No silent CPU fallback.
"""
from __future__ import annotations

import torch

from torch_spyre._C import launch_kernel_from_bytes
from ._embedding_host import build_embedding_launches, EmbeddingHostError  # EmbeddingHostError re-exported so callers can catch the op's host-side failure mode

__all__ = ["embedding", "EmbeddingHostError"]

_DTYPE_BITS = {torch.float16: 16}  # DL16 on 1p0; extend per generation


@torch.library.custom_op("spyre::embedding", mutates_args=(), device_types="spyre")
def embedding(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if weight.device.type != "spyre":
        raise ValueError(f"embedding weight must be on spyre device, got {weight.device}")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"embedding indices must be integer, got {indices.dtype}")
    if weight.dtype not in _DTYPE_BITS:
        raise ValueError(f"embedding weight dtype {weight.dtype} unsupported (1p0: float16)")

    vocab, d_model = int(weight.shape[0]), int(weight.shape[1])
    element_bits = _DTYPE_BITS[weight.dtype]
    flat_idx = indices.reshape(-1).cpu().tolist()

    launches, tokens_per_launch = build_embedding_launches(vocab, d_model, element_bits, flat_idx)

    # Assemble output on CPU then move to device once.
    # Device-side strided slice-assign (out[start:end] = ...) ignores the start
    # offset on spyre hardware — launch1's rows land at 0..43 instead of 256..299,
    # corrupting multi-launch output.  CPU staging avoids this entirely: each
    # launch's buffer is copied down and placed into the correct CPU rows; a single
    # .to(device) at the end moves the fully-assembled tensor to the device.
    out_cpu = torch.empty((len(flat_idx), d_model), dtype=weight.dtype)
    for (start, end), binary in launches:
        # Each binary writes exactly tokens_per_launch output rows regardless of
        # how many real tokens are in this launch (the final launch may be partial).
        # Allocate a full-sized device buffer so the binary never overruns a shorter
        # slice, then copy only the real rows to the CPU staging tensor.
        buf = torch.empty((tokens_per_launch, d_model), dtype=weight.dtype, device=weight.device)
        launch_kernel_from_bytes(binary, [weight, buf])
        out_cpu[start:end] = buf[: end - start].cpu()
    return out_cpu.to(weight.device).reshape(*indices.shape, d_model)


@embedding.register_fake
def _(weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return weight.new_empty((*indices.shape, weight.shape[1]))


# Route aten.embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False,
# sparse=False) on spyre to spyre::embedding. padding_idx is forward-irrelevant
# (only affects backward) so it is ignored; the training-only flags are rejected.
@torch.library.register_kernel("aten::embedding", ["spyre"])
def _aten_embedding_spyre(weight, indices, padding_idx=-1,
                           scale_grad_by_freq=False, sparse=False):
    if scale_grad_by_freq:
        raise NotImplementedError("spyre embedding: scale_grad_by_freq unsupported")
    if sparse:
        raise NotImplementedError("spyre embedding: sparse unsupported")
    return torch.ops.spyre.embedding(weight, indices)
