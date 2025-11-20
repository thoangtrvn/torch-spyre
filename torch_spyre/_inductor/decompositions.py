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

from typing import Optional, Sequence
import torch

from torch._inductor.decomposition import register_decomposition


# List of decompositions to be re-defined in this file
decomps_to_exclude = [
    torch.ops.aten.cat.default
]
torch._decomp.remove_decompositions(torch._inductor.decomposition.decompositions, decomps_to_exclude)

@register_decomposition([torch.ops.spyre.compact])
def compact_decomp(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.spyre.slice(torch.ops.spyre.swap(x))


@register_decomposition([torch.ops.spyre.layer_norm])
def layernorm_decomp(
    input: torch.Tensor,
    normalized_shape: list[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    mean = torch.ops.spyre.exx2(input, 1.0 / normalized_shape[0], False)
    norm_mean = torch.ops.spyre.layernormscale(mean, eps)
    return torch.ops.spyre.layernormnorm(input, mean, norm_mean, weight, bias)


@register_decomposition([torch.ops.aten.cat.default])
def decompose_cat(
    tensors: list[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    orig_decomp = torch._inductor.decomposition.cat(tensors, dim)
    if orig_decomp == NotImplemented:
        expanded_size = 0
        for t in tensors:
            expanded_size += t.size(dim)
        output_size = list(tensors[0].size())
        output_size[dim] = expanded_size
        output = tensors[0].new_empty(output_size)
        output.spyre_dci = output.get_dci()
        offset = 0
        for input in tensors:
            output = torch.ops.spyre.cat(input=input, output=output, dim=dim, offset=offset)
            offset += input.size(dim)
        return output
    else:
        return orig_decomp


"""
Hook torch.nn.functional.layer_norm to select spyre optimized version where applicable
"""
orig_layer_norm = torch.nn.functional.layer_norm


def spyre_layer_norm(
    input: torch.Tensor,
    normalized_shape: Sequence[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    if input.device.type == "spyre" and len(normalized_shape) == 1:
        return torch.ops.spyre.layer_norm(input, normalized_shape, weight, bias, eps)
    else:
        return orig_layer_norm(input, normalized_shape, weight, bias, eps)


torch.nn.functional.layer_norm = spyre_layer_norm
