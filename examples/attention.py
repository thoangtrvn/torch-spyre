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

"""Scaled-dot-product attention on the Spyre device, eager and compiled.

This example runs ``F.scaled_dot_product_attention`` at the route's
witnessed envelope — B=1, H=1, rows=8, head_dim=64, kv_len=128 (DL16,
``torch.float16``) — through the decode-tail host contract: the K/V
tensors hold only positions strictly prior to the query block, so every
staged position is visible to every query row and ``is_causal`` is left
False (the causal-at-rows>1 form is ATen's bottom-right-aligned tril,
which this route does not compute and refuses typed — the route's
``is_causal=True`` admission is faithful only at rows == 1, pure decode,
where the full-visibility construction coincides with the causal mask).

kv_len=128 is two 64-position blocks: the multiblock owner's HOST_SERIAL
sequence (per-block launches with an fp64 host-side fold) — the same
execution form as the eager silicon e-cell at this shape.

The compiled section (``torch.compile(..., dynamic=False)``) routes
through the SAME seam as eager dispatch: the top-level SDPA op is
decomposed to the overrideable backend, whose Spyre decomposition maps
the call to ``spyre::attention`` (the fused attention program). The
graph compiles with ZERO graph breaks at this cell and the out-of-
envelope refusal below surfaces at compile time with the admission
predicate's own clause name.

VERIFICATION STATUS (read before relying on this):
- The compiled path is verified OFF-BOX against the phase simulator
  (values vs the fp64 flash reference, the plan's own launch sequence);
  it is SILICON-UNWITNESSED — the compiled cell on the box is pending.
- EAGER ROUTING DISCLOSURE: registering the top-level decomposition
  reroutes EAGER ``F.scaled_dot_product_attention`` through the Spyre
  seam (it no longer runs the ATen C++ composite). The destination is
  the same overrideable backend the composite selects on the device,
  but the route differs from the pre-change e-cell path and must be
  re-witnessed on silicon before the eager claim is reused.
- This route is INFERENCE-ONLY: the underlying ``spyre::attention`` is
  a custom op with no autograd formula, and the overrideable contract's
  aux outputs (logsumexp, philox seed/offset) are unpopulated dummies —
  do not build training graphs through it.

Everything outside the v1 envelope (rows > 32, head_dim not a whole
stick or head_dim > 512, kv_len not a whole number of 64-position
blocks, dropout, explicit masks, dtypes other than fp16) refuses typed
— there is deliberately no CPU fallback.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("spyre")

ROWS = 8
HEAD_DIM = 64
KV_LEN = 128  # two 64-position blocks (the HOST_SERIAL path)

# DL16-friendly inputs (small integer magnitudes; the route's KAT recipe)
g = torch.Generator().manual_seed(0xAFFE)
q = torch.randint(1, 4, (ROWS, HEAD_DIM), generator=g, dtype=torch.float16)
k = torch.randint(-1, 2, (KV_LEN, HEAD_DIM), generator=g, dtype=torch.float16)
v = torch.randint(1, 16, (KV_LEN, HEAD_DIM), generator=g, dtype=torch.float16)

# fp64 flash reference on the host
qf = q.double().numpy()
kf = k.double().numpy()
vf = v.double().numpy()
scores = qf @ kf.T / math.sqrt(HEAD_DIM)
scores -= scores.max(axis=-1, keepdims=True)
exp = np.exp(scores)
probs = exp / exp.sum(axis=-1, keepdims=True)
reference = probs @ vf

# Send inputs to the device
q_dev = q.reshape(1, 1, ROWS, HEAD_DIM).to(DEVICE)
k_dev = k.reshape(1, 1, KV_LEN, HEAD_DIM).to(DEVICE)
v_dev = v.reshape(1, 1, KV_LEN, HEAD_DIM).to(DEVICE)

# Compute attention on the device in eager mode
eager_result = (
    F.scaled_dot_product_attention(q_dev, k_dev, v_dev).cpu().reshape(ROWS, HEAD_DIM)
)

# Compute attention on the device in compiled mode
compiled_attention = torch.compile(
    lambda q_, k_, v_: F.scaled_dot_product_attention(q_, k_, v_), dynamic=False
)
compiled_result = compiled_attention(q_dev, k_dev, v_dev).cpu().reshape(ROWS, HEAD_DIM)

# Print the results and compare them
print(f"Reference (fp64 flash)\n{reference}")
print(f"Spyre eager result\n{eager_result}")
print(f"Spyre compiled result\n{compiled_result}")
eager_delta = float((eager_result.double().numpy() - reference).abs().max())
compiled_delta = float((compiled_result.double().numpy() - reference).abs().max())
eager_vs_compiled = float((eager_result.float() - compiled_result.float()).abs().max())
print(f"Max delta eager vs. fp64 reference: {eager_delta:.4f}")
print(f"Max delta compiled vs. fp64 reference: {compiled_delta:.4f}")
print(f"Max delta compiled vs. eager: {eager_vs_compiled:.4f}")

# The refusal surface: one out-of-envelope call (rows=33 > the v1 row
# ceiling of 32) refuses typed with the predicate's own clause name —
# under torch.compile the refusal surfaces at compile time.
torch._dynamo.reset()
q33 = torch.randint(1, 4, (33, HEAD_DIM), generator=g, dtype=torch.float16)
try:
    compiled_attention(q33.reshape(1, 1, 33, HEAD_DIM).to(DEVICE), k_dev, v_dev)
    print("rows=33: ADMITTED (unexpected — the v1 row ceiling is 32)")
except Exception as e:  # the typed refusal carries the clause name
    if "[row-budget]" not in str(e):
        raise  # not the expected typed refusal — do not mask it
    print("rows=33 refuses typed: [row-budget] in the message below")
    print(f"  {e}")
