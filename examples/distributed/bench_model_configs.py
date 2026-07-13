# Copyright 2026 The Torch-Spyre Authors.
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

"""Real-model architecture tables for model-derived communication scenarios.

hidden_size/vocab_size/num_hidden_layers were pulled directly from each
model's HuggingFace config.json (raw-JSON fetch, not a summarizer -- one
lookup pass caught a summarization tool returning a different model's cached
values for an edge case, so all values here were re-verified against the
raw config). No hardware/torch.distributed dependency in this module -- it
is plain data, testable with pytest alone.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DenseModelConfig:
    """Architecture parameters needed to size TP AllReduce/AllGather messages."""

    name: str
    hidden_size: int
    vocab_size: int
    num_layers: int
    notes: str = ""


@dataclass(frozen=True)
class MoEModelConfig:
    """Architecture parameters needed to size an MoE all-to-all proxy message."""

    name: str
    hidden_size: int
    num_experts: int
    top_k: int
    expert_intermediate_size: int
    num_layers: int
    num_shared_experts: int = 0
    notes: str = ""


# hidden_size sizes the TP AllReduce message: the attention output projection
# and FFN down-projection each produce a [batch, seq_len, hidden_size] tensor
# that gets all-reduced across TP ranks.
#
# vocab_size sizes the vocab-parallel AllGather message: a sharded
# embedding/logits table gathered across TP ranks.
DENSE_MODELS: list[DenseModelConfig] = [
    DenseModelConfig("qwen3-0.6b", hidden_size=1024, vocab_size=151936, num_layers=28),
    DenseModelConfig(
        "granite-3.3-8b", hidden_size=4096, vocab_size=49159, num_layers=40
    ),
    DenseModelConfig(
        "granite-3.3-2b", hidden_size=2048, vocab_size=49159, num_layers=40
    ),
    DenseModelConfig(
        "granite-4.0-1b",
        hidden_size=2048,
        vocab_size=100352,
        num_layers=40,
        notes="granitemoehybrid model class, but this checkpoint is configured dense (0 experts)",
    ),
    DenseModelConfig("smollm3-3b", hidden_size=2048, vocab_size=128256, num_layers=36),
    DenseModelConfig(
        "llama-3.2-3b", hidden_size=3072, vocab_size=128256, num_layers=28
    ),
    DenseModelConfig(
        "tinyllama-1.1b", hidden_size=2048, vocab_size=32000, num_layers=22
    ),
    DenseModelConfig("qwen2.5-7b", hidden_size=3584, vocab_size=152064, num_layers=28),
    DenseModelConfig(
        "qwen2.5-1.5b", hidden_size=1536, vocab_size=151936, num_layers=28
    ),
    DenseModelConfig(
        "mistral-7b-v0.3", hidden_size=4096, vocab_size=32768, num_layers=32
    ),
    DenseModelConfig("phi-4-mini", hidden_size=3072, vocab_size=200064, num_layers=32),
    DenseModelConfig("phi-3.5-mini", hidden_size=3072, vocab_size=32064, num_layers=32),
    DenseModelConfig("olmo-1b", hidden_size=2048, vocab_size=50304, num_layers=16),
    DenseModelConfig("olmo2-1b", hidden_size=2048, vocab_size=100352, num_layers=16),
    DenseModelConfig("falcon3-1b", hidden_size=2048, vocab_size=131072, num_layers=18),
    DenseModelConfig(
        "deepseek-coder-1.3b", hidden_size=2048, vocab_size=32256, num_layers=24
    ),
    DenseModelConfig("yi-1.5-6b", hidden_size=4096, vocab_size=64000, num_layers=32),
    DenseModelConfig(
        "granite-vision-4.1-4b",
        hidden_size=2560,
        vocab_size=100353,
        num_layers=40,
        notes="text backbone only, extracted from text_config of a vision-language model",
    ),
    DenseModelConfig(
        "gemma-4-12b",
        hidden_size=3840,
        vocab_size=262144,
        num_layers=48,
        notes="text backbone only, extracted from text_config of a unified any-to-any model",
    ),
    DenseModelConfig("gemma-3-1b", hidden_size=1152, vocab_size=262144, num_layers=26),
    DenseModelConfig("gpt2-124m", hidden_size=768, vocab_size=50257, num_layers=12),
    DenseModelConfig("gpt-neo-125m", hidden_size=768, vocab_size=50257, num_layers=12),
    DenseModelConfig("pythia-70m", hidden_size=512, vocab_size=50304, num_layers=6),
    DenseModelConfig(
        "ministral-8b-instruct", hidden_size=4096, vocab_size=131072, num_layers=36
    ),
    DenseModelConfig(
        "mistral-small-3-24b", hidden_size=5120, vocab_size=131072, num_layers=40
    ),
    DenseModelConfig(
        "ministral-3-14b-instruct",
        hidden_size=5120,
        vocab_size=131072,
        num_layers=40,
        notes=(
            "UNVERIFIED CAVEAT: hidden_size/vocab_size/num_layers came back byte-identical to "
            "mistral-small-3-24b despite being a different-sized model. Plausible (shared "
            "backbone template, differing only in FFN width/other dims not tracked here), but "
            "not independently re-confirmed -- treat with extra skepticism vs. the other rows."
        ),
    ),
]

# num_experts/top_k/expert_intermediate_size size the MoE all-to-all PROXY
# message (see bench_alltoall_proxy in bench_distributed.py) -- this is a
# lower-bound proxy via pairwise sendrecv, not a native all-to-all collective
# (spyre-comms/torch_spyre has no working end-to-end all_to_all today).
MOE_MODELS: list[MoEModelConfig] = [
    MoEModelConfig(
        "mixtral-8x7b-style",
        hidden_size=4096,
        num_experts=8,
        top_k=2,
        expert_intermediate_size=14336,
        num_layers=32,
        num_shared_experts=0,
        notes="coarse-grained reference MoE (Mixtral 8x7B config)",
    ),
    MoEModelConfig(
        "deepseek-v2-lite-style",
        hidden_size=2048,
        num_experts=64,
        top_k=6,
        expert_intermediate_size=1408,
        num_layers=27,
        num_shared_experts=2,
        notes="fine-grained shared+routed expert MoE (DeepSeek-V2-Lite config)",
    ),
]
