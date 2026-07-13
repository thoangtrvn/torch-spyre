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

"""Model-derived communication scenario generation.

Turns a compact model-config table (bench_model_configs.py) plus a small set
of representative workload points into the full AllReduce/AllGather/
all-to-all-proxy scenario matrix, instead of hand-coding one dict entry per
(model, workload point, op) combination.

No torch.distributed/hardware dependency in this module -- world_size is
always passed in as a parameter rather than read from `dist`, so everything
here is testable with plain pytest, without torchrun or a Spyre device.
"""

import math
from dataclasses import dataclass

import torch

from bench_model_configs import DenseModelConfig, MoEModelConfig

DEFAULT_DTYPE = torch.float16


@dataclass(frozen=True)
class WorkloadPoint:
    """A representative (batch, seq_len) point on the decode/prefill curve."""

    name: str
    phase: str  # "decode" | "prefill"
    batch: int
    seq_len: int


# Decode: continuous-batching serving, one new token per step (seq_len=1).
# Prefill: time-to-first-token / throughput path, single request, long prompt.
WORKLOAD_POINTS: list[WorkloadPoint] = [
    WorkloadPoint("decode_b1", "decode", batch=1, seq_len=1),
    WorkloadPoint("decode_b8", "decode", batch=8, seq_len=1),
    WorkloadPoint("decode_b32", "decode", batch=32, seq_len=1),
    WorkloadPoint("prefill_s2048", "prefill", batch=1, seq_len=2048),
    WorkloadPoint("prefill_s4096", "prefill", batch=1, seq_len=4096),
]


@dataclass(frozen=True)
class Scenario:
    """Fully materialized scenario, ready to feed to a bench_* runner in
    bench_distributed.py."""

    name: str
    benchmark: str  # "allreduce" | "allgather" | "alltoall_proxy"
    model_name: str
    workload_point: str
    phase: str
    batch: int
    seq_len: int
    hidden_size: int
    elements: int
    dtype: torch.dtype
    description: str
    world_size: int
    num_layers: int | None = None
    vocab_size: int | None = None  # allgather only
    num_experts: int | None = None  # alltoall_proxy only
    top_k: int | None = None  # alltoall_proxy only
    tokens_to_expert: float | None = None  # alltoall_proxy only, pre-rounding


def gen_allreduce_scenario(
    model: DenseModelConfig,
    wp: WorkloadPoint,
    world_size: int,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> Scenario:
    """TP attention-output / FFN-down-proj AllReduce.

    elements = batch * seq_len * hidden_size -- the full, already-projected
    activation tensor that participates in the reduction (same shape on every
    rank; AllReduce doesn't shard by rank the way AllGather/all-to-all do).
    """
    elements = model.hidden_size * wp.batch * wp.seq_len
    return Scenario(
        name=f"ar_{model.name}_{wp.name}",
        benchmark="allreduce",
        model_name=model.name,
        workload_point=wp.name,
        phase=wp.phase,
        batch=wp.batch,
        seq_len=wp.seq_len,
        hidden_size=model.hidden_size,
        elements=elements,
        dtype=dtype,
        description=f"AllReduce: {model.name} {wp.phase} (batch={wp.batch}, seq_len={wp.seq_len})",
        world_size=world_size,
        num_layers=model.num_layers,
        vocab_size=model.vocab_size,
    )


def gen_allgather_scenario(
    model: DenseModelConfig,
    wp: WorkloadPoint,
    world_size: int,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> Scenario:
    """Vocab-parallel embedding/logits AllGather.

    Each rank holds an even shard of the vocab dimension; per-rank input
    elements = batch * seq_len * ceil(vocab_size / world_size). AllGather
    assembles the full [batch*seq_len, vocab_size] logits on every rank.
    """
    vocab_shard = math.ceil(model.vocab_size / world_size)
    elements = vocab_shard * wp.batch * wp.seq_len
    return Scenario(
        name=f"ag_{model.name}_{wp.name}",
        benchmark="allgather",
        model_name=model.name,
        workload_point=wp.name,
        phase=wp.phase,
        batch=wp.batch,
        seq_len=wp.seq_len,
        hidden_size=model.hidden_size,
        elements=elements,
        dtype=dtype,
        description=(
            f"AllGather: {model.name} vocab shard {wp.phase} (batch={wp.batch}, seq_len={wp.seq_len})"
        ),
        world_size=world_size,
        num_layers=model.num_layers,
        vocab_size=model.vocab_size,
    )


def gen_alltoall_proxy_scenario(
    model: MoEModelConfig,
    wp: WorkloadPoint,
    world_size: int,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> Scenario:
    """MoE expert-routing all-to-all PROXY (see bench_alltoall_proxy()).

    tokens_to_expert = batch*seq_len*top_k/num_experts is a uniform-routing
    approximation (real routing is data-dependent at runtime -- this is the
    expected value under a balanced router, not a measured distribution).
    elements = round(tokens_to_expert) * hidden_size = size of ONE pairwise
    message in the proxy's N-1 pairwise-sendrecv exchange.
    """
    tokens_to_expert = (wp.batch * wp.seq_len * model.top_k) / model.num_experts
    elements = max(1, round(tokens_to_expert)) * model.hidden_size
    return Scenario(
        name=f"a2a_{model.name}_{wp.name}",
        benchmark="alltoall_proxy",
        model_name=model.name,
        workload_point=wp.name,
        phase=wp.phase,
        batch=wp.batch,
        seq_len=wp.seq_len,
        hidden_size=model.hidden_size,
        elements=elements,
        dtype=dtype,
        description=(
            f"AllToAll PROXY: {model.name} routing {wp.phase} (batch={wp.batch}, "
            f"seq_len={wp.seq_len}, top_k={model.top_k}/{model.num_experts} experts)"
        ),
        world_size=world_size,
        num_layers=model.num_layers,
        num_experts=model.num_experts,
        top_k=model.top_k,
        tokens_to_expert=tokens_to_expert,
    )


def generate_dense_scenarios(
    models: list[DenseModelConfig],
    workload_points: list[WorkloadPoint],
    world_size: int,
    ops: tuple[str, ...] = ("allreduce", "allgather"),
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> list[Scenario]:
    """Cartesian product of models x workload points x ops.

    Replaces hand-coding one dict entry per combination (26 models x 5
    workload points x 2 ops = 260 scenarios today).
    """
    scenarios = []
    for model in models:
        for wp in workload_points:
            if "allreduce" in ops:
                scenarios.append(gen_allreduce_scenario(model, wp, world_size, dtype))
            if "allgather" in ops:
                scenarios.append(gen_allgather_scenario(model, wp, world_size, dtype))
    return scenarios


def generate_moe_scenarios(
    models: list[MoEModelConfig],
    workload_points: list[WorkloadPoint],
    world_size: int,
    dtype: torch.dtype = DEFAULT_DTYPE,
) -> list[Scenario]:
    """Cartesian product of MoE models x workload points (all-to-all proxy only)."""
    return [
        gen_alltoall_proxy_scenario(model, wp, world_size, dtype)
        for model in models
        for wp in workload_points
    ]


def filter_by_names(
    names_csv: str, table: list, key: str = "name", valid_names: set | None = None
) -> list:
    """Filter a table (DenseModelConfig/MoEModelConfig/WorkloadPoint list) by a
    comma-separated list of names, or return the whole table for "all".

    If `valid_names` is given, unknown-name validation checks against that
    broader set instead of just this table's own names. This lets one
    --models flag be applied to two different tables (e.g. dense models and
    MoE models) without raising just because a requested name belongs to the
    OTHER table -- it simply yields zero matches from THIS table in that
    case. Raises ValueError only if a name isn't in `valid_names` (or this
    table's own names, if valid_names is None) at all -- fails loudly on a
    genuine typo instead of silently running zero scenarios everywhere.
    """
    if names_csv is None or names_csv.strip().lower() == "all":
        return list(table)

    requested = {n.strip() for n in names_csv.split(",") if n.strip()}
    table_names = {getattr(item, key) for item in table}
    check_against = valid_names if valid_names is not None else table_names
    unknown = requested - check_against
    if unknown:
        raise ValueError(
            f"Unknown name(s) {sorted(unknown)!r}. Valid names: {sorted(check_against)!r}"
        )
    return [item for item in table if getattr(item, key) in requested]
