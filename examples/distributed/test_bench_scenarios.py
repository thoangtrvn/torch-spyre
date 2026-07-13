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

"""Hardware-free unit tests for bench_model_configs.py / bench_scenarios.py.

No torchrun, no Spyre device needed -- run with plain pytest:
    pytest examples/distributed/test_bench_scenarios.py
"""

import pytest
import torch

from bench_model_configs import DENSE_MODELS, MOE_MODELS
from bench_scenarios import (
    WORKLOAD_POINTS,
    filter_by_names,
    gen_allgather_scenario,
    gen_allreduce_scenario,
    gen_alltoall_proxy_scenario,
    generate_dense_scenarios,
    generate_moe_scenarios,
)

ALL_MODEL_NAMES = {m.name for m in DENSE_MODELS} | {m.name for m in MOE_MODELS}


def test_dense_model_names_are_unique():
    names = [m.name for m in DENSE_MODELS]
    assert len(names) == len(set(names))


def test_moe_model_names_are_unique():
    names = [m.name for m in MOE_MODELS]
    assert len(names) == len(set(names))


def test_workload_point_names_are_unique():
    names = [wp.name for wp in WORKLOAD_POINTS]
    assert len(names) == len(set(names))


def test_dense_and_moe_model_names_dont_collide():
    dense_names = {m.name for m in DENSE_MODELS}
    moe_names = {m.name for m in MOE_MODELS}
    assert dense_names.isdisjoint(moe_names)


def test_generate_dense_scenarios_total_count():
    scenarios = generate_dense_scenarios(DENSE_MODELS, WORKLOAD_POINTS, world_size=4)
    assert (
        len(scenarios) == len(DENSE_MODELS) * len(WORKLOAD_POINTS) * 2
    )  # allreduce + allgather


def test_generate_dense_scenarios_single_op():
    scenarios = generate_dense_scenarios(
        DENSE_MODELS, WORKLOAD_POINTS, world_size=4, ops=("allreduce",)
    )
    assert len(scenarios) == len(DENSE_MODELS) * len(WORKLOAD_POINTS)
    assert all(s.benchmark == "allreduce" for s in scenarios)


def test_generate_moe_scenarios_total_count():
    scenarios = generate_moe_scenarios(MOE_MODELS, WORKLOAD_POINTS, world_size=4)
    assert len(scenarios) == len(MOE_MODELS) * len(WORKLOAD_POINTS)
    assert all(s.benchmark == "alltoall_proxy" for s in scenarios)


def test_allreduce_element_count_matches_hand_computed_value():
    # GPT-2 124M: hidden_size=768. decode_b1: batch=1, seq_len=1.
    gpt2 = next(m for m in DENSE_MODELS if m.name == "gpt2-124m")
    decode_b1 = next(wp for wp in WORKLOAD_POINTS if wp.name == "decode_b1")
    scenario = gen_allreduce_scenario(gpt2, decode_b1, world_size=4)
    assert scenario.elements == 768
    assert scenario.dtype == torch.float16

    # decode_b8: batch=8, seq_len=1 -> 8 * 768
    decode_b8 = next(wp for wp in WORKLOAD_POINTS if wp.name == "decode_b8")
    scenario = gen_allreduce_scenario(gpt2, decode_b8, world_size=4)
    assert scenario.elements == 8 * 768

    # prefill_s2048: batch=1, seq_len=2048 -> 2048 * 768
    prefill = next(wp for wp in WORKLOAD_POINTS if wp.name == "prefill_s2048")
    scenario = gen_allreduce_scenario(gpt2, prefill, world_size=4)
    assert scenario.elements == 2048 * 768


def test_allgather_element_count_matches_hand_computed_value():
    # GPT-2 124M: vocab_size=50257. decode_b1 at world_size=4:
    # per-rank shard = ceil(50257/4) = 12565; elements = 1*1*12565.
    gpt2 = next(m for m in DENSE_MODELS if m.name == "gpt2-124m")
    decode_b1 = next(wp for wp in WORKLOAD_POINTS if wp.name == "decode_b1")
    scenario = gen_allgather_scenario(gpt2, decode_b1, world_size=4)
    assert scenario.elements == 12565  # ceil(50257 / 4)


def test_alltoall_proxy_tokens_to_expert_matches_hand_computed_value():
    # Mixtral-8x7b-style: hidden_size=4096, num_experts=8, top_k=2.
    # decode_b8: batch=8, seq_len=1 -> tokens_to_expert = 8*1*2/8 = 2.0
    mixtral = next(m for m in MOE_MODELS if m.name == "mixtral-8x7b-style")
    decode_b8 = next(wp for wp in WORKLOAD_POINTS if wp.name == "decode_b8")
    scenario = gen_alltoall_proxy_scenario(mixtral, decode_b8, world_size=4)
    assert scenario.tokens_to_expert == pytest.approx(2.0)
    assert scenario.elements == round(2.0) * 4096


def test_alltoall_proxy_elements_never_zero_even_for_tiny_ratios():
    # DeepSeek-V2-Lite-style: 64 experts, top_k=6. decode_b1 (batch=1,seq_len=1):
    # tokens_to_expert = 1*1*6/64 = 0.09375 -> rounds to 0, but elements must
    # still be at least 1*hidden_size, never zero.
    deepseek = next(m for m in MOE_MODELS if m.name == "deepseek-v2-lite-style")
    decode_b1 = next(wp for wp in WORKLOAD_POINTS if wp.name == "decode_b1")
    scenario = gen_alltoall_proxy_scenario(deepseek, decode_b1, world_size=4)
    assert scenario.tokens_to_expert == pytest.approx(6 / 64)
    assert scenario.elements == 1 * 2048  # max(1, round(0.09375)) * hidden_size


def test_filter_by_names_all_returns_full_table():
    assert filter_by_names("all", DENSE_MODELS) == list(DENSE_MODELS)
    assert filter_by_names(None, DENSE_MODELS) == list(DENSE_MODELS)


def test_filter_by_names_comma_list():
    result = filter_by_names("gpt2-124m,pythia-70m", DENSE_MODELS)
    assert {m.name for m in result} == {"gpt2-124m", "pythia-70m"}


def test_filter_by_names_raises_on_genuine_typo():
    with pytest.raises(ValueError, match="totally-bogus-model"):
        filter_by_names(
            "totally-bogus-model", DENSE_MODELS, valid_names=ALL_MODEL_NAMES
        )


def test_filter_by_names_cross_table_name_yields_empty_not_raise():
    # "gpt2-124m" is a valid name overall (in DENSE_MODELS), but filtering
    # MOE_MODELS by it should yield an empty list, not raise -- the same
    # --models flag is applied to both tables in bench_distributed.py.
    result = filter_by_names("gpt2-124m", MOE_MODELS, valid_names=ALL_MODEL_NAMES)
    assert result == []


def test_filter_by_names_without_valid_names_checks_only_this_table():
    # Without valid_names, a name from the OTHER table looks like a typo
    # against THIS table and should raise -- this is the single-table usage
    # (e.g. calling filter_by_names on just one table in isolation).
    with pytest.raises(ValueError):
        filter_by_names("mixtral-8x7b-style", DENSE_MODELS)
