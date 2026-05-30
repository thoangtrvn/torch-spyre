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

from torch_spyre.constants import DEVICE_NAME
from .patches import enable_spyre_context

import threading
from functools import wraps

_autoload_lock = threading.Lock()


def enable_spyre_compile_fx_wrapper():
    import torch._inductor.compile_fx as cfx
    import torch.fx as fx
    import torch

    if getattr(cfx, "_spyre_wrapped", False):
        return
    with _autoload_lock:
        if getattr(cfx, "_spyre_wrapped", False):
            return
        _orig = cfx.compile_fx

        # Iterate over producer nodes (supports nested containers of nodes)
        def iter_nodes(x):
            if isinstance(x, fx.Node):
                yield x
            elif isinstance(x, (tuple, list)):
                for e in x:
                    yield from iter_nodes(e)
            elif isinstance(x, dict):
                for e in x.values():
                    yield from iter_nodes(e)

        def iter_tensors(v):
            if isinstance(v, torch.Tensor):
                yield v  # FakeTensor is a Tensor subclass, so this works
            elif isinstance(v, (tuple, list)):
                for e in v:
                    yield from iter_tensors(e)
            elif isinstance(v, dict):
                for e in v.values():
                    yield from iter_tensors(e)

        def _uses_spyre(gm, example_inputs, device_name=DEVICE_NAME) -> bool:
            # Inputs
            if any(
                isinstance(x, torch.Tensor)
                and getattr(x.device, "type", None) == device_name
                for x in (example_inputs or ())
            ):
                return True
            # Output
            out_node = gm.graph.output_node()
            out_puts = out_node.args[0] if out_node.args else []
            for n in iter_nodes(out_puts):
                meta = getattr(n, "meta", {}) or {}
                mv = meta.get("val", None) or meta.get("example_value", None)
                if mv is None:
                    continue

                if any(
                    getattr(getattr(t, "device", None), "type", None) == device_name
                    for t in iter_tensors(mv)
                ):
                    return True

            # Graph nodes (covers tensorless factories)
            for n in gm.graph.nodes:
                dev = n.kwargs.get("device")
                if dev is None:
                    continue

                if isinstance(dev, torch.device) and dev.type == device_name:
                    return True
                if isinstance(dev, str) and dev.split(":")[0] == device_name:
                    return True
            return False

        @wraps(_orig)
        def _wrapper(gm, example_inputs, *args, **kwargs):
            decomps = kwargs.setdefault(
                "decompositions", torch._inductor.decomposition.decompositions
            )

            if _uses_spyre(gm, example_inputs):
                torch.spyre._impl._lazy_init()

                with enable_spyre_context(
                    example_inputs, decomps=decomps
                ) as spyre_context_decompositions:
                    # The `decomps` is the updated in the context manager
                    # with the appropriate spyre decompositions
                    # and yielded as `spyre_context_decompositions` from the CM

                    kwargs["decompositions"] = spyre_context_decompositions

                    return _orig(
                        gm,
                        example_inputs,
                        *args,
                        **kwargs,
                    )

            return _orig(gm, example_inputs, *args, **kwargs)

        cfx.compile_fx = _wrapper
        cfx._spyre_wrapped = True


def _light_autoload():
    from . import decompositions  # noqa: F401

    enable_spyre_compile_fx_wrapper()


def _autoload():
    if getattr(_autoload, "_ran", False):
        return

    with _autoload_lock:
        if getattr(_autoload, "_ran", False):
            return
        from torch._dynamo.device_interface import register_interface_for_device

        from torch_spyre.device.interface import SpyreInterface

        register_interface_for_device(DEVICE_NAME, SpyreInterface)

        from torch._inductor.codegen.common import (
            register_backend_for_device,
            register_device_op_overrides,
        )

        # Register in-tree CPU and CUDA device
        from torch._inductor.codegen import cpu_device_op_overrides  # noqa: F401  # usort: skip
        from torch._inductor.codegen.cuda import device_op_overrides  # noqa: F401  # usort: skip

        from torch_spyre.device.op_overrides import SpyreDeviceOpOverrides

        register_device_op_overrides(
            device=DEVICE_NAME, device_op_overrides=SpyreDeviceOpOverrides()
        )

        from torch._inductor.codegen.triton import TritonScheduling
        from torch._inductor import config
        from .choices import SpyreHeuristics
        from .wrapper import SpyrePythonWrapperCodegen

        register_backend_for_device(
            DEVICE_NAME,
            TritonScheduling,
            SpyrePythonWrapperCodegen,
        )

        # Register SpyreHeuristics as the Inductor choices class so
        # V.choices.triton_kernel_kwargs() is called during scheduling.
        # Without this, V.choices defaults to base InductorChoices and
        # problem_m/n/k are never set in kernel_kwargs.
        config.inductor_choices_class = SpyreHeuristics

        # Monkey-patch CachingAutotuner to transport Spyre-specific fields
        # (problem_m/n/k) from cfg.kwargs to backend.parse_options().
        #
        # Problem: SpyreHeuristics.triton_kernel_kwargs() embeds problem_m/n/k
        # inside FixedTritonConfig.config, which flows to cfg.kwargs at runtime.
        # But Inductor's CachingAutotuner has TWO issues:
        #
        # 1. _create_compile_meta() adds ALL cfg.kwargs to
        #    compile_meta["constants"] (line 751), then ASTSource.__init__()
        #    tries fn.arg_names.index(k) on each string key. problem_m/n/k are
        #    NOT Triton kernel parameters → ValueError: 'problem_m' is not in
        #    list.
        #
        # 2. _create_compile_options() only includes standard GPU options
        #    (num_warps, num_stages, debug, sanitize_overflow). Custom backend
        #    fields in cfg.kwargs are stripped out, so backend.parse_options()
        #    never sees problem_m/n/k, and SpyreOptions defaults to
        #    problem_m=0/n=0/k=0.
        #
        # Fix: Two patches following the HIP precedent (lines 746-750 and
        # 817-821 in triton_heuristics.py) — device-specific keys that aren't
        # kernel constexpr params get removed from constants and added to
        # options instead.
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        _SPYRE_PROBLEM_KEYS = frozenset({"problem_m", "problem_n", "problem_k"})

        _orig_create_compile_meta = CachingAutotuner._create_compile_meta

        def _spyre_create_compile_meta(self, cfg):
            compile_meta = _orig_create_compile_meta(self, cfg)
            # Remove Spyre-specific keys that were added to
            # compile_meta["constants"] by the original method. These are NOT
            # kernel constexpr params — they're backend-specific options.
            # Without this removal, ASTSource.__init__() would try
            # fn.arg_names.index("problem_m") and raise ValueError because
            # "problem_m" is not a kernel parameter name.
            for key in _SPYRE_PROBLEM_KEYS:
                compile_meta["constants"].pop(key, None)
            return compile_meta

        _orig_create_compile_options = CachingAutotuner._create_compile_options

        def _spyre_create_compile_options(self, cfg, compile_meta):
            options = _orig_create_compile_options(self, cfg, compile_meta)
            # Inject Spyre-specific keys from cfg.kwargs into the options dict.
            # The original _create_compile_options only includes standard GPU
            # options. Spyre's problem_m/n/k need to flow through to
            # backend.parse_options() → SpyreOptions.
            for key in _SPYRE_PROBLEM_KEYS:
                if key in cfg.kwargs:
                    options[key] = cfg.kwargs[key]
            return options

        if not getattr(CachingAutotuner, "_spyre_patched", False):
            CachingAutotuner._create_compile_meta = _spyre_create_compile_meta
            CachingAutotuner._create_compile_options = _spyre_create_compile_options
            CachingAutotuner._spyre_patched = True

        _autoload._ran = True
