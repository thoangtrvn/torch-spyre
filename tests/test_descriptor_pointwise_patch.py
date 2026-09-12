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

import os
import subprocess
import sys
import textwrap


def _run_script(script: str, *, descriptor_pointwise: bool) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["TORCH_SPYRE_CODEGEN_ONLY"] = "1"
    env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
    if descriptor_pointwise:
        env["TORCH_SPYRE_EXPERIMENTAL_DESCRIPTOR_POINTWISE"] = "1"
    else:
        env.pop("TORCH_SPYRE_EXPERIMENTAL_DESCRIPTOR_POINTWISE", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_spyre_codegen_only_autoload_registers_backend_and_light_compile_hooks():
    result = _run_script(
        """
        import torch
        from torch._inductor.codegen.common import device_op_overrides_dict

        assert torch._C._get_privateuse1_backend_name() == 'spyre'
        assert 'spyre' in device_op_overrides_dict
        assert getattr(torch._inductor.compile_fx, '_spyre_wrapped', False) is True
        """,
        descriptor_pointwise=False,
    )

    assert result.returncode == 0, result.stderr



def test_spyre_autoload_enables_descriptor_friendly_inductor_defaults():
    result = _run_script(
        """
        import torch

        assert torch._inductor.config.triton.use_tensor_descriptor is True
        assert torch._inductor.config.assume_aligned_inputs is True
        """,
        descriptor_pointwise=False,
    )

    assert result.returncode == 0, result.stderr


def test_spyre_context_preserves_descriptor_and_dtype_defaults_after_exit():
    result = _run_script(
        """
        import torch
        from torch_spyre._inductor.patches import enable_spyre_context

        original_use_tensor_descriptor = (
            torch._inductor.config.triton.use_tensor_descriptor
        )
        original_assume_aligned_inputs = (
            torch._inductor.config.assume_aligned_inputs
        )
        import torch._inductor.graph as inductor_graph
        import torch._inductor.utils as inductor_utils

        original_use_block_ptr = torch._inductor.config.triton.use_block_ptr
        original_codegen_upcast_to_fp32 = (
            torch._inductor.config.triton.codegen_upcast_to_fp32
        )
        original_graph_alignment_helper = inductor_graph.should_assume_input_aligned
        original_utils_alignment_helper = inductor_utils.should_assume_input_aligned
        spyre_input = torch.empty((512, 128), dtype=torch.float16).to('spyre')
        with enable_spyre_context([]):
            assert torch._inductor.config.triton.use_tensor_descriptor is True
            assert torch._inductor.config.assume_aligned_inputs is True
            assert torch._inductor.config.triton.use_block_ptr is original_use_block_ptr
            assert torch._inductor.config.triton.codegen_upcast_to_fp32 is False
            assert inductor_graph.should_assume_input_aligned(spyre_input) is True
            assert inductor_utils.should_assume_input_aligned(spyre_input) is True
        assert (
            torch._inductor.config.triton.use_tensor_descriptor
            is original_use_tensor_descriptor
        )
        assert (
            torch._inductor.config.assume_aligned_inputs
            is original_assume_aligned_inputs
        )
        assert torch._inductor.config.triton.use_block_ptr is original_use_block_ptr
        assert (
            torch._inductor.config.triton.codegen_upcast_to_fp32
            is original_codegen_upcast_to_fp32
        )
        assert inductor_graph.should_assume_input_aligned is original_graph_alignment_helper
        assert inductor_utils.should_assume_input_aligned is original_utils_alignment_helper
        """,
        descriptor_pointwise=False,
    )

    assert result.returncode == 0, result.stderr


def test_codegen_only_spyre_allocators_honor_fake_tensor_mode():
    result = _run_script(
        """
        import torch
        from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

        with FakeTensorMode():
            vector = torch.empty(4, device='spyre')
            matrix = torch.empty((4, 8), device='spyre')
            strided = torch.empty_strided((4, 8), (8, 1), device='spyre')
            moved = torch.empty((4, 8)).to('spyre')

        for tensor in (vector, matrix, strided, moved):
            assert isinstance(tensor, FakeTensor)
            assert tensor.device.type == 'spyre'
        assert tuple(vector.shape) == (4,)
        assert tuple(matrix.shape) == (4, 8)
        assert tuple(strided.stride()) == (8, 1)
        assert tuple(moved.shape) == (4, 8)
        """,
        descriptor_pointwise=False,
    )

    assert result.returncode == 0, result.stderr



def test_spyre_descriptor_env_disables_block_ptr_only_when_enabled():
    result = _run_script(
        """
        import torch
        from torch_spyre._inductor.patches import enable_spyre_context

        original_use_block_ptr = torch._inductor.config.triton.use_block_ptr
        with enable_spyre_context([]):
            assert torch._inductor.config.triton.use_block_ptr is False
        assert torch._inductor.config.triton.use_block_ptr is original_use_block_ptr
        """,
        descriptor_pointwise=True,
    )

    assert result.returncode == 0, result.stderr



def test_spyre_pointwise_identity_capture_does_not_implicitly_upcast_to_fp32():
    result = _run_script(
        """
        import torch
        from triton.backends.spyre.compiler import SpyreBackend

        captured = []

        class CapturedTTIR(RuntimeError):
            pass

        def capture_spyre_ir(mod, metadata, options):
            text = mod.str_nodebug() if hasattr(mod, 'str_nodebug') else str(mod)
            captured.append(text)
            raise CapturedTTIR('captured')

        SpyreBackend._make_spyre_ir = staticmethod(capture_spyre_ir)

        @torch.compile
        def identity(x):
            return x.clone()

        x = torch.empty((512, 128), dtype=torch.float16).to('spyre')
        try:
            identity(x)
        except Exception:
            if not captured:
                raise

        text = captured[0]
        assert 'tt.make_tensor_descriptor' in text
        assert 'tt.descriptor_load' in text
        assert 'tt.descriptor_store' in text
        assert 'tt.load' not in text
        assert 'tt.store' not in text
        assert 'arith.extf' not in text
        assert 'arith.truncf' not in text
        """,
        descriptor_pointwise=True,
    )

    assert result.returncode == 0, result.stderr



def test_spyre_descriptor_checker_is_context_scoped_and_env_gated():
    result = _run_script(
        """
        import torch
        from torch._inductor.codegen import triton as triton_codegen
        from torch_spyre._inductor.patches import enable_spyre_context
        orig = triton_codegen.TritonKernel.tma_compatibility_checker_cls
        with enable_spyre_context([]):
            patched = triton_codegen.TritonKernel.tma_compatibility_checker_cls
            assert patched is not orig
        assert triton_codegen.TritonKernel.tma_compatibility_checker_cls is orig
        """,
        descriptor_pointwise=True,
    )

    assert result.returncode == 0, result.stderr


def test_spyre_descriptor_checker_refuses_without_env_flag():
    result = _run_script(
        """
        import torch
        from torch._inductor.codegen import triton as triton_codegen
        from torch_spyre._inductor.patches import enable_spyre_context
        import torch_spyre._inductor.patches as patches

        class Graph:
            def get_current_device_or_throw(self):
                return torch.device('spyre')

        class Kernel:
            no_x_dim = False

        with enable_spyre_context([]):
            checker_cls = triton_codegen.TritonKernel.tma_compatibility_checker_cls
            checker = checker_cls(Kernel(), torch.float16, for_store=False, force=False, buffer_name='in_ptr0')
            with patches.V.set_graph_handler(Graph()):
                assert checker.can_use_tma() is False
        """,
        descriptor_pointwise=False,
    )

    assert result.returncode == 0, result.stderr


def test_spyre_descriptor_checker_accepts_guarded_spyre_case():
    result = _run_script(
        """
        import torch
        from torch._inductor.codegen import triton as triton_codegen
        from torch_spyre._inductor.patches import enable_spyre_context
        import torch_spyre._inductor.patches as patches

        class Buffer:
            def get_size(self):
                return [512, 128]

        class Graph:
            unaligned_buffers = set()
            graph_inputs = {'in_ptr0', 'out_ptr0'}
            constants = {}

            def get_current_device_or_throw(self):
                return torch.device('spyre')

            def try_get_buffer(self, name):
                return Buffer()

        class Kernel:
            no_x_dim = False

        with enable_spyre_context([]):
            checker_cls = triton_codegen.TritonKernel.tma_compatibility_checker_cls
            checker = checker_cls(Kernel(), torch.float16, for_store=False, force=False, buffer_name='in_ptr0')
            with patches.V.set_graph_handler(Graph()):
                assert checker.can_use_tma() is True
                assert checker.can_lift() is True
        """,
        descriptor_pointwise=True,
    )

    assert result.returncode == 0, result.stderr


def test_spyre_descriptor_checker_rejects_non_diagnostic_shape():
    result = _run_script(
        """
        import torch
        from torch._inductor.codegen import triton as triton_codegen
        from torch_spyre._inductor.patches import enable_spyre_context
        import torch_spyre._inductor.patches as patches

        class Buffer:
            def get_size(self):
                return [512, 256]

        class Graph:
            unaligned_buffers = set()
            graph_inputs = {'in_ptr0', 'out_ptr0'}
            constants = {}

            def get_current_device_or_throw(self):
                return torch.device('spyre')

            def try_get_buffer(self, name):
                return Buffer()

        class Kernel:
            no_x_dim = False

        with enable_spyre_context([]):
            checker_cls = triton_codegen.TritonKernel.tma_compatibility_checker_cls
            checker = checker_cls(Kernel(), torch.float16, for_store=False, force=False, buffer_name='in_ptr0')
            with patches.V.set_graph_handler(Graph()):
                assert checker.can_use_tma() is False
        """,
        descriptor_pointwise=True,
    )

    assert result.returncode == 0, result.stderr


def test_spyre_descriptor_checker_accepts_symbolic_block_shape_with_fixed_config():
    result = _run_script(
        """
        import sympy
        import torch
        from torch._inductor.codegen import triton as triton_codegen
        from torch._inductor.codegen.triton import BlockParameters, FixedTritonConfig
        from torch_spyre._inductor.patches import enable_spyre_context
        import torch_spyre._inductor.patches as patches

        class Buffer:
            def get_size(self):
                return [512, 128]

        class Graph:
            unaligned_buffers = set()
            graph_inputs = {'in_ptr0', 'out_ptr0'}
            constants = {}

            def get_current_device_or_throw(self):
                return torch.device('spyre')

            def try_get_buffer(self, name):
                return Buffer()

        class Kernel:
            no_x_dim = False
            persistent_reduction = False
            fixed_config = FixedTritonConfig({'XBLOCK': 64})
            range_trees = []

            def max_block(self, prefix):
                return self.fixed_config[f'{prefix.upper()}BLOCK']

            def index_to_str(self, expr):
                return str(expr)

        block_params = BlockParameters(
            shape=[sympy.Integer(512), sympy.Integer(128)],
            block_shape=[(sympy.Symbol('XBLOCK') + 127) // 128, sympy.Min(128, sympy.Symbol('XBLOCK'))],
            strides=[sympy.Integer(128), sympy.Integer(1)],
            offsets=[sympy.Integer(0), sympy.Integer(0)],
        )

        with enable_spyre_context([]):
            checker_cls = triton_codegen.TritonKernel.tma_compatibility_checker_cls
            checker = checker_cls(Kernel(), torch.float16, for_store=False, force=False, buffer_name='in_ptr0')
            with patches.V.set_graph_handler(Graph()):
                assert checker.can_use_tma() is True
                assert checker.are_block_parameters_compatible(block_params) is True
        """,
        descriptor_pointwise=True,
    )

    assert result.returncode == 0, result.stderr



def test_spyre_descriptor_checker_rejects_unaligned_buffer_name():
    result = _run_script(
        """
        import sympy
        import torch
        from torch._inductor.codegen import triton as triton_codegen
        from torch._inductor.codegen.triton import BlockParameters, FixedTritonConfig
        from torch_spyre._inductor.patches import enable_spyre_context
        import torch_spyre._inductor.patches as patches

        class Buffer:
            def get_size(self):
                return [512, 128]

        class Graph:
            unaligned_buffers = {'in_ptr0'}
            graph_inputs = {'in_ptr0'}
            constants = {}

            def get_current_device_or_throw(self):
                return torch.device('spyre')

            def try_get_buffer(self, name):
                return Buffer()

        class Kernel:
            no_x_dim = False
            persistent_reduction = False
            fixed_config = FixedTritonConfig({'XBLOCK': 64})
            range_trees = []

            def max_block(self, prefix):
                return self.fixed_config[f'{prefix.upper()}BLOCK']

            def index_to_str(self, expr):
                return str(expr)

        block_params = BlockParameters(
            shape=[sympy.Integer(512), sympy.Integer(128)],
            block_shape=[sympy.Integer(512), sympy.Integer(128)],
            strides=[sympy.Integer(128), sympy.Integer(1)],
            offsets=[sympy.Integer(0), sympy.Integer(0)],
        )

        with enable_spyre_context([]):
            checker_cls = triton_codegen.TritonKernel.tma_compatibility_checker_cls
            checker = checker_cls(Kernel(), torch.float16, for_store=False, force=False, buffer_name='in_ptr0')
            with patches.V.set_graph_handler(Graph()):
                assert checker.are_block_parameters_compatible(block_params) is False
        """,
        descriptor_pointwise=True,
    )

    assert result.returncode == 0, result.stderr


def test_spyre_descriptor_checker_rejects_unresolved_constant_offset():
    result = _run_script(
        """
        import sympy
        import torch
        from torch._inductor.codegen import triton as triton_codegen
        from torch._inductor.codegen.triton import BlockParameters, FixedTritonConfig
        from torch_spyre._inductor.patches import enable_spyre_context
        import torch_spyre._inductor.patches as patches

        class Buffer:
            def get_size(self):
                return [512, 128]

        class Graph:
            unaligned_buffers = set()
            graph_inputs = {'in_ptr0', 'out_ptr0'}
            constants = {}

            def get_current_device_or_throw(self):
                return torch.device('spyre')

            def try_get_buffer(self, name):
                return Buffer()

        class Kernel:
            no_x_dim = False
            persistent_reduction = False
            fixed_config = FixedTritonConfig({'XBLOCK': 64})
            range_trees = []

            def max_block(self, prefix):
                return self.fixed_config[f'{prefix.upper()}BLOCK']

            def index_to_str(self, expr):
                return str(expr)

        block_params = BlockParameters(
            shape=[sympy.Integer(512), sympy.Integer(128)],
            block_shape=[sympy.Integer(512), sympy.Integer(128)],
            strides=[sympy.Integer(128), sympy.Integer(1)],
            offsets=[sympy.Integer(0), sympy.Integer(0)],
        )

        with enable_spyre_context([]):
            checker_cls = triton_codegen.TritonKernel.tma_compatibility_checker_cls
            checker = checker_cls(Kernel(), torch.float16, for_store=False, force=False, buffer_name='in_ptr0')
            with patches.V.set_graph_handler(Graph()):
                assert checker.are_block_parameters_compatible(
                    block_params,
                    constant_offset=sympy.Symbol('unresolved_offset'),
                ) is False
        """,
        descriptor_pointwise=True,
    )

    assert result.returncode == 0, result.stderr


def test_spyre_descriptor_checker_rejects_no_x_dim_store():
    result = _run_script(
        """
        import torch
        from torch._inductor.codegen import triton as triton_codegen
        from torch_spyre._inductor.patches import enable_spyre_context
        import torch_spyre._inductor.patches as patches

        class Buffer:
            def get_size(self):
                return [512, 128]

        class Graph:
            unaligned_buffers = set()
            graph_inputs = {'in_ptr0', 'out_ptr0'}
            constants = {}

            def get_current_device_or_throw(self):
                return torch.device('spyre')

            def try_get_buffer(self, name):
                return Buffer()

        class Kernel:
            no_x_dim = True

        with enable_spyre_context([]):
            checker_cls = triton_codegen.TritonKernel.tma_compatibility_checker_cls
            checker = checker_cls(Kernel(), torch.float16, for_store=True, force=False, buffer_name='out_ptr0')
            with patches.V.set_graph_handler(Graph()):
                assert checker.can_use_tma() is False
        """,
        descriptor_pointwise=True,
    )

    assert result.returncode == 0, result.stderr
