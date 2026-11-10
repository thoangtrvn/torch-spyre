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
from types import SimpleNamespace


def _run_codegen_only_script(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["TORCH_SPYRE_CODEGEN_ONLY"] = "1"
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_codegen_only_backend_entrypoint_autoload_without_env():
    env = os.environ.copy()
    env.pop("TORCH_SPYRE_CODEGEN_ONLY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import torch
                import torch_spyre

                assert torch._C._get_privateuse1_backend_name() == "spyre"
                assert torch_spyre.codegen_only_available() is True
                assert torch.spyre.codegen_only_available() is True
                assert torch.spyre.is_available() is False
                assert torch.spyre.device_count() == 0
                """
            ),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_codegen_only_metadata_detection(monkeypatch):
    from torch_spyre import _runtime

    dist = SimpleNamespace(
        entry_points=(
            SimpleNamespace(
                group="torch.backends",
                name="torch_spyre",
                value="torch_spyre:_autoload_codegen_only",
            ),
        )
    )
    monkeypatch.delenv("TORCH_SPYRE_CODEGEN_ONLY", raising=False)
    monkeypatch.setattr(_runtime.metadata, "distribution", lambda name: dist)
    _runtime._installed_codegen_only_entrypoint.cache_clear()

    try:
        assert _runtime.codegen_only_available() is True
    finally:
        _runtime._installed_codegen_only_entrypoint.cache_clear()


def test_codegen_only_import_autoload_and_inductor_modules():
    result = _run_codegen_only_script(
        """
        import torch
        import torch_spyre

        torch_spyre._autoload()
        assert torch._C._get_privateuse1_backend_name() == "spyre"
        assert torch_spyre.codegen_only_available() is True
        assert torch.spyre.codegen_only_available() is True
        assert torch.spyre.is_available() is False
        assert torch.spyre.device_count() == 0

        import torch_spyre._inductor
        import torch_spyre._inductor.customops
        import torch_spyre._inductor.decompositions
        import torch_spyre._inductor.lowering
        import torch_spyre._inductor.ir
        assert hasattr(torch.ops.spyre, "embedding")
        assert hasattr(torch.ops.spyre, "softmax")
        """
    )
    assert result.returncode == 0, result.stderr


def test_codegen_only_runtime_use_fails_loud():
    result = _run_codegen_only_script(
        """
        import torch_spyre
        import torch_spyre._C as _C
        from torch_spyre._runtime import SpyreRuntimeUnavailableError

        torch_spyre._autoload()
        try:
            _C.launch_kernel_from_bytes(b"", [])
        except SpyreRuntimeUnavailableError as e:
            assert "codegen-only mode" in str(e)
            assert "TORCH_SPYRE_CODEGEN_ONLY" in str(e)
        else:
            raise AssertionError("runtime launch unexpectedly succeeded")
        """
    )
    assert result.returncode == 0, result.stderr


def test_codegen_only_layout_helper_is_importable():
    result = _run_codegen_only_script(
        """
        import torch
        import torch_spyre

        torch_spyre._autoload()
        layout = torch.spyre.SpyreTensorLayout([512, 128], torch.float16)
        assert layout.device_size == [512, 2, 64]
        assert layout.stride_map == [128, 64, 1]
        assert layout.elems_per_stick() == 64
        """
    )
    assert result.returncode == 0, result.stderr


def test_codegen_only_materializes_fake_spyre_tensor_without_runtime():
    result = _run_codegen_only_script(
        """
        import torch
        import torch_spyre

        x = torch.empty((512, 128), dtype=torch.float16)
        y = x.to("spyre")

        assert y.device.type == "spyre"
        assert y.device.index == 0
        assert tuple(y.shape) == (512, 128)
        assert tuple(y.stride()) == (128, 1)
        layout = y.device_tensor_layout()
        assert layout.device_size == [512, 2, 64]
        assert layout.stride_map == [128, 64, 1]
        assert layout.elems_per_stick() == 64
        assert torch.spyre.is_available() is False
        assert torch.spyre.device_count() == 0
        """
    )
    assert result.returncode == 0, result.stderr


def test_codegen_only_to_spyre_dtype_matches_layout_dtype():
    result = _run_codegen_only_script(
        """
        import torch
        import torch_spyre

        x = torch.empty((512, 128), dtype=torch.float16)
        y = x.to("spyre", dtype=torch.float32)

        assert y.dtype == torch.float32
        assert y.device.type == "spyre"
        layout = y.device_tensor_layout()
        assert "IEEE_FP32" in repr(layout)
        assert layout.device_size == [512, 4, 32]
        assert layout.stride_map == [128, 32, 1]
        """
    )
    assert result.returncode == 0, result.stderr


def test_codegen_only_to_spyre_bfloat16_dtype_matches_layout_dtype():
    result = _run_codegen_only_script(
        """
        import torch
        import torch_spyre

        x = torch.empty((512, 128), dtype=torch.float16)
        y = x.to("spyre", dtype=torch.bfloat16)

        assert y.dtype == torch.bfloat16
        assert y.device.type == "spyre"
        layout = y.device_tensor_layout()
        assert "BFLOAT16" in repr(layout)
        assert layout.device_size == [512, 2, 64]
        assert layout.stride_map == [128, 64, 1]
        """
    )
    assert result.returncode == 0, result.stderr


def test_codegen_only_torch_empty_spyre_allocates_fake_tensor():
    result = _run_codegen_only_script(
        """
        import torch
        import torch_spyre

        y = torch.empty((512, 128), device="spyre", dtype=torch.float16)

        assert y.device.type == "spyre"
        assert y.device.index == 0
        assert tuple(y.shape) == (512, 128)
        assert tuple(y.stride()) == (128, 1)
        layout = y.device_tensor_layout()
        assert layout.device_size == [512, 2, 64]
        assert layout.stride_map == [128, 64, 1]
        """
    )
    assert result.returncode == 0, result.stderr


def test_codegen_only_factory_family_materializes_fake_tensors():
    """zeros/ones/full/empty_strided materialize fake tensors via the
    PrivateUse1 kernels, and torch_spyre._C resolves to the pure-Python
    module even when a built _C.so sits next to it (mixed checkout)."""
    result = _run_codegen_only_script(
        """
        import torch
        import torch_spyre

        torch_spyre._autoload()
        for y in (
            torch.zeros((512, 128), device="spyre", dtype=torch.float16),
            torch.ones((512, 128), device="spyre", dtype=torch.float16),
            torch.full((512, 128), 1.5, device="spyre", dtype=torch.float16),
            torch.empty_strided(
                (512, 128), (128, 1), device="spyre", dtype=torch.float16
            ),
        ):
            assert y.device.type == "spyre"
            assert y.device.index == 0
            assert tuple(y.shape) == (512, 128)
            layout = y.device_tensor_layout()
            assert layout.device_size == [512, 2, 64]
            assert layout.stride_map == [128, 64, 1]

        import torch_spyre._C as _C

        assert _C.__file__.endswith("_C.py")
        """
    )
    assert result.returncode == 0, result.stderr


def test_codegen_only_C_first_import_before_torch():
    """Importing torch_spyre._C before torch must not trigger a circular import.

    Regression test for the module-scope `import torch` in _C.py (codegen-only
    branch) which, when _C is imported first in a fresh interpreter, triggered
    torch's backend autoload -> _autoload_codegen_only -> _autoload ->
    torch_spyre._inductor._autoload -> choices.py:24 `from torch_spyre._C import
    get_elem_in_stick` while _C was still frozen at its own `import torch`.
    """
    result = _run_codegen_only_script(
        """
        import torch_spyre._C as _C
        import torch

        # Verify the public surface is accessible and functional
        assert hasattr(_C, "get_elem_in_stick")
        assert hasattr(_C, "get_device_dtype")
        assert hasattr(_C, "DataFormats")
        assert hasattr(_C, "SpyreTensorLayout")
        assert hasattr(_C, "encode_constant")
        assert hasattr(_C, "copy_tensor")
        assert hasattr(_C, "empty_with_layout")
        assert hasattr(_C, "current_device")
        assert hasattr(_C, "device_count")
        assert hasattr(_C, "set_device")

        # Verify core functionality works
        assert _C.get_elem_in_stick(torch.float16) == 64
        assert _C.get_elem_in_stick(torch.float32) == 32
        assert _C.get_elem_in_stick(torch.bfloat16) == 64
        # IntEnum comparison with int is False in Python 3.11+; compare enum members directly
        assert _C.get_device_dtype(torch.float16) == _C.DataFormats.SEN169_FP16
        assert _C.get_device_dtype(torch.float32) == _C.DataFormats.IEEE_FP32
        assert _C.get_device_dtype(torch.bfloat16) == _C.DataFormats.BFLOAT16

        layout = _C.SpyreTensorLayout([512, 128], torch.float16)
        assert layout.device_size == [512, 2, 64]
        assert layout.stride_map == [128, 64, 1]
        assert layout.elems_per_stick() == 64
        """
    )
    assert result.returncode == 0, result.stderr
