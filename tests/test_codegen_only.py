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
            assert "TORCH_SPYRE_CODEGEN_ONLY=1" in str(e)
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
