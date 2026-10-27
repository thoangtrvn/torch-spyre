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

"""Loop-carried reduction accumulator dtype seam.

With ``triton.codegen_upcast_to_fp32`` disabled (DL16 pointwise emission pin),
torch's ``upcast_acc_dtype`` returns fp16 for fp16 inputs, so a looped matmul
(K > 64) materializes its loop-carried accumulator with
``tl.full(..., tl.float16)`` while ``tl.dot`` promotes to fp32. The upstream
loop-carried type assert in triton-spyre's ast_to_ttir then refuses every
shape. ``enable_spyre_context`` overrides ``upcast_acc_dtype`` so reduction
accumulators upcast to fp32 while pointwise compute types stay 16-bit.
"""

import os
import subprocess
import sys
import textwrap


def _run_script(script: str, extra_env: dict[str, str] | None = None):
    env = os.environ.copy()
    env["TORCH_SPYRE_CODEGEN_ONLY"] = "1"
    env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
    env.pop("TORCH_SPYRE_EXPERIMENTAL_DESCRIPTOR_POINTWISE", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_spyre_context_upcasts_reduction_accumulator_only():
    result = _run_script(
        """
        import torch
        import torch._inductor.codegen.triton as triton_codegen
        from torch._inductor.utils import upcast_compute_type
        from torch_spyre._inductor.patches import enable_spyre_context

        original = triton_codegen.upcast_acc_dtype
        with enable_spyre_context([]):
            # Reduction accumulators upcast to fp32: this is the seam the
            # looped-matmul loop-carried assert fires through.
            assert triton_codegen.upcast_acc_dtype(torch.float16) is torch.float32
            assert triton_codegen.upcast_acc_dtype(torch.bfloat16) is torch.float32
            # Pointwise compute types are untouched: the DL16 emission pin
            # (codegen_upcast_to_fp32=False) must keep 16-bit compute.
            assert upcast_compute_type(torch.float16) is torch.float16
            assert upcast_compute_type(torch.bfloat16) is torch.bfloat16
            # Integer accumulator upcasts keep their upstream behavior.
            assert triton_codegen.upcast_acc_dtype(torch.int16) is torch.int32
        assert triton_codegen.upcast_acc_dtype is original
        """
    )
    assert result.returncode == 0, result.stderr


def test_spyre_context_restores_accumulator_override_on_exception():
    result = _run_script(
        """
        import torch
        import torch._inductor.codegen.triton as triton_codegen
        from torch_spyre._inductor.patches import enable_spyre_context

        original = triton_codegen.upcast_acc_dtype
        try:
            with enable_spyre_context([]):
                assert triton_codegen.upcast_acc_dtype(torch.float16) is torch.float32
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert triton_codegen.upcast_acc_dtype is original
        """
    )
    assert result.returncode == 0, result.stderr


def test_spyre_context_accumulator_override_nests_correctly():
    result = _run_script(
        """
        import torch
        import torch._inductor.codegen.triton as triton_codegen
        from torch_spyre._inductor.patches import enable_spyre_context

        original = triton_codegen.upcast_acc_dtype
        with enable_spyre_context([]):
            with enable_spyre_context([]):
                assert triton_codegen.upcast_acc_dtype(torch.float16) is torch.float32
            # Inner exit must restore the outer (still spyre) override, not
            # the pristine upstream function.
            assert triton_codegen.upcast_acc_dtype(torch.float16) is torch.float32
        assert triton_codegen.upcast_acc_dtype is original
        """
    )
    assert result.returncode == 0, result.stderr


def test_looped_matmul_k512_accumulator_is_fp32():
    """End-to-end codegen-only check of the emitted Triton kernel.

    Before the accumulator override, this compile dies in triton-spyre's
    ast_to_ttir with "Loop-carried variable ... fp16 ... re-assigned to
    ... fp32". After it, the loop-carried accumulator is materialized in
    fp32 (``tl.full(..., tl.float32)``) in the emitted output code. The
    compile may still be refused downstream by the structured-route
    admission gate; that is a separate contract and not this seam.
    """
    result = _run_script(
        """
        import torch
        import torch_spyre
        torch_spyre._autoload()

        def fn(a, b):
            return a @ b

        M, K, N = 128, 512, 128
        a = torch.randn(M, K, dtype=torch.float16).to("spyre")
        b = torch.randn(K, N, dtype=torch.float16).to("spyre")
        try:
            cfn = torch.compile(fn, dynamic=False)
            cfn(a, b)
        except Exception:
            # Downstream refusals (e.g. structured-route admission) are out
            # of scope here; the accumulator seam is asserted from the
            # emitted output code below.
            pass
        """,
        extra_env={"TORCH_LOGS": "output_code"},
    )
    combined = result.stdout + result.stderr
    assert "Loop-carried variable" not in combined, (
        "loop-carried fp16->fp32 accumulator regression reproduced:\n"
        + combined[-4000:]
    )
    assert "= tl.full([YBLOCK, XBLOCK], 0, tl.float32)" in combined, (
        "looped-matmul accumulator was not materialized in fp32:\n"
        + combined[-4000:]
    )
