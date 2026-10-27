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

"""Offline guards for the looped-matmul accumulator silicon KAT.

The KAT itself (tests/scripts/looped_matmul_acc_silicon_kat.py) is the
pre-registered measurement; these tests pin its offline invariants so a
future shortcut cannot silently subvert them:

  1. PRE-REGISTRATION FREEZE — the shape set, distributions, seeds and the
     derived bound constants are pinned to the committed values (the
     derivation lives in the KAT docstring; any change must re-derive, not
     drift).
  2. SCORER RED-FIRST — the DL16 closed form passes the frozen criterion
     (no false positives), while the truncated-K and k-tile-permuted
     surrogates fail it (the instrument detects known-wrong outputs).
  3. DL16 ROUND EQUIVALENCE — the KAT's local round_to_dl16 agrees with
     sentient_codegen's W1d-pinned reference on this KAT's value band.
  4. R-EMISSION — with the accumulator override neutralized (pre-patch
     simulation), the compile dies at triton-spyre's loop-carried type
     assert; with it active (current source), the assert is gone and the
     emitted artifact carries the fp32 loop-carried accumulator + tl.dot.
  5. ADMISSION RECORD — the current compile verdict for a KAT row is the
     T-SR-29 typed structured-route refusal, recorded with its codes. The
     day this flips to `compiled_not_launched`, this test FAILS LOUD on
     purpose: structured admission for K>64 looped matmul has landed, the
     KAT is launchable, and a coordinated HW claim should be requested
     (never launched ad hoc).
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_KAT_PATH = (
    Path(__file__).resolve().parent / "scripts" / "looped_matmul_acc_silicon_kat.py"
)


def _load_kat():
    spec = importlib.util.spec_from_file_location("looped_matmul_acc_silicon_kat", _KAT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def kat():
    return _load_kat()


# --- 1. pre-registration freeze ------------------------------------------------
def test_pre_registration_shapes_frozen(kat):
    assert kat.SHAPE_ROWS == (
        ("LM1", 32, 256, 64),
        ("LM2", 8, 384, 64),
        ("LM3", 128, 512, 128),
    ), "pre-registered shape set changed: re-derive the registration, do not drift"


def test_pre_registration_distributions_frozen(kat):
    assert kat.DISTRIBUTIONS == (("A", 32), ("B", 64)), (
        "pre-registered distribution set changed: re-derive the registration"
    )


def test_pre_registration_bound_constants_frozen(kat):
    # Derived (KAT docstring): DL16 10-bit significand => 0.5 ulp <= |x|*2^-10
    # per round; output side 2 ulp <= |ref|*2^-8. Not tuned to any output.
    assert kat._ACCUM_TERM_SCALE == 2.0 ** -10
    assert kat._PRODUCT_TERM_SCALE == 2.0 ** -10
    assert kat._OUTPUT_TERM_SCALE == 2.0 ** -8
    assert kat._DL16_MANTISSA_BITS == 9  # 1-6-9 DL16 (W1d-pinned)
    assert kat._SEED_BASE == 20261026


def test_pre_registration_seed_is_deterministic(kat):
    assert kat.seed_for("LM1", 32, 256, 64, 32) == kat.seed_for("LM1", 32, 256, 64, 32)
    assert kat.seed_for("LM1", 32, 256, 64, 32) != kat.seed_for("LM1", 32, 256, 64, 64)
    assert kat.seed_for("LM1", 32, 256, 64, 32) != kat.seed_for("LM2", 8, 384, 64, 32)


def test_row_and_distribution_keys_fail_loud(kat):
    with pytest.raises(ValueError, match="unknown row key"):
        kat.row_shape("LM4")
    with pytest.raises(ValueError, match="unknown distribution key"):
        kat.dist_half_range("C")


# --- 2. scorer red-first -------------------------------------------------------
def test_red_first_scorer_checks(kat):
    checks = kat.red_first_self_checks()
    failed = [name for name, ok in checks if not ok]
    assert not failed, f"red-first self-checks failed: {failed}"
    # Census: 3 rows x 2 distributions x 4 checks (honest, honest with
    # product pre-round, truncated-K, k-tile-permuted).
    assert len(checks) == 24


def test_no_subnormal_band_holds(kat):
    """Recorded pre-registration fact: every nonzero value is >= 2^-6."""
    import torch

    for row_key, m, k, n in kat.SHAPE_ROWS:
        for dist_key, h in kat.DISTRIBUTIONS:
            seed = kat.seed_for(row_key, m, k, n, h)
            a, b = kat.build_inputs(m, k, n, h, seed)
            p = a.double().unsqueeze(2) * b.double().unsqueeze(0)
            nonzero = p[p != 0].abs()
            assert nonzero.numel() > 0
            assert nonzero.min().item() >= 2.0 ** -6, (
                f"{row_key}/{dist_key}: sub-2^-6 product — the no-subnormal "
                "pre-registration band is violated, re-derive"
            )
            ref = p.sum(dim=1)
            nonzero_ref = ref[ref != 0].abs()
            assert nonzero_ref.min().item() >= 2.0 ** -6
            assert nonzero_ref.max().item() < 65504.0, "fp16 overflow band"


# --- 3. DL16 round equivalence with the codegen reference ----------------------
def test_round_to_dl16_matches_codegen_reference(kat):
    pytest.importorskip("sentient_codegen", reason="codegen stack not on path")
    import numpy as np
    from sentient_codegen.emulator.dl16_round import round_to_dl16 as reference

    rng = np.random.default_rng(20261026)
    # This KAT's band: multiples of 2^-6 up to K*64, plus cancellation-scale
    # values, plus exact powers of two and the round-half-even ties.
    grid = rng.integers(-2 ** 18, 2 ** 18, size=4096) / 64.0
    powers = np.ldexp(np.ones(40), np.arange(-20, 20))
    values = np.concatenate([grid, powers, -powers, [0.0]])
    np.testing.assert_array_equal(kat.round_to_dl16(values), reference(values))


# --- 3b. run()-gate and exit-code contract (pure helpers) --------------------
def _authentic():
    return {
        "tl_dot": True,
        "fp32_loop_carried_acc": True,
        "fp16_loop_carried_acc": False,
    }


def test_gate_flags_inauthentic_rows_without_overwriting_status(kat):
    verdicts = [
        {"row": "LM1", "status": "refused", "route_authenticity": _authentic()},
        {"row": "LM2", "status": "WORKER_TIMEOUT"},  # no authenticity record
        {"row": "LM3", "status": "WORKER_CRASHED", "returncode": 1},
        {"row": "LM1", "status": "ENV_REFUSED_ALLOCATION"},
        {
            "row": "LM2",
            "status": "refused",
            "route_authenticity": {
                "tl_dot": False,
                "fp32_loop_carried_acc": False,
                "fp16_loop_carried_acc": False,
            },  # broken stderr capture
        },
    ]
    kat.apply_route_authenticity_gate(verdicts)
    # The inauthentic row is flagged...
    assert verdicts[4].get("route_authenticity_failed") is True
    # ...and NO status is laundered: crash/timeout/env keep their diagnosis.
    assert verdicts[1]["status"] == "WORKER_TIMEOUT"
    assert verdicts[2]["status"] == "WORKER_CRASHED"
    assert verdicts[3]["status"] == "ENV_REFUSED_ALLOCATION"
    assert verdicts[0].get("route_authenticity_failed") is None


def test_exit_code_contract(kat):
    clean_offline = [{"row": "LM1", "status": "refused", "tsr29_typed": True,
                      "route_authenticity": _authentic()}]
    # Offline, typed refusal: the recorded contract.
    assert kat.exit_code(clean_offline, [], launch=False) == 0
    # A NON-TYPED refusal (environment death, e.g. a missing codegen
    # module) must NOT satisfy the contract — r3 MAJOR-1r3.
    env_dead = [{"row": "LM1", "status": "refused", "tsr29_typed": False,
                 "route_authenticity": _authentic()}]
    assert kat.exit_code(env_dead, [], launch=False) == 2
    no_flag = [{"row": "LM1", "status": "refused",
                "route_authenticity": _authentic()}]
    assert kat.exit_code(no_flag, [], launch=False) == 2
    # Red-first self-check failure beats everything (instrument broken).
    assert kat.exit_code(clean_offline, ["some-check"], launch=False) == 1
    # Wrong route / crash / timeout / env: KAT failure, offline included.
    for status in ("WORKER_TIMEOUT", "WORKER_CRASHED", "ENV_REFUSED_ALLOCATION"):
        assert kat.exit_code([{"row": "LM1", "status": status}], [], launch=False) == 2
    laundered = [{"row": "LM1", "status": "refused",
                  "route_authenticity_failed": True}]
    assert kat.exit_code(laundered, [], launch=False) == 2
    # Launch mode: every row launched and numerically green -> 0.
    launched = [{"row": "LM1", "status": "compiled_and_launched",
                 "route_authenticity": _authentic(),
                 "score": {"pass": True}}]
    assert kat.exit_code(launched, [], launch=True) == 0
    # Launch mode: numeric criterion violation -> 3 (THE finding).
    red = [{"row": "LM1", "status": "compiled_and_launched",
            "route_authenticity": _authentic(),
            "score": {"pass": False, "n_fail": 3}}]
    assert kat.exit_code(red, [], launch=True) == 3
    # Launch mode: a row that could not launch -> 2 (recorded, not a pass).
    unlaunched = [{"row": "LM1", "status": "refused",
                   "route_authenticity": _authentic()}]
    assert kat.exit_code(unlaunched, [], launch=True) == 2
    # Launch mode: a TYPED refusal is clean in `broken` (it is the recorded
    # contract offline) but still cannot satisfy a launch run (r4 MINOR-1).
    typed_unlaunched = [{"row": "LM1", "status": "refused",
                         "tsr29_typed": True,
                         "route_authenticity": _authentic()}]
    assert kat.exit_code(typed_unlaunched, [], launch=True) == 2


def test_route_authenticity_requires_the_accumulate_chain(kat):
    """A 2-D fp32 zero-init ALONE does not authenticate; the chain does.

    Guards the false-positive window: an unrelated fp32 full (mask staging,
    epilogue zero) plus a tl.dot must NOT pass, while the real emission
    shape (init, self-read add, back-edge re-assignment) must.
    """
    unrelated = (
        "ymask = tl.full([YBLOCK], True, tl.int1)\n"
        "zero = tl.full([XBLOCK, YBLOCK], 0, tl.float32)\n"
        "tmp = tl.dot(a, b, input_precision=\"ieee\")\n"
        "out = zero + tmp\n"
    )
    assert kat.route_authenticity(unrelated)["fp32_loop_carried_acc"] is False
    real = (
        "_tmp6 = tl.full([YBLOCK, XBLOCK], 0, tl.float32)\n"
        "tmp4 = tl.dot(tmp2, tmp3, input_precision=\"ieee\")\n"
        "tmp7 = _tmp6 + tmp5\n"
        "_tmp6 = tmp7\n"
    )
    assert kat.route_authenticity(real) == kat._EXPECTED_ROUTE_AUTHENTICITY
    # The fp16 spelling of the same chain must be caught (pre-patch class).
    fp16 = real.replace("tl.float32", "tl.float16")
    assert kat.route_authenticity(fp16)["fp16_loop_carried_acc"] is True


def test_honest_closed_form_margins_are_recorded(kat):
    """The docstring's margin claim is a witnessed number, not prose."""
    margins = kat.honest_closed_form_margins()
    assert set(margins) == {
        f"{row}/{dist}" for row, _, _, _ in kat.SHAPE_ROWS
        for dist, _ in kat.DISTRIBUTIONS
    }
    for tag, forms in margins.items():
        assert set(forms) == {"closed_form", "pre_round"}
        for form, slack in forms.items():
            assert 0.0 <= slack <= 0.2, (
                f"{tag}/{form}: honest closed-form slack {slack} outside the "
                "recorded margin band — the bound or the model drifted"
            )



def _run_kat_subprocess(script: str, extra_env: dict[str, str] | None = None,
                        timeout: int = 420):
    env = os.environ.copy()
    env["TORCH_SPYRE_CODEGEN_ONLY"] = "1"
    env["TORCH_LOGS"] = "output_code"
    env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
    env.pop("TORCH_SPYRE_EXPERIMENTAL_DESCRIPTOR_POINTWISE", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


_EMISSION_SCRIPT_TEMPLATE = """
    import torch
    import torch_spyre
    import torch_spyre._inductor as spyre_inductor
    from contextlib import contextmanager
    import torch._inductor.codegen.triton as triton_codegen

    real_ctx = spyre_inductor.enable_spyre_context

    @contextmanager
    def neutralized_ctx(*args, **kwargs):
        with real_ctx(*args, **kwargs) as value:
            saved = triton_codegen.upcast_acc_dtype
            # Pre-patch simulation: upstream upcast_acc_dtype leaves fp16
            # accumulators in fp16 (codegen_upcast_to_fp32 is pinned off).
            triton_codegen.upcast_acc_dtype = lambda dtype: dtype
            try:
                yield value
            finally:
                triton_codegen.upcast_acc_dtype = saved

    if {neutralize}:
        spyre_inductor.enable_spyre_context = neutralized_ctx

    torch_spyre._autoload()

    def fn(a, b):
        return a @ b

    M, K, N = 32, 256, 64
    a = torch.randn(M, K, dtype=torch.float16).to("spyre")
    b = torch.randn(K, N, dtype=torch.float16).to("spyre")
    try:
        torch.compile(fn, dynamic=False)(a, b)
    except Exception:
        pass  # downstream refusals are out of scope; the seam is read from
              # the emitted output code / the loop-carried assert below
    """


def test_r_emission_prepatch_red_reproduces():
    """R-EMISSION: override neutralized => the loop-carried assert fires.

    This is the defect ff74490 fixed, reproduced from current source by
    simulating the pre-patch upcast_acc_dtype at the exact seam the patch
    patches (the spyre context's install site, wrapped around compile_fx).
    """
    pytest.importorskip(
        "sentient_codegen",
        reason="codegen stack not on path (the compile route requires it)",
    )
    result = _run_kat_subprocess(
        _EMISSION_SCRIPT_TEMPLATE.format(neutralize="True")
    )
    combined = result.stdout + result.stderr
    assert "Loop-carried variable" in combined, (
        "pre-patch simulation did not reproduce the loop-carried fp16->fp32 "
        "assert — the KAT row no longer exercises the accumulator seam:\n"
        + combined[-4000:]
    )


def test_r_emission_postpatch_acc_is_fp32():
    """Current source: no loop-carried assert; the emitted accumulator is fp32.

    The compile may (and today does) refuse downstream at the structured
    admission gate — that is recorded separately; this test pins the
    EMISSION seam: the artifact is an Inductor looped matmul (tl.dot) with
    an fp32 loop-carried accumulator and no fp16 one.
    """
    pytest.importorskip(
        "sentient_codegen",
        reason="codegen stack not on path (the compile route requires it)",
    )
    result = _run_kat_subprocess(
        _EMISSION_SCRIPT_TEMPLATE.format(neutralize="False")
    )
    combined = result.stdout + result.stderr
    assert "Loop-carried variable" not in combined, (
        "loop-carried fp16->fp32 accumulator regression reproduced:\n"
        + combined[-4000:]
    )
    kat = _load_kat()
    import re

    assert "tl.dot" in combined, "emitted kernel is not a matmul (no tl.dot)"
    assert re.search(kat.FP32_ACC_INIT_RE, combined), (
        "looped-matmul accumulator was not materialized in fp32:\n"
        + combined[-4000:]
    )
    assert not re.search(kat.FP16_ACC_INIT_RE, combined), (
        "looped-matmul accumulator was materialized in fp16:\n"
        + combined[-4000:]
    )


def test_admission_refusal_is_recorded_fail_loud_on_flip():
    """The KAT row's current compile verdict is the T-SR-29 typed refusal.

    Runs the KAT's own single-row worker (offline, codegen-only) for
    LM1/A — the same subprocess the KAT parent spawns — and asserts the
    recorded contract. If the verdict ever becomes `compiled_not_launched`,
    this test fails ON PURPOSE: structured admission for K>64 looped
    matmul has landed, the pre-registered KAT is launchable, and a
    coordinated HW claim should be requested (HW Probe Claim Coordination
    is mandatory; never launch ad hoc).
    """
    import tempfile

    pytest.importorskip(
        "sentient_codegen",
        reason="codegen stack not on path (the compile route requires it)",
    )
    kat = _load_kat()
    with tempfile.TemporaryDirectory() as tmp:
        result_out = os.path.join(tmp, "result.json")
        stderr_capture = os.path.join(tmp, "stderr.txt")
        env = os.environ.copy()
        env["TORCH_LOGS"] = "output_code"
        env["_LM_KAT_STDERR_CAPTURE"] = stderr_capture
        env["TORCH_SPYRE_CODEGEN_ONLY"] = "1"
        env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
        env.pop("TORCH_SPYRE_EXPERIMENTAL_DESCRIPTOR_POINTWISE", None)
        with open(stderr_capture, "w") as stderr_fh:
            proc = subprocess.run(
                [
                    sys.executable, str(_KAT_PATH),
                    "--worker", "LM1", "A", "--result-out", result_out,
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=stderr_fh,
                text=True,
                timeout=420,
            )
        assert proc.returncode == 0, proc.stdout[-2000:]
        with open(result_out) as fh:
            record = json.load(fh)

    if record["status"] == "compiled_not_launched":
        pytest.fail(
            "STRUCTURED ADMISSION FOR K>64 LOOPED MATMUL HAS LANDED: the "
            "pre-registered KAT row compiled without refusal. Update the "
            "KAT docstring's CURRENT ADMISSION STATE, re-run the offline "
            "KAT, and request a coordinated HW claim for the launch — do "
            "not launch ad hoc."
        )
    assert record["status"] == "refused", (
        f"unexpected worker status {record['status']!r}"
    )
    # The typed T-SR-29 refusal specifically (an arbitrary compile death
    # — e.g. a missing dependency — must NOT satisfy this guard).
    assert record.get("tsr29_typed") is True, (
        f"refusal was not the typed T-SR-29 structured-route refusal: "
        f"{record.get('refusal_type')}: {record.get('refusal_detail')}"
    )
    assert record.get("refusal_codes"), "typed refusal must record its codes"
    assert all(
        code and not code.startswith("the ") for code in record["refusal_codes"]
    ), f"refusal codes look unparsed: {record['refusal_codes']}"
    # Route authenticity is witnessable even on a refused compile (the
    # output code is emitted before the admission gate runs).
    assert record["route_authenticity"] == {
        "tl_dot": True,
        "fp32_loop_carried_acc": True,
        "fp16_loop_carried_acc": False,
    }, (
        "the refused artifact is not the frontend-routed fp32-accumulator "
        f"looped matmul: {record['route_authenticity']}"
    )
