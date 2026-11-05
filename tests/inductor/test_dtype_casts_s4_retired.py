"""S4 retirement record — the pin-era in-graph typecast tests are RETIRED.

The 8 held tests (class TestToDtypeMapping, preserved verbatim on the
pin-era lane branch a3d8ef74 as tests/inductor/test_dtype_casts_s4_pending.py)
convicted a surface that no longer exists on this base: the SDSC op-spec
machinery in torch_spyre/_inductor/spyre_kernel.py was deleted in 02a8c8b4
(SDSC Phase 4B, new-runtime restructure). There is no in-graph typecast
path in torch-spyre on this base — dtype casts in the new stack surface as
TTIR ops (tt.fp_to_fp / arith.extf / arith.truncf) and are adjudicated
KB-side by the sentient_codegen admission policy.

Adjudication (lead ruling 2026-11-05, option (a)): receipted absence here +
KB-side red-first probe of the TTIR cast surface. Probe receipts (spyre-
knowledgebase, branch bf16-fp32-pivot-lane):
  - 92473034  admit-then-drop P1 witness (arith truncf/extf admitted by
              policy, routed to the pointwise_identity copy pattern —
              the cast is dropped; tt.fp_to_fp non-F8 spellings refuse
              typed at parse).
  - a417afb9  refuse-typed fix: tt.fp_to_fp rejects on rcudd1a with a
              typed admission diagnostic (no conversion lowering; a
              dropped cast would silently corrupt); sen1p5 keeps accept.

The pin-era S4 rulings are LOCALIZED, not overturned: correct for the
pin-era SDSC surface, vacuously satisfied where that surface no longer
exists.
"""

import importlib.util
import unittest


class TestS4SurfaceRetirement(unittest.TestCase):
    def test_spyre_kernel_surface_absent(self):
        """Receipt: the convicted surface (spyre_kernel SDSC op-spec
        machinery) does not exist on this base — deleted in 02a8c8b4."""
        self.assertIsNone(
            importlib.util.find_spec("torch_spyre._inductor.spyre_kernel"),
            "torch_spyre/_inductor/spyre_kernel.py exists again — the S4 "
            "retirement receipt (02a8c8b4 deletion) is stale; re-adjudicate "
            "the in-graph typecast tests before resurrecting this surface.",
        )


if __name__ == "__main__":
    unittest.main()
