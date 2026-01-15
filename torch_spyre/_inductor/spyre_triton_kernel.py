import sympy
from typing import Optional
from torch._inductor.codegen.triton import TritonKernel, FixedTritonConfig


class SpyreTritonKernel(TritonKernel):
    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        min_elem_per_thread=0,
        optimize_mask=True,
        fixed_config: Optional[FixedTritonConfig] = None,
        hint_override: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            tiling,
            min_elem_per_thread,
            optimize_mask,
            fixed_config,
            hint_override,
            **kwargs,
        )

    def codegen_body(self):
        self.triton_meta["spyre_options"] = {}
        return super().codegen_body()
