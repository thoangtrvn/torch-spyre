import sympy
from typing import Optional, Sequence
from torch._inductor.codegen.triton import TritonKernel, FixedTritonConfig
from torch._inductor.virtualized import StoreMode, V
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.utils import IndentedBuffer
from . import Unsupported
from .ir import FixedTiledLayout
from .spyre_kernel import (
    create_tensor_arg,
    create_kernel_spec,
    DimensionInfo,
)
from .runtime import KernelSpec, TensorArg


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
        self.di: list[DimensionInfo] = []
        self.tensor_args: dict[str, TensorArg] = {}
        self.scales: dict[str, list[int]] = {}
        self.kernel_specs: list[KernelSpec] = []

    def codegen_kernel(self, name=None) -> str:
        original_code = super().codegen_kernel(name)
        code = IndentedBuffer()
        code.splice("from torch_spyre._inductor.runtime import TensorArg, KernelSpec")
        code.splice("import torch")
        code.splice(
            "from torch_spyre._C import DataFormats, SpyreTensorLayout, StickFormat"
        )
        return code.getvalue() + original_code

    def codegen_body(self):
        self.triton_meta["spyre_options"] = {"kernel_specs": self.kernel_specs}
        return super().codegen_body()

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        print(f"load name={name} var={var}")
        self.tensor_args[var] = create_tensor_arg(True, -1, layout)
        self.scales[var] = self.analyze_tensor_access(self.get_dimension_info(), index)
        return super().load(name, index)

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        var = self.args.output(name)
        buf = V.graph.get_buffer(name)
        layout = buf.get_layout()
        if not isinstance(layout, FixedTiledLayout):
            raise Unsupported(f"{name} does not have FixedTiledLayout")
        print(f"store name={name} index={index} value={value} var={var}")
        self.tensor_args[var] = create_tensor_arg(False, -1, layout)
        self.scales[var] = self.analyze_tensor_access(self.get_dimension_info(), index)
        args, scales = self.create_args_and_scales()
        self.kernel_specs.append(
            create_kernel_spec("triton", False, self.di, args, scales, {})
        )

        return super().store(name, index, value, mode)

    def get_dimension_info(self) -> list[DimensionInfo]:
        if len(self.di) == 0:
            var_ranges = self.var_ranges()
            symbols = reversed(sorted(var_ranges.keys(), key=lambda x: str(x)))
            for s in symbols:
                self.di.append(DimensionInfo(s, int(var_ranges[s])))
        return self.di

    def analyze_tensor_access(
        self,
        op_dimensions: Sequence[DimensionInfo],
        index: sympy.Expr,
    ) -> list[int]:
        """
        Return the scale implied by the given iteration space and indexing expression
        """
        return [1 if di.var in index.free_symbols else -1 for di in op_dimensions]

    def create_args_and_scales(self) -> tuple[list[TensorArg], list[list[int]]]:
        args: list[TensorArg] = []
        scales: list[list[int]] = []
        actuals = self.args.python_argdefs()[1]
        print(f"create_args actuals={actuals} args={self.args}")
        for index, name in enumerate(actuals):
            if name.startswith("buf"):
                var = self.args.output(name)
            else:
                var = self.args.input(name)
            arg = self.tensor_args[var]
            arg.arg_index = index
            args.append(arg)
            scale = self.scales[var]
            scales.append(scale)
        return args, scales
