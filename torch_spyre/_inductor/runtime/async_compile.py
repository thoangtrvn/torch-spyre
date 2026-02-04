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

import json
import tempfile
from typing import Any, Union
import os
import subprocess

from torch._inductor.runtime.runtime_utils import cache_dir
from torch_spyre._C import convert_artifacts
from torch_spyre._inductor.codegen.superdsc import generate_sdsc
from torch_spyre._inductor.constants import SEGMENT_OFFSETS
from . import KernelSpec, ConstantArg, UnimplementedOp
from .kernel_runner import (
    SpyreSDSCKernelRunner,
    SpyreUnimplementedRunner,
)

from torch._inductor.codecache import PyCodeCache

_argument_names = ["arg0", "arg1", "arg2", "arg3", "arg4", "arg5", "arg6"]


def get_output_dir(kernel_name: str):
    spyre_dir = os.path.join(cache_dir(), "inductor-spyre")
    os.makedirs(spyre_dir, exist_ok=True)
    kernel_output_dir = tempfile.mkdtemp(dir=spyre_dir, prefix=f"{kernel_name}_")
    return kernel_output_dir


class SpyreAsyncCompile:
    def __init__(self) -> None:
        pass

    def sdsc(self, kernel_name: str, ks: Union[KernelSpec | UnimplementedOp]):
        if isinstance(ks, UnimplementedOp):
            print(f"WARNING: Compiling unimplemented {ks.op} to runtime exception")
            return SpyreUnimplementedRunner(kernel_name, ks.op)

        kernel_output_dir, arg_mapping = self.generate_sdsc(kernel_name, ks)
        return self.run_dxp_standalone(kernel_name, kernel_output_dir, arg_mapping)

    def generate_sdsc(self, kernel_name: str, ks: KernelSpec) -> tuple[str, list[int]]:
        inputs = []
        outputs = []
        arg_mapping: list[int] = []
        for index, ts in enumerate(ks.args):
            if isinstance(ts, ConstantArg):
                raise RuntimeError("TOOO: implement SDSC generation for constants")
            elif ts.is_input:
                inputs.append(
                    {
                        "name": _argument_names[index],
                        "scale": ks.scales[index],
                        "ddtype": ts.device_layout.device_dtype,
                    }
                )
                arg_mapping.append(ts.arg_index)
            else:
                outputs.append(
                    {
                        "name": _argument_names[index],
                        "scale": ks.scales[index],
                        "ddtype": ts.device_layout.device_dtype,
                    }
                )
                arg_mapping.append(ts.arg_index)
        kernel_descriptor = {
            "name": kernel_name,
            "reduction": ks.is_reduction,
            "op": ks.op,
            "dimensions": ks.dimensions,
            "inputs": inputs,
            "outputs": outputs,
        }
        if ks.op_info is not None:
            kernel_descriptor["op_info"] = ks.op_info
        pointers = dict(zip(_argument_names, SEGMENT_OFFSETS))
        dt_sdsc = generate_sdsc(pointers, **kernel_descriptor)
        kernel_output_dir = get_output_dir(kernel_name)
        subdir = os.path.join(kernel_output_dir, "execute", kernel_name)
        os.makedirs(subdir, exist_ok=True)
        with open(os.path.join(subdir, "sdsc.json"), "w") as file:
            print(f"Generating {file.name}")
            json.dump(dt_sdsc, file, indent=2)
        return kernel_output_dir, arg_mapping

    def run_dxp_standalone(
        self, kernel_name: str, kernel_output_dir: str, arg_mapping: list[int]
    ):
        subprocess.run(["dxp_standalone", "-d", kernel_output_dir], check=True)
        convert_artifacts(kernel_output_dir)
        return SpyreSDSCKernelRunner(kernel_name, kernel_output_dir, arg_mapping)

    def wait(self, scope: dict[str, Any]) -> None:
        pass

    def triton(self, kernel_name: str, source_code: str, device_str: str):
        cat = getattr(PyCodeCache.load(source_code), kernel_name)
        cfg = cat.configs[0]
        compile_meta = cat.triton_meta
        compile_meta["device_type"] = cat.device_props.type
        compile_meta["cc"] = cat.device_props.cc
        compile_meta["constants"].update(cfg.kwargs)
        # compile_args = (
        #     ASTSource(
        #         cat.fn,
        #         compile_meta["signature"],
        #         compile_meta["constants"],
        #         compile_meta["configs"][0],
        #     ),
        # )
        # target = GPUTarget(
        #     compile_meta["device_type"],
        #     compile_meta["cc"],
        #     cc_warp_size(compile_meta["cc"]),
        # )
        # options = {
        #     "spyre_options": compile_meta["spyre_options"],
        # }
        # compile_kwargs = {
        #     "target": target,
        #     "options": options,
        # }

        ks: KernelSpec = compile_meta["spyre_options"]["kernel_specs"][0]
        kernel_output_dir, arg_mapping = self.generate_sdsc(kernel_name, ks)
        # tkc = triton.compile(*compile_args, **compile_kwargs)
        return self.run_dxp_standalone(kernel_name, kernel_output_dir, arg_mapping)
