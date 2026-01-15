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

import os
import json
import torch
from typing import Any
from torch_spyre._C import launch_kernel


class SpyreUnimplementedRunner:
    def __init__(self, name: str, op: str):
        self.kernel_name = name
        self.op = op

    def run(self, *args, **kw_args):
        raise RuntimeError(
            f"Invoked {self.kernel_name} which contains unimplemented operation {self.op}"
        )


class SpyreSDSCKernelRunner:
    def __init__(self, name: str, code_dir: str, arg_mapping: list[int]):
        self.kernel_name = name
        self.code_dir = code_dir
        self.arg_mapping = arg_mapping

    def run(self, *args, **kw_args):
        g2 = os.path.join(self.code_dir, "g2.graph.cbor")
        print(f"RUN: {self.kernel_name} {g2}")
        actuals = [args[i] for i in self.arg_mapping]
        return launch_kernel(g2, actuals)


class SpyreTritonKernelRunner:
    def __init__(self, name: str, kernel: Any):
        self.kernel_name = name
        self.kernel = kernel
        compiler_artifacts = json.loads(self.kernel.asm["dtir"])
        self.code_dir = compiler_artifacts["code_dir"]
        print(f"SpyreTritonKernelRunner.init: {self.kernel_name} {self.code_dir}")

    def run(self, *args, **kw_args):
        g2 = os.path.join(self.code_dir, "g2.graph.cbor")
        print(f"RUN: {self.kernel_name} {g2}")
        return launch_kernel(
            g2,
            [tensor for tensor in args if torch.is_tensor(tensor)],
        )
