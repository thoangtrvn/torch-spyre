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

from __future__ import annotations

import functools
import inspect
import operator

import torch


# Decorator to keep track of compiled variant.
def compile_once(op, **compile_kwargs):
    def decorator(fn):
        compiled = None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal compiled
            nonlocal op
            if compiled is None:
                if isinstance(op, str):
                    op = operator.attrgetter(op)(torch.ops)
                compiled = torch.compile(op, **compile_kwargs)
            return fn(*args, compiled=compiled, **kwargs)

        # We remove the `compiled` arg from the signature to have
        # a clean signature.
        old_signature = inspect.signature(fn)
        params = dict(old_signature.parameters)
        params.pop("compiled", None)
        new_signature = old_signature.replace(parameters=params.values())
        wrapper.__signature__ = new_signature

        return wrapper

    return decorator


def dispatch_to_torch_compile(*args, compiled=None, **kwargs):
    return compiled(*args, **kwargs)
