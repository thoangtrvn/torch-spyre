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
import sys
import threading
import types
import importlib
from pathlib import Path

from .constants import DEVICE_NAME
from ._runtime import (
    SpyreRuntimeUnavailableError,
    codegen_only_available,
)

from . import profiler
from .raw_copy import raw_copy_from_device, raw_copy_to_device

_runtime_init_lock = threading.Lock()


def _install_codegen_only_C() -> types.ModuleType:
    """Bind torch_spyre._C to the pure-Python module in codegen-only mode.

    An environment that carries both _codegen_backend.so and a built _C.so
    (mixed build state; see PLAN_COMPILE_ONLY_FLAG.md) must not load _C.so
    in codegen-only mode: its spyre_mem aten::empty PrivateUse1
    registrations override the codegen fake kernels, and its
    @rpath/libc10.dylib dependency fails to dlopen when torch has not been
    imported first. The pure-Python _C re-exports the fake backend surface
    when _codegen_backend is importable and keeps runtime-only APIs
    fail-loud otherwise. The real module executes lazily on first
    attribute access so the backend resolves against whichever import
    order the process used.
    """
    existing = sys.modules.get("torch_spyre._C")
    if existing is not None:
        return existing

    import importlib.util

    placeholder = types.ModuleType("torch_spyre._C")
    placeholder.__doc__ = (
        "Lazy alias: executes torch_spyre/_C.py on first attribute access."
    )
    placeholder.__file__ = str(Path(__file__).parent / "_C.py")

    def _lazy_exec(name):
        current = sys.modules.get("torch_spyre._C")
        if current is not None and current is not placeholder:
            return getattr(current, name)
        spec = importlib.util.spec_from_file_location(
            "torch_spyre._C", Path(__file__).parent / "_C.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["torch_spyre._C"] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            # Restore the placeholder: a half-executed _C.py left in
            # sys.modules would make later imports "succeed" and fail with
            # misleading AttributeErrors instead of the original error.
            sys.modules["torch_spyre._C"] = placeholder
            raise
        return getattr(module, name)

    placeholder.__getattr__ = _lazy_exec
    sys.modules["torch_spyre._C"] = placeholder
    return placeholder


_factory_kernel_lib = None


def _register_codegen_only_factory_kernels() -> None:
    """Register fake-materializing kernels for the aten factory family on
    PrivateUse1 in codegen-only mode.

    Fake Spyre tensors carry no data (their storage is a 1-byte token), so
    torch.zeros / torch.ones / torch.full materialize a fake tensor whose
    contents are undefined — consistent with copy_tensor's no-op semantics.
    Two registrations matter:

    - zero_ / fill_: the C-level tail of torch.zeros et al. Without a
      kernel they raise "Could not run aten::fill_".
    - ones / full / new_ones: registered BEFORE the spyre decompositions
      load, so register_spyre_decomposition's has-kernel check skips
      them — otherwise eager torch.ones routes into the compiled
      spyre.ones_scalar custom op and dies at the launch seam instead
      of materializing.

    Launching on a fake tensor still fails loudly (launch-ABI seam).
    """
    global _factory_kernel_lib
    if _factory_kernel_lib is not None:
        return
    import torch

    _factory_kernel_lib = torch.library.Library("aten", "IMPL", "PrivateUse1")

    def _fill_noop(self, value):
        return self

    def _zero_noop(self):
        return self

    def _ones(
        size,
        *,
        dtype=None,
        layout=None,
        device=None,
        pin_memory=None,
        memory_format=None,
    ):
        return torch.ops.aten.empty.memory_format(
            size,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            memory_format=memory_format,
        )

    def _full(
        size,
        fill_value,
        *,
        dtype=None,
        layout=None,
        device=None,
        pin_memory=None,
    ):
        return torch.ops.aten.empty.memory_format(
            size,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )

    def _new_ones(self, size, *, dtype=None, layout=None, device=None, pin_memory=None):
        return torch.ops.aten.empty.memory_format(
            size,
            dtype=dtype if dtype is not None else self.dtype,
            layout=layout,
            device=device if device is not None else self.device,
            pin_memory=pin_memory,
        )

    _factory_kernel_lib.impl("fill_.Scalar", _fill_noop)
    _factory_kernel_lib.impl("fill_.Tensor", _fill_noop)
    _factory_kernel_lib.impl("zero_", _zero_noop)
    _factory_kernel_lib.impl("ones", _ones)
    _factory_kernel_lib.impl("full", _full)
    _factory_kernel_lib.impl("new_ones", _new_ones)

    # View family — mirrors the full runtime's spyre_views.cpp registration
    # set (view / _unsafe_view / _reshape_alias / alias / as_strided), which
    # lives in _C.so and never loads in codegen-only mode. A view of a fake
    # tensor materializes as a new fake tensor with the requested
    # sizes/strides and the source's SpyreTensorLayout tag (same
    # layout-preservation semantics as reinterpret_tensor_with_layout).
    from torch_spyre import _codegen_backend as _cb

    def _fake_view(self, size):
        # Derive the view's strides with torch's own view-stride algorithm:
        # a meta tensor carrying the source's sizes/strides, viewed to the
        # new size. Fabricating contiguous strides silently diverges from
        # real view semantics whenever the true result is non-contiguous
        # (e.g. a size-1-dim insertion on a transposed base). A non-viewable
        # request raises here, loudly, exactly like the real op.
        base = torch.empty_strided(
            tuple(self.size()), tuple(self.stride()), device="meta"
        )
        viewed = base.view(size)
        return _cb.spyre_empty_with_layout(
            list(size),
            list(viewed.stride()),
            self.dtype,
            _cb.get_spyre_tensor_layout(self),
        )

    def _fake_as_strided(self, size, stride, storage_offset=0):
        if storage_offset != 0:
            # Fake tensors carry a 1-byte token, not backing storage — an
            # offset-aliased view cannot be represented. Fail loudly rather
            # than fabricate metadata that diverges from the real op.
            raise RuntimeError(
                "codegen-only mode: as_strided with a non-zero "
                "storage_offset is not supported on fake Spyre tensors "
                "(no backing storage to offset into); the full runtime "
                "build is required for offset-aliased views"
            )
        return _cb.spyre_empty_with_layout(
            list(size),
            list(stride),
            self.dtype,
            _cb.get_spyre_tensor_layout(self),
        )

    def _fake_alias(self):
        return _cb.spyre_empty_with_layout(
            list(self.size()),
            list(self.stride()),
            self.dtype,
            _cb.get_spyre_tensor_layout(self),
        )

    _factory_kernel_lib.impl("view", _fake_view)
    _factory_kernel_lib.impl("_unsafe_view", _fake_view)
    _factory_kernel_lib.impl("_reshape_alias", _fake_as_strided)
    _factory_kernel_lib.impl("alias", _fake_alias)
    _factory_kernel_lib.impl("as_strided", _fake_as_strided)

    # Inductor constant-folds scalar outputs of fallback kernels via
    # .item() (aten::_local_scalar_dense). Fake tensors carry no data, so
    # the true scalar is unknowable in codegen-only mode; return 0 with a
    # per-call-site warning. A constant folded from fake data is a
    # placeholder: a compiled artifact embedding it must not be treated as
    # deployable. TORCH_SPYRE_STRICT_ITEM=1 turns the placeholder into a
    # hard raise (SpyreRuntimeUnavailableError) so pipelines can
    # machine-check that no fake constant entered their artifacts.
    import os
    import warnings

    _local_scalar_warned_sites = set()

    def _local_scalar_dense(self):
        if os.environ.get("TORCH_SPYRE_STRICT_ITEM", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise SpyreRuntimeUnavailableError(
                ".item() on a fake Spyre tensor (TORCH_SPYRE_STRICT_ITEM=1): "
                "the true scalar is unknowable in codegen-only mode; a "
                "constant folded from fake data is a placeholder and a "
                "compiled artifact embedding it is not deployable"
            )
        site = (tuple(self.size()), self.dtype)
        if site not in _local_scalar_warned_sites:
            _local_scalar_warned_sites.add(site)
            warnings.warn(
                "codegen-only mode: .item() on a fake Spyre tensor returned "
                "0; constants folded from fake data are placeholders "
                "(torch_spyre/__init__.py::_local_scalar_dense)",
                stacklevel=2,
            )
        return 0

    _factory_kernel_lib.impl("_local_scalar_dense", _local_scalar_dense)


def _activate_spyre_triton_driver() -> None:
    try:
        from triton.runtime.driver import driver as triton_driver
        from triton.backends.spyre.driver import SpyreDriver

        triton_driver.set_active(SpyreDriver())

        # Patch triton_helpers.set_driver_to_gpu to recognize the already-active
        # SpyreDriver. Without this, generated Triton kernel wrappers call
        # set_driver_to_gpu(), which cannot discover SpyreDriver because
        # is_active() returns False by design to avoid CUDA auto-discovery conflicts.
        import torch._inductor.runtime.triton_helpers as _triton_helpers

        _original_set_driver_to_gpu = _triton_helpers.set_driver_to_gpu

        def _spyre_aware_set_driver_to_gpu():
            if isinstance(triton_driver.active, SpyreDriver):
                return
            _original_set_driver_to_gpu()

        _triton_helpers.set_driver_to_gpu = _spyre_aware_set_driver_to_gpu
    except ImportError:
        pass  # triton-spyre not installed — Triton path unavailable


class _SpyreImpl:
    def __init__(self):
        self._initialized = False
        self._in_bad_fork = False
        self._pending_device_idx = None

        # When spawning a supprocess from inductor, ensure that IS_INDUCTOR_SPAWNED_SUBPROCESS=1
        # This will avoid additional initialization when processes are spawned from torch inductor (This happens in Triton pathway)
        # TODO: This may require monkey-patching the method where torch-inductor spawns a subprocess
        if int(os.getenv("IS_INDUCTOR_SPAWNED_SUBPROCESS", "0")):
            # NOTE (tmhoangt): currently, Spyre can't be used by more than one process
            # so, we want only the main process can have access to the actual device
            self._in_bad_fork = True
            self._initialized = True
        try:
            os.register_at_fork(after_in_child=self._mark_after_fork)
        except Exception:
            pass

    def __getattr__(self, name):
        if name == "_C":
            return self.__dict__.get(name, None)
        self._lazy_init()
        return super().__getattribute__(name)

    def _lazy_init(self):
        if self._initialized:
            return
        # Codegen-only mode: init is admission, not a runtime operation —
        # proceed so the compile-only branch below (device_count() == 0)
        # installs the monkey patches and dispatch-key kernels. Runtime-only
        # APIs (start_runtime, launch_kernel, DMA fill/copy) keep their own
        # fail-loud seams in torch_spyre._C.
        with _runtime_init_lock:
            if self._initialized:
                return
            # Load the C++ Module
            # put any light, once-per-process setup here
            self._C = importlib.import_module("torch_spyre._C")
            # Apply pending device index before runtime init
            pending = self._pending_device_idx
            if pending is not None:
                self._C.set_device(pending)
            # this will create the allocator
            if self.device_count() > 0:
                self._C.start_runtime()
            else:
                # compile-only builds (no runtime libs / no devices): skip
                # runtime init — _C is importable and kernel generation
                # (inductor/triton-spyre/codegen) never calls into it.
                pass
            self._initialized = True

            ## Run patch on import
            from ._monkey_patch import _patch_tensor_for_spyre

            _patch_tensor_for_spyre()

            # Activate SpyreDriver as Triton's active driver BEFORE registering
            # the Inductor backend, so any Triton compilation triggered during
            # registration finds the correct driver.
            _activate_spyre_triton_driver()

            from torch_spyre._inductor import _autoload as ts_autoload

            ts_autoload()

            # Permanently register PrivateUse1 kernels for DispatchKeys
            # so that eager-mode dispatch reaches the Spyre implementations
            # without requiring global monkey-patching.
            # Customops must be imported here because decompositions.py references
            # torch.ops.spyre.* at module level (e.g. torch.ops.spyre.rms_norm).
            import torch_spyre._inductor.customops  # noqa: F401
            from torch_spyre._inductor.decompositions import (
                _register_spyre_dispatchkey_kernels_permanently,
            )

            _register_spyre_dispatchkey_kernels_permanently()

    def _is_in_bad_fork(self) -> bool:
        return self._in_bad_fork

    def manual_seed(self, seed: int, device: int | None = None) -> None:
        fn = getattr(self._C, "manual_seed", None)
        if fn:
            fn(int(seed), -1 if device is None else int(device))

    def manual_seed_all(self, seed: int) -> None:
        _C = self._C
        if hasattr(_C, "manual_seed_all"):
            _C.manual_seed_all(int(seed))
        else:
            # Otherwise, fan out:
            for idx in range(self.device_count()):
                self.manual_seed(seed, device=idx)

    def is_available(self) -> bool:
        if codegen_only_available():
            return False
        if self._is_in_bad_fork():
            return True
        else:
            return self.device_count() > 0

    def is_initialized(self):
        return self._initialized and not self._is_in_bad_fork()

    def device_count(self) -> int:
        if codegen_only_available():
            return 0
        from . import _hooks

        return _hooks.device_count()

    def current_device(self) -> int:
        return getattr(self._C, "current_device", lambda: 0)()

    def set_device(self, idx: int) -> None:
        self._pending_device_idx = int(idx)
        # If runtime is already initialized, also set it on the C++ side.
        if self._initialized:
            fn = getattr(self._C, "set_device", None)
            if fn:
                fn(int(idx))

    def _mark_after_fork(self):
        self._initialized = True
        self._in_bad_fork = True


# Codegen-only mode must resolve torch_spyre._C to the pure-Python module
# before anything (this package, torch's backend autoload, or a direct
# `import torch_spyre._C`) can load a built _C.so from a mixed checkout.
if codegen_only_available():
    _install_codegen_only_C()


def make_spyre_module() -> types.ModuleType:
    """Return a real module object backed by a single _SpyreImpl instance."""
    impl = _SpyreImpl()

    mod = types.ModuleType(DEVICE_NAME)
    mod.__doc__ = "Spyre backend module (wrapped around a stateful implementation)."

    # Expose bound methods directly — they look like plain functions on the module.
    # These are *bound* to `impl`, so `self` is already captured.
    mod._is_in_bad_fork = lambda: impl._is_in_bad_fork()
    mod.manual_seed = lambda s: impl.manual_seed(s)
    mod.manual_seed_all = lambda s: impl.manual_seed_all(s)
    mod.is_available = lambda: impl.is_available()
    mod.is_initialized = lambda: impl.is_initialized()
    mod.device_count = lambda: impl.device_count()
    mod.codegen_only_available = codegen_only_available
    mod.current_device = lambda: impl.current_device()
    mod.set_device = lambda idx: impl.set_device(idx)
    mod._is_compiled = lambda: True
    if codegen_only_available():
        mod.memory = None
    else:
        from . import memory

        mod.memory = memory

    import torch  # noqa: E402

    mod.get_amp_supported_dtype = lambda: [torch.float16, torch.bfloat16]

    # Optional: forward unknown attrs to the impl or _C for convenience
    def __getattr__(name):
        if name in ["__file__", "_C"]:
            # Important: raising AttributeError ensures hasattr() returns False
            # without triggering our lazy loader.
            raise AttributeError(name)
        try:
            return object.__getattribute__(impl, name)
        except AttributeError:
            pass
        if name in {
            "Stream",
            "stream",
            "current_stream",
            "default_stream",
            "synchronize",
            "set_launch_context_provider",
        }:
            impl._lazy_init()
            from torch_spyre.streams import (
                Stream,
                stream,
                current_stream,
                default_stream,
                synchronize,
            )
            from torch_spyre.launch_context import set_launch_context_provider

            streams_map = {
                "Stream": Stream,
                "stream": stream,
                "current_stream": current_stream,
                "default_stream": default_stream,
                "synchronize": synchronize,
                "set_launch_context_provider": set_launch_context_provider,
            }
            return streams_map[name]
        if codegen_only_available():
            from torch_spyre import _C

            if hasattr(_C, name):
                return getattr(_C, name)
            raise AttributeError(name)
        if not hasattr(impl, "_C"):
            impl._lazy_init()
        if hasattr(impl._C, name):
            return getattr(impl._C, name)
        raise AttributeError(name)

    mod.__getattr__ = __getattr__
    _OPTIONAL_HOOKS = {
        "Scheduling",
        "GraphLowering",
        "Lowering",
        "Codegen",
        "Compile",
    }
    for _name in _OPTIONAL_HOOKS:
        setattr(mod, _name, None)

    # Keep a hidden handle to the impl (handy for tests/debugging)
    mod._impl = impl

    return mod


def _autoload_codegen_only():
    from ._runtime import force_codegen_only

    force_codegen_only()
    return _autoload()


def _autoload():
    # guard if autoload may run more than once
    if getattr(_autoload, "_ran", False):
        return
    _autoload._ran = True

    import torch  # noqa: E402
    if codegen_only_available():
        from . import _codegen_backend  # noqa: F401
    else:
        from . import _hooks  # noqa: F401

    # Set all the appropriate state on PyTorch
    torch.utils.rename_privateuse1_backend(DEVICE_NAME)
    torch._register_device_module(DEVICE_NAME, make_spyre_module())
    if codegen_only_available():
        import torch_spyre._C  # noqa: F401
        _register_codegen_only_factory_kernels()
        _activate_spyre_triton_driver()
        from torch_spyre._inductor import _autoload as ts_autoload
        from torch_spyre._inductor import _light_autoload

        ts_autoload()
        _light_autoload()
    else:
        import torch_spyre.ops.eager  # noqa: F401
        from torch_spyre._inductor import _light_autoload

        _light_autoload()

    # Set correct state for dynamo to support eager ops
    import torch._dynamo.config

    # Increasing the default number of graphs from 8 to 1024
    # to have enough cache space for all eager ops
    # You'll get recursion errors if this is exceeded
    torch._dynamo.config.cache_size_limit = 1024

    _orig_isAllocatorInitialized = torch._C._accelerator_isAllocatorInitialized

    def _patched_isAllocatorInitialized():
        try:
            return _orig_isAllocatorInitialized()
        except RuntimeError as e:
            if "not a DeviceAllocator" in str(e):
                return False
            raise
            return False

    torch._C._accelerator_isAllocatorInitialized = _patched_isAllocatorInitialized

    # set the default backend debugging to quiet
    # enable these if you would like to see runtime/compiler logging
    os.environ.setdefault("TORCH_SENDNN_LOG", "CRITICAL")
    os.environ.setdefault("DT_DEEPRT_VERBOSE", "-1")
    os.environ.setdefault("DTLOG_LEVEL", "error")


if not profiler.is_available():
    profiler = None
