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
import shutil
import sys
from pathlib import Path

os.environ.setdefault(
    "TORCH_DEVICE_BACKEND_AUTOLOAD", "0"
)  # must be before torch import

import glob

from setuptools import Command, setup

PATH_NAME = "torch_spyre"
PACKAGE_NAME = "torch_spyre"
DISTRIBUTED_PACKAGE_NAME = "spyre_ccl"


def check_libflex():
    ld_library_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    for path in ld_library_paths:
        if glob.glob(os.path.join(path, "libflex.so")):
            return True
    return False


ROOT_DIR = Path(__file__).absolute().parent
CSRC_DIR = ROOT_DIR / PATH_NAME / "csrc"
DISTRIBUTED_SRC_DIR = CSRC_DIR / "distributed"
PROFILER_SRC_DIR = CSRC_DIR / "profiler"


# Automatically download json.hpp if not present
def maybe_download_nlohmann_json():
    """return path to header files"""
    import urllib.request

    NLOHMANN_URL = "https://raw.githubusercontent.com/nlohmann/json/v3.11.2/single_include/nlohmann/json.hpp"
    SHARED_PATH = Path(
        os.environ.get("SHARED_DEPS_DIR", ROOT_DIR / PATH_NAME / "csrc" / "external")
    )
    NLOHMANN_INC_DIR = SHARED_PATH / "nlohmann" / "include"
    NLOHMANN_DIR = NLOHMANN_INC_DIR / "nlohmann"

    NLOHMANN_HEADER = os.path.join(NLOHMANN_DIR, "json.hpp")
    if not os.path.exists(NLOHMANN_HEADER):
        os.makedirs(NLOHMANN_DIR, exist_ok=True)
        print("Downloading nlohmann/json.hpp...")
        urllib.request.urlretrieve(NLOHMANN_URL, NLOHMANN_HEADER)
    return NLOHMANN_INC_DIR


INCLUDE_DIRS = [
    CSRC_DIR,
    # "tracy/public"
]
LIBRARY_DIRS = []


INCLUDE_DIRS += [maybe_download_nlohmann_json()]

cmake_include_path = os.environ.get("CMAKE_INCLUDE_PATH", "")
extra_include_dirs = cmake_include_path.split(":") if cmake_include_path else []
INCLUDE_DIRS += [Path(p) for p in extra_include_dirs if p]

cmake_library_path = os.environ.get("CMAKE_LIBRARY_PATH", "")
extra_library_dirs = cmake_library_path.split(":") if cmake_library_path else []
LIBRARY_DIRS += [Path(p) for p in extra_library_dirs if p]

# Kernel-generation-only build: compile/import the extension without probing or
# initializing Spyre hardware. macOS uses generated runtime stubs because the
# real runtime libraries are not available there; Linux is expected to link the
# real runtime libraries and only bypass hardware discovery/initialization.
COMPILE_ONLY_BUILD = os.environ.get("TORCH_SPYRE_COMPILE_ONLY", "0") == "1"
RUNTIME_STUB_BUILD = COMPILE_ONLY_BUILD and sys.platform == "darwin"

COMPILE_AIUPTI = (
    os.environ.get("USE_SPYRE_PROFILER", "0" if COMPILE_ONLY_BUILD else "1") == "1"
)

if "RUNTIME_INSTALL_DIR" in os.environ:
    # take lower precedence than CMAKE_LIBRARY_PATH and CMAKE_INCLUDE_PATH
    RUNTIME_DIR = Path(os.environ["RUNTIME_INSTALL_DIR"])
    SENLIB_DIR = Path(os.environ["SENLIB_INSTALL_DIR"])
    DEEPTOOLS_DIR = Path(os.environ["DEEPTOOLS_INSTALL_DIR"])
    INCLUDE_DIRS += [
        RUNTIME_DIR / "include",
    ]
    INCLUDE_DIRS += [
        RUNTIME_DIR / "include" / "concurrentqueue" / "moodycamel",
    ]
    INCLUDE_DIRS += [
        SENLIB_DIR / "include",
    ]
    INCLUDE_DIRS += [
        DEEPTOOLS_DIR / "include",
    ]
    LIBRARY_DIRS += [RUNTIME_DIR / "lib"]

# The USE_SPYRE_CCL environment variable can be used to build torch-spyre
# without support for Multi-Spyre. This is for developers only.
# If set to '0' then Multi-Spyre support is disabled.
# Otherwise (default) Multi-Spyre support is enabled.
use_spyre_ccl_default = "0" if COMPILE_ONLY_BUILD else "1"
use_spyre_ccl = os.environ.get("USE_SPYRE_CCL", use_spyre_ccl_default) != "0"

if not use_spyre_ccl:
    print("=" * 80)
    print("WARNING: Multi-Spyre support has been disabled")
    print("=" * 80)
else:
    if "SPYRE_COMMS_INSTALL_DIR" in os.environ:
        SPYRE_COMMS_DIR = Path(os.environ["SPYRE_COMMS_INSTALL_DIR"])
        if not SPYRE_COMMS_DIR.exists():
            raise RuntimeError(
                f"SPYRE_COMMS_INSTALL_DIR directory does not exist: {SPYRE_COMMS_DIR}"
            )
        SPYRE_COMMS_INCLUDE_DIR = SPYRE_COMMS_DIR / "include"
        if not SPYRE_COMMS_INCLUDE_DIR.exists():
            raise RuntimeError(
                "SPYRE_COMMS_INSTALL_DIR include directory does not exist: "
                f"{SPYRE_COMMS_INCLUDE_DIR}"
            )
        SPYRE_COMMS_LIB_DIR = SPYRE_COMMS_DIR / "lib"
        if not SPYRE_COMMS_LIB_DIR.exists():
            raise RuntimeError(
                "SPYRE_COMMS_INSTALL_DIR lib directory does not exist: "
                f"{SPYRE_COMMS_LIB_DIR}"
            )
        INCLUDE_DIRS += [
            SPYRE_COMMS_INCLUDE_DIR,
        ]
        LIBRARY_DIRS += [SPYRE_COMMS_LIB_DIR]
    else:
        raise RuntimeError(
            "SPYRE_COMMS_INSTALL_DIR not set. "
            "Set USE_SPYRE_CCL=0 to build without Multi-Spyre support, "
            "or set the SPYRE_COMMS_INSTALL_DIR to the Spyre Comms install directory."
        )

if RUNTIME_STUB_BUILD:
    LIBRARIES = []
else:
    LIBRARIES = ["flex"]

if COMPILE_AIUPTI:  # Include kineto and libaiupti headers
    import torch

    KINETO_INCLUDE_DIR = Path(torch.__path__[0]) / "include" / "kineto"
    INCLUDE_DIRS += [KINETO_INCLUDE_DIR]

    # On default system install, aiupti is installed in runtime folders
    if os.environ.get("LIBAIUPTI_INSTALL_DIR", "") != "":
        LIBAIUPTI_DIR = Path(os.environ["LIBAIUPTI_INSTALL_DIR"])
        LIBRARY_DIRS += [LIBAIUPTI_DIR / "lib"]
        INCLUDE_DIRS += [LIBAIUPTI_DIR / "include"]

    if not RUNTIME_STUB_BUILD:
        LIBRARIES.insert(-1, "aiupti")  # Build dependency on flex

if use_spyre_ccl and not RUNTIME_STUB_BUILD:
    LIBRARIES.append("spyre_comms")

# FIXME: added no-deprecated as this fails in sentensor_shape.hpp
# - we need to fix there
# Note that we always compile with debug info
# EXTRA_CXX_FLAGS = ["-g", "-Wall", "-Werror", "-Wno-deprecated"]
# Set TORCH_SPYRE_DEBUG=1 to build with -O0 for easier debugging
NO_OPT_BUILD = os.environ.get("TORCH_SPYRE_DEBUG", "0") == "1"

EXTRA_CXX_FLAGS = ["-g", "-Wall", "-Wno-deprecated", "-std=c++20"]
if NO_OPT_BUILD:
    EXTRA_CXX_FLAGS += ["-O0"]

if COMPILE_ONLY_BUILD:
    # Compile-only bypasses hardware probing/runtime init. Darwin additionally
    # builds without runtime libraries and uses generated stubs.
    EXTRA_CXX_FLAGS += [
        "-std=c++20",
        "-DTORCH_SPYRE_COMPILE_ONLY",
        "-DSENLIB_X86_SUPPORT=1",
        "-DBOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED",
    ]
    if sys.platform == "darwin":
        EXTRA_CXX_FLAGS.append("-Doff64_t=off_t")


class clean(Command):
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Remove torch_spyre extension
        for path in (ROOT_DIR / PATH_NAME).glob("**/*.so"):
            path.unlink()
        # Remove build directory
        build_dirs = [
            ROOT_DIR / "build",
        ]
        for path in build_dirs:
            if path.exists():
                shutil.rmtree(str(path), ignore_errors=True)


def _runtime_stub_path():
    return ROOT_DIR / PATH_NAME / "_runtime_stubs.dylib"


def _remove_runtime_stub_dylib():
    stub = _runtime_stub_path()
    if stub.exists():
        stub.unlink()


def _build_runtime_stub_dylib():
    """Emit the macOS compile-only dylib that lets _C.so import.

    Compile-only builds link _C.so without the real runtime libraries, but dyld
    still requires unresolved runtime symbols to exist at import time. The dylib
    provides dummy definitions for unused runtime paths plus a few real helpers
    needed by import, layout formatting, and constant encoding.
    """
    import subprocess  # noqa: PLC0415

    build_lib = ROOT_DIR / "build"
    exts = list(build_lib.glob("lib.*/torch_spyre/_C.so"))
    in_place_ext = ROOT_DIR / PATH_NAME / "_C.so"
    if in_place_ext.exists():
        exts.append(in_place_ext)
    if not exts:
        return
    pkg_dir = exts[-1].parent

    # Use dyld's import list: it includes eager load-time fixups that nm misses.
    syms: set[str] = set()
    so = pkg_dir / "_C.so"
    dyld_info = subprocess.run(
        ["dyld_info", "-imports", str(so)],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in dyld_info.stdout.splitlines():
        tok = line.split()
        if len(tok) >= 2 and tok[1] == "(from" and tok[0].startswith("__Z"):
            syms.add(tok[0])

    # Only stub the runtime namespaces — torch/python/libc++ symbols resolve
    # from the host process.
    runtime_ns = (
        "flex",
        "senlib",
        "deeptools",
        "data_conversion_info",
        "boost",
        "moodycamel",
        "sendnn",
        "EnumsConversion",
        "DataFormats",
        "Hcm",
        "Sen",
    )
    stub_syms = sorted(
        s
        for s in syms
        if s.startswith("__Z") and any(f"{len(ns)}{ns}" in s for ns in runtime_ns)
    )
    if not stub_syms:
        return

    lines = [".section __TEXT,__text", ".globl ___stub_marker", "___stub_marker: ret"]
    data_lines = [".section __DATA,__data"]
    cpp_syms = {
        "__ZN15EnumsConversion21dataFormatsToBitWidthE",
        "__ZN15EnumsConversion19dataFormatsToStringE11DataFormats",
        "__ZN4flex13getNumDevicesEv",
        "__ZN9deeptools14FloatToFp16BinEf",
        "__ZlsRNSt3__113basic_ostreamIcNS_11char_traitsIcEEEERK11DataFormats",
    }
    for s in stub_syms:
        if s in cpp_syms:
            continue  # real implementations live in runtime_stubs.cpp
        if s.startswith("__ZTV") or s.startswith("__ZTI") or s.startswith("__ZTS"):
            data_lines.append(f".globl {s}")
            data_lines.append(f"{s}:")
            data_lines.append("    .quad 0, 0, 0, 0")
        else:
            lines.append(f".globl {s}")
            lines.append(f"{s}:")
            lines.append("    ret")
    lines.extend(data_lines)
    asm = build_lib / "runtime_stubs.s"
    asm.write_text("\n".join(lines) + "\n")

    # Some import/codegen helpers need real behavior; everything else is a dummy.
    cpp = build_lib / "runtime_stubs.cpp"
    cpp.write_text(
        """#include <cstdint>
#include <cstddef>
#include <map>
#include <ostream>
#include <string>

enum class DataFormats {
  SEN169_FP16,
  IEEE_FP32,
  INVALID,
  SEN143_FP8,
  SEN152_FP8,
  SEN153_FP9,
  SENINT2,
  SENINT4,
  SENINT8,
  SENINT16,
  SENINT24,
  IEEE_INT64,
  IEEE_INT32,
  SENUINT32,
  SENUINT2,
  IEEE_FP16,
  BOOL,
  BFLOAT16,
  SEN18F_FP24,
  SEN080_FP8,
  SEN053_FP8,
  SEN121_FP4,
  SENUINT8,
  NUM_DATA_FORMATS,
};

namespace EnumsConversion {
extern const std::map<DataFormats, int> dataFormatsToBitWidth = {
    {DataFormats::SEN169_FP16, 16}, {DataFormats::IEEE_FP32, 32},
    {DataFormats::SEN143_FP8, 8},   {DataFormats::SEN152_FP8, 8},
    {DataFormats::SEN153_FP9, 9},   {DataFormats::SENINT2, 2},
    {DataFormats::SENINT4, 4},      {DataFormats::SENINT8, 8},
    {DataFormats::SENINT16, 16},    {DataFormats::SENINT24, 24},
    {DataFormats::IEEE_INT64, 64},  {DataFormats::IEEE_INT32, 32},
    {DataFormats::SENUINT32, 32},   {DataFormats::SENUINT2, 2},
    {DataFormats::IEEE_FP16, 16},   {DataFormats::BOOL, 8},
    {DataFormats::BFLOAT16, 16},    {DataFormats::SEN18F_FP24, 24},
    {DataFormats::SEN080_FP8, 8},   {DataFormats::SEN053_FP8, 8},
    {DataFormats::SEN121_FP4, 4},   {DataFormats::SENUINT8, 8},
};

std::string dataFormatsToString(DataFormats df) {
  switch (df) {
    case DataFormats::SEN169_FP16:
      return "SEN169_FP16";
    case DataFormats::IEEE_FP32:
      return "IEEE_FP32";
    case DataFormats::INVALID:
      return "INVALID";
    case DataFormats::SEN143_FP8:
      return "SEN143_FP8";
    case DataFormats::SEN152_FP8:
      return "SEN152_FP8";
    case DataFormats::SEN153_FP9:
      return "SEN153_FP9";
    case DataFormats::SENINT2:
      return "SENINT2";
    case DataFormats::SENINT4:
      return "SENINT4";
    case DataFormats::SENINT8:
      return "SENINT8";
    case DataFormats::SENINT16:
      return "SENINT16";
    case DataFormats::SENINT24:
      return "SENINT24";
    case DataFormats::IEEE_INT64:
      return "IEEE_INT64";
    case DataFormats::IEEE_INT32:
      return "IEEE_INT32";
    case DataFormats::SENUINT32:
      return "SENUINT32";
    case DataFormats::SENUINT2:
      return "SENUINT2";
    case DataFormats::IEEE_FP16:
      return "IEEE_FP16";
    case DataFormats::BOOL:
      return "BOOL";
    case DataFormats::BFLOAT16:
      return "BFLOAT16";
    case DataFormats::SEN18F_FP24:
      return "SEN18F_FP24";
    case DataFormats::SEN080_FP8:
      return "SEN080_FP8";
    case DataFormats::SEN053_FP8:
      return "SEN053_FP8";
    case DataFormats::SEN121_FP4:
      return "SEN121_FP4";
    case DataFormats::SENUINT8:
      return "SENUINT8";
    case DataFormats::NUM_DATA_FORMATS:
      return "NUM_DATA_FORMATS";
  }
  return "UNKNOWN";
}
}  // namespace EnumsConversion

std::ostream& operator<<(std::ostream& os, const DataFormats& df) {
  os << EnumsConversion::dataFormatsToString(df);
  return os;
}

namespace flex {
int getNumDevices() { return 0; }  // compile-only: honestly no devices
}

namespace deeptools {
static inline uint16_t FloatToFp16BinInternal(float f) {
  uint32_t f32;
  {  // BinaryConvert<uint32_t>: bit-level float->uint32
    __builtin_memcpy(&f32, &f, 4);
  }
  int32_t exponent = ((f32 >> 23) & 0xFF) - 127;
  uint32_t fraction = f32 & 0x7FFFFF;
  // fp32: 1 (sign) + 8 (exp) + 23 (fra) -> sentient fp16: 1 + 6 + 9

  uint32_t rounding = 0;
  if (((fraction >> (23 - 9 - 1)) & 1) && ((fraction >> (23 - 9)) != 0x1FF)) {
    rounding = 1;
  }
  fraction = (fraction >> (23 - 9)) + rounding;
  uint32_t f16;
  if (exponent > 31) {
    f16 = (f32 >> 16) | 0x7FFF;  // INF with sign bit

    // Typical DL use-case is mask operation where saturating does not
    // impact accuracy and avoids potential NaN values during computation
    if (f < 0) {
      // for large negative values, cap saturation at 0xEFFF
      f16 = f16 & 0xEFFF;
    } else {
      // for large positive values, cap saturation at 0x6FFF
      f16 = f16 & 0x6FFF;
    }
  } else if (exponent < -30) {
    f16 = (f32 >> 16) & 0x8000;  // Zero with sign bit
  } else {
    f16 = ((f32 >> 16) & 0x8000) |
          static_cast<uint32_t>(exponent + 31) << 9u | fraction;
  }
  return uint16_t(f16);
}

uint16_t FloatToFp16Bin(float f) { return FloatToFp16BinInternal(f); }
}  // namespace deeptools
"""
    )

    out = _runtime_stub_path()
    subprocess.run(
        [
            "clang++",
            "-arch",
            "arm64",
            "-shared",
            "-std=c++17",
            str(cpp),
            str(asm),
            "-o",
            str(out),
        ],
        check=True,
    )
    print(
        f"compile-only build: wrote runtime stub dylib ({len(stub_syms)} symbols) "
        f"-> {out}"
    )


if __name__ == "__main__":
    import sys

    is_meta = any(
        cmd in sys.argv for cmd in ["dist_info", "egg_info", "install_egg_info"]
    )

    if is_meta:
        setup(
            entry_points={
                "torch.backends": [
                    "torch_spyre = torch_spyre:_autoload",
                ],
            },
        )
    else:
        from torch.utils.cpp_extension import BuildExtension, CppExtension

        sources = list(CSRC_DIR.glob("*.cpp"))
        profiler_sources = (
            list(PROFILER_SRC_DIR.glob("*.cpp")) if COMPILE_AIUPTI else []
        )
        distributed_sources = (
            list(DISTRIBUTED_SRC_DIR.glob("*.cpp")) if use_spyre_ccl else []
        )

        core_src_paths = [p.relative_to(ROOT_DIR).as_posix() for p in sorted(sources)]
        distributed_src_paths = [
            p.relative_to(ROOT_DIR).as_posix() for p in sorted(distributed_sources)
        ]
        profiler_src_paths = [
            p.relative_to(ROOT_DIR).as_posix() for p in sorted(profiler_sources)
        ]

        # Build define_macros list conditionally
        base_define_macros = [
            ("PACKAGE_NAME", f'"{PACKAGE_NAME}"'),
            ("SPYRE_DOWNCAST_ENV", '"TORCH_SPYRE_DOWNCAST_WARN"'),
            ("EAGER_MODE_ENV", '"EAGER_MODE"'),
            ("BOOST_ALL_DYN_LINK", None),  # avoid static link to boost
        ]
        if use_spyre_ccl:
            base_define_macros.append(("USE_SPYRE_CCL", None))

        base_define_macros.append(("USE_FLEX_NAMESPACE", None))

        if COMPILE_AIUPTI:
            base_define_macros.append(("HAS_AIUPTI", None))
            base_define_macros.append(("USE_KINETO", None))
            base_define_macros.append(("FMT_HEADER_ONLY", None))

        ext_modules = [
            CppExtension(
                name=f"{PACKAGE_NAME}._C",
                sources=core_src_paths + distributed_src_paths + profiler_src_paths,
                include_dirs=[str(p) for p in INCLUDE_DIRS],
                library_dirs=[str(p) for p in LIBRARY_DIRS],
                libraries=LIBRARIES,
                extra_compile_args={"cxx": EXTRA_CXX_FLAGS},
                define_macros=base_define_macros
                + [
                    ("MODULE_NAME", f'"{PACKAGE_NAME}._C"'),
                ],
            ),
        ]

        BUILD_DIR = ROOT_DIR / "build"

        _BuildExtension = BuildExtension.with_options(
            no_python_abi_suffix=True, verbose=True
        )

        class PermanentBuildExtension(_BuildExtension):
            def finalize_options(self):
                super().finalize_options()
                self.build_temp = str(BUILD_DIR)

            def build_extension(self, ext):
                # Use a per-extension subdirectory so each gets its own build.ninja
                original_build_temp = self.build_temp
                self.build_temp = os.path.join(original_build_temp, ext.name)
                os.makedirs(self.build_temp, exist_ok=True)
                try:
                    super().build_extension(ext)
                finally:
                    self.build_temp = original_build_temp

            def run(self):
                super().run()
                if RUNTIME_STUB_BUILD:
                    _build_runtime_stub_dylib()
                else:
                    _remove_runtime_stub_dylib()

        setup(
            ext_modules=ext_modules,
            cmdclass={
                "build_ext": PermanentBuildExtension,
                "clean": clean,
            },
            entry_points={
                "torch.backends": [
                    "torch_spyre = torch_spyre:_autoload",
                ],
            },
        )
