"""Offline TDD coverage for the ragged-pointwise compile-cache-clear fixture (#168).

PURPOSE
-------
`tests/conftest.py` defines `_clear_dir_contents` and the autouse
`_clear_compile_caches_for_ragged` fixture that clears the Inductor FxGraphCache,
Triton cache, and Sentient codegen cache before a fixed set of ragged M>1
pointwise/scalar-affine HW tests run (`_RAGGED_CACHE_CLEAR_TESTS`). This guards
against a stale pre-fix compiled kernel masking a correct choices.py fix on HW
(see #167/#168).

The fixture body itself only does anything useful on the HW box (it is a no-op
plugin hook otherwise). These tests cover what CAN be verified offline, with no
torch/board import required:
  1. The `_clear_dir_contents` helper actually empties a directory (keeping the
     directory itself) and tolerates files + a subdirectory.
  2. The helper is a silent no-op on a non-existent path.
  3. `_RAGGED_CACHE_CLEAR_TESTS` has not drifted from the real test function
     names in test_op_pointwise.py / test_op_scalar_affine.py (the exact drift
     class that caused a masked fix this session) — checked via AST parse of
     the file TEXT, so this runs with no torch/board import at all.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).parent
_CONFTEST_PATH = _TESTS_DIR / "conftest.py"


def _load_conftest_module():
    """Load tests/conftest.py by file path under a private module name.

    Avoids relying on `import conftest` picking up whatever pytest's own
    conftest-loading machinery may (or may not) have registered in
    sys.modules, and avoids re-triggering pytest_addoption/pytest_configure
    registration side effects that a second real import could cause.
    """
    spec = importlib.util.spec_from_file_location(
        "_torch_spyre_tests_conftest_under_test", _CONFTEST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_conftest = _load_conftest_module()
_clear_dir_contents = _conftest._clear_dir_contents
_RAGGED_CACHE_CLEAR_TESTS = _conftest._RAGGED_CACHE_CLEAR_TESTS


def test_clear_dir_contents_removes_children_keeps_dir(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("world")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested")

    _clear_dir_contents(tmp_path)

    assert tmp_path.exists()
    assert tmp_path.is_dir()
    assert list(tmp_path.iterdir()) == []


def test_clear_dir_contents_absent_dir_is_noop(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert not missing.exists()

    _clear_dir_contents(missing)  # must not raise

    assert not missing.exists()


def _def_names_in_file(path: Path) -> set[str]:
    """Top-level and nested `def <name>` function names, found via ast.parse
    of the raw file text (no import of the module — offline, no torch/board
    needed)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def test_fixture_name_set_matches_real_tests():
    pointwise_path = _TESTS_DIR / "test_op_pointwise.py"
    scalar_affine_path = _TESTS_DIR / "test_op_scalar_affine.py"
    assert pointwise_path.is_file()
    assert scalar_affine_path.is_file()

    defined_names = _def_names_in_file(pointwise_path) | _def_names_in_file(scalar_affine_path)

    missing = _RAGGED_CACHE_CLEAR_TESTS - defined_names
    assert not missing, (
        f"_RAGGED_CACHE_CLEAR_TESTS in tests/conftest.py names test function(s) "
        f"that no longer exist in test_op_pointwise.py / test_op_scalar_affine.py: "
        f"{sorted(missing)} (frozenset has drifted from the real function names)"
    )
