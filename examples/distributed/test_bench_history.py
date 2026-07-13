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

"""Hardware-free unit tests for bench_history.py.

No torchrun, no Spyre device needed -- run with plain pytest:
    pytest examples/distributed/test_bench_history.py
"""

import csv
import json

from bench_history import (
    DEPLOYED_STATE_FILENAME,
    HISTORY_CSV_COLUMNS,
    RELEVANT_ENV_VARS,
    append_history_rows,
    capture_relevant_env_vars,
    get_deployed_state,
    get_git_commit,
    is_git_dirty,
)


def test_get_git_commit_returns_a_string(tmp_path):
    # This repo IS a git checkout, so this should resolve to a real hash --
    # but the function must never raise even if it can't (checked below).
    commit = get_git_commit(cwd=str(tmp_path.parent.parent))
    assert isinstance(commit, str)
    assert commit != ""


def test_get_git_commit_falls_back_to_unknown_outside_a_repo(tmp_path):
    # tmp_path is a fresh directory, not a git checkout.
    assert get_git_commit(cwd=str(tmp_path)) == "unknown"


def test_is_git_dirty_falls_back_to_none_outside_a_repo(tmp_path):
    assert is_git_dirty(cwd=str(tmp_path)) is None


def test_append_history_rows_creates_file_with_header(tmp_path):
    path = tmp_path / "history.csv"
    append_history_rows(
        str(path), [{"benchmark": "allreduce", "scenario_name": "ar_test"}]
    )

    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)

    assert header == HISTORY_CSV_COLUMNS
    # Missing columns fill in as empty string (DictWriter restval).
    assert row[HISTORY_CSV_COLUMNS.index("benchmark")] == "allreduce"
    assert row[HISTORY_CSV_COLUMNS.index("scenario_name")] == "ar_test"
    assert row[HISTORY_CSV_COLUMNS.index("git_commit")] == ""


def test_append_history_rows_second_call_does_not_rewrite_header(tmp_path):
    path = tmp_path / "history.csv"
    append_history_rows(str(path), [{"benchmark": "allreduce"}])
    append_history_rows(str(path), [{"benchmark": "allgather"}])

    with open(path, newline="") as f:
        lines = f.readlines()

    # Exactly one header line + two data rows -- not two headers.
    assert lines[0].strip() == ",".join(HISTORY_CSV_COLUMNS)
    assert len(lines) == 3


def test_append_history_rows_multiple_rows_in_one_call(tmp_path):
    path = tmp_path / "history.csv"
    append_history_rows(
        str(path),
        [
            {"benchmark": "allreduce"},
            {"benchmark": "allgather"},
            {"benchmark": "alltoall_proxy"},
        ],
    )

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    assert [r["benchmark"] for r in rows] == [
        "allreduce",
        "allgather",
        "alltoall_proxy",
    ]


def test_append_history_rows_noop_on_empty_list(tmp_path):
    path = tmp_path / "history.csv"
    append_history_rows(str(path), [])
    assert not path.exists()


def test_get_deployed_state_falls_back_to_git_without_a_marker(tmp_path):
    # No DEPLOYED_STATE_FILENAME present -- falls back to asking git directly
    # for torch-spyre (this repo checkout) and "unknown"/None for
    # flex-opensource, since there's no reliable way to locate a sibling repo
    # from a bare script_dir.
    ts_commit, ts_dirty, flex_commit, flex_dirty = get_deployed_state(str(tmp_path))
    assert ts_commit == "unknown"  # tmp_path is not a git checkout
    assert ts_dirty is None
    assert flex_commit == "unknown"
    assert flex_dirty is None


def test_get_deployed_state_prefers_the_marker_when_present(tmp_path):
    marker = {
        "torch_spyre_commit": "abc123",
        "torch_spyre_dirty": False,
        "flex_opensource_commit": "def456",
        "flex_opensource_dirty": True,
    }
    (tmp_path / DEPLOYED_STATE_FILENAME).write_text(json.dumps(marker))

    ts_commit, ts_dirty, flex_commit, flex_dirty = get_deployed_state(str(tmp_path))
    assert ts_commit == "abc123"
    assert ts_dirty is False
    assert flex_commit == "def456"
    assert flex_dirty is True


def test_get_deployed_state_falls_back_on_malformed_marker(tmp_path):
    (tmp_path / DEPLOYED_STATE_FILENAME).write_text("not valid json{")
    ts_commit, ts_dirty, flex_commit, flex_dirty = get_deployed_state(str(tmp_path))
    assert ts_commit == "unknown"
    assert flex_commit == "unknown"
    assert flex_dirty is None


def test_capture_relevant_env_vars_empty_when_nothing_set(monkeypatch):
    for name in RELEVANT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    assert capture_relevant_env_vars() == ""


def test_capture_relevant_env_vars_includes_only_whats_set(monkeypatch):
    for name in RELEVANT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BENCH_SKIP_WARMUP", "1")

    result = capture_relevant_env_vars()
    assert result == "BENCH_SKIP_WARMUP=1"


def test_capture_relevant_env_vars_joins_multiple_with_semicolons(monkeypatch):
    for name in RELEVANT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BENCH_SKIP_WARMUP", "1")
    monkeypatch.setenv("FLEX_HDMA_P2PSIZE", "65536")

    result = capture_relevant_env_vars()
    assert "BENCH_SKIP_WARMUP=1" in result
    assert "FLEX_HDMA_P2PSIZE=65536" in result
    assert result.count(";") == 1
