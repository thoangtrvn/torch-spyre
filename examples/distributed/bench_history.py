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

"""Append-only CSV history ledger for benchmark results.

One row per scenario per run, so results are comparable over time and
traceable to the exact code state that produced them. Only rank 0 should
ever call append_history_rows() (mirrors the existing --json rank-gating in
bench_distributed.py) -- this module adds a file lock on top of that as
belt-and-suspenders protection against two separate torchrun invocations
(e.g. a parallel CI matrix) landing on the same history file at once.

Commit tracking has a real wrinkle in this project's dev workflow: benchmarks
run on a remote Spyre host that a local machine pushes code to via `rsync`
(see sync_to_sentient.sh), not via `git push`/`git pull`. `git rev-parse
HEAD` run ON the remote checkout reflects the remote's OWN last-cloned
commit, which the sync workflow never advances -- it can be arbitrarily
stale relative to what rsync actually just deployed there, including
uncommitted local changes and commits the remote's `.git` has never seen.
On top of that, the runtime fixes this benchmark exercises often live in a
SEPARATE repo (flex-opensource, built into a shared library torch-spyre
links against), which plain `git rev-parse` inside torch-spyre can't see at
all. Two rows can show the identical git_commit/git_dirty while having run
against completely different flex-opensource states.

get_deployed_state() addresses this: sync_to_sentient.sh stamps a small JSON
marker (DEPLOYED_STATE_FILENAME) with the LOCAL (source-of-truth) commit/
dirty state for both repos at sync time, and pushes it alongside the code.
If that marker is present, it is authoritative and used instead of running
git commands on the (possibly stale) remote checkout. Falls back to the old
git-on-cwd behavior for torch-spyre (and "unknown" for flex-opensource) when
running somewhere the marker was never deployed, e.g. a pure local dev loop.
"""

import csv
import fcntl
import json
import os
import subprocess

HISTORY_CSV_COLUMNS = [
    "timestamp_utc",
    "git_commit",
    "git_dirty",
    "benchmark",
    "scenario_name",
    "description",
    "is_proxy",
    "notes",
    "model_name",
    "workload_point",
    "phase",
    "batch",
    "seq_len",
    "hidden_size",
    "vocab_size",
    "num_layers",
    "num_experts",
    "top_k",
    "tokens_to_expert",
    "world_size",
    "rank",
    "iterations",
    "warmup",
    "dtype",
    "elements",
    "message_bytes",
    "aggregate_bytes_per_rank",
    "e2e_us_mean",
    "e2e_us_p50",
    "e2e_us_p99",
    "e2e_us_min",
    "e2e_us_max",
    "throughput_gbps",
    "coll_allreduce_algo",
    "flex_opensource_commit",
    "flex_opensource_dirty",
    "env_vars",
]

# Name of the deployment marker sync_to_sentient.sh stamps next to this file.
DEPLOYED_STATE_FILENAME = ".deployed_state.json"

# Env vars that have materially changed what a scenario actually exercises
# during this project's HDMA/fail-fast investigation work (e.g.
# BENCH_SKIP_WARMUP disables the benchmark-level P2P warm-up pairing, so a
# passing row means the underlying runtime fix -- not the workaround -- is
# what's being measured; FLEX_HDMA_P2PSIZE controls whether the HDMA P2P pool
# is large enough for a given message). Extend this list as new env vars
# become relevant -- deliberately a single free-form column rather than one
# CSV column per var, so adding one doesn't require a schema migration.
RELEVANT_ENV_VARS = (
    "BENCH_SKIP_WARMUP",
    "FLEX_HDMA_P2PSIZE",
)


def capture_relevant_env_vars(names=RELEVANT_ENV_VARS):
    """Return a compact "name=value;name2=value2" string of whichever of
    `names` are actually set in the environment, or "" if none are. Only sets
    vars are included, so an empty result means "nothing unusual configured"
    rather than a wall of blank key=value pairs."""
    parts = [
        f"{name}={value}"
        for name in names
        if (value := os.environ.get(name)) is not None
    ]
    return ";".join(parts)


def get_git_commit(cwd=None):
    """Return the current git commit hash, or "unknown" if it can't be
    determined (e.g. not a git checkout). Never raises -- a git failure must
    not crash a benchmark run."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def is_git_dirty(cwd=None):
    """Return True if the working tree has uncommitted changes, False if
    clean, or None if this can't be determined. Never raises."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return len(result.stdout.strip()) > 0
    except Exception:
        return None


def get_deployed_state(script_dir):
    """Return (torch_spyre_commit, torch_spyre_dirty, flex_opensource_commit,
    flex_opensource_dirty) for the code that actually produced this
    benchmark run.

    Prefers the marker sync_to_sentient.sh stamps at
    "<script_dir>/DEPLOYED_STATE_FILENAME" -- written from the LOCAL machine
    at sync time, so it is accurate even though the remote's own `.git`
    (which the sync workflow never advances) is not. Falls back to running
    git directly in `script_dir` for torch-spyre only (the historical
    behavior) when no marker is present, e.g. a pure local dev loop that
    never went through the sync script; flex-opensource is "unknown" in
    that fallback since there is no reliable way to locate it from here.

    Never raises -- a missing or malformed marker must not crash a
    benchmark run, it should just fall back.
    """
    marker_path = os.path.join(script_dir, DEPLOYED_STATE_FILENAME)
    try:
        with open(marker_path) as f:
            state = json.load(f)
        return (
            state.get("torch_spyre_commit", "unknown"),
            state.get("torch_spyre_dirty"),
            state.get("flex_opensource_commit", "unknown"),
            state.get("flex_opensource_dirty"),
        )
    except (OSError, ValueError):
        # No marker (or unreadable/malformed one) -- fall back to asking git
        # directly. This only ever reflects torch-spyre, and only correctly
        # reflects it when this script's own checkout's git history is what
        # is actually running (true for local dev, NOT true after an rsync
        # deploy without a marker).
        return (
            get_git_commit(cwd=script_dir),
            is_git_dirty(cwd=script_dir),
            "unknown",
            None,
        )


def append_history_rows(path, rows):
    """Append one CSV row per entry in `rows` (each a dict, keys a subset of
    HISTORY_CSV_COLUMNS) to `path`, creating the file (with header) if it
    doesn't exist yet or is empty.

    Wraps the open-append-close in an exclusive flock (POSIX-only -- this
    targets Linux Spyre hosts, no new dependency) so two processes appending
    to the same file at the same time don't interleave/corrupt each other's
    rows. Caller is responsible for only calling this from rank 0.
    """
    if not rows:
        return

    # Open with "a+" (create-if-missing, never truncates) and determine
    # write_header AFTER acquiring the lock, not before -- otherwise two
    # processes racing to create the file could both decide to write a
    # header (checking os.path.exists/getsize before the lock is held is not
    # atomic with respect to a concurrent writer doing the same).
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            write_header = f.tell() == 0
            writer = csv.DictWriter(f, fieldnames=HISTORY_CSV_COLUMNS, restval="")
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
