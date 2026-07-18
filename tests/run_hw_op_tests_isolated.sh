#!/usr/bin/env bash
# run_hw_op_tests_isolated.sh — cascade-proof HW runner for the test_op_<opname>.py suite.
#
# WHY: On a Spyre box a single device fault mid-file throws StreamInErrorState, which
# POISONS every subsequent test in the same pytest process — they all fake-fail even though
# they pass in isolation ("batch red != real red", see the run-torchspyre-tests-before-commit
# memory). A wedged compute-CB (e.g. matmul #164) makes this routine. xdist `-n1` does NOT
# help: the fault is an exception, not a segfault, so the worker survives and keeps cascading.
# Only a FRESH PROCESS clears the per-process device stream.
#
# STRATEGY (adaptive — fast when healthy, correct when wedged):
#   Phase 1: run each file once in a fresh process. If it exits clean, done (cheap).
#   Phase 2: for every FAILED id, RE-RUN it alone in a fresh process (pkill + cache-clear
#            before each). Exit 0 => the batch failure was a CASCADE FAKE (real pass / xfail).
#            Exit 1 => a REAL failure. This is the manual isolate-before-believing step,
#            mechanized.
#
# USAGE (run ON the box after sync):
#   source /home/tmhoangt/AIU/env.sh
#   bash tests/run_hw_op_tests_isolated.sh tests/test_op_pointwise.py [tests/test_op_*.py ...]
#   bash tests/run_hw_op_tests_isolated.sh tests/test_op_pointwise.py -- -k ragged   # extra pytest args after --
#
# EXIT: 0 iff every file has ZERO real failures (cascade fakes do not count). 1 otherwise.
set -uo pipefail

# Inductor defaults to a 32-worker compile pool (TORCHINDUCTOR_COMPILE_THREADS=32),
# forked PER pytest process. With one-shape-per-process each test compiles ~one kernel,
# so 32 fork workers is pure overhead (slow spin-up, resource contention on an already
# device-stressed box). Pin to 1 — matches the SP1 probe convention. Caller may override
# by exporting TORCHINDUCTOR_COMPILE_THREADS before invoking.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

# ---- args: test files up to "--", then extra pytest args --------------------
FILES=()
EXTRA=()
_parsing=1
for a in "$@"; do
    if [[ "$a" == "--" ]]; then _parsing=0; continue; fi
    if [[ $_parsing -eq 1 && -f "$a" ]]; then FILES+=("$a"); else EXTRA+=("$a"); fi
done
if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "Usage: bash $0 <test_file.py> [more...] [-- extra pytest args]" >&2
    exit 2
fi

CACHE_DIRS=(~/.sentient_cache ~/.triton/cache /tmp/torchinductor_* ~/.cache/torch)
_reset_device() {
    # Fresh process is what actually clears the stream; pkill mops up any segfaulted
    # stragglers, cache-clear avoids validating a stale kernel. Safe here: this runner
    # is bash, so pkill python3 does not kill the runner itself.
    pkill -9 python3 2>/dev/null || true
    sleep 1
    rm -rf "${CACHE_DIRS[@]}" 2>/dev/null || true
}

PYTEST=(python3 -m pytest -q -p no:cacheprovider --no-header)

TOTAL_REAL_FAIL=0
declare -a SUMMARY_LINES=()

for f in "${FILES[@]}"; do
    echo "========================================================================"
    echo "[iso] FILE: $f"
    echo "========================================================================"
    _reset_device
    out="$("${PYTEST[@]}" "$f" "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1)"
    file_exit=$?
    # Last non-empty line is pytest's summary (e.g. "6 passed, 2 xfailed in 9s").
    summary="$(printf '%s\n' "$out" | grep -E '(passed|failed|error|xfailed|xpassed|no tests ran)' | tail -1)"
    echo "[iso]   phase-1 summary: ${summary:-<none>} (exit $file_exit)"

    if [[ $file_exit -eq 0 ]]; then
        SUMMARY_LINES+=("$f: CLEAN — ${summary}")
        continue
    fi
    if [[ $file_exit -eq 5 ]]; then
        SUMMARY_LINES+=("$f: no tests collected")
        continue
    fi

    # Phase 2: extract FAILED ids and re-run each alone.
    mapfile -t failed_ids < <(printf '%s\n' "$out" | grep -oE 'FAILED [^ ]+' | sed 's/^FAILED //' | sort -u)
    if [[ ${#failed_ids[@]} -eq 0 ]]; then
        # Non-zero exit but no FAILED lines (e.g. collection/segfault) — surface raw.
        echo "[iso]   phase-1 non-zero exit with no FAILED ids — possible crash; last lines:"
        printf '%s\n' "$out" | tail -8
        SUMMARY_LINES+=("$f: NON-ZERO exit $file_exit, no FAILED ids (crash?) — INSPECT")
        TOTAL_REAL_FAIL=$((TOTAL_REAL_FAIL + 1))
        continue
    fi

    echo "[iso]   phase-2: re-running ${#failed_ids[@]} batch-failure(s) in isolation..."
    real_fail=()
    cascade=()
    for id in "${failed_ids[@]}"; do
        _reset_device
        "${PYTEST[@]}" "$id" "${EXTRA[@]+"${EXTRA[@]}"}" >/dev/null 2>&1
        iso_exit=$?
        if [[ $iso_exit -eq 0 ]]; then
            cascade+=("$id"); echo "[iso]     CASCADE-FAKE (passes alone): $id"
        else
            real_fail+=("$id"); echo "[iso]     REAL FAIL (fails alone, exit $iso_exit): $id"
        fi
    done

    if [[ ${#real_fail[@]} -eq 0 ]]; then
        SUMMARY_LINES+=("$f: PASS after isolation — ${#cascade[@]} cascade-fake(s), 0 real fail")
    else
        SUMMARY_LINES+=("$f: ${#real_fail[@]} REAL FAIL, ${#cascade[@]} cascade-fake(s)")
        TOTAL_REAL_FAIL=$((TOTAL_REAL_FAIL + ${#real_fail[@]}))
        for id in "${real_fail[@]}"; do SUMMARY_LINES+=("      REAL: $id"); done
    fi
done

echo ""
echo "========================================================================"
echo "[iso] FINAL REPORT (cascade fakes excluded)"
echo "========================================================================"
for line in "${SUMMARY_LINES[@]}"; do echo "  $line"; done
echo ""
if [[ $TOTAL_REAL_FAIL -eq 0 ]]; then
    echo "[iso] RESULT: ALL CLEAN — 0 real failures."
    exit 0
else
    echo "[iso] RESULT: $TOTAL_REAL_FAIL real failure(s) — see REAL ids above."
    exit 1
fi
