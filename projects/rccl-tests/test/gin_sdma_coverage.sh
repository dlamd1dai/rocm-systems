#!/usr/bin/env bash
# Run the GIN-SDMA host policy unit tests and emit a gcov line+branch coverage
# report for the shared policy header. Invoked by the `gin_sdma_coverage` CMake
# target (configure with -DBUILD_TESTS=ON -DENABLE_COVERAGE=ON). Works with both
# GNU gcov (g++/gcc builds) and `llvm-cov gcov` (amdclang/hipcc builds).
#
# Usage: gin_sdma_coverage.sh <test_binary> <cmake_binary_dir> <header_path> [gcov_tool] [min_pct]
set -euo pipefail

TEST_BIN="${1:?test binary}"
BUILD_DIR="${2:?cmake binary dir}"
HEADER="${3:?policy header path}"
GCOV_TOOL="${4:-gcov}"          # "gcov" or "/opt/rocm/llvm/bin/llvm-cov gcov"
MIN_PCT="${5:-95}"              # gate threshold for line + branch-taken %

HDR_NAME="$(basename "$HEADER")"

echo "== running $TEST_BIN to generate coverage data =="
"$TEST_BIN" >/dev/null

GCDA="$(find "$BUILD_DIR" -name '*gin_sdma_policy_test*.gcda' 2>/dev/null | head -1)"
if [[ -z "$GCDA" ]]; then
  echo "ERROR: no .gcda found under $BUILD_DIR — was -DENABLE_COVERAGE=ON set?" >&2
  exit 1
fi
GDIR="$(dirname "$GCDA")"
cd "$GDIR"

echo "== $GCOV_TOOL -b (in $GDIR) =="
# shellcheck disable=SC2086
$GCOV_TOOL -b -o "$GDIR" "$GCDA" > /tmp/gin_sdma_gcov.out 2>/dev/null \
  || $GCOV_TOOL -b "$GCDA" > /tmp/gin_sdma_gcov.out 2>/dev/null || true

echo "==================== GIN-SDMA policy coverage ===================="
BLOCK="$(grep -A3 "File '.*${HDR_NAME}'" /tmp/gin_sdma_gcov.out 2>/dev/null || true)"
if [[ -z "$BLOCK" ]]; then
  echo "WARN: could not find $HDR_NAME summary in gcov output; raw output:" >&2
  cat /tmp/gin_sdma_gcov.out >&2 || true
  exit 1
fi
echo "$BLOCK"

# Extract line % and branch-taken % and gate on MIN_PCT.
LINE_PCT="$(sed -n 's/^Lines executed:\([0-9.]*\)%.*/\1/p' <<<"$BLOCK" | head -1)"
BR_PCT="$(sed -n 's/^Taken at least once:\([0-9.]*\)%.*/\1/p' <<<"$BLOCK" | head -1)"
echo "================================================================="
echo "line=${LINE_PCT:-?}%  branch(taken)=${BR_PCT:-?}%  threshold=${MIN_PCT}%"

fail=0
awk "BEGIN{exit !(${LINE_PCT:-0} >= ${MIN_PCT})}" || { echo "FAIL: line coverage below ${MIN_PCT}%"; fail=1; }
awk "BEGIN{exit !(${BR_PCT:-0} >= ${MIN_PCT})}"   || { echo "FAIL: branch coverage below ${MIN_PCT}%"; fail=1; }
[[ "$fail" == 0 ]] && echo "COVERAGE_OK (>= ${MIN_PCT}%)"
exit "$fail"
