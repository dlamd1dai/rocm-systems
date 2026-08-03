#!/usr/bin/env bash
# Build all rccl-tests perf binaries in /rt-build, then run the GIN-SDMA GPU
# functional gate for every collective. Each collective runs independently so a
# single failure does not abort the rest; a per-collective PASS/FAIL summary is
# printed at the end. Runs INSIDE the rccl-gin-gda-sdma-713 container with:
#   -v ~/rt-build:/rt-build -v ~/rocm-systems:/rocm-systems -v ~/rocm-systems/projects/rccl-tests:/src-tests
set -uo pipefail

GATE=/rocm-systems/projects/rccl-tests/test/gin_sdma_gpu_functional.sh
export GIN_SDMA_MAX_BYTES="${GIN_SDMA_MAX_BYTES:-64M}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "===== BUILD: make -j all perf binaries ====="
cd /rt-build || { echo "no /rt-build"; exit 2; }
if ! make -j"$(nproc)"; then
  echo "BUILD_FAILED"
  exit 3
fi
echo "BUILD_OK"

COLLS=(broadcast all_gather alltoall scatter gather sendrecv reduce all_reduce)
declare -A RC
for c in "${COLLS[@]}"; do
  echo ""
  echo "########## GATE START: $c ##########"
  if bash "$GATE" "$c" /rt-build 8 mpirun; then
    RC[$c]=0
    echo "########## GATE DONE: $c PASS ##########"
  else
    RC[$c]=$?
    echo "########## GATE DONE: $c FAIL rc=${RC[$c]} ##########"
  fi
done

echo ""
echo "===================== SUMMARY ====================="
fail=0
for c in "${COLLS[@]}"; do
  if [[ "${RC[$c]}" -eq 0 ]]; then
    echo "  PASS  $c"
  else
    echo "  FAIL  $c (rc=${RC[$c]})"
    fail=1
  fi
done
echo "==================================================="
[[ "$fail" -eq 0 ]] && echo "ALL_PASS" || echo "SOME_FAILED"
exit "$fail"
