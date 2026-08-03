#!/usr/bin/env bash
# Incrementally rebuild all_reduce_perf in the existing /rt-build tree, then run
# the GIN-SDMA all_reduce GPU functional gate (-D 5 one-shot/two-shot, op/type
# matrix, forced two-shot tier, -c 1).
# Intended to run INSIDE the rccl-gin-gda-sdma-713 container with:
#   -v ~/rt-build:/rt-build -v ~/rocm-systems:/rocm-systems -v ~/rocm-systems/projects/rccl-tests:/src-tests
set -uo pipefail

echo "===== BUILD: make all_reduce_perf ====="
cd /rt-build || { echo "no /rt-build"; exit 2; }
if ! make all_reduce_perf -j"$(nproc)"; then
  echo "BUILD_FAILED"
  exit 3
fi
echo "BUILD_OK"
ls -l /rt-build/all_reduce_perf

echo "===== GATE: gin_sdma_gpu_functional.sh all_reduce ====="
# Keep the sweep meaningful but bounded; gate script defaults MAX_BYTES=64M.
export GIN_SDMA_MAX_BYTES="${GIN_SDMA_MAX_BYTES:-128M}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
if bash /rocm-systems/projects/rccl-tests/test/gin_sdma_gpu_functional.sh all_reduce /rt-build 8 mpirun; then
  echo "GATE_PASS"
else
  echo "GATE_FAIL rc=$?"
  exit 4
fi
