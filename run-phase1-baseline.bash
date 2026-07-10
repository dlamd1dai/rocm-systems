#!/usr/bin/env bash
# Phase 1 — Lock perf baseline on MI355 (Test#5 GIN Anvil SDMA).
# Run from rocm-systems repo root on smci355-ccs-aus-m03-17.
#
# Usage:
#   ./run-phase1-baseline.bash [BUILD]
#   BUILD=0  use cached docker build (default)
#   BUILD=1  full docker rebuild (--no-cache)
#
# Outputs (in current directory):
#   perfopt-baseline.log   — full sweep 128B–128M, Test#5 only
#   perfopt-128k.log       — 128K spot check (correctness)
#   perfopt-128m-only.log  — 128M only (quick perf check)

set -euo pipefail

BUILD="${1:-0}"
TAG="${BASELINE_TAG:-perfopt-$(date +%Y%m%d-%H%M)}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,2,4,6,8,10,12,14}"
# export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,2,4,6,8,10,12,14}"
export RCCL_GIN_RUN_TESTS=5

if [[ ! -f ./docker-gin-gda-sdma-build.bash ]]; then
  echo "error: run from rocm-systems repo root (docker-gin-gda-sdma-build.bash not found)" >&2
  exit 1
fi

echo "=== Phase 1 baseline: branch=$(git branch --show-current) commit=$(git rev-parse --short HEAD) ==="
echo "=== HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} tag=${TAG} ==="

if [[ "${SKIP_DOCKER_BUILD:-0}" != 1 ]]; then
  RCCL_CACHE_BUST="${TAG}" ROCSHMEM_CACHE_BUST="${TAG}" \
    ./docker-gin-gda-sdma-build.bash "${BUILD}" 2>&1 | tee "ddai-gin-docker-build-${TAG}.log"
fi

echo "=== Test#5 full sweep 128B–128M ==="
./docker-gin-gda-sdma-test.bash 8 128M 2>&1 | tee perfopt-baseline.log

echo "=== Test#5 128K correctness spot check ==="
MIN_BYTES=128K MAX_BYTES=128K ./docker-gin-gda-sdma-test.bash 8 128K 2>&1 | tee perfopt-128k.log

echo "=== Test#5 128M-only quick perf ==="
MIN_BYTES=128M MAX_BYTES=128M ./docker-gin-gda-sdma-test.bash 8 128M 2>&1 | tee perfopt-128m-only.log

echo "=== Done. Copy logs to work-plans if needed: ==="
echo "  scp perfopt-*.log <host>:work-plans/gin-sdma-backend/perfopt-07092026/"
