#! /usr/bin/env bash
# 1p4g gin-sdma A2A matrix on the MI455 SUT. IB withheld.
# Usage: bash gin-sdma-a2a-mi455-run-all.bash
set -uo pipefail
fail=0

export DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gin-sdma-a2a-mi455}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE=1
export DOCKER_UVERBS=0
export DOCKER_EXTRA="${DOCKER_EXTRA:---tmpfs /dev/infiniband}"
export A2A_WARMUP="${A2A_WARMUP:-2}"
export A2A_ITERS="${A2A_ITERS:-5}"
export TEST5_MLX5_PREFLIGHT=0
export RCCL_IMAGE_INFO="${RCCL_IMAGE_INFO:-1}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
HARNESS="${ROOT}/gin-sdma-a2a-test.bash"
LOGDIR="${LOGDIR:-$HOME/gin-a2a-runs/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOGDIR}"
echo "LOGDIR=${LOGDIR} IMAGE=${DOCKER_IMAGE}"

run_logged() {
  local name="$1"; shift
  echo "===== ${name} =====" | tee -a "${LOGDIR}/summary.txt"
  if "$@" >"${LOGDIR}/${name}.log" 2>&1; then
    echo "PASS ${name}" | tee -a "${LOGDIR}/summary.txt"
  else
    echo "FAIL ${name} exit=$?" | tee -a "${LOGDIR}/summary.txt"
    fail=1
  fi
}

# Host gtests (no GPU).
run_logged ut-ll-policy docker run --rm --init "${DOCKER_IMAGE}" \
  bash -lc 'set -e; f=$(find / -name rccl-UnitTestsGinFabricLLPolicy -type f 2>/dev/null | head -1); test -n "$f"; "$f"'

run_logged ut-lsa-policy docker run --rm --init "${DOCKER_IMAGE}" \
  bash -lc 'f=$(find / -name rccl-UnitTestsGinFabricLsaPolicy -type f 2>/dev/null | head -1); if test -z "$f"; then echo SKIP no LSA policy binary; exit 0; fi; "$f"'

run_logged ut-anvil-plugin docker run --rm --init "${DOCKER_IMAGE}" \
  bash -lc 'f=$(find / -name rccl-UnitTestsGinAnvilPlugin -type f 2>/dev/null | head -1); if test -z "$f"; then echo SKIP no plugin UT; exit 0; fi; "$f"'

# Harness Test#1 host ring, Test#2 proxy, Test#4 GDA (may skip), Test#5 Anvil-SDMA.
# Small-to-1M first so a backend failure is cheap.
RCCL_GIN_RUN_TESTS=1,2,4,5 TEST1_MODE=ring \
  run_logged harness-1m bash "${HARNESS}" 4 1M

# Test#5 128B-4GiB (LL then gin.put/SDMA). Keep iters modest.
RCCL_GIN_RUN_TESTS=5 TEST5_ITERS="${A2A_ITERS}" TEST5_WARMUP="${A2A_WARMUP}" \
  run_logged test5-4g bash "${HARNESS}" 4 4G

# MPI fabric-LL gtests if the image ships them.
run_logged ut-mpi-fabric-ll docker run --rm --init \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
  --shm-size 64G --network host --ipc host \
  --tmpfs /dev/infiniband \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES}" \
  -e NCCL_IB_DISABLE=1 -e NCCL_GIN_TYPE=6 -e NCCL_MNNVL_ENABLE=1 \
  -e NCCL_CUMEM_ENABLE=1 -e RCCL_ENABLE_INTRANET=1 -e NCCL_GIN_ENABLE=1 \
  "${DOCKER_IMAGE}" bash -lc \
  'f=$(find / -name rccl-UnitTestsMPI -type f 2>/dev/null | head -1); \
   if test -z "$f"; then echo SKIP no rccl-UnitTestsMPI; exit 0; fi; \
   mpirun --allow-run-as-root -n 4 "$f" --gtest_filter=GinMPIDeviceTests.Alltoall_FabricLL_*'

echo "===== done fail=${fail} =====" | tee -a "${LOGDIR}/summary.txt"
cat "${LOGDIR}/summary.txt"
exit "${fail}"
