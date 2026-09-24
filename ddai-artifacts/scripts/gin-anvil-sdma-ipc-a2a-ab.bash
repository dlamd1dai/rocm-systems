#! /usr/bin/env bash
# A/B alltoall_perf -D 3 with NCCL_GIN_TYPE=7 (Anvil SDMA) for PR 12137.
# Overlay binaries come from gin-anvil-sdma-ipc-rebuild-overlay.bash.
#
# Usage: OUT=/home/dondai/pr12137 ./gin-anvil-sdma-ipc-a2a-ab.bash skip|always
set -euo pipefail
VARIANT="${1:?skip|always}"
OUT="${OUT:-/home/dondai/pr12137}"
IMAGE="${DOCKER_IMAGE:-rccl-ginsdma101}"
LOG="$OUT/ab/output-d3all2all-${VARIANT}.txt"
: > "$LOG"
BNXT1=/usr/local/lib/libbnxt_re-rdmav34.so
BNXT2=/usr/local/lib/libbnxt_re.so
VOLS=(-v "$OUT/ab/$VARIANT:/pr12137" -v "$OUT/ab:/logs")
if [ -f "$BNXT1" ]; then VOLS+=(-v "$BNXT1:/usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so:ro"); fi
if [ -f "$BNXT2" ]; then VOLS+=(-v "$BNXT2:/usr/local/lib/libbnxt_re.so:ro"); fi

docker run --rm --name "pr12137-a2a-${VARIANT}" \
  --shm-size 64G --network host \
  --device /dev/dri --device /dev/kfd --device /dev/infiniband \
  --ipc host --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  "${VOLS[@]}" \
  "$IMAGE" \
  bash -lc "
    set -euo pipefail
    export PATH=/usr/bin:/bin:\${PATH:-}
    export OPAL_PREFIX=/usr
    export LD_LIBRARY_PATH=/pr12137:\${LD_LIBRARY_PATH:-}
    ldd /pr12137/alltoall_perf | grep rccl || true
    mpirun --allow-run-as-root -n 8 -mca pml ob1 -mca btl ^openib \
      -x LD_LIBRARY_PATH \
      -x OPAL_PREFIX \
      -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=7 \
      -x NCCL_GIN_ANVIL_SDMA_THRESHOLD=256 \
      -x NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL=0 \
      -x NCCL_DEBUG=WARN -x NCCL_DEBUG_SUBSYS=INIT,NET \
      -x NCCL_CUMEM_ENABLE=1 -x RCCL_ENABLE_INTRANET=0 -x NCCL_P2P_DISABLE=1 \
      -x NCCL_DMABUF_ENABLE=1 -x NCCL_MSCCL_ENABLE=0 \
      -x HSA_NO_SCRATCH_RECLAIM=1 \
      -x RCCL_CHEAP_POST_SEND_FENCE_OFF=1 \
      /pr12137/alltoall_perf -b 128 -e 1024M -f 2 -g 1 -R 2 -D 3 -A 1 -V 1
  " 2>&1 | tee -a "$LOG"
echo "RUN_${VARIANT}_DONE"
