#!/bin/bash
# Does the plugin assertion in test_Broadcast.py actually discriminate? Capture
# NCCL debug the same way the test does (per-rank files, INIT,NET) for a GIN
# Anvil-SDMA run and for a non-GIN host-path run, then count the marker in each.
# The assertion is only meaningful if it is present in the first and absent in
# the second.
set -e
cd /workspace/rccl-tests
SRC=$(grep -m1 CMAKE_HOME_DIRECTORY CMakeCache.txt | cut -d= -f2)
cp /fix/broadcast.cu "$SRC/src/broadcast.cu"
cp /fix/gin_sdma_broadcast_policy.h "$SRC/src/gin_sdma_broadcast_policy.h"
cmake --build . --target broadcast_perf -j 32 > /tmp/b.log 2>&1 || { tail -20 /tmp/b.log; exit 1; }

MPI="mpirun --allow-run-as-root -np 8 -mca pml ob1 -mca btl ^openib"
BASE="-x NCCL_CUMEM_ENABLE=1 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1 -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_MSCCL_ENABLE=0"

probe() {  # $1 = label, $2 = deviceImpl, rest = extra -x env
  local label="$1"; local dimpl="$2"; shift 2
  local dir; dir=$(mktemp -d /tmp/dbg-XXXX)
  local rc=0
  $MPI $BASE "$@" \
    -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,NET \
    -x NCCL_DEBUG_FILE="$dir/nccl-debug.%h.%p.log" \
    ./broadcast_perf -b 64M -e 64M -f 2 -g 1 -R 2 -D "$dimpl" -A 1 -c 1 -w 1 -n 3 \
    > "$dir/stdout.txt" 2>&1 || rc=$?
  local hits; hits=$(cat "$dir"/nccl-debug.* 2>/dev/null | grep -c "gin-anvil-sdma" || true)
  local rows; rows=$(grep -cE "^ +[0-9]+ +[0-9]+ +(float|int32)" "$dir/stdout.txt" || true)
  echo "$label: exit=$rc  debug_files=$(ls "$dir"/nccl-debug.* 2>/dev/null | wc -l)  gin-anvil-sdma_hits=$hits  result_rows=$rows"
  rm -rf "$dir"
}

probe "GIN Anvil-SDMA (-D 3, TYPE=5)" 3 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=5 \
      -x NCCL_GIN_ANVIL_SDMA_THRESHOLD=0 -x NCCL_GIN_ANVIL_SDMA_THRESHOLD_BROADCAST=0
probe "host path    (-D 0, no GIN)" 0 -x NCCL_GIN_ENABLE=0
