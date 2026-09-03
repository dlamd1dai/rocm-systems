#!/bin/bash
# Regression check for the LSA ring gate: on a single node lsaSize == nRanks, so
# the ring must still be selected and nothing should change. The check is by
# bandwidth, because a silent demotion to SAG is exactly the failure mode a gate
# like this can introduce: SAG plateaus (~229 GB/s at >=256 MiB) while the ring
# keeps climbing (~322-350 GB/s at 2 GiB), so the two tiers are separable.
#
# Run A: default env  -> gate open, ring expected.
# Run B: ring disabled -> SAG, the tier the gate would wrongly fall back to.
# A must be clearly faster than B, and both must be correct.
set -e
cd /workspace/rccl-tests
SRC=$(grep -m1 CMAKE_HOME_DIRECTORY CMakeCache.txt | cut -d= -f2)
cp /fix/broadcast.cu "$SRC/src/broadcast.cu"
cp /fix/gin_sdma_broadcast_policy.h "$SRC/src/gin_sdma_broadcast_policy.h"
cmake --build . --target broadcast_perf -j 32 > /tmp/b.log 2>&1 || { tail -30 /tmp/b.log; exit 1; }
echo "BUILD=OK (lsaSize gate compiled)"

MPI="mpirun --allow-run-as-root -np 8 -mca pml ob1 -mca btl ^openib"
E="-x NCCL_CUMEM_ENABLE=1 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1"
E="$E -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=5"
E="$E -x NCCL_GIN_ANVIL_SDMA_THRESHOLD=0 -x NCCL_GIN_ANVIL_SDMA_THRESHOLD_BROADCAST=0"

run() {  # $1 label, rest extra env
  local label="$1"; shift
  local out
  out=$($MPI $E "$@" ./broadcast_perf -b 2G -e 2G -f 2 -g 1 -R 2 -D 3 -A 1 -c 1 -w 1 -n 5 2>&1)
  local row wrong bw
  row=$(echo "$out" | grep -E "^ +[0-9]+ +[0-9]+ +" | tail -1)
  bw=$(echo "$row" | awk '{print $8}')
  wrong=$(echo "$row" | awk '{print $9}')
  echo "$label: busbw=${bw:-NONE} GB/s  #wrong=${wrong:-NONE}  oob=$(echo "$out" | grep -o 'Out of bounds values : [0-9]*' | awk '{print $NF}')"
}

echo "--- lsaSize reported by the runtime (expect 8 on this single node) ---"
$MPI $E -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT ./broadcast_perf -b 8 -e 8 -f 2 -g 1 -R 2 -D 3 -A 1 2>&1 \
  | grep -o "lsaSize=[0-9]*" | sort | uniq -c | head -3

run "A ring enabled (default, gate open)"
run "B ring disabled (forced SAG)" -x NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=0
