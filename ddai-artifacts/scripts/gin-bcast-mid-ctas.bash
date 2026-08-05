#!/usr/bin/env bash
# Broadcast MID-BAND (SAG regime, 2 MiB-64 MiB) CTA-ladder diagnostic. Warm
# (discard first mpirun, keep second). float, -c 0, OOP. The SAG tier launches on
# -V/deviceCtaCount (broadcast.cu line ~1124, plain testLaunchDeviceKernel), so if
# the mid-band gap vs host is occupancy/CTA-bound (the RS story), raising -V should
# lift it; if it is root-egress-bound, the ladder should be flat. Ring forced OFF
# (BCAST_RING_MIN_BYTES=0) so every size here is pure SAG.
#   host    : ncclBroadcast -D 0 reference (clean env).
#   V16..V64: GIN -D 3 SAG at each deviceCtaCount.
# Prints OOP busbw (col before the N/A out-of-place algbw sentinel) per size.
# Run INSIDE rccl-gin-gda-sdma-713-rs (-v ~/rt-build:/rt-build).
set -uo pipefail
BIN_DIR="${1:-/rt-build}"; NP="${2:-8}"
SIZES_LO="${3:-2M}"; SIZES_HI="${4:-64M}"
EXE="$BIN_DIR/broadcast_perf"
MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
COMMON=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        -x NCCL_DEBUG=WARN -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_CUMEM_ENABLE=1)
GINENV=(-x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1
        -x RCCL_ROCSHMEM_THRESHOLD=134217728 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
        -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_PLUGIN=none -x NCCL_NET_PLUGIN=none
        -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
        -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1
        -x NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=0)
ARGS=(-b "$SIZES_LO" -e "$SIZES_HI" -f 2 -g 1 -R 2 -c 0 -n 20 -w 5 -z 0 -d float -r 0)
tbl() { awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ { for(i=4;i<=NF;i++) if($i=="N/A"){printf "  %12s  OOP=%8s\n",$1,$(i-1); break} }'; }

echo "===== host  -D 0  (reference) ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$EXE" "${ARGS[@]}" -D 0 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$EXE" "${ARGS[@]}" -D 0 2>&1 | tbl

for V in 16 32 48 64; do
  echo "===== GIN SAG  -V $V ====="
  mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$EXE" "${ARGS[@]}" -V "$V" -D 3 >/dev/null 2>&1 || true
  mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$EXE" "${ARGS[@]}" -V "$V" -D 3 2>&1 | tbl
done
echo "MID_CTAS_DONE"
