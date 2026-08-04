#!/usr/bin/env bash
# Broadcast 7-ring CHUNK-depth diagnostic (env-only, existing binary). The ring's
# chunk count is clamp(msgBytes/2MiB, 2, kBroadcastRingMaxChunks=64) -- a HARD 64
# cap, so the doc's "chunks 64/128/256 flat" sweep all clamped to 64 and never
# tested a deeper pipeline. This sweeps BCAST_RING_CHUNKS within the allowed range
# (8..64): if busbw still RISES toward 64, the cap is the limiter and raising it
# (reduce-style per-CTA ~64 KiB sizing) should unlock the ring past the 232 plateau.
# NP=8, float, table kernel defaults (SM+MULTI+STREAM on, P2P off, 128 CTAs), OOP.
# Run INSIDE rccl-gin-gda-sdma-713 (-v ~/rt-build:/rt-build).
set -uo pipefail
BIN_DIR="${1:-/rt-build}"; NP="${2:-8}"
SIZES_LO="${3:-512M}"; SIZES_HI="${4:-2G}"
CHUNKS_LIST="${CHUNKS:-8 16 32 64}"
RING_MIN="${RING_MIN:-1048576}"
EXE="$BIN_DIR/broadcast_perf"

MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
COMMON=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        -x NCCL_DEBUG=WARN -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_CUMEM_ENABLE=1)
GINENV=(-x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1
        -x RCCL_ROCSHMEM_THRESHOLD=134217728 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
        -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_PLUGIN=none -x NCCL_NET_PLUGIN=none
        -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
        -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1)
ARGS=(-b "$SIZES_LO" -e "$SIZES_HI" -f 4 -g 1 -R 2 -c 0 -n 20 -w 5 -z 0 -d float -r 0)
# OOP busbw = the numeric column immediately before the first "N/A" (#wrong w/ -c 0).
tbl() { awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ { for(i=4;i<=NF;i++) if($i=="N/A"){printf "  %12s  OOP=%8s\n",$1,$(i-1); break} }'; }

echo "===== host  -D 0  (clean env, default tuner) ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$EXE" "${ARGS[@]}" -D 0 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$EXE" "${ARGS[@]}" -D 0 2>&1 | tbl

echo "===== GIN SAG (ring OFF) -V 32 ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" -x NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=0 "$EXE" "${ARGS[@]}" -V 32 -D 3 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" -x NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=0 "$EXE" "${ARGS[@]}" -V 32 -D 3 2>&1 | tbl

for CK in $CHUNKS_LIST; do
  echo "===== GIN 7-ring  CHUNKS=$CK (clamped to <=64)  128 CTAs ====="
  E=(-x NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=$RING_MIN -x NCCL_GIN_ANVIL_BCAST_RING_CHUNKS=$CK)
  mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "${E[@]}" "$EXE" "${ARGS[@]}" -D 3 >/dev/null 2>&1 || true
  mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "${E[@]}" "$EXE" "${ARGS[@]}" -D 3 2>&1 | tbl
done
echo "SWEEP_DONE"
