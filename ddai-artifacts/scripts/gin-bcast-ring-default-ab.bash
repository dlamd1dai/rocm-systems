#!/usr/bin/env bash
# Broadcast shipped-DEFAULT crossover A/B after the adaptive-chunk change (ring
# default-on >=2 MiB, per-CTA ~64 KiB chunks, cap 256). Warm. float. Configs:
#   host : ncclBroadcast -D 0 CLEAN env, default tuner (~322 @2G baseline).
#   SAG  : GIN -D 3, ring FORCED OFF (BCAST_RING_MIN_BYTES=0), -V 32.
#   ring : GIN -D 3, shipped defaults (adaptive chunks). No CHUNKS env.
# Verifies the adaptive default beats SAG/host across the whole >=2 MiB tier and
# does NOT regress the mid-range where fixed deep chunks tanked (128M x256 = 69).
# Run INSIDE rccl-gin-gda-sdma-713 (-v ~/rt-build:/rt-build).
set -uo pipefail
BIN_DIR="${1:-/rt-build}"; NP="${2:-8}"
SIZES_LO="${3:-2M}"; SIZES_HI="${4:-2G}"
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
ARGS=(-b "$SIZES_LO" -e "$SIZES_HI" -f 2 -g 1 -R 2 -c 0 -n 20 -w 5 -z 0 -d float -r 0)
tbl() { awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ { for(i=4;i<=NF;i++) if($i=="N/A"){printf "  %12s  OOP=%8s\n",$1,$(i-1); break} }'; }

echo "===== host  -D 0  (clean env, default tuner) ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$EXE" "${ARGS[@]}" -D 0 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$EXE" "${ARGS[@]}" -D 0 2>&1 | tbl

echo "===== GIN SAG (ring FORCED OFF) -V 32 ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" -x NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=0 "$EXE" "${ARGS[@]}" -V 32 -D 3 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" -x NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=0 "$EXE" "${ARGS[@]}" -V 32 -D 3 2>&1 | tbl

echo "===== GIN ring (shipped defaults: on>=2M, adaptive chunks, 128 CTAs) ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$EXE" "${ARGS[@]}" -D 3 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$EXE" "${ARGS[@]}" -D 3 2>&1 | tbl
echo "AB_DONE"
