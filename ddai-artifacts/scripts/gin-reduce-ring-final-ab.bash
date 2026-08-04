#!/usr/bin/env bash
# Final crossover A/B for the multi-ring Reduce (now default-ON for OOP >= 8 MiB,
# size-adaptive chunk depth). Warm (discard first, keep second). float sum. Configs:
#   host  : ncclReduce (-D 0) CLEAN env (~285 baseline).
#   fused : GIN -D 3, ring FORCED OFF (NCCL_GIN_ANVIL_REDUCE_RING_MIN_BYTES=0), -V 32.
#   ring  : GIN -D 3, shipped defaults (ring on >=8 MiB, adaptive chunks, 128 CTAs).
# Prints OOP (field 8) and in-place (field 12) busbw per size, over 32M..2G to find
# the ring>fused / ring>host crossover and validate the 8 MiB default threshold.
# Run INSIDE rccl-gin-gda-sdma-713 (-v ~/rt-build:/rt-build).
set -uo pipefail
BIN_DIR="${1:-/rt-build}"; NP="${2:-8}"
SIZES_LO="${3:-32M}"; SIZES_HI="${4:-2G}"
MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
COMMON=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        -x NCCL_DEBUG=WARN -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_CUMEM_ENABLE=1)
GINENV=(-x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1
        -x RCCL_ROCSHMEM_THRESHOLD=134217728 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
        -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_PLUGIN=none -x NCCL_NET_PLUGIN=none
        -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
        -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1)
ARGS=(-b "$SIZES_LO" -e "$SIZES_HI" -f 2 -g 1 -R 2 -c 0 -n 20 -w 5 -z 0 -o sum -d float -r 0)
RD="$BIN_DIR/reduce_perf"
tbl() { awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ {printf "  %12s  OOP=%8s  INPLACE=%8s\n",$1,$8,$12}'; }

echo "===== host  -D 0  (clean env) ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$RD" "${ARGS[@]}" -D 0 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$RD" "${ARGS[@]}" -D 0 2>&1 | tbl

echo "===== GIN fused (ring FORCED OFF) -V 32 ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" -x NCCL_GIN_ANVIL_REDUCE_RING_MIN_BYTES=0 "$RD" "${ARGS[@]}" -V 32 -D 3 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" -x NCCL_GIN_ANVIL_REDUCE_RING_MIN_BYTES=0 "$RD" "${ARGS[@]}" -V 32 -D 3 2>&1 | tbl

echo "===== GIN ring (shipped defaults: on>=8M, adaptive chunks, 128 CTAs) ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$RD" "${ARGS[@]}" -D 3 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$RD" "${ARGS[@]}" -D 3 2>&1 | tbl
echo "AB_DONE"
