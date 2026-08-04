#!/usr/bin/env bash
# Warm A/B for the pipelined large-tier Reduce (-D 3, OOP). Warm (discard the
# first run, keep the second). Three configs:
#   host    : ncclReduce (-D 0) reference -- CLEAN env (COMMON only; NO GIN/MSCCL
#             overrides). Applying the GIN env (NCCL_GIN_ENABLE / NCCL_MSCCL_ENABLE=0
#             / *_PLUGIN=none) to the host path cripples it (~52 vs ~284 GB/s), so
#             the host baseline MUST run with the clean env -- mirrors
#             gin-reduce-cta-sweep.bash.
#   pipe ON : GIN -D 3 with the pipelined SM-reduce || SDMA-put path.
#   pipe OFF: GIN -D 3 with NCCL_GIN_ANVIL_REDUCE_PIPE_MIN_BYTES=0 (fused read-reduce).
# Prints per size the OUT-OF-PLACE busbw (field 8) and in-place busbw (field 12).
# Run INSIDE the rccl-gin-gda-sdma-713 container (-v ~/rt-build:/rt-build).
set -uo pipefail
BIN_DIR="${1:-/rt-build}"; NP="${2:-8}"
SIZES_LO="${3:-128M}"; SIZES_HI="${4:-2G}"

MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
COMMON=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        -x NCCL_DEBUG=WARN -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_CUMEM_ENABLE=1)
GINENV=(-x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1
        -x RCCL_ROCSHMEM_THRESHOLD=134217728 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
        -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_PLUGIN=none -x NCCL_NET_PLUGIN=none
        -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
        -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1)

# size  OOP-busbw(f8)  inplace-busbw(f12)
tbl() { awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ {printf "  %12s  OOP=%8s  INPLACE=%8s\n", $1, $8, $12}'; }

RD="$BIN_DIR/reduce_perf"
ARGS=(-b "$SIZES_LO" -e "$SIZES_HI" -f 2 -g 1 -R 2 -V 32 -c 0 -n 20 -w 5 -z 0 -o sum -d float -r 0)

# HOST: clean env (COMMON only), warm.
echo "===== host  -D 0  (clean env) ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$RD" "${ARGS[@]}" -D 0 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$RD" "${ARGS[@]}" -D 0 2>&1 | tbl

# GIN pipe ON: GIN env, warm.
echo "===== GIN   -D 3  pipe ON ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$RD" "${ARGS[@]}" -D 3 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$RD" "${ARGS[@]}" -D 3 2>&1 | tbl

# GIN pipe OFF (fused): GIN env + disable pipeline, warm.
echo "===== GIN   -D 3  pipe OFF(fused) ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" -x NCCL_GIN_ANVIL_REDUCE_PIPE_MIN_BYTES=0 "$RD" "${ARGS[@]}" -D 3 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" -x NCCL_GIN_ANVIL_REDUCE_PIPE_MIN_BYTES=0 "$RD" "${ARGS[@]}" -D 3 2>&1 | tbl
echo "AB_DONE"
