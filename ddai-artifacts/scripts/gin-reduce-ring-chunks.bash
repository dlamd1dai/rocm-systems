#!/usr/bin/env bash
# Ring-reduce CHUNK-depth sweep (env-only, existing binary). The ring is an
# (N-1)-hop pipeline; with C chunks its fill/drain efficiency is C/(C+N-1), so the
# default C=4 on a 7-hop pipeline (~36%) starves it. Sweep C to find the ring's
# true ceiling vs the flat fused kernel (214) and host (~285). float sum, OOP.
# Run INSIDE rccl-gin-gda-sdma-713 (-v ~/rt-build:/rt-build).
set -uo pipefail
BIN_DIR="${1:-/rt-build}"; NP="${2:-8}"
SIZES_LO="${3:-512M}"; SIZES_HI="${4:-2G}"
CHUNKS_LIST="${CHUNKS:-4 8 16 32 64}"
RCTAS="${RCTAS:-128}"

MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
COMMON=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        -x NCCL_DEBUG=WARN -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_CUMEM_ENABLE=1)
GINENV=(-x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1
        -x RCCL_ROCSHMEM_THRESHOLD=134217728 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
        -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_PLUGIN=none -x NCCL_NET_PLUGIN=none
        -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
        -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1)
ARGS=(-b "$SIZES_LO" -e "$SIZES_HI" -f 4 -g 1 -R 2 -c 0 -n 20 -w 5 -z 0 -o sum -d float -r 0)
RD="$BIN_DIR/reduce_perf"
tbl() { awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ {printf "  %12s  OOP=%8s\n",$1,$8}'; }

echo "===== host  -D 0  (clean env) ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$RD" "${ARGS[@]}" -D 0 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$RD" "${ARGS[@]}" -D 0 2>&1 | tbl

echo "===== GIN fused (ring OFF) -V 32 ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$RD" "${ARGS[@]}" -V 32 -D 3 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$RD" "${ARGS[@]}" -V 32 -D 3 2>&1 | tbl

for CK in $CHUNKS_LIST; do
  echo "===== GIN ring  CTAS=$RCTAS  CHUNKS=$CK ====="
  E=(-x NCCL_GIN_ANVIL_REDUCE_RING_MIN_BYTES=1048576 -x NCCL_GIN_ANVIL_REDUCE_RING_CTAS=$RCTAS -x NCCL_GIN_ANVIL_REDUCE_RING_CHUNKS=$CK)
  mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "${E[@]}" "$RD" "${ARGS[@]}" -D 3 >/dev/null 2>&1 || true
  mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "${E[@]}" "$RD" "${ARGS[@]}" -D 3 2>&1 | tbl
done
echo "SWEEP_DONE"
