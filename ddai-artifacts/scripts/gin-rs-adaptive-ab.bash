#!/usr/bin/env bash
# ReduceScatter shipped-default verification A/B (warm: discard first mpirun, keep
# second). float sum, OOP + in-place, -c 0. The -D 3 RS kernel now self-selects a
# size-adaptive CTA count (gin_sdma::reduceScatterCtas: ~48 in the grid-stride
# mid-band [8,48) MiB, 32 elsewhere) DECOUPLED from -V, mirroring the broadcast/
# reduce rings. Configs:
#   host        : ncclReduceScatter (-D 0) reference.
#   gin(default): GIN -D 3, NO -V  -> exercises the shipped self-select (the bare
#                 default used to under-launch the mid-band at deviceCtaCount=16).
#   gin(-V 32)  : GIN -D 3, -V 32  -> must MATCH gin(default): -V no longer changes
#                 the launched CTA count (only raises the barrier alloc floor).
# Prints OOP (field 8) + in-place (field 12) busbw per size.
# Run INSIDE rccl-gin-gda-sdma-713-rs (-v ~/rt-build:/rt-build).
set -uo pipefail
BIN_DIR="${1:-/rt-build}"; NP="${2:-8}"
SIZES_LO="${3:-1M}"; SIZES_HI="${4:-256M}"
MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
COMMON=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        -x NCCL_DEBUG=WARN -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_CUMEM_ENABLE=1)
GINENV=(-x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1
        -x RCCL_ROCSHMEM_THRESHOLD=134217728 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
        -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_PLUGIN=none -x NCCL_NET_PLUGIN=none
        -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
        -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1)
ARGS=(-b "$SIZES_LO" -e "$SIZES_HI" -f 2 -g 1 -R 2 -c 0 -n 20 -w 5 -z 0 -o sum -d float)
RS="$BIN_DIR/reduce_scatter_perf"
tbl() { awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ {printf "  %12s  OOP=%8s  INPLACE=%8s\n",$1,$8,$12}'; }

echo "===== host  -D 0  (reference) ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$RS" "${ARGS[@]}" -D 0 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$RS" "${ARGS[@]}" -D 0 2>&1 | tbl

echo "===== GIN -D 3  shipped default (NO -V, self-select adaptive CTAs) ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$RS" "${ARGS[@]}" -D 3 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$RS" "${ARGS[@]}" -D 3 2>&1 | tbl

echo "===== GIN -D 3  -V 32 (must match default: -V decoupled) ====="
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$RS" "${ARGS[@]}" -V 32 -D 3 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$RS" "${ARGS[@]}" -V 32 -D 3 2>&1 | tbl
echo "AB_DONE"
