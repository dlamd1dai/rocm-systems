#!/usr/bin/env bash
# ReduceScatter host (-D 0) vs GIN-SDMA (-D 3) time sweep, 128 B .. 2 GB, warm.
# GIN is run with NCCL_GIN_ANVIL_DEVICE_TIMING=1 (augment): the normal graph-timed
# loop still runs AND an extra "#[rs-devtime]" line reports the IN-KERNEL
# wall_clock64 latency (launch/graph overhead stripped), using the same adaptive
# CTA count the perf path self-selects. Host has no device-timing hook (library
# path), so it is graph-timed only. float/sum, OOP.
# Emits three tagged streams the caller can merge by size:
#   HOSTG <bytes> <time_us> <algbw>          (host graph time)
#   GING  <bytes> <time_us> <algbw>          (GIN graph time)
#   GIND  <bytes> <devtime_us> <algbw>       (GIN in-kernel device time)
# Run INSIDE rccl-gin-gda-sdma-713-rs (-v ~/rt-build:/rt-build).
set -uo pipefail
BIN_DIR="${1:-/rt-build}"; NP="${2:-8}"
B="${3:-128}"; E="${4:-2G}"; F="${5:-2}"; N="${6:-20}"; W="${7:-5}"
MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
COMMON=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        -x NCCL_DEBUG=WARN -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_CUMEM_ENABLE=1)
GINENV=(-x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1
        -x RCCL_ROCSHMEM_THRESHOLD=134217728 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
        -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_PLUGIN=none -x NCCL_NET_PLUGIN=none
        -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
        -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1)
ARGS=(-b "$B" -e "$E" -f "$F" -g 1 -R 2 -c 0 -n "$N" -w "$W" -z 0 -o sum -d float)
RS="$BIN_DIR/reduce_scatter_perf"

echo "########## HOST -D 0 (graph time) ##########"
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$RS" "${ARGS[@]}" -D 0 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "$RS" "${ARGS[@]}" -D 0 2>&1 \
  | awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+float/ {printf "HOSTG %s %s %s\n",$1,$6,$7}'

echo "########## GIN -D 3 (graph time + in-kernel device time) ##########"
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" -x NCCL_GIN_ANVIL_DEVICE_TIMING=1 "$RS" "${ARGS[@]}" -D 3 >/dev/null 2>&1 || true
mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" -x NCCL_GIN_ANVIL_DEVICE_TIMING=1 "$RS" "${ARGS[@]}" -D 3 2>&1 \
  | awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+float/ {printf "GING %s %s %s\n",$1,$6,$7}
         /#\[rs-devtime\]/ {for(i=1;i<=NF;i++){if($i=="size")b=$(i+1); if($i=="devtime")d=$(i+1); if($i=="algbw")a=$(i+1)} printf "GIND %s %s %s\n",b,d,a}'
echo "SWEEP_DONE"
