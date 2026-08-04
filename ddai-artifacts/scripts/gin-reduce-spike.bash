#!/usr/bin/env bash
# Cheap env-only diagnostic spike for the Reduce large tier (no kernel changes).
# Goal: bound the achievable headroom before deciding on a ring/tree rewrite.
#
#   Ceiling note: phase-isolation (plan 9.5) measured RS-to-local 4.88 ms (440 GB/s)
#   and SDMA gather-to-root 4.29 ms (500 GB/s) at 2 GiB. Perfect overlap ->
#   max(4.88,4.29) ~= 4.88 ms -> ~440 GB/s, i.e. ABOVE host (285). The pipeline only
#   reached 202 because SDMA is pinned to 1 channel with tiny per-CTA puts. So:
#
#   PART A (device): enable the pipeline via env and sweep SDMA channels x pipe
#     chunks to see if multi-channel SDMA moves the pipeline toward the 440 ceiling
#     (or deadlocks). Each run is timeout-guarded so a hang does not block the spike.
#   PART B (host): is host's 285 from the TREE or the 64 WarpSpeed channels? Force
#     NCCL_ALGO and cap channels to find host's floor.
#
# Run INSIDE the rccl-gin-gda-sdma-713 container (-v ~/rt-build:/rt-build).
set -uo pipefail
BIN_DIR="${1:-/rt-build}"; NP="${2:-8}"
RD="$BIN_DIR/reduce_perf"
TMO="${TMO:-150}"   # per-mpirun timeout (s); a deadlocked GIN run is killed

MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
COMMON=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        -x NCCL_DEBUG=WARN -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_CUMEM_ENABLE=1)
GINENV=(-x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1
        -x RCCL_ROCSHMEM_THRESHOLD=134217728 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
        -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_PLUGIN=none -x NCCL_NET_PLUGIN=none
        -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
        -x HSA_FORCE_FINE_GRAIN_PCIE=1)
ARGS=(-b 512M -e 2G -f 4 -g 1 -R 2 -c 0 -n 15 -w 5 -z 0 -o sum -d float -r 0)
tbl() { awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ {printf "  %12s  OOP=%8s  INPLACE=%8s\n",$1,$8,$12}'; }

warm() {  # $1 label; rest = full mpirun argv
  local label="$1"; shift
  timeout "$TMO" "$@" >/dev/null 2>&1
  echo "===== $label ====="
  if ! timeout "$TMO" "$@" 2>&1 | tbl; then echo "  (TIMEOUT/xxx -- likely deadlock)"; fi
}

echo "########## PART A: device pipeline -- SDMA channels x pipe chunks (-V 16) ##########"
for CH in 1 2 4; do
  for CK in 2 4 8; do
    warm "pipe ON  SDMA_CH=$CH  CHUNKS=$CK" \
      mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" \
        -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=$CH \
        -x NCCL_GIN_ANVIL_REDUCE_PIPE_MIN_BYTES=1048576 \
        -x NCCL_GIN_ANVIL_REDUCE_PIPE_CHUNKS=$CK \
        "$RD" "${ARGS[@]}" -V 16 -D 3
  done
done
echo "########## reference: fused (pipe OFF) SDMA_CH=1 -V 32 ##########"
warm "fused pipe OFF" \
  mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" \
    -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x NCCL_GIN_ANVIL_REDUCE_PIPE_MIN_BYTES=0 \
    "$RD" "${ARGS[@]}" -V 32 -D 3

echo "########## PART B: host algo/channel sensitivity (clean env) ##########"
warm "host default"        mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}"                         "$RD" "${ARGS[@]}" -D 0
warm "host ALGO=Tree"      mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" -x NCCL_ALGO=Tree        "$RD" "${ARGS[@]}" -D 0
warm "host ALGO=Ring"      mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" -x NCCL_ALGO=Ring        "$RD" "${ARGS[@]}" -D 0
warm "host MAX_NCH=8"      mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" -x NCCL_MAX_NCHANNELS=8  "$RD" "${ARGS[@]}" -D 0
warm "host MAX_NCH=16"     mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" -x NCCL_MAX_NCHANNELS=16 "$RD" "${ARGS[@]}" -D 0
echo "SPIKE_DONE"
