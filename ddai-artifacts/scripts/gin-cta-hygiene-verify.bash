#!/usr/bin/env bash
# Verify the -V CTA-default hygiene fix (global default 16 + broadcast ring
# decoupled to self-select 128):
#   * broadcast (-D 3) must hit ~350 GB/s BOTH at the bare default (no -V) AND
#     under -V 32 (before decoupling the ring was clamped to -V => 238 @ -V 32);
#   * reduce (-D 3) bare default (now 16 CTAs) should match/beat its -V 32 number
#     (more CTAs hurt reduce), and not regress.
# Run INSIDE the rccl-gin-gda-sdma-713 container (-v ~/rt-build:/rt-build).
set -uo pipefail
BIN_DIR="${1:-/rt-build}"; NP="${2:-8}"

MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
COMMON=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        -x NCCL_DEBUG=WARN -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_CUMEM_ENABLE=1)
GINENV=(-x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1
        -x RCCL_ROCSHMEM_THRESHOLD=134217728 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
        -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_PLUGIN=none -x NCCL_NET_PLUGIN=none
        -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
        -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1)

busbw() { awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ { for(i=4;i<=NF;i++) if($i=="N/A"){print $1,$(i-1);break} }'; }
# warm: discard first, keep second
run() {  # $1 label ; rest = mpirun args after exe
  local label="$1"; shift
  local exe="$1"; shift
  mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$exe" "$@" >/dev/null 2>&1 || true
  echo "===== $label ====="
  mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" "$exe" "$@" 2>&1 | busbw
}

BC="$BIN_DIR/broadcast_perf"; RD="$BIN_DIR/reduce_perf"
run "broadcast -D 3  DEFAULT (no -V)"  "$BC" -b 512M -e 2G -f 2 -g 1 -R 2 -D 3 -c 0 -n 20 -w 5 -z 0 -d float -r 0
run "broadcast -D 3  -V 32"            "$BC" -b 512M -e 2G -f 2 -g 1 -R 2 -V 32 -D 3 -c 0 -n 20 -w 5 -z 0 -d float -r 0
run "reduce    -D 3  DEFAULT (no -V)"  "$RD" -b 128M -e 2G -f 2 -g 1 -R 2 -D 3 -c 0 -n 20 -w 5 -z 0 -o sum -d float -r 0
run "reduce    -D 3  -V 32"            "$RD" -b 128M -e 2G -f 2 -g 1 -R 2 -V 32 -D 3 -c 0 -n 20 -w 5 -z 0 -o sum -d float -r 0
echo "VERIFY_DONE"
