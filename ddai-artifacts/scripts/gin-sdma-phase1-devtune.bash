#!/usr/bin/env bash
# Phase-1 device (-D 3, GIN Anvil-SDMA) knob sweep for sendrecv/scatter/gather.
# Sweeps the two untuned device levers to pick performant defaults:
#   NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS  (SDMA queues; only bites in the GIN tier)
#   -V CTA count                      (kernel grid; bites in both LSA and GIN tiers)
# Runs at the tuned default thresholds (scatter uses GIN tier at these sizes;
# sendrecv/gather are LSA-always, so NUM_CHANNELS should be a no-op for them).
# Usage: gin-sdma-phase1-devtune.bash <bin_dir> <np> <out_dir> [min] [max]
set -euo pipefail
BIN_DIR="${1:?}"; NP="${2:-8}"; OUT_DIR="${3:?}"; MIN_BYTES="${4:-1M}"; MAX_BYTES="${5:-128M}"
ITERS="${ITERS:-15}"; WARMUP="${WARMUP:-3}"
CHAN_LIST="${CHAN_LIST:-1 2 4}"; CTA_LIST="${CTA_LIST:-16 32 64}"
mkdir -p "$OUT_DIR"
MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
BASE=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 -x RCCL_ROCSHMEM_ENABLE=0
      -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1 -x RCCL_ROCSHMEM_THRESHOLD=$((128*1024*1024))
      -x NCCL_DEBUG=WARN -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1 -x NCCL_MSCCL_ENABLE=0
      -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_GIN_PLUGIN=none -x NCCL_CUMEM_ENABLE=1 -x NCCL_NET_PLUGIN=none
      -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6 -x HSA_FORCE_FINE_GRAIN_PCIE=1)

run() {  # $1=coll $2=chan $3=cta
  local coll="$1" chan="$2" cta="$3"
  local exe="${BIN_DIR}/${coll}_perf" out="${OUT_DIR}/${coll}.c${chan}.v${cta}.txt"
  local -a rootarg=(); [[ "$coll" != sendrecv ]] && rootarg=(-r 0)
  echo "=== ${coll} chan=${chan} cta=${cta} ==="
  mpirun -n "$NP" "${MPI_OPT[@]}" "${BASE[@]}" -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS="$chan" \
    "$exe" -b "$MIN_BYTES" -e "$MAX_BYTES" -f 2 -g 1 -R 2 -V "$cta" -D 3 -n "$ITERS" -w "$WARMUP" "${rootarg[@]}" > "$out" 2>&1
}
for coll in sendrecv scatter gather; do
  for chan in $CHAN_LIST; do for cta in $CTA_LIST; do run "$coll" "$chan" "$cta"; done; done
done
echo DEVTUNE_DONE
