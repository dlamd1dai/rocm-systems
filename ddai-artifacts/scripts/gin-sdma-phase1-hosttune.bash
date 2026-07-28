#!/usr/bin/env bash
# Phase-1 host-baseline (-D 0) tuning sweep for sendrecv/scatter/gather.
# Compares single-node host-initiated variants to pick the performant C1 path:
#   default : tuner, NCCL_CUMEM_ENABLE=0, -R 0
#   ce      : copy-engine/SDMA, NCCL_CUMEM_ENABLE=1, -R 2, NCCL_CTA_POLICY=ZERO
#   chan    : default + pinned channels (NCCL_MIN/MAX_NCHANNELS=NCH)
# Usage: gin-sdma-phase1-hosttune.bash <bin_dir> <np> <out_dir> [min] [max]
set -euo pipefail
BIN_DIR="${1:?}"; NP="${2:-8}"; OUT_DIR="${3:?}"; MIN_BYTES="${4:-8}"; MAX_BYTES="${5:-128M}"
NCH="${NCH:-32}"; ITERS="${ITERS:-20}"; WARMUP="${WARMUP:-5}"
mkdir -p "$OUT_DIR"
MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
BASE=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 -x RCCL_ROCSHMEM_ENABLE=0
      -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1 -x RCCL_ROCSHMEM_THRESHOLD=$((128*1024*1024))
      -x NCCL_DEBUG=WARN -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1 -x NCCL_MSCCL_ENABLE=0
      -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_GIN_PLUGIN=none -x NCCL_GIN_ENABLE=0 -x NCCL_GIN_TYPE=0 -x ROCSHMEM_SDMA_ENABLED=0)

run() {  # $1=coll $2=variant $3..=extra perf/env args passed via arrays below
  local coll="$1" var="$2"; shift 2
  local exe="${BIN_DIR}/${coll}_perf" out="${OUT_DIR}/${coll}.${var}.txt"
  local -a env=() perf=()
  case "$var" in
    default) env=(-x NCCL_CUMEM_ENABLE=0); perf=(-R 0) ;;
    ce)      env=(-x NCCL_CUMEM_ENABLE=1 -x NCCL_CTA_POLICY=ZERO); perf=(-R 2) ;;
    chan)    env=(-x NCCL_CUMEM_ENABLE=0 -x NCCL_MIN_NCHANNELS="${NCH}" -x NCCL_MAX_NCHANNELS="${NCH}"); perf=(-R 0) ;;
  esac
  local -a rootarg=(); [[ "$coll" != sendrecv ]] && rootarg=(-r 0)
  echo "=== ${coll} ${var} ==="
  mpirun -n "$NP" "${MPI_OPT[@]}" "${BASE[@]}" "${env[@]}" \
    "$exe" -b "$MIN_BYTES" -e "$MAX_BYTES" -f 2 -g 1 -D 0 -n "$ITERS" -w "$WARMUP" "${rootarg[@]}" "${perf[@]}" > "$out" 2>&1
  echo "  -> $out"
}
for coll in sendrecv scatter gather; do
  for var in default ce chan; do run "$coll" "$var"; done
done
echo HOSTTUNE_DONE
