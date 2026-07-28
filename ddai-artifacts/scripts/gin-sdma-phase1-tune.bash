#!/usr/bin/env bash
# Phase-1 (scatter/gather/sendrecv) LSA<->GIN threshold tuning sweep.
#
# For each collective, runs the SAME size sweep twice under -D 3:
#   * LSA-forced : NCCL_GIN_ANVIL_SDMA_THRESHOLD_<COLL>=<huge>  (kernel always LSA)
#   * GIN-forced : NCCL_GIN_ANVIL_SDMA_THRESHOLD_<COLL>=0       (kernel always GIN/SDMA)
# The collective-specific env only changes the kernel tier decision; the backend's
# own IPC<->SDMA put threshold (NCCL_GIN_ANVIL_SDMA_THRESHOLD) is left at default.
# The crossover size (where GIN busbw overtakes LSA) is the tuned threshold.
#
# Usage: gin-sdma-phase1-tune.bash <bin_dir> <np> <out_dir> [min] [max]
set -euo pipefail

BIN_DIR="${1:?build dir with *_perf}"
NP="${2:-8}"
OUT_DIR="${3:?output dir}"
MIN_BYTES="${4:-8}"
MAX_BYTES="${5:-64M}"
CTA="${GIN_SDMA_CTA:-32}"
ITERS="${GIN_SDMA_ITERS:-20}"
WARMUP="${GIN_SDMA_WARMUP:-5}"
HUGE=1073741824  # 1 GiB: every chunk <= this -> LSA

mkdir -p "$OUT_DIR"

MPI_OPT=(--allow-run-as-root
         -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none
         -mca hwloc_base_binding_policy none)

MPI_ENV=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
         -x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc
         -x ROCSHMEM_DISABLE_MIXED_IPC=1 -x ROCSHMEM_DEBUG_LEVEL=info:noversion
         -x RCCL_ROCSHMEM_THRESHOLD=$((128*1024*1024))
         -x NCCL_DEBUG=WARN -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
         -x NCCL_MSCCL_ENABLE=0 -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_GIN_PLUGIN=none
         -x NCCL_CUMEM_ENABLE=1 -x NCCL_NET_PLUGIN=none -x ROCSHMEM_SDMA_ENABLED=0
         -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
         -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1)

run_one() {  # $1=coll $2=envname $3=threshold $4=mode-tag $5..=extra perf args
  local coll="$1" envname="$2" thr="$3" tag="$4"; shift 4
  local exe="${BIN_DIR}/${coll}_perf"
  local out="${OUT_DIR}/${coll}.${tag}.txt"
  echo "=== ${coll} ${tag} (${envname}=${thr}) ==="
  mpirun -n "$NP" "${MPI_OPT[@]}" "${MPI_ENV[@]}" -x "${envname}=${thr}" \
    "$exe" -b "$MIN_BYTES" -e "$MAX_BYTES" -f 2 -g 1 -R 2 -V "$CTA" \
    -D 3 -n "$ITERS" -w "$WARMUP" "$@" > "$out" 2>&1
  echo "  -> $out"
}

for coll in scatter gather sendrecv; do
  case "$coll" in
    scatter)  ENVN=NCCL_GIN_ANVIL_SDMA_THRESHOLD_SCATTER;  EXTRA=(-r 0) ;;
    gather)   ENVN=NCCL_GIN_ANVIL_SDMA_THRESHOLD_GATHER;   EXTRA=(-r 0) ;;
    sendrecv) ENVN=NCCL_GIN_ANVIL_SDMA_THRESHOLD_SENDRECV; EXTRA=() ;;
  esac
  run_one "$coll" "$ENVN" "$HUGE" lsa "${EXTRA[@]}"
  run_one "$coll" "$ENVN" 0       gin "${EXTRA[@]}"
done
echo "PHASE1_TUNE_DONE"
