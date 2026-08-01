#!/usr/bin/env bash
# GIN Anvil SDMA AllReduce gate + perf (docker or bare-metal).
# Primary SUT: smci355-ccs-aus-m03-17 (MI355X, ~/rocm-systems). Sibling of
# gin-sdma-reducescatter-test.bash; AllReduce is the P3 reduction collective and
# composes ReduceScatter + AllGather, so like RS it carries an op dimension (-o).
#
# Usage: ./gin-sdma-ar-test.bash [NP] [MAX_BYTES]
#   NP         default 8
#   MAX_BYTES  default 128M   (whole-message size; send == recv == size)
#
# Checks (order: AR-C1 host baseline -> AR-C2 GIN):
#   AR-C1  Host-initiated ncclAllReduce                 (all_reduce_perf -D 0)
#   AR-C2  GIN Anvil-SDMA AllReduce (-D 5, NCCL_GIN_TYPE=6, -V 32): size-hybrid
#            * total <= threshold AND out-of-place: one-shot direct LSA read-reduce
#              (every rank reads the whole buffer from each peer, folds, writes own
#              recvbuf; no scratch/signals);
#            * total >  threshold OR in-place: two-shot ReduceScatter + AllGather,
#              each per-rank slice tiled across CTAs (self-contained RS+AG, per-CTA
#              GIN signal). RS variant A (direct-LSA read-reduce, no scratch) by
#              default; set RS_PUTPARTIALS=1 for variant B (put-partials into the
#              resource-window scratch + SM reduce).
#          Default cutover 256 KiB total (provisional; retune per the design plan).
#          Override with THRESHOLD (bytes total; 0 = all-GIN two-shot).
#
# The perf binary runs BOTH in-place and out-of-place per size and reports #wrong
# for each, so IP (two-shot) and OOP (one-shot small / two-shot large) correctness
# are both covered. The SM reduction mirrors rccl-tests' verifiable oracle exactly
# (gin_sdma_reduce.h), so correctness holds across ops/types; fp8 {prod,avg} and
# PreMulSum (mulsum) are out of scope (skipped by the test driver / dispatch).
# Sweep ops via OP (default "sum"; e.g. OP=all or OP="sum,avg,max"); types via
# DTYPE (default all -> the perf binary sweeps every supported type when -d is
# omitted).
#
# Full gate (default): AR-C1 + AR-C2.
# Skip sections: RUN_HOST_BASELINE=0, RUN_GIN_SDMA=0
# GPU reset first: GPU_RESET_BEFORE_TEST=1

set -euo pipefail

NP="${1:-8}"
MAX_BYTES="${2:-128M}"
# AllReduce count has no per-rank-slice alignment constraint (unlike RS), so start
# small to exercise the one-shot small path and the degenerate tiny two-shot tiling.
MIN_BYTES="${MIN_BYTES:-8}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gin-gda-sdma-713}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
USE_DOCKER="${USE_DOCKER:-1}"
HOST_RANKS="${HOST_RANKS:-0}"
GIN_RANKS="${GIN_RANKS:-2}"
# Device (-D 5) knobs. DEVICE_CTA_COUNT (-V) sets the launch grid the one-shot /
# LSA tiers use and the per-CTA GIN signal count requested; 32 mirrors the RS
# default. The two-shot (large/in-place) path internally caps its grid to
# kAllReduceTwoShotMaxCtas (16) regardless of -V: at -V 32 its dense per-CTA world
# barrier + AllGather puts deadlock on 8x MI355X (NUM_CHANNELS=1), while <=16 is
# stable to 128 MiB. NUM_CHANNELS: keep 1 (extra SDMA queues give no gain and can
# deadlock the GIN put tier at chan>=2).
DEVICE_CTA_COUNT="${DEVICE_CTA_COUNT:-32}"
NUM_CHANNELS="${NUM_CHANNELS:-1}"
FACTOR="${FACTOR:-2}"
HOST_NCHANNELS="${HOST_NCHANNELS:-32}"
OP="${OP:-sum}"                 # reduction op(s): sum|prod|min|max|avg|all or CSV
DTYPE="${DTYPE:-}"              # element type; empty -> perf binary sweeps all types
THRESHOLD="${THRESHOLD:-}"      # kernel LSA<->GIN cutover (bytes total); empty = tuned 256 KiB
RS_PUTPARTIALS="${RS_PUTPARTIALS:-0}"  # 1 = two-shot RS variant B (scratch put-partials)

_HOST="$(hostname -s 2>/dev/null || hostname)"
ALL_REDUCE_PERF="${ALL_REDUCE_PERF:-rccl-tests/all_reduce_perf}"
RUN_HOST_BASELINE="${RUN_HOST_BASELINE:-1}"
RUN_GIN_SDMA="${RUN_GIN_SDMA:-1}"
GPU_RESET_BEFORE_TEST="${GPU_RESET_BEFORE_TEST:-0}"

MPI_OPT_RCCL="${MPI_OPT_RCCL:---allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none}"

_parse_size_to_bytes() {
  local s="$1" n u
  if [[ "${s}" =~ ^([0-9]+)([KkMmGg]?)[Bb]?$ ]]; then
    n="${BASH_REMATCH[1]}"; u="${BASH_REMATCH[2]}"
    case "${u,,}" in
      k) echo $((n * 1024)) ;; m) echo $((n * 1024 * 1024)) ;;
      g) echo $((n * 1024 * 1024 * 1024)) ;; *) echo "${n}" ;;
    esac
  else echo "${s}"; fi
}
MAX_BYTES_INT="$(_parse_size_to_bytes "${MAX_BYTES}")"
ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD:-${MAX_BYTES_INT}}"
OP_ARG=(-o "${OP}")
DTYPE_ARG=(); [[ -n "${DTYPE}" ]] && DTYPE_ARG=(-d "${DTYPE}")

DOCKER_GPU="--rm --init --ulimit memlock=-1:-1 --shm-size 64G --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged"
if getent group rdma >/dev/null 2>&1; then DOCKER_GPU+=" --group-add rdma"; fi

MPI_BASE=(
  -x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  -x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc
  -x ROCSHMEM_DISABLE_MIXED_IPC=1 -x ROCSHMEM_DEBUG_LEVEL=info:noversion
  -x RCCL_ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD}"
  -x NCCL_DEBUG="${NCCL_DEBUG:-VERSION}" -x NCCL_DEBUG_SUBSYS=INIT,NET
  -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1 -x NCCL_MSCCL_ENABLE=0
  -x HSA_NO_SCRATCH_RECLAIM=1
)
MPI_BASE_HOST=(
  -x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  -x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc
  -x ROCSHMEM_DISABLE_MIXED_IPC=1 -x ROCSHMEM_DEBUG_LEVEL=info:noversion
  -x RCCL_ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD}"
  -x NCCL_DEBUG="${NCCL_DEBUG:-VERSION}" -x NCCL_DEBUG_SUBSYS=INIT,NET
  -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1 -x NCCL_MSCCL_ENABLE=0
  -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_GIN_PLUGIN=none -x NCCL_CUMEM_ENABLE=0
  -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=0 -x NCCL_GIN_TYPE=0
)

_run() {
  echo "=== $* ==="
  if [[ "${USE_DOCKER}" == "1" ]]; then ${DOCKER_CMD} run ${DOCKER_GPU} "${DOCKER_IMAGE}" "$@"
  else "$@"; fi
}

_gpu_reset_before_gate() {
  [[ "${GPU_RESET_BEFORE_TEST}" == "0" ]] && return 0
  command -v rocm-smi >/dev/null 2>&1 || { echo "WARN: rocm-smi not found; skip reset"; return 0; }
  echo "GPU reset before gate..."; for g in $(seq 0 $((NP-1))); do rocm-smi -d "$g" --gpureset >/dev/null 2>&1 || true; done; sleep "${GPU_RESET_WAIT_SEC:-8}"
}

echo "AllReduce gate: NP=${NP} host=${_HOST} op=${OP} dtype=${DTYPE:-all} rs_variant=$([[ "${RS_PUTPARTIALS}" == 1 ]] && echo B || echo A)"
_gpu_reset_before_gate

POLICY_UT="${POLICY_UT:-rccl-tests/gin_sdma_policy_test}"
if [[ "${RUN_POLICY_UT:-1}" != "0" ]]; then
  _run "${POLICY_UT}" || { echo "FATAL: GIN-SDMA policy unit tests failed"; exit 1; }
fi

AR_C1_STATUS="skipped"
# --- AR-C1: host-initiated ncclAllReduce (-D 0, no GIN) ---
if [[ "${RUN_HOST_BASELINE}" != "0" ]]; then
  echo "AR-C1: host AllReduce -D 0, ${MIN_BYTES}..${MAX_BYTES} (-R ${HOST_RANKS}, nch ${HOST_NCHANNELS}, op ${OP})"
  _run mpirun -n "${NP}" ${MPI_OPT_RCCL} "${MPI_BASE_HOST[@]}" \
    -x NCCL_MIN_NCHANNELS="${HOST_NCHANNELS}" -x NCCL_MAX_NCHANNELS="${HOST_NCHANNELS}" \
    "${ALL_REDUCE_PERF}" -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${FACTOR}" -g 1 -R "${HOST_RANKS}" -D 0 "${OP_ARG[@]}" "${DTYPE_ARG[@]}"
  AR_C1_STATUS="passed"
  sleep "${TEST_GAP_SEC:-3}"
fi

# --- AR-C2: GIN Anvil-SDMA AllReduce kernel (-D 5, NCCL_GIN_TYPE=6) ---
if [[ "${RUN_GIN_SDMA}" != "0" ]]; then
  echo "AR-C2: GIN AllReduce -D 5, ${MIN_BYTES}..${MAX_BYTES} (-R ${GIN_RANKS}, -V ${DEVICE_CTA_COUNT}, op ${OP}, threshold=${THRESHOLD:-256K default}, RS variant=$([[ "${RS_PUTPARTIALS}" == 1 ]] && echo B || echo A))"
  _run mpirun -n "${NP}" ${MPI_OPT_RCCL} "${MPI_BASE[@]}" \
    -x NCCL_GIN_PLUGIN=none -x NCCL_CUMEM_ENABLE=1 -x NCCL_NET_PLUGIN=none \
    -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6 \
    -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS="${NUM_CHANNELS}" \
    ${THRESHOLD:+-x NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLREDUCE="${THRESHOLD}"} \
    ${RS_PUTPARTIALS:+-x NCCL_GIN_ANVIL_AR_RS_PUTPARTIALS="${RS_PUTPARTIALS}"} \
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    "${ALL_REDUCE_PERF}" -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${FACTOR}" -g 1 -R "${GIN_RANKS}" -V "${DEVICE_CTA_COUNT}" -D 5 "${OP_ARG[@]}" "${DTYPE_ARG[@]}"
  sleep "${TEST_GAP_SEC:-3}"
fi

echo "PASS: gin-sdma-ar-test np=${NP} op=${OP} AR-C1=${AR_C1_STATUS} AR-C2=${MIN_BYTES}..${MAX_BYTES}"
