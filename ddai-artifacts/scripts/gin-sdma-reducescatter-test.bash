#!/usr/bin/env bash
# GIN Anvil SDMA ReduceScatter gate + perf (docker or bare-metal).
# Companion to gin-sdma-ag-test.bash / gin-sdma-a2a-test.bash for the GIN
# Anvil-SDMA ReduceScatter (reduce_scatter_perf -D 3, NCCL_GIN_TYPE=5).
# ReduceScatter is the first REDUCTION collective, so it adds an op dimension
# (-o) over the two-way host-vs-GIN comparison.
#
# Usage: ./gin-sdma-reducescatter-test.bash [NP] [MAX_BYTES]
#   NP         default 8
#   MAX_BYTES  default 128M   (total send size; per-rank output slice = size/NP)
#
# Checks (order: RS-C1 host baseline -> RS-C2 GIN):
#   RS-C1  Host-initiated ncclReduceScatter          (reduce_scatter_perf -D 0)
#   RS-C2  GIN Anvil-SDMA ReduceScatter (-D 3, NCCL_GIN_TYPE=5, -V 32): every rank
#          reads its owned output slice directly from EVERY peer's sendbuff via LSA
#          and folds the N contributions in ascending source-rank order (bit-for-bit
#          matching rccl-tests' verifiable oracle). SINGLE-TIER balanced LSA
#          read-reduce for all sizes (the load SCHEDULE adapts by total bytes:
#          grid-stride 4-peer ILP < 48 MiB, warp-strided pack+peer unroll above);
#          no scratch, no signals, entry LSA barrier only.
#
# The SM reduction mirrors rccl-tests' verifiable oracle exactly (see
# gin_sdma_reduce.h), so correctness holds across ops/types; fp8 prod & mulsum
# are out of scope (skipped by the test driver / dispatch). Sweep ops via OP
# (default "sum"; e.g. OP=all or OP="sum,avg,max"); types via DTYPE (default
# "all" -> the perf binary sweeps every supported type when -d is omitted).
#
# The GIN-SDMA policy host unit test (rccl-UnitTestsGinSdmaReduceScatterPolicy) is
# built and run via `ctest -L unit` in the rccl build, not from this perf image.
#
# Full gate (default): RS-C1 + RS-C2.
# Skip sections: RUN_HOST_BASELINE=0, RUN_GIN_SDMA=0
# GPU reset first: GPU_RESET_BEFORE_TEST=1

set -euo pipefail

NP="${1:-8}"
MAX_BYTES="${2:-128M}"
# Start at 128 B (= NP*16): reduce_scatter's per-rank slice is (total/NP) rounded
# down to a 16-byte-aligned element count, so any total < NP*16 collapses to a
# 0-element slice and prints empty "size 0 / count 0" rows. 128 B is the first
# total that yields a nonzero aligned per-rank slice on NP=8.
MIN_BYTES="${MIN_BYTES:-128}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gin-gda-sdma-rs}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
USE_DOCKER="${USE_DOCKER:-1}"
HOST_RANKS="${HOST_RANKS:-0}"
GIN_RANKS="${GIN_RANKS:-2}"
# Device (-D 3) knobs, tuned on 8x MI355X (see sibling scripts):
#   DEVICE_CTA_COUNT (-V): 32 is near-optimal (the kernel also self-selects a
#     size-adaptive CTA count decoupled from -V; see reduceScatterCtas).
#   NUM_CHANNELS: keep 1 (extra SDMA queues give no gain and can deadlock the
#     GIN put tier at chan>=2).
DEVICE_CTA_COUNT="${DEVICE_CTA_COUNT:-32}"
NUM_CHANNELS="${NUM_CHANNELS:-1}"
FACTOR="${FACTOR:-2}"
HOST_NCHANNELS="${HOST_NCHANNELS:-32}"
OP="${OP:-sum}"                 # reduction op(s): sum|prod|min|max|avg|all or CSV
DTYPE="${DTYPE:-}"              # element type; empty -> perf binary sweeps all types
# Reserved: NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCESCATTER (bytes/rank-slice). The
# shipped kernel is SINGLE-TIER LSA read-reduce and does NOT branch on it, so
# setting THRESHOLD is currently a no-op; retained for the future put-partials
# large tier (see reducescatter-gin-sdma-phase2.md).
THRESHOLD="${THRESHOLD:-}"

_HOST="$(hostname -s 2>/dev/null || hostname)"
REDUCE_SCATTER_PERF="${REDUCE_SCATTER_PERF:-rccl-tests/reduce_scatter_perf}"
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

echo "ReduceScatter gate: NP=${NP} host=${_HOST} op=${OP} dtype=${DTYPE:-all}"
_gpu_reset_before_gate

RS_C1_STATUS="skipped"
# --- RS-C1: host-initiated ncclReduceScatter (-D 0, no GIN) ---
if [[ "${RUN_HOST_BASELINE}" != "0" ]]; then
  echo "RS-C1: host ReduceScatter -D 0, ${MIN_BYTES}..${MAX_BYTES} (-R ${HOST_RANKS}, nch ${HOST_NCHANNELS}, op ${OP})"
  _run mpirun -n "${NP}" ${MPI_OPT_RCCL} "${MPI_BASE_HOST[@]}" \
    -x NCCL_MIN_NCHANNELS="${HOST_NCHANNELS}" -x NCCL_MAX_NCHANNELS="${HOST_NCHANNELS}" \
    "${REDUCE_SCATTER_PERF}" -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${FACTOR}" -g 1 -R "${HOST_RANKS}" -D 0 "${OP_ARG[@]}" "${DTYPE_ARG[@]}"
  RS_C1_STATUS="passed"
  sleep "${TEST_GAP_SEC:-3}"
fi

# --- RS-C2: GIN Anvil-SDMA ReduceScatter kernel (-D 3, NCCL_GIN_TYPE=5) ---
if [[ "${RUN_GIN_SDMA}" != "0" ]]; then
  echo "RS-C2: GIN ReduceScatter -D 3, ${MIN_BYTES}..${MAX_BYTES} (-R ${GIN_RANKS}, -V ${DEVICE_CTA_COUNT}, op ${OP}, single-tier LSA read-reduce)"
  _run mpirun -n "${NP}" ${MPI_OPT_RCCL} "${MPI_BASE[@]}" \
    -x NCCL_GIN_PLUGIN=none -x NCCL_CUMEM_ENABLE=1 -x NCCL_NET_PLUGIN=none \
    -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=5 \
    -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS="${NUM_CHANNELS}" \
    ${THRESHOLD:+-x NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCESCATTER="${THRESHOLD}"} \
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    "${REDUCE_SCATTER_PERF}" -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${FACTOR}" -g 1 -R "${GIN_RANKS}" -V "${DEVICE_CTA_COUNT}" -D 3 "${OP_ARG[@]}" "${DTYPE_ARG[@]}"
  sleep "${TEST_GAP_SEC:-3}"
fi

echo "PASS: gin-sdma-reducescatter-test np=${NP} op=${OP} RS-C1=${RS_C1_STATUS} RS-C2=${MIN_BYTES}..${MAX_BYTES}"
