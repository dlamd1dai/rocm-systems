#!/usr/bin/env bash
# GIN Anvil SDMA SendRecv gate + perf (docker or bare-metal).
# Primary SUT: smci355-ccs-aus-m03-17 (MI355X, ~/rocm-systems). Sibling of
# gin-sdma-a2a-test.bash / gin-sdma-bcast-test.bash, trimmed to the two-way
# host-vs-GIN comparison (no GDA/host-proxy NIC paths, so no RDMA volume setup).
#
# Usage: ./gin-sdma-sendrecv-test.bash [NP] [MAX_BYTES]
#   NP         default 8
#   MAX_BYTES  default 128M
#
# Checks (order: SR-C1 host baseline -> SR-C2 GIN):
#   SR-C1  Host-initiated ncclSend/ncclRecv ring  (sendrecv_perf -D 0)
#   SR-C2  GIN Anvil-SDMA SendRecv (-D 3, NCCL_GIN_TYPE=6, -V 32): each rank
#          sends to (rank+1)%N and receives from (rank-1+N)%N. The kernel is a
#          size-hybrid LSA(small)/GIN-SDMA(large); tuning (2026-07-27, 8x MI355X)
#          found direct LSA wins at every size to 512 MiB (ring writes spread
#          across all ranks), so the default threshold is LSA-always. Force the
#          GIN/SDMA tier with THRESHOLD=0 to A/B the SDMA path.
#
# Full gate (default): SR-C1 + SR-C2.
# Skip sections: RUN_HOST_BASELINE=0, RUN_GIN_SDMA=0
# Force GIN tier:  THRESHOLD=0 ./gin-sdma-sendrecv-test.bash 8 128M
# GPU reset first: GPU_RESET_BEFORE_TEST=1

set -euo pipefail

NP="${1:-8}"
MAX_BYTES="${2:-128M}"
MIN_BYTES="${MIN_BYTES:-8}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gin-gda-sdma-713}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
USE_DOCKER="${USE_DOCKER:-1}"
HOST_RANKS="${HOST_RANKS:-0}"   # -R register mode for host path (0 = none)
GIN_RANKS="${GIN_RANKS:-2}"     # -R symmetric register mode (REQUIRED for -D 3)
# Device (-D 3) knobs, tuned 2026-07-28 on 8x MI355X (busbw sweep, 1M..128M):
#   DEVICE_CTA_COUNT (-V): 32 is near-optimal. -V 16 loses ~20% (sendrecv 51 vs
#     62, gather 347 vs 422 GB/s @128M); -V 64 is at most a few % better (helps
#     scatter's GIN tier slightly, marginal elsewhere).
#   NUM_CHANNELS: keep 1. Extra SDMA queues give no gain on the LSA-tier
#     collectives (sendrecv/gather identical for 1/2/4) and DEADLOCK scatter's
#     GIN tier (chan>=2 hangs), so do not raise it.
DEVICE_CTA_COUNT="${DEVICE_CTA_COUNT:-32}"
NUM_CHANNELS="${NUM_CHANNELS:-1}"
FACTOR="${FACTOR:-2}"
# Host baseline channel pin. SendRecv is P2P (send/recv); the stock tuner
# collapses channels at mid/large sizes and cliffs (8x MI355X, 2026-07-27: 16M
# 27->56 GB/s, no small-size loss). Pinning MIN=MAX lifts it uniformly; CE/SDMA
# (-R 2 CTA_POLICY=ZERO) is actually slower for P2P here, so it is not used.
HOST_NCHANNELS="${HOST_NCHANNELS:-32}"
# Kernel LSA<->GIN cutover override (compared against the full message). Empty =
# tuned default (LSA-always). Set 0 to force GIN/SDMA, or a byte value to sweep.
THRESHOLD="${THRESHOLD:-}"

_HOST="$(hostname -s 2>/dev/null || hostname)"
SENDRECV_PERF="${SENDRECV_PERF:-rccl-tests/sendrecv_perf}"
RUN_HOST_BASELINE="${RUN_HOST_BASELINE:-1}"
RUN_GIN_SDMA="${RUN_GIN_SDMA:-1}"
GPU_RESET_BEFORE_TEST="${GPU_RESET_BEFORE_TEST:-0}"

MPI_OPT_RCCL="${MPI_OPT_RCCL:---allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none}"

_parse_size_to_bytes() {
  local s="$1" n u
  if [[ "${s}" =~ ^([0-9]+)([KkMmGg]?)[Bb]?$ ]]; then
    n="${BASH_REMATCH[1]}"; u="${BASH_REMATCH[2]}"
    case "${u,,}" in
      k) echo $((n * 1024)) ;;
      m) echo $((n * 1024 * 1024)) ;;
      g) echo $((n * 1024 * 1024 * 1024)) ;;
      *) echo "${n}" ;;
    esac
  else echo "${s}"; fi
}
MAX_BYTES_INT="$(_parse_size_to_bytes "${MAX_BYTES}")"
ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD:-${MAX_BYTES_INT}}"

DOCKER_GPU="--rm --init --ulimit memlock=-1:-1 --shm-size 64G --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged"
if getent group rdma >/dev/null 2>&1; then DOCKER_GPU+=" --group-add rdma"; fi

# GIN Anvil SDMA env (NCCL_GIN_TYPE=6).
MPI_BASE=(
  -x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  -x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc
  -x ROCSHMEM_DISABLE_MIXED_IPC=1 -x ROCSHMEM_DEBUG_LEVEL=info:noversion
  -x RCCL_ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD}"
  -x NCCL_DEBUG="${NCCL_DEBUG:-VERSION}" -x NCCL_DEBUG_SUBSYS=INIT,NET
  -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1 -x NCCL_MSCCL_ENABLE=0
  -x HSA_NO_SCRATCH_RECLAIM=1
)
# Host path (-D 0): no GIN/Anvil SDMA env; intranet/xGMI only.
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

echo "SendRecv gate: NP=${NP} host=${_HOST}"
_gpu_reset_before_gate

# --- UT: GIN-SDMA host policy unit tests (no GPU); hard preflight gate ---
POLICY_UT="${POLICY_UT:-rccl-tests/gin_sdma_policy_test}"
if [[ "${RUN_POLICY_UT:-1}" != "0" ]]; then
  _run "${POLICY_UT}" || { echo "FATAL: GIN-SDMA policy unit tests failed"; exit 1; }
fi

SR_C1_STATUS="skipped"
# --- SR-C1: host-initiated ncclSend/ncclRecv ring (-D 0, no GIN) ---
if [[ "${RUN_HOST_BASELINE}" != "0" ]]; then
  echo "SR-C1: host SendRecv -D 0, ${MIN_BYTES}..${MAX_BYTES} (-R ${HOST_RANKS}, nch ${HOST_NCHANNELS})"
  _run mpirun -n "${NP}" ${MPI_OPT_RCCL} "${MPI_BASE_HOST[@]}" \
    -x NCCL_MIN_NCHANNELS="${HOST_NCHANNELS}" -x NCCL_MAX_NCHANNELS="${HOST_NCHANNELS}" \
    "${SENDRECV_PERF}" -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${FACTOR}" -g 1 -R "${HOST_RANKS}" -D 0
  SR_C1_STATUS="passed"
  sleep "${TEST_GAP_SEC:-3}"
fi

# --- SR-C2: GIN Anvil-SDMA SendRecv kernel (-D 3, NCCL_GIN_TYPE=6) ---
if [[ "${RUN_GIN_SDMA}" != "0" ]]; then
  echo "SR-C2: GIN SendRecv -D 3, ${MIN_BYTES}..${MAX_BYTES} (-R ${GIN_RANKS}, -V ${DEVICE_CTA_COUNT}, threshold=${THRESHOLD:-LSA-always default})"
  _run mpirun -n "${NP}" ${MPI_OPT_RCCL} "${MPI_BASE[@]}" \
    -x NCCL_GIN_PLUGIN=none -x NCCL_CUMEM_ENABLE=1 -x NCCL_NET_PLUGIN=none \
    -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6 \
    -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS="${NUM_CHANNELS}" \
    ${THRESHOLD:+-x NCCL_GIN_ANVIL_SDMA_THRESHOLD_SENDRECV="${THRESHOLD}"} \
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    "${SENDRECV_PERF}" -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${FACTOR}" -g 1 -R "${GIN_RANKS}" -V "${DEVICE_CTA_COUNT}" -D 3
  sleep "${TEST_GAP_SEC:-3}"
fi

echo "PASS: gin-sdma-sendrecv-test np=${NP} SR-C1=${SR_C1_STATUS} SR-C2=${MIN_BYTES}..${MAX_BYTES}"
