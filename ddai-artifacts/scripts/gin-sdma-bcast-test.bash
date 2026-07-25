#!/usr/bin/env bash
# GIN Anvil SDMA Broadcast gate + smoke (docker or bare-metal).
# Primary SUT: smci355-ccs-aus-m03-17 (MI355X, ~/rocm-systems). Mirrors gin-sdma-ag-test.bash.
#
# Usage: ./gin-sdma-bcast-test.bash [NP] [MAX_BYTES]
#   NP         default 8 (formal gate); use 2 or 4 for smoke
#   MAX_BYTES  default 128M (max broadcast size for BC-C2)
#
# Checks (order: BC-C1 -> BC-C2; BC-C1 host baseline runs first):
#   BC-C1  Host-initiated ncclBroadcast          (broadcast_perf -D 0, -R 0)
#          Always NCCL_CUMEM_ENABLE=0 (matches AllGather AG-C1 / docker Test#1).
#          Requires broadcast host device kernels: the image includes them via
#          ONLY_FUNCS="...|Broadcast" (Dockerfile / docker-gin-gda-sdma-build.bash). On an RCCL
#          build without them it fails "ncclDevFuncId ... not found for coll:0"; disable with
#          RUN_HOST_BASELINE=0 in that case.
#   BC-C2  GIN hybrid Broadcast (-D 3, NCCL_GIN_TYPE=6), root sweep (-r all by default)
#
# Semantic model (see gin-anvil-sdma-broadcast-design-plan.md §4.4):
#   flat/star fan-out from root; non-roots complete via receiver-side waitSignal(+1).
#
# Full gate (default): BC-C1 + BC-C2 (both fatal on failure).
# Skip sections:   RUN_HOST_BASELINE=0, RUN_GIN_SDMA=0
# All-SDMA (BC-D4): NCCL_GIN_ANVIL_SDMA_THRESHOLD=0 RUN_HOST_BASELINE=0 ./gin-sdma-bcast-test.bash 8 128M
# Single root:     ROOT=0 ./gin-sdma-bcast-test.bash 8 128M
# GPU reset first: GPU_RESET_BEFORE_TEST=1

set -euo pipefail

NP="${1:-8}"
MAX_BYTES="${2:-128M}"
MIN_BYTES="${MIN_BYTES:-128}"
# Host broadcast runs at full size on MI355X (validated to 256M), so BC-C1 sweeps the same
# range as BC-C2 by default. There is no 64M host cap (that was an AllGather-specific limit).
HOST_MAX_BYTES="${HOST_MAX_BYTES:-${MAX_BYTES}}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gin-gda-sdma-713}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
USE_DOCKER="${USE_DOCKER:-1}"
HOST_RANKS="${HOST_RANKS:-0}"     # -R register mode for host path (0 = none)
GIN_RANKS="${GIN_RANKS:-2}"       # -R register mode for GIN path (2 = symmetric, REQUIRED for -D 3)
DEVICE_CTA_COUNT="${DEVICE_CTA_COUNT:-8}"
ROOT="${ROOT:-all}"               # broadcast root: 'all' sweeps 0..NP-1, or pin an integer
FACTOR="${FACTOR:-2}"

_HOST="$(hostname -s 2>/dev/null || hostname)"

# RCCL / *_perf MPI (matches docker-gin-gda-sdma-test.bash Test#5)
MPI_OPT_RCCL="${MPI_OPT_RCCL:---allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none}"

_parse_size_to_bytes() {
  local s="$1" n u
  if [[ "${s}" =~ ^([0-9]+)([KkMmGg]?)[Bb]?$ ]]; then
    n="${BASH_REMATCH[1]}"
    u="${BASH_REMATCH[2]}"
    case "${u,,}" in
      k) echo $((n * 1024)) ;;
      m) echo $((n * 1024 * 1024)) ;;
      g) echo $((n * 1024 * 1024 * 1024)) ;;
      *) echo "${n}" ;;
    esac
  else
    echo "${s}"
  fi
}

MAX_BYTES_INT="$(_parse_size_to_bytes "${MAX_BYTES}")"
HOST_MAX_BYTES_INT="$(_parse_size_to_bytes "${HOST_MAX_BYTES}")"

# BC-C1 sweep: min(HOST_MAX_BYTES, MAX_BYTES).
HOST_MAX_BYTES_EFFECTIVE_INT="${HOST_MAX_BYTES_INT}"
if [[ "${MAX_BYTES_INT}" -lt "${HOST_MAX_BYTES_EFFECTIVE_INT}" ]]; then
  HOST_MAX_BYTES_EFFECTIVE_INT="${MAX_BYTES_INT}"
fi

_format_bytes() {
  local b="$1"
  if (( b >= 1024 * 1024 * 1024 )); then
    echo "$((b / 1024 / 1024 / 1024))G"
  elif (( b >= 1024 * 1024 )); then
    echo "$((b / 1024 / 1024))M"
  elif (( b >= 1024 )); then
    echo "$((b / 1024))K"
  else
    echo "${b}"
  fi
}
HOST_MAX_BYTES_EFFECTIVE="$(_format_bytes "${HOST_MAX_BYTES_EFFECTIVE_INT}")"

ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD:-${MAX_BYTES_INT}}"
HOST_ROCSHMEM_THRESHOLD="${HOST_ROCSHMEM_THRESHOLD:-${HOST_MAX_BYTES_EFFECTIVE_INT}}"
BROADCAST_PERF="${BROADCAST_PERF:-rccl-tests/broadcast_perf}"

# BC-C1 host baseline (perf reference) is on by default now that the image ships broadcast
# host kernels (ONLY_FUNCS includes Broadcast). Set RUN_HOST_BASELINE=0 to skip it.
RUN_HOST_BASELINE="${RUN_HOST_BASELINE:-1}"
RUN_GIN_SDMA="${RUN_GIN_SDMA:-1}"

# Optional GPU reset before gate (off by default). Enable with GPU_RESET_BEFORE_TEST=1.
GPU_RESET_BEFORE_TEST="${GPU_RESET_BEFORE_TEST:-0}"

DOCKER_GPU="--rm --init --ulimit memlock=-1:-1 --shm-size 64G --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged"
if getent group rdma >/dev/null 2>&1; then
  DOCKER_GPU+=" --group-add rdma"
fi

# GIN Anvil SDMA env (matches AllGather AG-C2 / docker Test#5).
MPI_BASE=(
  -x OMPI_ALLOW_RUN_AS_ROOT=1
  -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  -x RCCL_ROCSHMEM_ENABLE=0
  -x ROCSHMEM_BACKEND=ipc
  -x ROCSHMEM_DISABLE_MIXED_IPC=1
  -x ROCSHMEM_DEBUG_LEVEL=info:noversion
  -x RCCL_ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD}"
  -x NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"
  -x NCCL_DEBUG_SUBSYS=INIT,NET
  -x RCCL_ENABLE_INTRANET=1
  -x NCCL_DMABUF_ENABLE=1
  -x NCCL_MSCCL_ENABLE=0
  -x HSA_NO_SCRATCH_RECLAIM=1
)

# Host ncclBroadcast (-D 0): no GIN/Anvil SDMA env; intranet/xGMI only.
MPI_BASE_HOST=(
  -x OMPI_ALLOW_RUN_AS_ROOT=1
  -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  -x RCCL_ROCSHMEM_ENABLE=0
  -x ROCSHMEM_BACKEND=ipc
  -x ROCSHMEM_DISABLE_MIXED_IPC=1
  -x ROCSHMEM_DEBUG_LEVEL=info:noversion
  -x RCCL_ROCSHMEM_THRESHOLD="${HOST_ROCSHMEM_THRESHOLD}"
  -x NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"
  -x NCCL_DEBUG_SUBSYS=INIT,NET
  -x RCCL_ENABLE_INTRANET=1
  -x NCCL_DMABUF_ENABLE=1
  -x NCCL_MSCCL_ENABLE=0
  -x HSA_NO_SCRATCH_RECLAIM=1
  -x NCCL_GIN_PLUGIN=none
  -x NCCL_CUMEM_ENABLE=0
  -x ROCSHMEM_SDMA_ENABLED=0
  -x NCCL_GIN_ENABLE=0
  -x NCCL_GIN_TYPE=0
)

_run() {
  echo "=== $* ==="
  if [[ "${USE_DOCKER}" == "1" ]]; then
    ${DOCKER_CMD} run ${DOCKER_GPU} "${DOCKER_IMAGE}" "$@"
  else
    "$@"
  fi
}

_docker_cleanup_stale() {
  if [[ "${USE_DOCKER}" != "1" ]] || [[ "${DOCKER_CLEANUP_BEFORE_TEST:-1}" == "0" ]]; then
    return 0
  fi
  local ids
  ids="$(${DOCKER_CMD} ps -q --filter ancestor="${DOCKER_IMAGE}" 2>/dev/null || true)"
  if [[ -n "${ids}" ]]; then
    echo "Cleaning stale ${DOCKER_IMAGE} containers before gate..."
    # shellcheck disable=SC2086
    ${DOCKER_CMD} kill ${ids} >/dev/null 2>&1 || true
    sleep 2
  fi
}

_gpu_reset() {
  local reason="$1"
  if ! command -v rocm-smi >/dev/null 2>&1; then
    echo "WARN: ${reason}: rocm-smi not found; skipping GPU reset"
    return 0
  fi
  echo "${reason}..."
  local ngpu g ok=0 fail=0
  ngpu="$(rocm-smi --showid 2>/dev/null | sed -n 's/^GPU\[\([0-9][0-9]*\)\].*/\1/p' | sort -u | wc -l)"
  ngpu="${ngpu// /}"
  if [[ "${ngpu}" -lt 1 ]]; then
    ngpu=8
  fi
  for ((g = 0; g < ngpu; g++)); do
    if rocm-smi -d "${g}" --gpureset >/dev/null 2>&1; then
      ok=$((ok + 1))
    elif sudo -n rocm-smi -d "${g}" --gpureset >/dev/null 2>&1; then
      ok=$((ok + 1))
    else
      fail=$((fail + 1))
      echo "WARN: GPU ${g} --gpureset failed (try: sudo rocm-smi -d ${g} --gpureset)"
    fi
  done
  if [[ "${ok}" -gt 0 && "${fail}" -eq 0 ]]; then
    sleep "${GPU_RESET_WAIT_SEC:-8}"
    return 0
  fi
  echo "WARN: GPU reset failed on ${fail}/${ngpu} device(s)"
  if [[ "${GPU_RESET_FATAL:-0}" != "0" ]]; then
    return 1
  fi
  return 0
}

_maybe_gpu_reset_before_gate() {
  if [[ "${GPU_RESET_BEFORE_TEST}" == "0" ]]; then
    return 0
  fi
  _gpu_reset "GPU reset before gate (GPU_RESET_BEFORE_TEST=1)"
}

BC_C1_STATUS="skipped"

echo "Broadcast gate: NP=${NP} host=${_HOST} root=${ROOT}"
if [[ "${RUN_HOST_BASELINE}" != "0" ]]; then
  echo "  BC-C1 host:  ${MIN_BYTES} .. ${HOST_MAX_BYTES_EFFECTIVE} (NCCL_CUMEM_ENABLE=0)"
else
  echo "  BC-C1 host:  skipped (RUN_HOST_BASELINE=0)"
fi
if [[ "${RUN_GIN_SDMA}" != "0" ]]; then
  echo "  BC-C2 gin:   ${MIN_BYTES} .. ${MAX_BYTES} (hybrid -D 3, CTAs=${DEVICE_CTA_COUNT}, THRESHOLD=${NCCL_GIN_ANVIL_SDMA_THRESHOLD:-128})"
else
  echo "  BC-C2 gin:   skipped (RUN_GIN_SDMA=0)"
fi

if [[ "${MAX_BYTES_INT}" -lt $((1024 * 1024)) ]]; then
  echo "WARN: MAX_BYTES=${MAX_BYTES} (<1M) — BC-C2 mostly exercises the LSA/GIN-setup floor; use 128M for full SDMA path"
fi

_docker_cleanup_stale
_maybe_gpu_reset_before_gate

# --- BC-C1: host-initiated ncclBroadcast (-D 0, no GIN); runs first (hard gate) ---
if [[ "${RUN_HOST_BASELINE}" != "0" ]]; then
  echo "BC-C1: host ncclBroadcast -D 0, ${MIN_BYTES}..${HOST_MAX_BYTES_EFFECTIVE} (-R ${HOST_RANKS}, -r ${ROOT})"
  _run mpirun -n "${NP}" ${MPI_OPT_RCCL} \
    "${MPI_BASE_HOST[@]}" \
    "${BROADCAST_PERF}" -b "${MIN_BYTES}" -e "${HOST_MAX_BYTES_EFFECTIVE}" -f "${FACTOR}" -g 1 -R "${HOST_RANKS}" -D 0 -r "${ROOT}"
  BC_C1_STATUS="passed"
  sleep "${TEST_GAP_SEC:-3}"
fi

# --- BC-C2: GIN hybrid Broadcast kernel (-D 3, NCCL_GIN_TYPE=6) ---
if [[ "${RUN_GIN_SDMA}" != "0" ]]; then
  echo "BC-C2: GIN hybrid Broadcast -D 3, ${MIN_BYTES}..${MAX_BYTES} (-R ${GIN_RANKS}, -V ${DEVICE_CTA_COUNT}, -r ${ROOT})"
  _run mpirun -n "${NP}" ${MPI_OPT_RCCL} \
    "${MPI_BASE[@]}" \
    -x NCCL_GIN_PLUGIN=none \
    -x NCCL_CUMEM_ENABLE=1 \
    -x NCCL_NET_PLUGIN=none \
    -x ROCSHMEM_SDMA_ENABLED=0 \
    -x NCCL_GIN_ENABLE=1 \
    -x NCCL_GIN_TYPE=6 \
    -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS="${NUM_CHANNELS:-1}" \
    -x NCCL_GIN_ANVIL_SDMA_THRESHOLD="${NCCL_GIN_ANVIL_SDMA_THRESHOLD:-128}" \
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    "${BROADCAST_PERF}" -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f "${FACTOR}" -g 1 -R "${GIN_RANKS}" -V "${DEVICE_CTA_COUNT}" -D 3 -r "${ROOT}"
  sleep "${TEST_GAP_SEC:-3}"
fi

echo "PASS: gin-sdma-bcast-test np=${NP} root=${ROOT} BC-C1=${BC_C1_STATUS} BC-C2=${MIN_BYTES}..${MAX_BYTES}"
