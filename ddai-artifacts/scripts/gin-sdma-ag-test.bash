#!/usr/bin/env bash
# GIN Anvil SDMA AllGather gate + smoke (docker or bare-metal).
# Primary SUT: smci355-ccs-aus-m03-17 (MI355X, ~/rocm-systems). Also validated on smci350 (MI350X).
#
# Usage: ./gin-anvil-allgather-test.bash [NP] [MAX_BYTES]
#   NP         default 8 (formal gate); use 2 or 4 for smoke
#   MAX_BYTES  default 128M (max total gathered size for AG-C2/C3)
#
# Checks (order: AG-C1 → AG-C2 → AG-C3; AG-C1 runs first to avoid GPU state
# contamination from rocSHMEM fcollect):
#   AG-C1  Host-initiated ncclAllGather       (all_gather_perf -D 0, -R 0)
#          Full range by default (no 64M cap): earlier images faulted >64M only because the
#          AllGather ring device kernel was absent (ncclDevFuncId not found for coll:2); the
#          image now ships it via ONLY_FUNCS="...|AllGather". Note >64M no longer uses the
#          rcclDirectAllGather fast path, so busbw drops to the ring kernel rate.
#          Always NCCL_CUMEM_ENABLE=0 (matches docker-gin-gda-sdma-test.bash Test#1).
#   AG-C2  GIN hybrid AllGather (-D 3, NCCL_GIN_TYPE=6)
#   AG-C3  rocSHMEM team fcollect cross-check  (rocshmem_functional_tests -a teamfcollect)
#
# Full gate (default): AG-C1 + AG-C2 + AG-C3 — all fatal on failure.
# Skip sections: RUN_HOST_BASELINE=0, RUN_GIN_SDMA=0, RUN_FCOLLECT=0
#
# mi350 AG-C3 @ 128M may segfault — GIN-only gate: RUN_FCOLLECT=0 ./gin-anvil-allgather-test.bash 8 128M
# Enable GPU reset if needed: GPU_RESET_BEFORE_TEST=1
# AG-C1 only: RUN_GIN_SDMA=0 RUN_FCOLLECT=0 ./gin-anvil-allgather-test.bash 8 128M

set -euo pipefail

NP="${1:-8}"
MAX_BYTES="${2:-128M}"
MIN_BYTES="${MIN_BYTES:-128}"
# Host ncclAllGather runs the full size range now that the AllGather ring kernel is built
# (ONLY_FUNCS includes AllGather); AG-C1 sweeps the same range as AG-C2 by default.
HOST_MAX_BYTES="${HOST_MAX_BYTES:-${MAX_BYTES}}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gin-gda-sdma-713}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
USE_DOCKER="${USE_DOCKER:-1}"
HOST_RANKS="${HOST_RANKS:-0}"
GIN_RANKS="${GIN_RANKS:-2}"
DEVICE_CTA_COUNT="${DEVICE_CTA_COUNT:-1}"

_HOST="$(hostname -s 2>/dev/null || hostname)"

# RCCL / all_gather_perf MPI (matches docker-gin-gda-sdma-test.bash Test#5)
MPI_OPT_RCCL="${MPI_OPT_RCCL:---allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none}"

# rocSHMEM functional tests need ^openib (self,vader,tcp breaks MPI_Win_create)
MPI_OPT_ROCSHMEM="${MPI_OPT_ROCSHMEM:---allow-run-as-root -mca pml ob1 -mca btl ^openib}"

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
MIN_BYTES_INT="$(_parse_size_to_bytes "${MIN_BYTES}")"

# AG-C1 sweep: min(HOST_MAX_BYTES, MAX_BYTES).
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
ALL_GATHER_PERF="${ALL_GATHER_PERF:-rccl-tests/all_gather_perf}"
FCOLLECT_BIN="${FCOLLECT_BIN:-/opt/rocm/core-7.13/bin/rocshmem_functional_tests}"
FCOLLECT_ALGO="${FCOLLECT_ALGO:-teamfcollect}"
FCOLLECT_WGS="${FCOLLECT_WGS:-1}"
FCOLLECT_WG_SIZE="${FCOLLECT_WG_SIZE:-64}"
FCOLLECT_MAX_VOL="${FCOLLECT_MAX_VOL:-${MAX_BYTES_INT}}"
FCOLLECT_MAX_MSG="${FCOLLECT_MAX_MSG:-$((FCOLLECT_MAX_VOL / NP / FCOLLECT_WGS))}"

RUN_HOST_BASELINE="${RUN_HOST_BASELINE:-1}"
RUN_GIN_SDMA="${RUN_GIN_SDMA:-1}"
RUN_FCOLLECT="${RUN_FCOLLECT:-1}"

# Optional GPU reset before gate (off by default). Enable with GPU_RESET_BEFORE_TEST=1
# after hung GIN/fcollect runs leave GPUs in a bad state for AG-C1.
GPU_RESET_BEFORE_TEST="${GPU_RESET_BEFORE_TEST:-0}"

DOCKER_GPU="--rm --init --ulimit memlock=-1:-1 --shm-size 64G --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged"
if getent group rdma >/dev/null 2>&1; then
  DOCKER_GPU+=" --group-add rdma"
fi

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

# Host ncclAllGather (-D 0): no GIN/Anvil SDMA env; intranet/xGMI only.
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

AG_C1_STATUS="skipped"

echo "AllGather gate: NP=${NP} host=${_HOST}"
if [[ "${RUN_HOST_BASELINE}" != "0" ]]; then
  echo "  AG-C1 host:  ${MIN_BYTES} .. ${HOST_MAX_BYTES_EFFECTIVE} (NCCL_CUMEM_ENABLE=0)"
else
  echo "  AG-C1 host:  skipped (RUN_HOST_BASELINE=0)"
fi
if [[ "${RUN_GIN_SDMA}" != "0" ]]; then
  echo "  AG-C2 gin:   ${MIN_BYTES} .. ${MAX_BYTES} (hybrid -D 3, CTAs=${DEVICE_CTA_COUNT})"
else
  echo "  AG-C2 gin:   skipped (RUN_GIN_SDMA=0)"
fi
if [[ "${RUN_FCOLLECT}" != "0" ]]; then
  echo "  AG-C3 fcol:  ${MIN_BYTES} .. ${MAX_BYTES} (teamfcollect)"
else
  echo "  AG-C3 fcol:  skipped (RUN_FCOLLECT=0)"
fi

if [[ "${MAX_BYTES_INT}" -lt $((1024 * 1024)) ]]; then
  echo "WARN: MAX_BYTES=${MAX_BYTES} (<1M) — AG-C2 will not reach GIN SDMA ring path (chunk >128 B); use 128M for full gate"
fi

_docker_cleanup_stale
_maybe_gpu_reset_before_gate

# --- AG-C1: host-initiated ncclAllGather (-D 0, no GIN); runs first (hard gate) ---
if [[ "${RUN_HOST_BASELINE}" != "0" ]]; then
  echo "AG-C1: host ncclAllGather -D 0, ${MIN_BYTES}..${HOST_MAX_BYTES_EFFECTIVE} (-R ${HOST_RANKS})"
  _run mpirun -n "${NP}" ${MPI_OPT_RCCL} \
    "${MPI_BASE_HOST[@]}" \
    "${ALL_GATHER_PERF}" -b "${MIN_BYTES}" -e "${HOST_MAX_BYTES_EFFECTIVE}" -f 2 -g 1 -R "${HOST_RANKS}" -D 0 -A 1 -V 1
  AG_C1_STATUS="passed"
  sleep "${TEST_GAP_SEC:-3}"
fi

# --- AG-C2: GIN hybrid AllGather kernel (-D 3, NCCL_GIN_TYPE=6) ---
if [[ "${RUN_GIN_SDMA}" != "0" ]]; then
  echo "AG-C2: GIN hybrid AllGather -D 3, ${MIN_BYTES}..${MAX_BYTES} (-R ${GIN_RANKS}, -V ${DEVICE_CTA_COUNT})"
  _run mpirun -n "${NP}" ${MPI_OPT_RCCL} \
    "${MPI_BASE[@]}" \
    -x NCCL_GIN_PLUGIN=none \
    -x NCCL_CUMEM_ENABLE=1 \
    -x NCCL_NET_PLUGIN=none \
    -x ROCSHMEM_SDMA_ENABLED=0 \
    -x NCCL_GIN_ENABLE=1 \
    -x NCCL_GIN_TYPE=6 \
    -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS="${TEST5_NUM_CHANNELS:-1}" \
    ${NCCL_GIN_ANVIL_SDMA_THRESHOLD:+-x NCCL_GIN_ANVIL_SDMA_THRESHOLD="${NCCL_GIN_ANVIL_SDMA_THRESHOLD}"} \
    ${NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER:+-x NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER="${NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER}"} \
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    "${ALL_GATHER_PERF}" -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f 2 -g 1 -R "${GIN_RANKS}" -V "${DEVICE_CTA_COUNT}" -D 3 -A 1 -C 0
  sleep "${TEST_GAP_SEC:-3}"
fi

# --- AG-C3: rocSHMEM fcollect (runs last; may leave GPUs in bad state for a follow-up AG-C1) ---
if [[ "${RUN_FCOLLECT}" != "0" ]]; then
  if [[ "${USE_DOCKER}" == "1" ]] || [[ -x "${FCOLLECT_BIN}" ]]; then
    echo "AG-C3: rocSHMEM teamfcollect NP=${NP}, max total vol=${MAX_BYTES} (per-rank msg max=${FCOLLECT_MAX_MSG} B)"
    _run mpirun -n "${NP}" ${MPI_OPT_ROCSHMEM} \
      -x OMPI_ALLOW_RUN_AS_ROOT=1 \
      -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
      -x ROCSHMEM_BACKEND=ipc \
      -x ROCSHMEM_DISABLE_MIXED_IPC=1 \
      -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
      "${FCOLLECT_BIN}" \
      -a "${FCOLLECT_ALGO}" \
      -w "${FCOLLECT_WGS}" \
      -z "${FCOLLECT_WG_SIZE}" \
      -s "${FCOLLECT_MAX_MSG}" \
      -v "${FCOLLECT_MAX_VOL}"
  else
    echo "skip: rocSHMEM fcollect (${FCOLLECT_BIN} not found; set FCOLLECT_BIN or USE_DOCKER=1)"
  fi
fi

echo "PASS: gin-anvil-allgather-test np=${NP} AG-C1=${AG_C1_STATUS} AG-C2/C3=${MIN_BYTES}..${MAX_BYTES}"
