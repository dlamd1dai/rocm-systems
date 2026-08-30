#! /usr/bin/env bash
# Build the GIN Anvil-SDMA docker image for gin-sdma A2A bring-up (MI455 / MI355).
#
# Usage (from repo root):
#   bash ddai-artifacts/scripts/docker-gin-gda-sdma-build.bash
#   bash ddai-artifacts/scripts/docker-gin-gda-sdma-build.bash 1          # --no-cache
#   GPU_TARGETS=gfx1250 COLLECTIVE=a2a bash ddai-artifacts/scripts/docker-gin-gda-sdma-build.bash
#
# Test#5 (GIN Anvil-SDMA, NCCL_GIN_TYPE=6) needs rocSHMEM built with USE_SDMA=ON.
# Optional: RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS=1 for strict libmlx5 symbol check.

DOCKER_NO_CACHE=${1:-0}
DOCKER_CMD="${DOCKER_CMD:-docker}"
DOCKERFILE="Dockerfile-rccl-gin-gda-sdma"
export DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gin-sdma-a2a-mi455}"
TARGET_GPU_ARCH="${GPU_TARGETS:-gfx1250}"
USE_LOCAL_SRC=1
ROCSHMEM_USE_SDMA=1
# CentOS Stream 9 default (MI455 SUT rdma-core 61 / bng_re ABI). Ubuntu: BASE_OS=ubuntu BASE_IMAGE=ubuntu:24.04
BASE_OS="${BASE_OS:-centos}"
BASE_IMAGE="${BASE_IMAGE:-quay.io/centos/centos:stream9}"
RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS="${RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS:-0}"
ROCM_NIGHTLY_VERSION="${ROCM_NIGHTLY_VERSION:-10.1.0a20260822}"
ROCM_NIGHTLY_BASE_URL="${ROCM_NIGHTLY_BASE_URL:-https://rocm.nightlies.amd.com/tarball-multi-arch}"
case "${TARGET_GPU_ARCH}" in
  gfx1250|gfx125*) ROCM_NIGHTLY_FAMILY="${ROCM_NIGHTLY_FAMILY:-gfx125X-dcgpu}" ;;
  gfx950*)        ROCM_NIGHTLY_FAMILY="${ROCM_NIGHTLY_FAMILY:-gfx950-dcgpu}" ;;
  *)              ROCM_NIGHTLY_FAMILY="${ROCM_NIGHTLY_FAMILY:-gfx125X-dcgpu}" ;;
esac

# COLLECTIVE=a2a narrows ONLY_FUNCS to AllToAll* + SendRecv and enables the A2A smoke gate.
COLLECTIVE="${COLLECTIVE:-a2a}"
case "${COLLECTIVE}" in
  a2a) : "${ONLY_FUNCS:=SendRecv|AlltoAllPivot|AlltoAllGda|AlltoAllvGda}"
       : "${RCCL_IMAGE_GIN_SMOKE:=1}" ;;
  full) : "${ONLY_FUNCS:=SendRecv|AlltoAllPivot|AlltoAllGda|AlltoAllvGda|Broadcast|AllGather|AllReduce|Reduce}"
       : "${RCCL_IMAGE_GIN_SMOKE:=1}" ;;
  *)   echo "WARN: unknown COLLECTIVE='${COLLECTIVE}' (use a2a|full); building full set" >&2
       ONLY_FUNCS="${ONLY_FUNCS:-SendRecv|AlltoAllPivot|AlltoAllGda|AlltoAllvGda|Broadcast|AllGather|AllReduce|Reduce}" ;;
esac

GIT_CLONE_ROOT="${GIT_CLONE_ROOT:-$PWD}"
DEV_ARTI_DIR="${GIT_CLONE_ROOT}/ddai-artifacts"
DOCKERFILE_PATH="${DEV_ARTI_DIR}/docker/${DOCKERFILE}"

mkdir -p extra-rdma-debs

if [ "$DOCKER_NO_CACHE" -eq 1 ]; then
  DOCKER_CACHE_OPT="--no-cache"
  RCCL_CACHE_BUST=1
  ROCSHMEM_CACHE_BUST=1
else
  DOCKER_CACHE_OPT=""
fi

export RCCL_CACHE_BUST=${RCCL_CACHE_BUST:-1}
export ROCSHMEM_CACHE_BUST=${ROCSHMEM_CACHE_BUST:-1}

DOCKER_NET_OPT=()
[[ -n "${DOCKER_BUILD_NETWORK:-}" ]] && DOCKER_NET_OPT=(--network="${DOCKER_BUILD_NETWORK}")

${DOCKER_CMD} build -f ${DOCKERFILE_PATH} -t ${DOCKER_IMAGE} \
    "${DOCKER_NET_OPT[@]}" \
    ${DOCKER_CACHE_OPT} \
    --build-arg BASE_OS=${BASE_OS} \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    --build-arg GPU_TARGETS=${TARGET_GPU_ARCH} \
    --build-arg ROCM_NIGHTLY_VERSION=${ROCM_NIGHTLY_VERSION} \
    --build-arg ROCM_NIGHTLY_FAMILY=${ROCM_NIGHTLY_FAMILY} \
    --build-arg ROCM_NIGHTLY_BASE_URL=${ROCM_NIGHTLY_BASE_URL} \
    --build-arg USE_LOCAL_SRC=${USE_LOCAL_SRC} \
    --build-arg ONLY_FUNCS="${ONLY_FUNCS}" \
    --build-arg ROCSHMEM_USE_SDMA=${ROCSHMEM_USE_SDMA} \
    --build-arg RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS=${RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS} \
    --build-arg RCCL_CACHE_BUST=$((RCCL_CACHE_BUST++)) \
    --build-arg ROCSHMEM_CACHE_BUST=$((ROCSHMEM_CACHE_BUST++)) \
    .
${DOCKER_CMD} image inspect "${DOCKER_IMAGE}" >/dev/null

# Post-build runtime smoke: minimal Test#5 (NCCL_GIN_TYPE=6, GinAlltoAllKernel -D 3).
RCCL_IMAGE_GIN_SMOKE="${RCCL_IMAGE_GIN_SMOKE:-1}"
GIN_SMOKE_NP="${GIN_SMOKE_NP:-4}"
GIN_SMOKE_SIZE="${GIN_SMOKE_SIZE:-1M}"
if [ "${RCCL_IMAGE_GIN_SMOKE}" = "1" ]; then
  if [ ! -e /dev/kfd ]; then
    echo "WARN: GIN smoke assert skipped (no /dev/kfd; GPU-less builder). Set RCCL_IMAGE_GIN_SMOKE=0 to silence." >&2
  else
    echo "=== Build hardening: GIN Anvil-SDMA A2A smoke (NP=${GIN_SMOKE_NP}, NCCL_GIN_TYPE=6, -e ${GIN_SMOKE_SIZE}) ==="
    GIN_SMOKE_LOG="$(mktemp)"
    RCCL_GIN_RUN_TESTS=5 TEST5_MLX5_PREFLIGHT=0 \
      bash "${DEV_ARTI_DIR}/scripts/gin-sdma-a2a-test.bash" "${GIN_SMOKE_NP}" "${GIN_SMOKE_SIZE}" \
      > "${GIN_SMOKE_LOG}" 2>&1
    if grep -qE "GIN support is not enabled for this communicator|Failed to initialize any GIN plugin|Test failure|FATAL:" "${GIN_SMOKE_LOG}" \
       || ! grep -q "Out of bounds values : 0 OK" "${GIN_SMOKE_LOG}"; then
      echo "ERROR: GIN Anvil-SDMA A2A bring-up FAILED for image '${DOCKER_IMAGE}'." >&2
      tail -n 40 "${GIN_SMOKE_LOG}" >&2
      rm -f "${GIN_SMOKE_LOG}"
      exit 1
    fi
    echo "GIN smoke OK: $(grep 'Avg bus bandwidth' "${GIN_SMOKE_LOG}" | tail -n1 | sed 's/^# *//')"
    rm -f "${GIN_SMOKE_LOG}"
  fi
else
  echo "NOTE: GIN smoke assert disabled (RCCL_IMAGE_GIN_SMOKE=0)."
fi
