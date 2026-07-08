#! /usr/bin/env bash

# Test#5 (GIN Anvil-SDMA, NCCL_GIN_TYPE=5) needs rocSHMEM built with USE_SDMA=ON. Default
# ROCSHMEM_USE_SDMA=1 passes --build-arg to the Dockerfile; set ROCSHMEM_USE_SDMA=0 to opt out.
# The Dockerfile upgrades rdma-core/libmlx5 from ${VERSION_CODENAME}-updates when available.
# Optional CI: pass --build-arg RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS=1 to fail the build unless
# libmlx5/libmlx5dv export mlx5dv_reg_dmabuf_mr (MOFED / newer rdma-core). Default 0: stock Ubuntu 24.04
# often lacks those symbols (ddai-gin-build.log); test scripts can skip Test#5 via MLX5 preflight.
# Optional: export RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS=1 for strict MLX5 DMA-BUF symbol check at docker build.
# ROCSHMEM_USE_SDMA="${ROCSHMEM_USE_SDMA:-1}"

DOCKER_NO_CACHE=${1:-0}
DOCKER_CMD="${DOCKER_CMD:-docker}"
DOCKERFILE="Dockerfile-rccl-gin-gda-sdma"
DOCKER_IMAGE="rccl-gin-gda-sdma-713"
TARGET_GPU_ARCH="${GPU_TARGETS:-gfx950}"
USE_LOCAL_SRC=1
ROCSHMEM_USE_SDMA=1
RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS="${RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS:-0}"

# Dockerfile COPY extra-rdma-debs/ requires the directory in build context (optional .deb install).
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

${DOCKER_CMD} build -f ${DOCKERFILE} -t ${DOCKER_IMAGE} \
    ${DOCKER_CACHE_OPT} \
    --build-arg GPU_TARGETS=${TARGET_GPU_ARCH} \
    --build-arg USE_LOCAL_SRC=${USE_LOCAL_SRC} \
    --build-arg ROCSHMEM_USE_SDMA=${ROCSHMEM_USE_SDMA} \
    --build-arg RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS=${RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS} \
    --build-arg RCCL_CACHE_BUST=$((RCCL_CACHE_BUST++)) \
    --build-arg ROCSHMEM_CACHE_BUST=$((ROCSHMEM_CACHE_BUST++)) \
    .
${DOCKER_CMD} image inspect "${DOCKER_IMAGE}" >/dev/null
