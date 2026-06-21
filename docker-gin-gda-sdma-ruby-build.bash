#! /usr/bin/env bash

# Test#5 (GIN Anvil-SDMA, NCCL_GIN_TYPE=6) needs rocSHMEM built with USE_SDMA=ON. Default
# ROCSHMEM_USE_SDMA=1 passes --build-arg to the Dockerfile; set ROCSHMEM_USE_SDMA=0 to opt out.
# The Dockerfile pulls rdma-core/libmlx5 from ${VERSION_CODENAME}-updates when available so
# libmlx5 exports mlx5dv_reg_dmabuf_mr (MLX5_1.25); the image build fails early if symbols are missing.
# BuildKit: hosts without docker0 fail with "network bridge not found" — default
# DOCKER_BUILD_NETWORK=host. Override: DOCKER_BUILD_NETWORK=default

DOCKER_CMD="sudo docker"
DOCKERFILE="Dockerfile-rccl-gin-gda-sdma-ruby"
DOCKER_IMAGE="rccl-gin-gda-sdma-713"
TARGET_GPU_ARCH="gfx950"
USE_LOCAL_SRC=1
ROCSHMEM_USE_SDMA="${ROCSHMEM_USE_SDMA:-1}"

N=1
DOCKER_BUILD_NETWORK="${DOCKER_BUILD_NETWORK:-host}"
${DOCKER_CMD} build --network="${DOCKER_BUILD_NETWORK}" -f ${DOCKERFILE} -t ${DOCKER_IMAGE} \
    --no-cache \
    --build-arg GPU_TARGETS=${TARGET_GPU_ARCH} \
    --build-arg USE_LOCAL_SRC=${USE_LOCAL_SRC} \
    --build-arg ROCSHMEM_USE_SDMA=${ROCSHMEM_USE_SDMA} \
    --build-arg ROCSHMEM_CACHE_BUST=$((N++)) .
${DOCKER_CMD} image inspect "${DOCKER_IMAGE}" >/dev/null

