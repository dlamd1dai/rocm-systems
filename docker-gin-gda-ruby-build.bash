#! /usr/bin/env bash

# Hosts without docker0: BuildKit fails with "network bridge not found". Default: --network=host.
# Override: DOCKER_BUILD_NETWORK=default

DOCKER_CMD="sudo docker"
DOCKERFILE="Dockerfile-rccl-gin-gda-ruby"
DOCKER_IMAGE="rccl-gingda713"
TARGET_GPU_ARCH="gfx950"
USE_LOCAL_SRC=1

N=1
# BuildKit needs a network driver; hosts without docker0 fail with "network bridge not found".
# Use host networking for the build. Override: DOCKER_BUILD_NETWORK=default
DOCKER_BUILD_NETWORK="${DOCKER_BUILD_NETWORK:-host}"
${DOCKER_CMD} build --network="${DOCKER_BUILD_NETWORK}" -f ${DOCKERFILE} -t ${DOCKER_IMAGE} \
    --no-cache \
    --build-arg GPU_TARGETS=${TARGET_GPU_ARCH} \
    --build-arg USE_LOCAL_SRC=${USE_LOCAL_SRC} \
    --build-arg ROCSHMEM_CACHE_BUST=$((N++)) .
${DOCKER_CMD} image inspect "${DOCKER_IMAGE}" >/dev/null

