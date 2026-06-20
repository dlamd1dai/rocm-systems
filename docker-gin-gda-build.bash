#! /usr/bin/env bash

DOCKER_CMD=docker
DOCKERFILE="Dockerfile-rccl-gin-gda"
DOCKER_IMAGE="rccl-gingda713"
TARGET_GPU_ARCH=gfx950
USE_LOCAL_SRC=1

N=1
${DOCKER_CMD} build -f ${DOCKERFILE} -t ${DOCKER_IMAGE} \
    --no-cache \
    --build-arg GPU_TARGETS=${TARGET_GPU_ARCH} \
    --build-arg USE_LOCAL_SRC=${USE_LOCAL_SRC} \
    --build-arg ROCSHMEM_CACHE_BUST=$((N++)) .
${DOCKER_CMD} image inspect "${DOCKER_IMAGE}" >/dev/null

