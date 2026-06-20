#! /usr/bin/env bash

NP=${1:-8}

DOCKER_CMD=docker
DOCKER_IMAGE="rccl-gingda713"

# DOCKER_GPU="--rm --init --shm-size 64G --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged"
DOCKER_GPU="-it --rm --shm-size 64G   --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host   --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged"
MPI_OPT="-mca pml ob1 -mca btl ^openib"
# RCCL_LD_PATH="/workspace/rocshmem/lib:/workspace/rccl/lib:/opt/ucx/lib:/opt/ompi/lib:/opt/rocm/lib:/opt/rocm/core/lib/rocm_sysdeps/lib"
# HFILE="my_hostfile"
# MPIRUN_BASE="-n ${NP} --allow-run-as-root -mca pml ob1 -mca btl ^openib"
# MPIRUN_BASE_HFILE="-n ${NP} --hostfile /workspace/${HFILE} --allow-run-as-root -mca pml ob1 -mca btl ^openib"

# for ((NP = 2; NP <= 8; NP <<= 1)); do
set -x
  echo "=== Test#1: A2A, ${NP} gpus, Host Initiated ==="
  ${DOCKER_CMD} run ${DOCKER_GPU}  ${DOCKER_IMAGE} \
    mpirun -n ${NP} ${MPI_OPT} \
    -x RCCL_ROCSHMEM_ENABLE=0 \
    -x ROCSHMEM_BACKEND=ipc \
    -x ROCSHMEM_DISABLE_MIXED_IPC=1 \
    -x ROCSHMEM_SDMA_ENABLED=0 \
    -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
    -x RCCL_ROCSHMEM_THRESHOLD=$((128*1024*1024)) \
    -x NCCL_DEBUG=NONE \
    -x NCCL_GIN_ENABLE=1 \
    -x NCCL_GIN_TYPE=0 \
    -x NCCL_DEBUG_SUBSYS=INIT,NET \
    -x NCCL_CUMEM_ENABLE=1 \
    -x RCCL_ENABLE_INTRANET=1  \
    -x NCCL_DMABUF_ENABLE=1 \
    -x NCCL_MSCCL_ENABLE=0 \
    -x HSA_NO_SCRATCH_RECLAIM=1  \
    rccl-tests/alltoall_perf  \
    -b 128 -e 1024M -f 2 -g 1 -R 2 -D 0 -A 1 -V 1
set +x

set -x
  echo "=== Test#2: A2A, ${NP} gpus, GIN Host Proxy ==="
  ${DOCKER_CMD} run ${DOCKER_GPU}  ${DOCKER_IMAGE} \
    mpirun -n ${NP} ${MPI_OPT} \
    -x RCCL_ROCSHMEM_ENABLE=0 \
    -x ROCSHMEM_BACKEND=ipc \
    -x ROCSHMEM_DISABLE_MIXED_IPC=1 \
    -x ROCSHMEM_SDMA_ENABLED=0 \
    -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
    -x RCCL_ROCSHMEM_THRESHOLD=$((128*1024*1024)) \
    -x NCCL_DEBUG=NONE \
    -x NCCL_GIN_ENABLE=1 \
    -x NCCL_GIN_TYPE=2 \
    -x NCCL_DEBUG_SUBSYS=INIT,NET \
    -x NCCL_CUMEM_ENABLE=1 \
    -x RCCL_ENABLE_INTRANET=1  \
    -x NCCL_DMABUF_ENABLE=1 \
    -x NCCL_MSCCL_ENABLE=0 \
    -x HSA_NO_SCRATCH_RECLAIM=1  \
    rccl-tests/alltoall_perf  \
    -b 128 -e 1024M -f 2 -g 1 -R 2 -D 2 -A 1 -V 1
set +x

set -x
  echo "=== Test#3: A2A, ${NP} gpus, GIN ROCSHMEM+SDMA ==="
  ${DOCKER_CMD} run ${DOCKER_GPU}  ${DOCKER_IMAGE} \
    mpirun -n ${NP} ${MPI_OPT} \
    -x RCCL_ROCSHMEM_ENABLE=0 \
    -x ROCSHMEM_BACKEND=ipc \
    -x ROCSHMEM_DISABLE_MIXED_IPC=1 \
    -x ROCSHMEM_SDMA_ENABLED=1 \
    -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
    -x RCCL_ROCSHMEM_THRESHOLD=$((128*1024*1024)) \
    -x NCCL_DEBUG=NONE \
    -x NCCL_GIN_ENABLE=1 \
    -x NCCL_GIN_TYPE=4 \
    -x NCCL_DEBUG_SUBSYS=INIT,NET \
    -x NCCL_CUMEM_ENABLE=1 \
    -x RCCL_ENABLE_INTRANET=1  \
    -x NCCL_DMABUF_ENABLE=1 \
    -x NCCL_MSCCL_ENABLE=0 \
    -x HSA_NO_SCRATCH_RECLAIM=1  \
    rccl-tests/alltoall_perf  \
    -b 128 -e 1024M -f 2 -g 1 -R 2 -D 3 -A 1 -V 1
set +x

set -x
  echo "=== Test#4: A2A, ${NP} gpus, GIN GDA ==="
  ${DOCKER_CMD} run ${DOCKER_GPU}  ${DOCKER_IMAGE} \
    mpirun -n ${NP} ${MPI_OPT} \
    -x RCCL_ROCSHMEM_ENABLE=0 \
    -x ROCSHMEM_BACKEND=ipc \
    -x ROCSHMEM_DISABLE_MIXED_IPC=1 \
    -x ROCSHMEM_SDMA_ENABLED=1 \
    -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
    -x RCCL_ROCSHMEM_THRESHOLD=$((128*1024*1024)) \
    -x NCCL_DEBUG=NONE \
    -x NCCL_GIN_ENABLE=1 \
    -x NCCL_GIN_TYPE=5 \
    -x NCCL_DEBUG_SUBSYS=INIT,NET \
    -x NCCL_CUMEM_ENABLE=1 \
    -x RCCL_ENABLE_INTRANET=1  \
    -x NCCL_DMABUF_ENABLE=1 \
    -x NCCL_MSCCL_ENABLE=0 \
    -x HSA_NO_SCRATCH_RECLAIM=1  \
    rccl-tests/alltoall_perf  \
    -b 128 -e 1024M -f 2 -g 1 -R 2 -D 3 -A 1 -V 1
set +x

# done

