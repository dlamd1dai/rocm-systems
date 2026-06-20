#! /usr/bin/env bash
#
# Deploy: copy this file from rocm-systems.git onto the compute node, or rsync the repo.
# Quick check: the traced "docker run" line must include "--group-add render" and
# "mpirun ... --allow-run-as-root". If not, you are running an old script.
#
# Optional env:
#   RCCL_GIN_GDA_DEBUG_MPI=1       → mpirun --tag-output --display-map --report-bindings
#   RCCL_GIN_GDA_MAX_BYTES=32M     → smaller -e for smoke (default 1024M)
#   RCCL_GIN_GDA_NCCL_DEBUG=INFO   → louder RCCL logs for Test#1
#

NP=${1:-8}

DOCKER_CMD="sudo docker"
DOCKER_IMAGE="rccl-gingda713"
RCCL_GIN_GDA_SCRIPT_MARK="${RCCL_GIN_GDA_SCRIPT_MARK:-ruby-20260618d}"
MAX_BYTES="${RCCL_GIN_GDA_MAX_BYTES:-1024M}"

# Batch / Slurm / non-interactive SSH: do not use docker -it (no TTY → docker can appear hung).
# For an interactive shell: export RCCL_GIN_GDA_DOCKER_IT=1 before running this script.
# Default NCCL log level for perf runs: VERSION (quiet). Deep debug: RCCL_GIN_GDA_NCCL_DEBUG=TRACE
RCCL_GIN_GDA_NCCL_DEBUG="${RCCL_GIN_GDA_NCCL_DEBUG:-VERSION}"
if [[ "${RCCL_GIN_GDA_DOCKER_IT:-0}" == 1 ]]; then
  DOCKER_GPU="-it --rm --init --shm-size 64G --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged"
else
  DOCKER_GPU="--rm --init --shm-size 64G --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged"
fi
# Root inside --privileged containers: without --allow-run-as-root, mpirun can block on the root warning.
MPI_OPT="--allow-run-as-root -mca pml ob1 -mca btl ^openib"
if [[ "${RCCL_GIN_GDA_DEBUG_MPI:-0}" == 1 ]]; then
  MPI_OPT="--allow-run-as-root --tag-output --display-map --report-bindings -mca pml ob1 -mca btl ^openib"
fi
# RCCL_LD_PATH="/workspace/rocshmem/lib:/workspace/rccl/lib:/opt/ucx/lib:/opt/ompi/lib:/opt/rocm/lib:/opt/rocm/core/lib/rocm_sysdeps/lib"
# HFILE="my_hostfile"
# MPIRUN_BASE="-n ${NP} --allow-run-as-root -mca pml ob1 -mca btl ^openib"
# MPIRUN_BASE_HFILE="-n ${NP} --hostfile /workspace/${HFILE} --allow-run-as-root -mca pml ob1 -mca btl ^openib"

# for ((NP = 2; NP <= 8; NP <<= 1)); do
echo "=== ${RCCL_GIN_GDA_SCRIPT_MARK} ===" >&2
echo "If the next +sudo docker line omits --group-add render or mpirun --allow-run-as-root, re-copy this script from rocm-systems.git." >&2
if [[ "${MPI_OPT}" != *"--allow-run-as-root"* ]]; then
  echo "error: MPI_OPT missing --allow-run-as-root (internal)" >&2
  exit 1
fi

set -x
  echo "=== Test#1: A2A, ${NP} gpus, Host Initiated ==="
  ${DOCKER_CMD} run ${DOCKER_GPU}  ${DOCKER_IMAGE} \
    mpirun -n ${NP} ${MPI_OPT} \
    -x OMPI_ALLOW_RUN_AS_ROOT=1 \
    -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    -x RCCL_ROCSHMEM_ENABLE=0 \
    -x ROCSHMEM_BACKEND=ipc \
    -x ROCSHMEM_DISABLE_MIXED_IPC=1 \
    -x ROCSHMEM_SDMA_ENABLED=0 \
    -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
    -x RCCL_ROCSHMEM_THRESHOLD=$((128*1024*1024)) \
    -x NCCL_DEBUG="${RCCL_GIN_GDA_NCCL_DEBUG}" \
    -x NCCL_GIN_ENABLE=1 \
    -x NCCL_GIN_TYPE=0 \
    -x NCCL_DEBUG_SUBSYS=INIT,NET \
    -x NCCL_CUMEM_ENABLE=1 \
    -x RCCL_ENABLE_INTRANET=1 \
    -x NCCL_DMABUF_ENABLE=1 \
    -x NCCL_MSCCL_ENABLE=0 \
    -x HSA_NO_SCRATCH_RECLAIM=1 \
    rccl-tests/alltoall_perf -b 128 -e "${MAX_BYTES}" -f 2 -g 1 -R 2 -D 0 -A 1 -V 1
set +x

set -x
  echo "=== Test#2: A2A, ${NP} gpus, GIN Host Proxy ==="
  ${DOCKER_CMD} run ${DOCKER_GPU}  ${DOCKER_IMAGE} \
    mpirun -n ${NP} ${MPI_OPT} \
    -x OMPI_ALLOW_RUN_AS_ROOT=1 \
    -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
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
    -x RCCL_ENABLE_INTRANET=1 \
    -x NCCL_DMABUF_ENABLE=1 \
    -x NCCL_MSCCL_ENABLE=0 \
    -x HSA_NO_SCRATCH_RECLAIM=1 \
    rccl-tests/alltoall_perf -b 128 -e "${MAX_BYTES}" -f 2 -g 1 -R 2 -D 2 -A 1 -V 1
set +x

set -x
  echo "=== Test#3: A2A, ${NP} gpus, GIN ROCSHMEM+SDMA ==="
  ${DOCKER_CMD} run ${DOCKER_GPU}  ${DOCKER_IMAGE} \
    mpirun -n ${NP} ${MPI_OPT} \
    -x OMPI_ALLOW_RUN_AS_ROOT=1 \
    -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
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
    -x RCCL_ENABLE_INTRANET=1 \
    -x NCCL_DMABUF_ENABLE=1 \
    -x NCCL_MSCCL_ENABLE=0 \
    -x HSA_NO_SCRATCH_RECLAIM=1 \
    rccl-tests/alltoall_perf -b 128 -e "${MAX_BYTES}" -f 2 -g 1 -R 2 -D 3 -A 1 -V 1
set +x

set -x
  echo "=== Test#4: A2A, ${NP} gpus, GIN GDA ==="
  ${DOCKER_CMD} run ${DOCKER_GPU}  ${DOCKER_IMAGE} \
    mpirun -n ${NP} ${MPI_OPT} \
    -x OMPI_ALLOW_RUN_AS_ROOT=1 \
    -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
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
    -x RCCL_ENABLE_INTRANET=1 \
    -x NCCL_DMABUF_ENABLE=1 \
    -x NCCL_MSCCL_ENABLE=0 \
    -x HSA_NO_SCRATCH_RECLAIM=1 \
    rccl-tests/alltoall_perf -b 128 -e "${MAX_BYTES}" -f 2 -g 1 -R 2 -D 3 -A 1 -V 1
set +x

# done

