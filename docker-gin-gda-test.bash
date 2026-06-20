#! /usr/bin/env bash
# Optional: RCCL_GIN_GDA_DEBUG_MPI=1, RCCL_GIN_GDA_MAX_BYTES=32M (see docker-gin-gda-ruby-test.bash).
# Test#4 (GIN GDA): RCCL_GIN_GDA_TEST4_MODE=auto|run|skip; RCCL_GIN_GDA_MIN_BNXT_FW_FOR_GDA (default 233.2.104.0).
# IB / GIN host proxy (Test#2): by default each host /dev/infiniband/uverbs* is passed through and
# --group-add rdma is added when the host has an "rdma" group. RCCL_GIN_GDA_DOCKER_UVERBS=0 or
# RCCL_GIN_GDA_DOCKER_RDMA_GROUP=0 disables those. See docker-gin-gda-ruby-test.bash for more env knobs.
#
# Test#4 auto-skips when no bnxt_en NIC is present (GDA probe path; use RCCL_GIN_GDA_TEST4_MODE=run to force).

NP=${1:-8}

DOCKER_CMD=docker
DOCKER_IMAGE="rccl-gingda713"
MAX_BYTES="${RCCL_GIN_GDA_MAX_BYTES:-1024M}"

# See docker-gin-gda-ruby-test.bash: avoid docker -it without a TTY. RCCL_GIN_GDA_DOCKER_IT=1 for interactive.
RCCL_GIN_GDA_NCCL_DEBUG="${RCCL_GIN_GDA_NCCL_DEBUG:-VERSION}"
if [[ "${RCCL_GIN_GDA_DOCKER_ULIMIT_MEMLOCK:-1}" != 0 ]]; then
  D_MEMLOCK=(--ulimit memlock=-1:-1)
else
  D_MEMLOCK=()
fi
DOCKER_GPU_COMMON="${D_MEMLOCK[*]} --shm-size 64G --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged ${RCCL_GIN_GDA_DOCKER_EXTRA:-}"
if [[ "${RCCL_GIN_GDA_DOCKER_UVERBS:-1}" != 0 ]]; then
  for _rccl_uverbs in /dev/infiniband/uverbs*; do
    [[ -c "$_rccl_uverbs" ]] && DOCKER_GPU_COMMON+=" --device ${_rccl_uverbs}"
  done
fi
if [[ "${RCCL_GIN_GDA_DOCKER_RDMA_GROUP:-1}" != 0 ]] && getent group rdma >/dev/null 2>&1; then
  DOCKER_GPU_COMMON+=" --group-add rdma"
fi
if [[ "${RCCL_GIN_GDA_DOCKER_IT:-0}" == 1 ]]; then
  DOCKER_GPU="-it --rm --init ${DOCKER_GPU_COMMON}"
else
  DOCKER_GPU="--rm --init ${DOCKER_GPU_COMMON}"
fi
MPI_CORE_MCA="-mca pml ob1 -mca btl self,vader,tcp -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none ${RCCL_GIN_GDA_MPI_MCA_EXTRA:-}"
MPI_OPT="--allow-run-as-root ${MPI_CORE_MCA}"
if [[ "${RCCL_GIN_GDA_DEBUG_MPI:-0}" == 1 ]]; then
  MPI_OPT="--allow-run-as-root --tag-output --display-map --report-bindings ${MPI_CORE_MCA}"
fi

# True (status 0) if dotted version $1 >= $2 (four numeric fields).
_rccl_gin_gda_ver_ge() {
  local IFS=.
  local -a a b
  read -r -a a <<< "$1"
  read -r -a b <<< "$2"
  local i ai bi
  for i in 0 1 2 3; do
    ai=${a[i]:-0}
    bi=${b[i]:-0}
    (( 10#$ai > 10#$bi )) && return 0
    (( 10#$ai < 10#$bi )) && return 1
  done
  return 0
}

# Returns 0 → skip Test#4; 1 → run Test#4 (auto mode only).
_rccl_gin_gda_should_skip_test4_auto() {
  local nic driver fw any_bnxt=0
  local min="${RCCL_GIN_GDA_MIN_BNXT_FW_FOR_GDA:-233.2.104.0}"
  for nic in /sys/class/net/*; do
    [[ -e "$nic" ]] || continue
    nic=${nic##*/}
    [[ "$nic" == lo ]] && continue
    driver=$(readlink -f "/sys/class/net/$nic/device/driver" 2>/dev/null)
    driver=${driver##*/}
    [[ "$driver" == bnxt_en ]] || continue
    any_bnxt=1
    command -v ethtool >/dev/null 2>&1 || continue
    fw=$(ethtool -i "$nic" 2>/dev/null | awk -F': +' '/firmware-version:/{v=$2; gsub(/ .*/,"",v); gsub(/\/.*/,"",v); print v; exit}')
    [[ -n "$fw" ]] || continue
    if ! _rccl_gin_gda_ver_ge "$fw" "$min"; then
      echo "=== RCCL_GIN_GDA: bnxt NIC '${nic}' firmware ${fw} < ${min} (RCCL GIN GDA probe minimum). Skipping Test#4. ===" >&2
      echo "=== Upgrade BNXT firmware or force: RCCL_GIN_GDA_TEST4_MODE=run ===" >&2
      return 0
    fi
  done
  if [[ "$any_bnxt" == 0 ]]; then
    echo "=== RCCL_GIN_GDA: no host bnxt_en interface (e.g. MLX-only node). Skipping Test#4 (GIN GDA probe). ===" >&2
    echo "=== To run anyway: RCCL_GIN_GDA_TEST4_MODE=run ===" >&2
    return 0
  fi
  return 1
}

# RCCL_LD_PATH="/workspace/rocshmem/lib:/workspace/rccl/lib:/opt/ucx/lib:/opt/ompi/lib:/opt/rocm/lib:/opt/rocm/core/lib/rocm_sysdeps/lib"
# HFILE="my_hostfile"
# MPIRUN_BASE="-n ${NP} --allow-run-as-root -mca pml ob1 -mca btl ^openib"
# MPIRUN_BASE_HFILE="-n ${NP} --hostfile /workspace/${HFILE} --allow-run-as-root -mca pml ob1 -mca btl ^openib"

# for ((NP = 2; NP <= 8; NP <<= 1)); do
if [[ "${RCCL_GIN_GDA_PREFLIGHT:-0}" == 1 ]]; then
  echo "=== Preflight (RCCL_GIN_GDA_PREFLIGHT=1) ===" >&2
  ${DOCKER_CMD} run ${DOCKER_GPU} --entrypoint /bin/bash "${DOCKER_IMAGE}" -c "
set -e
cd /workspace
command -v rocm-smi >/dev/null 2>&1 && rocm-smi -l || true
mpirun -n 2 --allow-run-as-root ${MPI_CORE_MCA} -x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 hostname
echo '[preflight] ok'
" || exit 2
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
    -x NCCL_GIN_ENABLE=0 \
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
  echo "=== Test#2: A2A, ${NP} gpus, GIN Host Proxy (Ib proxy; GinAlltoAllKernel; -D 3) ==="
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
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    rccl-tests/alltoall_perf -b 128 -e "${MAX_BYTES}" -f 2 -g 1 -R 2 -D 3 -A 1 -V 1
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

RCCL_GIN_GDA_RUN_TEST4=1
case "${RCCL_GIN_GDA_TEST4_MODE:-auto}" in
  skip) RCCL_GIN_GDA_RUN_TEST4=0 ;;
  run) RCCL_GIN_GDA_RUN_TEST4=1 ;;
  auto)
    if _rccl_gin_gda_should_skip_test4_auto; then
      RCCL_GIN_GDA_RUN_TEST4=0
    fi
    ;;
  *)
    echo "error: RCCL_GIN_GDA_TEST4_MODE must be auto, run, or skip (got: ${RCCL_GIN_GDA_TEST4_MODE})" >&2
    exit 1
    ;;
esac

if [[ "${RCCL_GIN_GDA_RUN_TEST4}" == 0 ]]; then
  echo "=== Test#4: A2A, ${NP} gpus, GIN GDA (skipped) ===" >&2
else
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
fi

# done

