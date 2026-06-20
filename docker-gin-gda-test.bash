#! /usr/bin/env bash
#
# Single-node GIN / alltoall_perf harness (non-Ruby nodes: uses `docker` not `sudo docker`).
# See docs/docker-gin-gda-ruby-gin-backends-and-tests.md for GIN design (types vs -D).
#
# Optional env (same semantics as docker-gin-gda-ruby-test.bash where noted):
#   RCCL_GIN_GDA_DOCKER_IT=1          → add docker -it (interactive TTY)
#   RCCL_GIN_GDA_DOCKER_ULIMIT_MEMLOCK=0 → omit --ulimit memlock=-1:-1
#   RCCL_GIN_GDA_DOCKER_UVERBS=0      → omit per-uverbs --device (Ib proxy / Test#2)
#   RCCL_GIN_GDA_DOCKER_RDMA_GROUP=0  → omit --group-add rdma
#   RCCL_GIN_GDA_DOCKER_EXTRA         → extra docker run flags
#   RCCL_GIN_GDA_MPI_MCA_EXTRA        → extra mpirun -mca tokens
#   RCCL_GIN_GDA_TEST4_MODE=auto|run|skip → GIN GDA (Test#4): auto-skip if no bnxt_en or fw < min
#   RCCL_GIN_GDA_MIN_BNXT_FW_FOR_GDA  → BNXT firmware floor for auto (default 233.2.104.0)
#   RCCL_GIN_USE_EXTERNAL_PLUGIN=1    → do NOT pass NCCL_GIN_PLUGIN=none (external libnccl-gin.so)
#   RCCL_GIN_GDA_TEST2_BIND_HOST_LIBS=1 (default) → Test#2: -v HOSTDIR:HOSTDIR:ro for dirs in
#       RCCL_GIN_GDA_TEST2_BIND_HOST_LIBDIRS (default: /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu)
#       so container sees host rdma-core/libmlx5 (fixes MLX5_1.25 dlvsym + 0 IB devices in ddai-gin-perf.log).
#       Set to 0 if host/container glibc mismatch causes instability.
#

NP=${1:-2}
MAX_BYTES="${2:-128M}"

DOCKER_CMD="${DOCKER_CMD:-docker}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gingda713}"

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

GIN_PLUGIN_X=()
if [[ "${RCCL_GIN_USE_EXTERNAL_PLUGIN:-0}" != 1 ]]; then
  GIN_PLUGIN_X=(-x NCCL_GIN_PLUGIN=none)
fi

# Test#2 (GIN IB proxy): image rdma-core is often older than RCCL's IB path needs (mlx5dv_* @ MLX5_1.25).
DOCKER_TEST2_VOLUMES=""
if [[ "${RCCL_GIN_GDA_TEST2_BIND_HOST_LIBS:-1}" != 0 ]]; then
  _rccl_t2_libdirs="${RCCL_GIN_GDA_TEST2_BIND_HOST_LIBDIRS:-/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu}"
  for _rccl_t2_d in ${_rccl_t2_libdirs}; do
    if [[ -d "${_rccl_t2_d}" ]]; then
      DOCKER_TEST2_VOLUMES+=" -v ${_rccl_t2_d}:${_rccl_t2_d}:ro"
    fi
  done
  unset _rccl_t2_d _rccl_t2_libdirs
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

# for ((NP = 2; NP <= 8; NP <<= 1)); do
set -x
  echo "=== Test#1: A2A, ${NP} gpus, Host Initiated ==="
  ${DOCKER_CMD} run ${DOCKER_GPU} "${DOCKER_IMAGE}" \
    mpirun -n "${NP}" ${MPI_OPT} \
    -x OMPI_ALLOW_RUN_AS_ROOT=1 \
    -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    -x RCCL_ROCSHMEM_ENABLE=0 \
    -x ROCSHMEM_BACKEND=ipc \
    -x ROCSHMEM_DISABLE_MIXED_IPC=1 \
    -x ROCSHMEM_SDMA_ENABLED=0 \
    -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
    -x RCCL_ROCSHMEM_THRESHOLD=$((128*1024*1024)) \
    -x NCCL_DEBUG=VERSION \
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
  ${DOCKER_CMD} run ${DOCKER_GPU}${DOCKER_TEST2_VOLUMES} "${DOCKER_IMAGE}" \
    mpirun -n "${NP}" ${MPI_OPT} \
    -x OMPI_ALLOW_RUN_AS_ROOT=1 \
    -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    "${GIN_PLUGIN_X[@]}" \
    -x NCCL_NET_PLUGIN=none \
    -x RCCL_ROCSHMEM_ENABLE=0 \
    -x ROCSHMEM_BACKEND=ipc \
    -x ROCSHMEM_DISABLE_MIXED_IPC=1 \
    -x ROCSHMEM_SDMA_ENABLED=0 \
    -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
    -x RCCL_ROCSHMEM_THRESHOLD=$((128*1024*1024)) \
    -x NCCL_DEBUG=VERSION \
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
  ${DOCKER_CMD} run ${DOCKER_GPU} "${DOCKER_IMAGE}" \
    mpirun -n "${NP}" ${MPI_OPT} \
    -x OMPI_ALLOW_RUN_AS_ROOT=1 \
    -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    "${GIN_PLUGIN_X[@]}" \
    -x NCCL_NET_PLUGIN=none \
    -x RCCL_ROCSHMEM_ENABLE=0 \
    -x ROCSHMEM_BACKEND=ipc \
    -x ROCSHMEM_DISABLE_MIXED_IPC=1 \
    -x ROCSHMEM_SDMA_ENABLED=1 \
    -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
    -x RCCL_ROCSHMEM_THRESHOLD=$((128*1024*1024)) \
    -x NCCL_DEBUG=VERSION \
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

if [[ "${RCCL_GIN_GDA_RUN_TEST4}" == 0 ]]; then
  echo "=== Test#4: A2A, ${NP} gpus, GIN GDA (skipped) ===" >&2
else
  set -x
  echo "=== Test#4: A2A, ${NP} gpus, GIN GDA ==="
  ${DOCKER_CMD} run ${DOCKER_GPU} "${DOCKER_IMAGE}" \
    mpirun -n "${NP}" ${MPI_OPT} \
    -x OMPI_ALLOW_RUN_AS_ROOT=1 \
    -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    "${GIN_PLUGIN_X[@]}" \
    -x NCCL_NET_PLUGIN=none \
    -x RCCL_ROCSHMEM_ENABLE=0 \
    -x ROCSHMEM_BACKEND=ipc \
    -x ROCSHMEM_DISABLE_MIXED_IPC=1 \
    -x ROCSHMEM_SDMA_ENABLED=1 \
    -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
    -x RCCL_ROCSHMEM_THRESHOLD=$((128*1024*1024)) \
    -x NCCL_DEBUG=VERSION \
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
