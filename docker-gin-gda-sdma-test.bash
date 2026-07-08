#! /usr/bin/env bash
# Single-node GIN alltoall perf harness (docker, non-Ruby).
# Usage: ./docker-gin-gda-sdma-test.bash [NP] [MAX_BYTES]
# Options: docs/gin-anvil-sdma-backend-tests.md

NP=${1:-8}
MAX_BYTES="${2:-128M}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gin-gda-sdma-713}"
RCCL_GIN_RUN_TESTS="${RCCL_GIN_RUN_TESTS:-${RUN_TESTS:-1,5}}"
GDA_HOST_LIB_DIRS="${TEST2_HOST_SO_SEARCH_DIRS:-${TEST2_HOST_SO_SEARCH_DIRS:-/lib64 /usr/lib64 /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu}}"
ROCSHMEM_THRESHOLD=$((128 * 1024 * 1024))

_run_test() {
  [[ ",${RCCL_GIN_RUN_TESTS}," == *",$1,"* ]]
}

_trace_on() {
  [[ "${RCCL_GIN_ECHO:-1}" == 1 ]] && set -x
}

_trace_off() {
  [[ "${RCCL_GIN_ECHO:-1}" == 1 ]] && set +x
}

if [[ "${DOCKER_ULIMIT_MEMLOCK:-1}" != 0 ]]; then
  D_MEMLOCK=(--ulimit memlock=-1:-1)
else
  D_MEMLOCK=()
fi

# DOCKER_GPU_COMMON="${D_MEMLOCK[*]} --shm-size 64G --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged ${DOCKER_EXTRA:-}"
DOCKER_GPU_COMMON="--shm-size 64G --network host --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged ${DOCKER_EXTRA:-}"

GDA_UVERBS_ADDED=0
_rccl_uverbs_seen=""
_rccl_add_uverbs() {
  local p="$1" rp
  [[ -e "${p}" ]] || return 1
  rp=$(readlink -f "${p}" 2>/dev/null) || return 1
  [[ -n "${rp}" ]] || rp="${p}"
  [[ -c "${rp}" ]] || return 1
  [[ " ${_rccl_uverbs_seen} " == *" ${rp} "* ]] && return 0
  DOCKER_GPU_COMMON+=" --device ${rp}"
  _rccl_uverbs_seen+=" ${rp} "
  GDA_UVERBS_ADDED=$((GDA_UVERBS_ADDED + 1))
}
if [[ "${DOCKER_UVERBS:-1}" != 0 ]]; then
  shopt -s nullglob
  for _u in /dev/infiniband/uverbs* /dev/uverbs*; do
    _rccl_add_uverbs "${_u}" || true
  done
  shopt -u nullglob
  if [[ "${GDA_UVERBS_ADDED}" -eq 0 ]]; then
    for _i in $(seq 0 31); do
      _rccl_add_uverbs "/dev/infiniband/uverbs${_i}" || true
      _rccl_add_uverbs "/dev/uverbs${_i}" || true
    done
  fi
fi
unset _rccl_uverbs_seen _u _i

if [[ "${DOCKER_RDMA_GROUP:-1}" != 0 ]] && getent group rdma >/dev/null 2>&1; then
  DOCKER_GPU_COMMON+=" --group-add rdma"
fi

if [[ "${GIN_GDA_DOCKER_IT:-0}" == 1 ]]; then
  DOCKER_GPU="-it --rm --init ${DOCKER_GPU_COMMON}"
else
  DOCKER_GPU="--rm --init ${DOCKER_GPU_COMMON}"
fi

MPI_OPT="--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none ${MPI_MCA_EXTRA:-}"

GIN_PLUGIN_X=()
[[ "${USE_EXTERNAL_PLUGIN:-0}" != 1 ]] && GIN_PLUGIN_X=(-x NCCL_GIN_PLUGIN=none)

MPI_BASE=(
  -x OMPI_ALLOW_RUN_AS_ROOT=1
  -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  -x RCCL_ROCSHMEM_ENABLE=0
  -x NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"
  -x NCCL_DEBUG_SUBSYS=INIT,NET
  -x NCCL_DMABUF_ENABLE=1
  -x NCCL_MSCCL_ENABLE=0
  -x HSA_NO_SCRATCH_RECLAIM=1
)

DOCKER_TEST2_VOLUMES=""
DOCKER_TEST5_MLX5_VOLUMES=""
_rccl_bind_seen=""

_rccl_bind_add() {
  local src="$1" dst="$2"
  [[ -n "${src}" && -e "${src}" && -n "${dst}" ]] || return 0
  [[ " ${_rccl_bind_seen} " == *" ${dst} "* ]] && return 0
  DOCKER_TEST2_VOLUMES+=" -v ${src}:${dst}:ro"
  _rccl_bind_seen+=" ${dst} "
}

_rccl_resolve_lib() {
  local base="$1" d cand
  for d in ${GDA_HOST_LIB_DIRS}; do
    cand="${d}/${base}"
    [[ -e "${cand}" ]] || continue
    readlink -f "${cand}" 2>/dev/null || echo "${cand}"
    return 0
  done
  return 1
}

_rccl_bind_lib_paths() {
  local base="$1" real d cand
  real=$(_rccl_resolve_lib "${base}") || return 1
  for d in ${GDA_HOST_LIB_DIRS}; do
    cand="${d}/${base}"
    [[ -e "${cand}" ]] || continue
    _rccl_bind_add "${real}" "${cand}"
    [[ "${real}" != "${cand}" ]] && _rccl_bind_add "${real}" "${real}"
  done
}

_rccl_bind_mlx5_adjacent() {
  local vbdir="$1" mlx cand real d
  for mlx in libmlx5.so libmlx5.so.1 libmlx5-infiniband.so.1 libmlx5dv.so libmlx5dv.so.1; do
    cand="${vbdir}/${mlx}"
    [[ -e "${cand}" ]] || continue
    real=$(readlink -f "${cand}" 2>/dev/null || echo "${cand}")
    for d in ${GDA_HOST_LIB_DIRS}; do
      _rccl_bind_add "${real}" "${d}/${mlx}"
    done
  done
}

_rccl_setup_rdma_volumes() {
  local d verbs vbdir plug base nl3 nlrt

  if [[ "${TEST2_BIND_HOST_GNU_DIRS:-${TEST2_BIND_HOST_GNU_DIRS:-0}}" != 0 ]]; then
    echo "warning: TEST2_BIND_HOST_GNU_DIRS=1 can break librccl if host glibc is older than the image." >&2
    for d in ${TEST2_BIND_HOST_LIBDIRS:-${TEST2_BIND_HOST_LIBDIRS:-/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu}}; do
      [[ -d "${d}" ]] && DOCKER_TEST2_VOLUMES+=" -v ${d}:${d}:ro"
    done
  fi

  if [[ "${TEST2_BIND_HOST_RDMA_SO:-${TEST2_BIND_HOST_RDMA_SO:-1}}" != 0 ]]; then
    _rccl_bind_seen=""
    if [[ -n "${TEST2_BIND_HOST_RDMA_BASE+x}" || -n "${TEST2_BIND_HOST_RDMA_BASE+x}" ]]; then
      for base in ${TEST2_BIND_HOST_RDMA_BASE:-${TEST2_BIND_HOST_RDMA_BASE:-}} \
                  ${TEST2_BIND_HOST_RDMA_EXTRA:-${TEST2_BIND_HOST_RDMA_EXTRA:-}} \
                  libnl-3.so.200 libnl-route-3.so.200; do
        [[ -n "${base// }" ]] && _rccl_bind_lib_paths "${base}" || true
      done
    else
      for base in librdmacm.so librdmacm.so.1 libibumad.so.3 \
                  ${TEST2_BIND_HOST_RDMA_EXTRA:-${TEST2_BIND_HOST_RDMA_EXTRA:-}}; do
        [[ -n "${base// }" ]] && _rccl_bind_lib_paths "${base}" || true
      done
      case "${TEST2_BIND_HOST_MLX5_SO:-${TEST2_BIND_HOST_MLX5_SO:-adjacent}}" in
        1)
          for base in libmlx5.so libmlx5.so.1 libmlx5-infiniband.so.1 libmlx5dv.so libmlx5dv.so.1; do
            _rccl_bind_lib_paths "${base}" || true
          done
          ;;
      esac
    fi

    if verbs=$(_rccl_resolve_lib libibverbs.so.1); then
      vbdir=$(dirname "${verbs}")
      for d in ${GDA_HOST_LIB_DIRS}; do
        _rccl_bind_add "${verbs}" "${d}/libibverbs.so"
        _rccl_bind_add "${verbs}" "${d}/libibverbs.so.1"
      done
      _rccl_bind_add "${verbs}" "${verbs}"

      if nl3=$(_rccl_resolve_lib libnl-3.so.200) && nlrt=$(_rccl_resolve_lib libnl-route-3.so.200); then
        for d in ${GDA_HOST_LIB_DIRS}; do
          _rccl_bind_add "${nl3}" "${d}/libnl-3.so.200"
          _rccl_bind_add "${nlrt}" "${d}/libnl-route-3.so.200"
        done
      fi

      plug="${vbdir}/libibverbs"
      if [[ -d "${plug}" ]]; then
        for d in ${GDA_HOST_LIB_DIRS}; do
          _rccl_bind_add "${plug}" "${d}/libibverbs"
        done
      fi

      case "${TEST2_BIND_HOST_MLX5_SO:-${TEST2_BIND_HOST_MLX5_SO:-adjacent}}" in
        0) ;;
        1) ;;
        *) _rccl_bind_mlx5_adjacent "${vbdir}" ;;
      esac
    fi
    unset _rccl_bind_seen
  fi

  if [[ "${TEST2_BIND_HOST_IB_SYSFS:-${TEST2_BIND_HOST_IB_SYSFS:-1}}" != 0 ]]; then
    for d in /sys/class/infiniband /etc/libibverbs.d; do
      [[ -e "${d}" ]] && DOCKER_TEST2_VOLUMES+=" -v ${d}:${d}:ro"
    done
  fi

  case "${TEST2_BIND_HOST_DEV_IFB:-${TEST2_BIND_HOST_DEV_INFINIBAND:-auto}}" in
    on) DOCKER_TEST2_VOLUMES+=" -v /dev/infiniband:/dev/infiniband" ;;
    off) ;;
    auto)
      if [[ "${GDA_UVERBS_ADDED:-0}" -eq 0 ]] && { [[ -d /dev/infiniband ]] || [[ -d /sys/class/infiniband ]]; }; then
        DOCKER_TEST2_VOLUMES+=" -v /dev/infiniband:/dev/infiniband"
      fi
      ;;
    *)
      echo "error: TEST2_BIND_HOST_DEV_IFB must be auto, on, or off" >&2
      exit 1
      ;;
  esac
}

_rccl_ver_ge() {
  local IFS=. a b i ai bi
  read -r -a a <<< "$1"
  read -r -a b <<< "$2"
  for i in 0 1 2 3; do
    ai=${a[i]:-0}
    bi=${b[i]:-0}
    (( 10#${ai} > 10#${bi} )) && return 0
    (( 10#${ai} < 10#${bi} )) && return 1
  done
  return 0
}

_rccl_skip_test4_auto() {
  local nic driver fw any_bnxt=0 min="${MIN_BNXT_FW_FOR_GDA:-233.2.104.0}"
  for nic in /sys/class/net/*; do
    [[ -e "${nic}" ]] || continue
    nic=${nic##*/}
    [[ "${nic}" == lo ]] && continue
    driver=$(readlink -f "/sys/class/net/${nic}/device/driver" 2>/dev/null)
    driver=${driver##*/}
    [[ "${driver}" == bnxt_en ]] || continue
    any_bnxt=1
    command -v ethtool >/dev/null 2>&1 || continue
    fw=$(ethtool -i "${nic}" 2>/dev/null | awk -F': +' '/firmware-version:/{v=$2; gsub(/ .*/,"",v); gsub(/\/.*/,"",v); print v; exit}')
    [[ -n "${fw}" ]] || continue
    if ! _rccl_ver_ge "${fw}" "${min}"; then
      echo "=== RCCL_GIN_GDA: skipping Test#4 (bnxt ${nic} fw ${fw} < ${min}); set TEST4_MODE=run to force ===" >&2
      return 0
    fi
  done
  if [[ "${any_bnxt}" == 0 ]]; then
    echo "=== RCCL_GIN_GDA: skipping Test#4 (no bnxt_en); set TEST4_MODE=run to force ===" >&2
    return 0
  fi
  return 1
}

_should_run_test4() {
  _run_test 4 || return 1
  case "${TEST4_MODE:-${TEST4_MODE:-auto}}" in
    skip) return 1 ;;
    run) return 0 ;;
    auto) ! _rccl_skip_test4_auto ;;
    *)
      echo "error: TEST4_MODE must be auto, run, or skip" >&2
      exit 1
      ;;
  esac
}

_rccl_setup_test5_mlx5_volumes() {
  local dir="$1" cand real base d
  [[ -d "${dir}" ]] || {
    echo "error: TEST5_HOST_MLX5_LIB_DIR is not a directory: ${dir}" >&2
    return 1
  }
  _rccl_bind_seen=""
  shopt -s nullglob
  for cand in "${dir}"/libmlx5*.so* "${dir}"/libmlx5dv*.so*; do
    [[ -f "${cand}" ]] || continue
    real=$(readlink -f "${cand}" 2>/dev/null || echo "${cand}")
    base=$(basename "${cand}")
    for d in ${GDA_HOST_LIB_DIRS}; do
      [[ " ${_rccl_bind_seen} " == *" ${d}/${base} "* ]] && continue
      DOCKER_TEST5_MLX5_VOLUMES+=" -v ${real}:${d}/${base}:ro"
      _rccl_bind_seen+=" ${d}/${base} "
    done
  done
  shopt -u nullglob
  unset _rccl_bind_seen
  [[ -n "${DOCKER_TEST5_MLX5_VOLUMES}" ]] || {
    echo "error: no libmlx5*.so* or libmlx5dv*.so* in ${dir}" >&2
    return 1
  }
}

_rccl_setup_rdma_volumes
if [[ -n "${TEST5_HOST_MLX5_LIB_DIR:-${TEST5_HOST_MLX5_LIB_DIR:-}}" ]]; then
  _rccl_setup_test5_mlx5_volumes "${TEST5_HOST_MLX5_LIB_DIR:-${TEST5_HOST_MLX5_LIB_DIR}}" || exit 1
fi

_rccl_test5_mlx5_ok() {
  [[ -n "${TEST5_HOST_MLX5_LIB_DIR:-${TEST5_HOST_MLX5_LIB_DIR:-}}" ]] && return 0
  ${DOCKER_CMD} run --rm --init ${DOCKER_TEST5_MLX5_VOLUMES} "${DOCKER_IMAGE}" sh -lc \
    'f=/lib/x86_64-linux-gnu/libmlx5.so.1; test -e "$f" || f=/usr/lib/x86_64-linux-gnu/libmlx5.so.1; \
     rf=$(readlink -f "$f"); test -f "$rf" && objdump -T "$rf" | grep -q mlx5dv_reg_dmabuf_mr' \
    >/dev/null 2>&1
}

_should_run_test5() {
  _run_test 5 || return 1
  [[ "${TEST5_MODE:-${TEST5_MODE:-run}}" != skip ]] || return 1
  if [[ "${TEST5_MLX5_PREFLIGHT:-${TEST5_MLX5_PREFLIGHT:-0}}" != 0 ]]; then
    if ! _rccl_test5_mlx5_ok; then
      echo "=== RCCL_GIN_GDA: skipping Test#5 (image libmlx5 lacks mlx5dv_reg_dmabuf_mr); set TEST5_MLX5_PREFLIGHT=0 or TEST5_HOST_MLX5_LIB_DIR ===" >&2
      return 1
    fi
  fi
  return 0
}

if _run_test 1; then
  _trace_on
  echo "=== Test#1: A2A, ${NP} gpus, Host Initiated ==="
  ${DOCKER_CMD} run ${DOCKER_GPU} "${DOCKER_IMAGE}" \
    mpirun -n "${NP}" ${MPI_OPT} \
    "${MPI_BASE[@]}" \
    -x NCCL_CUMEM_ENABLE=0 \
    -x ROCSHMEM_SDMA_ENABLED=0 \
    -x NCCL_GIN_ENABLE=0 \
    -x NCCL_GIN_TYPE=0 \
    rccl-tests/alltoall_perf -b 128 -e "${MAX_BYTES}" -f 2 -g 1 -R 0 -D 0 -A 1 -V 1
  _trace_off
fi

if _run_test 2; then
  _trace_on
  echo "=== Test#2: A2A, ${NP} gpus, GIN Host Proxy (NCCL_GIN_TYPE=2) ==="
  ${DOCKER_CMD} run ${DOCKER_GPU} ${D_MEMLOCK[*]} ${DOCKER_TEST2_VOLUMES} "${DOCKER_IMAGE}" \
    mpirun -n "${NP}" ${MPI_OPT} \
    "${MPI_BASE[@]}" \
    "${GIN_PLUGIN_X[@]}" \
    -x NCCL_CUMEM_ENABLE=1 \
    -x RCCL_ENABLE_INTRANET=1 \
    -x NCCL_NET_PLUGIN=none \
    -x NCCL_ENV_PLUGIN=none \
    -x ROCSHMEM_SDMA_ENABLED=0 \
    -x NCCL_DEBUG="${NCCL_DEBUG:-VERSION}" \
    -x NCCL_GIN_ENABLE=1 \
    -x NCCL_GIN_TYPE=2 \
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    rccl-tests/alltoall_perf -b 128 -e "${MAX_BYTES}" -f 2 -g 1 -R 2 -D 3 -A 1 -V 1
  _trace_off
fi

if _should_run_test4; then
  _trace_on
  echo "=== Test#4: A2A, ${NP} gpus, GIN GDA ==="
  ${DOCKER_CMD} run ${DOCKER_GPU} ${D_MEMLOCK[*]} "${DOCKER_IMAGE}" \
    mpirun -n "${NP}" ${MPI_OPT} \
    "${MPI_BASE[@]}" \
    "${GIN_PLUGIN_X[@]}" \
    -x NCCL_CUMEM_ENABLE=1 \
    -x RCCL_ENABLE_INTRANET=1 \
    -x NCCL_NET_PLUGIN=none \
    -x ROCSHMEM_BACKEND=ipc \
    -x ROCSHMEM_DISABLE_MIXED_IPC=1 \
    -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
    -x RCCL_ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD}" \
    -x ROCSHMEM_SDMA_ENABLED=1 \
    -x NCCL_GIN_ENABLE=1 \
    -x NCCL_GIN_TYPE=4 \
    rccl-tests/alltoall_perf -b 128 -e "${MAX_BYTES}" -f 2 -g 1 -R 2 -D 3 -A 1 -V 1
  _trace_off
fi

if _should_run_test5; then
  _trace_on
  echo "=== Test#5: A2A, ${NP} gpus, GIN Anvil SDMA (NCCL_GIN_TYPE=5); -D 3 requires -V 1 ==="
  TEST5_MPI_EXTRA=()
  if [[ -n "${NCCL_GIN_ANVIL_SDMA_THRESHOLD:-}" ]]; then
    TEST5_MPI_EXTRA+=(-x "NCCL_GIN_ANVIL_SDMA_THRESHOLD=${NCCL_GIN_ANVIL_SDMA_THRESHOLD}")
  fi
  ${DOCKER_CMD} run ${DOCKER_GPU} ${D_MEMLOCK[*]} ${DOCKER_TEST5_MLX5_VOLUMES} "${DOCKER_IMAGE}" \
    mpirun -n "${NP}" ${MPI_OPT} \
    "${MPI_BASE[@]}" \
    "${GIN_PLUGIN_X[@]}" \
    -x NCCL_CUMEM_ENABLE=1 \
    -x RCCL_ENABLE_INTRANET=1 \
    -x NCCL_NET_PLUGIN=none \
    -x ROCSHMEM_BACKEND=ipc \
    -x ROCSHMEM_DISABLE_MIXED_IPC=1 \
    -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
    -x RCCL_ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD}" \
    -x ROCSHMEM_SDMA_ENABLED=0 \
    -x NCCL_DEBUG="${NCCL_DEBUG:-VERSION}" \
    -x NCCL_GIN_ENABLE=1 \
    -x NCCL_GIN_TYPE=5 \
    -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS="${TEST5_NUM_CHANNELS:-1}" \
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    "${TEST5_MPI_EXTRA[@]}" \
    rccl-tests/alltoall_perf -b 128 -e "${MAX_BYTES}" -f 2 -g 1 -R 2 -D 3 -A 1 -V 1
  _trace_off
fi
