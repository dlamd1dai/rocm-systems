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
#   RCCL_GIN_GDA_PREFLIGHT=0       → skip docker preflight (default 1: mpirun hostname + rocm-smi)
#   RCCL_GIN_GDA_MPI_MCA_EXTRA     → extra mpirun -mca ... tokens (quoted on your shell if needed)
#   RCCL_GIN_GDA_DOCKER_EXTRA      → extra docker run flags (e.g. --pid=host)
#   RCCL_GIN_GDA_DOCKER_UVERBS=0   → do not add --device for each /dev/infiniband/uverbs* (GIN Ib proxy / Test#2)
#   RCCL_GIN_GDA_DOCKER_RDMA_GROUP=0 → do not add --group-add rdma when host has that group
#   RCCL_GIN_GDA_TEST4_MODE=auto   → GIN GDA (Test#4): auto-skip if host bnxt_en fw < min (default auto; run|skip)
#   RCCL_GIN_GDA_MIN_BNXT_FW_FOR_GDA → BNXT firmware floor for that auto check (default 233.2.104.0, matches RCCL GIN probe)
#   RCCL_GIN_GDA_TEST2_BIND_HOST_RDMA_SO / _BASE / _EXTRA / _GNU_DIRS → same as docker-gin-gda-test.bash (per-.so default)
#
# If perf still prints nothing for a long time: rebuild the image with --build-arg GPU_TARGETS matching
# the node (e.g. gfx942 on MI300); default image builds often target gfx950 only.
#

NP=${1:-8}

DOCKER_CMD="sudo docker"
DOCKER_IMAGE="rccl-gingda713"
RCCL_GIN_GDA_SCRIPT_MARK="${RCCL_GIN_GDA_SCRIPT_MARK:-ruby-20260618e}"
MAX_BYTES="${RCCL_GIN_GDA_MAX_BYTES:-1024M}"

# Batch / Slurm / non-interactive SSH: do not use docker -it (no TTY → docker can appear hung).
# For an interactive shell: export RCCL_GIN_GDA_DOCKER_IT=1 before running this script.
# Default NCCL log level for perf runs: VERSION (quiet). Deep debug: RCCL_GIN_GDA_NCCL_DEBUG=TRACE
RCCL_GIN_GDA_NCCL_DEBUG="${RCCL_GIN_GDA_NCCL_DEBUG:-VERSION}"
# memlock: IB / pinned host memory paths often need unlimited lock inside the container.
# Set RCCL_GIN_GDA_DOCKER_ULIMIT_MEMLOCK=0 if your docker rejects --ulimit memlock=-1:-1.
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
# Explicit BTL + vader single-copy + no hwloc binding: fewer silent hangs in ROCm+Docker than bare "^openib".
MPI_CORE_MCA="-mca pml ob1 -mca btl self,vader,tcp -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none ${RCCL_GIN_GDA_MPI_MCA_EXTRA:-}"
# Root inside --privileged containers: without --allow-run-as-root, mpirun can block on the root warning.
MPI_OPT="--allow-run-as-root ${MPI_CORE_MCA}"
if [[ "${RCCL_GIN_GDA_DEBUG_MPI:-0}" == 1 ]]; then
  MPI_OPT="--allow-run-as-root --tag-output --display-map --report-bindings ${MPI_CORE_MCA}"
fi

GIN_PLUGIN_X=()
if [[ "${RCCL_GIN_USE_EXTERNAL_PLUGIN:-0}" != 1 ]]; then
  GIN_PLUGIN_X=(-x NCCL_GIN_PLUGIN=none)
fi

_rccl_gin_gda_host_so_add_mount() {
  local p="$1"
  [[ -n "$p" && -e "$p" ]] || return 0
  if [[ " ${_rccl_t2_mounted} " == *" ${p} "* ]]; then
    return 0
  fi
  DOCKER_TEST2_VOLUMES+=" -v ${p}:${p}:ro"
  _rccl_t2_mounted+=" ${p} "
}

_rccl_gin_gda_host_so_mount_from_base() {
  local base="$1" d cand real
  for d in /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu; do
    cand="${d}/${base}"
    [[ -e "${cand}" ]] || continue
    _rccl_gin_gda_host_so_add_mount "${cand}"
    real=$(readlink -f "${cand}" 2>/dev/null || true)
    [[ -n "${real}" && "${real}" != "${cand}" ]] && _rccl_gin_gda_host_so_add_mount "${real}"
    return 0
  done
  return 1
}

DOCKER_TEST2_VOLUMES=""
if [[ "${RCCL_GIN_GDA_TEST2_BIND_HOST_GNU_DIRS:-0}" != 0 ]]; then
  echo "warning: RCCL_GIN_GDA_TEST2_BIND_HOST_GNU_DIRS=1 bind-mounts full GNU lib dirs; can break librccl if host glibc is older than the image." >&2
  _rccl_t2_libdirs="${RCCL_GIN_GDA_TEST2_BIND_HOST_LIBDIRS:-/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu}"
  for _rccl_t2_d in ${_rccl_t2_libdirs}; do
    if [[ -d "${_rccl_t2_d}" ]]; then
      DOCKER_TEST2_VOLUMES+=" -v ${_rccl_t2_d}:${_rccl_t2_d}:ro"
    fi
  done
  unset _rccl_t2_d _rccl_t2_libdirs
fi
if [[ "${RCCL_GIN_GDA_TEST2_BIND_HOST_RDMA_SO:-1}" != 0 ]]; then
  _rccl_t2_mounted=""
  _rccl_t2_bases="${RCCL_GIN_GDA_TEST2_BIND_HOST_RDMA_BASE:-libmlx5.so.1 libibverbs.so.1 librdmacm.so.1 libibumad.so.3} ${RCCL_GIN_GDA_TEST2_BIND_HOST_RDMA_EXTRA:-}"
  for _rccl_t2_base in ${_rccl_t2_bases}; do
    [[ -n "${_rccl_t2_base// }" ]] || continue
    _rccl_gin_gda_host_so_mount_from_base "${_rccl_t2_base}" || true
  done
  unset _rccl_t2_base _rccl_t2_bases _rccl_t2_mounted
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

if [[ "${RCCL_GIN_GDA_PREFLIGHT:-1}" == 1 ]]; then
  echo "=== Preflight: container, rocm-smi, mpirun hostname (RCCL_GIN_GDA_PREFLIGHT=0 to skip) ===" >&2
  ${DOCKER_CMD} run ${DOCKER_GPU} --entrypoint /bin/bash "${DOCKER_IMAGE}" -c "
set -e
cd /workspace
echo '[preflight] in-container cwd=/workspace'
command -v rocm-smi >/dev/null 2>&1 && rocm-smi -l || echo '[preflight] rocm-smi unavailable'
mpirun -n 2 --allow-run-as-root ${MPI_CORE_MCA} -x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 hostname
echo '[preflight] ok'
" || {
    echo "error: preflight failed — fix docker/rocm/openmpi before RCCL perf will run." >&2
    exit 2
  }
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
  ${DOCKER_CMD} run ${DOCKER_GPU}${DOCKER_TEST2_VOLUMES}  ${DOCKER_IMAGE} \
    mpirun -n ${NP} ${MPI_OPT} \
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
    "${GIN_PLUGIN_X[@]}" \
    -x NCCL_NET_PLUGIN=none \
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
    "${GIN_PLUGIN_X[@]}" \
    -x NCCL_NET_PLUGIN=none \
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

