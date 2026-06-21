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
#   RCCL_GIN_GDA_DOCKER_UVERBS=0   → do not add --device for uverbs (GIN Ib proxy / Test#2; default: readlink -f, /dev/uverbs*, nullglob, uverbs0..31 fallback, warnings if none)
#   RCCL_GIN_GDA_DOCKER_RDMA_GROUP=0 → do not add --group-add rdma when host has that group
#   RCCL_GIN_GDA_TEST4_MODE=auto   → GIN GDA (Test#4): auto-skip if host bnxt_en fw < min (default auto; run|skip)
#   RCCL_GIN_GDA_MIN_BNXT_FW_FOR_GDA → BNXT firmware floor for that auto check (default 233.2.104.0, matches RCCL GIN probe)
#   RCCL_GIN_GDA_TEST5_MODE=skip   → skip Test#5 (GIN Anvil SDMA direct path, NCCL_GIN_TYPE=5; default run)
#   RCCL_GIN_SDMA_TEST5_NUM_CHANNELS → NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS for Test#5 (default 1)
#   RCCL_GIN_GDA_TEST2_ADJACENT_MLX5_IGNORE_SO1_MINOR_CHECK=1 → always bind adjacent host libmlx5* (see docker-gin-gda-sdma-test.bash).
#   RCCL_GIN_GDA_TEST2_BIND_HOST_RDMA_SO / _BASE / _EXTRA / _MLX5_SO / _GNU_DIRS / _IB_SYSFS / _BIND_HOST_DEV_INFINIBAND / _HOST_SO_SEARCH_DIRS → see docker-gin-gda-sdma-test.bash header
#
# If perf still prints nothing for a long time: rebuild the image with --build-arg GPU_TARGETS matching
# the node (e.g. gfx942 on MI300); default image builds often target gfx950 only.
#

NP=${1:-8}

DOCKER_CMD="sudo docker"
DOCKER_IMAGE="rccl-gin-gda-sdma-713"
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
RCCL_GIN_GDA_UVERBS_ADDED=0
_rccl_gin_gda_uverbs_real_seen=""
_rccl_gin_gda_docker_add_uverbs_resolved() {
  local p="$1" rp
  [[ -e "${p}" ]] || return 1
  rp=$(readlink -f "${p}" 2>/dev/null) || return 1
  [[ -n "${rp}" ]] || rp="${p}"
  [[ -c "${rp}" ]] || return 1
  if [[ " ${_rccl_gin_gda_uverbs_real_seen} " == *" ${rp} "* ]]; then
    return 0
  fi
  DOCKER_GPU_COMMON+=" --device ${rp}"
  _rccl_gin_gda_uverbs_real_seen+=" ${rp} "
  RCCL_GIN_GDA_UVERBS_ADDED=$((RCCL_GIN_GDA_UVERBS_ADDED + 1))
}
if [[ "${RCCL_GIN_GDA_DOCKER_UVERBS:-1}" != 0 ]]; then
  shopt -s nullglob
  for _rccl_uverbs in /dev/infiniband/uverbs* /dev/uverbs*; do
    _rccl_gin_gda_docker_add_uverbs_resolved "${_rccl_uverbs}" || true
  done
  shopt -u nullglob
  if [[ "${RCCL_GIN_GDA_UVERBS_ADDED}" -eq 0 ]]; then
    for _rccl_uvi in $(seq 0 31); do
      _rccl_gin_gda_docker_add_uverbs_resolved "/dev/infiniband/uverbs${_rccl_uvi}" || true
      _rccl_gin_gda_docker_add_uverbs_resolved "/dev/uverbs${_rccl_uvi}" || true
    done
  fi
fi
unset _rccl_gin_gda_uverbs_real_seen
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

_rccl_gin_gda_host_so_add_bind() {
  local src="$1" dst="$2"
  [[ -n "${src}" && -e "${src}" && -n "${dst}" ]] || return 0
  if [[ " ${_rccl_t2_dst_mounted} " == *" ${dst} "* ]]; then
    return 0
  fi
  DOCKER_TEST2_VOLUMES+=" -v ${src}:${dst}:ro"
  _rccl_t2_dst_mounted+=" ${dst} "
}

_rccl_gin_gda_host_so_mount_from_base() {
  local base="$1" d cand real any=0
  local _dirs="${RCCL_GIN_GDA_TEST2_HOST_SO_SEARCH_DIRS:-/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /lib64 /usr/lib64}"
  for d in ${_dirs}; do
    cand="${d}/${base}"
    [[ -e "${cand}" ]] || continue
    any=1
    real=$(readlink -f "${cand}" 2>/dev/null || true)
    [[ -z "${real}" || ! -e "${real}" ]] && real="${cand}"
    _rccl_gin_gda_host_so_add_bind "${real}" "${cand}"
    if [[ "${real}" != "${cand}" ]]; then
      _rccl_gin_gda_host_so_add_bind "${real}" "${real}"
    fi
  done
  [[ "${any}" -eq 1 ]] && return 0
  return 1
}

# True if ELF exports mlx5dv_reg_dmabuf_mr in dynamic symbol table (RCCL MLX5_1.25 path; needs binutils objdump on host).
_rccl_gin_gda_host_so_objdump_has_mlx5_dmabuf_mr() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  command -v objdump >/dev/null 2>&1 || return 1
  objdump -T "$f" 2>/dev/null | grep -q mlx5dv_reg_dmabuf_mr
}

# Bind host libmlx5* from the directory of resolved libibverbs.so.1 (same rdma-core install). RCCL dlopens
# "libmlx5.so" and dlvsym's MLX5_1.25 — image libmlx5 can be too old while host libibverbs is new (ddai-gin-perf.log).
# Host adjacent libmlx5.so.1.N.* with small N can be too old vs RCCL while image mlx5 is new (ddai-gin-perf.log).
_rccl_gin_gda_host_so_mount_mlx5_adjacent_to_ibverbs() {
  local _dirs="${RCCL_GIN_GDA_TEST2_HOST_SO_SEARCH_DIRS:-/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /lib64 /usr/lib64}"
  local d cand verbs_real vbdir mlx cand2 real check cand5 bn min_impl impl_n dv dvr
  verbs_real=""
  for d in ${_dirs}; do
    cand="${d}/libibverbs.so.1"
    [[ -e "${cand}" ]] || continue
    verbs_real=$(readlink -f "${cand}" 2>/dev/null || true)
    [[ -z "${verbs_real}" || ! -e "${verbs_real}" ]] && verbs_real="${cand}"
    break
  done
  [[ -n "${verbs_real}" ]] || return 0
  vbdir=$(dirname "${verbs_real}")
  min_impl="${RCCL_GIN_GDA_TEST2_HOST_MLX5_MIN_SO1_MINOR:-25}"
  check=""
  for cand5 in "${vbdir}/libmlx5.so.1" "${vbdir}/libmlx5.so"; do
    [[ -e "${cand5}" ]] || continue
    check=$(readlink -f "${cand5}" 2>/dev/null || true)
    [[ -z "${check}" || ! -e "${check}" ]] && check="${cand5}"
    break
  done
  if [[ -n "${check}" ]]; then
    bn=$(basename "${check}")
    if [[ "${RCCL_GIN_GDA_TEST2_ADJACENT_MLX5_IGNORE_SO1_MINOR_CHECK:-0}" == 1 ]]; then
      echo "=== RCCL_GIN_GDA: RCCL_GIN_GDA_TEST2_ADJACENT_MLX5_IGNORE_SO1_MINOR_CHECK=1 — binding adjacent host libmlx5* (${bn}). ===" >&2
    elif [[ "${bn}" =~ ^libmlx5.*\.so\.1\.([0-9]+) ]]; then
      impl_n="${BASH_REMATCH[1]}"
      if (( 10#${impl_n} < 10#${min_impl} )); then
        if _rccl_gin_gda_host_so_objdump_has_mlx5_dmabuf_mr "${check}"; then
          echo "=== RCCL_GIN_GDA: adjacent host libmlx5 ${bn} has impl ${impl_n} < ${min_impl} but objdump shows mlx5dv_reg_dmabuf_mr — binding adjacent libmlx5* (RCCL MLX5_1.25). ===" >&2
        else
          dvr=""
          for dv in "${vbdir}/libmlx5dv.so.1" "${vbdir}/libmlx5dv.so"; do
            [[ -e "${dv}" ]] || continue
            dvr=$(readlink -f "${dv}" 2>/dev/null || true)
            [[ -z "${dvr}" || ! -e "${dvr}" ]] && dvr="${dv}"
            if _rccl_gin_gda_host_so_objdump_has_mlx5_dmabuf_mr "${dvr}"; then
              echo "=== RCCL_GIN_GDA: adjacent libmlx5dv exports mlx5dv_reg_dmabuf_mr (${dvr}); libmlx5 impl ${impl_n} < ${min_impl} — binding adjacent libmlx5* anyway. ===" >&2
              break
            fi
            dvr=""
          done
          if [[ -z "${dvr}" ]]; then
            if ! command -v objdump >/dev/null 2>&1; then
              echo "=== RCCL_GIN_GDA: warning: objdump not found — binding adjacent host libmlx5* anyway (${bn}; impl ${impl_n} < ${min_impl}). Install binutils for a safer symbol probe. ===" >&2
            else
              echo "=== RCCL_GIN_GDA: Test#2 skipping adjacent host libmlx5* bind (${bn}; impl ${impl_n} < ${min_impl}; no mlx5dv_reg_dmabuf_mr on host libmlx5/libmlx5dv; upgrade host rdma-core or set RCCL_GIN_GDA_TEST2_ADJACENT_MLX5_IGNORE_SO1_MINOR_CHECK=1; ddai-gin-perf.log). ===" >&2
              return 0
            fi
          fi
        fi
      fi
    fi
  fi
  for mlx in libmlx5.so libmlx5.so.1 libmlx5-infiniband.so.1 libmlx5dv.so libmlx5dv.so.1; do
    cand2="${vbdir}/${mlx}"
    [[ -e "${cand2}" ]] || continue
    real=$(readlink -f "${cand2}" 2>/dev/null || true)
    [[ -z "${real}" || ! -e "${real}" ]] && real="${cand2}"
    for d in ${_dirs}; do
      _rccl_gin_gda_host_so_add_bind "${real}" "${d}/${mlx}"
    done
    _rccl_gin_gda_host_so_add_bind "${real}" "${real}"
  done
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
  _rccl_t2_dst_mounted=""
  if [[ -n "${RCCL_GIN_GDA_TEST2_BIND_HOST_RDMA_BASE+x}" ]]; then
    _rccl_t2_bases="${RCCL_GIN_GDA_TEST2_BIND_HOST_RDMA_BASE} ${RCCL_GIN_GDA_TEST2_BIND_HOST_RDMA_EXTRA:-} libnl-3.so.200 libnl-route-3.so.200"
  else
    _rccl_t2_mlx5_part=""
    if [[ "${RCCL_GIN_GDA_TEST2_BIND_HOST_MLX5_SO:-}" == 1 ]]; then
      _rccl_t2_mlx5_part="libmlx5.so libmlx5.so.1 libmlx5-infiniband.so.1 libmlx5dv.so libmlx5dv.so.1 "
    fi
    _rccl_t2_bases="${_rccl_t2_mlx5_part}libibverbs.so libibverbs.so.1 librdmacm.so librdmacm.so.1 libibumad.so.3 ${RCCL_GIN_GDA_TEST2_BIND_HOST_RDMA_EXTRA:-} libnl-3.so.200 libnl-route-3.so.200"
  fi
  for _rccl_t2_base in ${_rccl_t2_bases}; do
    [[ -n "${_rccl_t2_base// }" ]] || continue
    _rccl_gin_gda_host_so_mount_from_base "${_rccl_t2_base}" || true
  done
  if [[ "${RCCL_GIN_GDA_TEST2_BIND_HOST_MLX5_SO:-}" != 0 ]] && [[ "${RCCL_GIN_GDA_TEST2_BIND_HOST_MLX5_SO:-}" != 1 ]]; then
    _rccl_gin_gda_host_so_mount_mlx5_adjacent_to_ibverbs
  fi
  unset _rccl_t2_base _rccl_t2_bases _rccl_t2_dst_mounted _rccl_t2_mlx5_part
fi
if [[ "${RCCL_GIN_GDA_TEST2_BIND_HOST_IB_SYSFS:-1}" != 0 ]]; then
  for _rccl_ibsys in /sys/class/infiniband /etc/libibverbs.d; do
    [[ -e "${_rccl_ibsys}" ]] && DOCKER_TEST2_VOLUMES+=" -v ${_rccl_ibsys}:${_rccl_ibsys}:ro"
  done
  unset _rccl_ibsys
fi

RCCL_GIN_GDA_TEST2_DEV_INF_MOUNTED=0
RCCL_GIN_GDA_TEST2_BIND_HOST_DEV_INFINIBAND="${RCCL_GIN_GDA_TEST2_BIND_HOST_DEV_INFINIBAND:-auto}"
_rccl_gin_gda_t2_dev_inf_bind=0
case "${RCCL_GIN_GDA_TEST2_BIND_HOST_DEV_INFINIBAND}" in
  on) _rccl_gin_gda_t2_dev_inf_bind=1 ;;
  off) _rccl_gin_gda_t2_dev_inf_bind=0 ;;
  auto)
    if [[ "${RCCL_GIN_GDA_UVERBS_ADDED:-0}" -eq 0 ]] && { [[ -d /dev/infiniband ]] || [[ -d /sys/class/infiniband ]]; }; then
      _rccl_gin_gda_t2_dev_inf_bind=1
    fi
    ;;
  *)
    echo "error: RCCL_GIN_GDA_TEST2_BIND_HOST_DEV_INFINIBAND must be auto, on, or off (got: ${RCCL_GIN_GDA_TEST2_BIND_HOST_DEV_INFINIBAND})" >&2
    exit 1
    ;;
esac
if [[ "${_rccl_gin_gda_t2_dev_inf_bind}" -eq 1 ]]; then
  DOCKER_TEST2_VOLUMES+=" -v /dev/infiniband:/dev/infiniband"
  RCCL_GIN_GDA_TEST2_DEV_INF_MOUNTED=1
fi
unset _rccl_gin_gda_t2_dev_inf_bind

DOCKER_TEST5_MLX5_VOLUMES=""
_rccl_gin_gda_test5_so_add_bind() {
  local src="$1" dst="$2"
  [[ -n "${src}" && -e "${src}" && -n "${dst}" ]] || return 0
  if [[ " ${_rccl_t5_dst_mounted} " == *" ${dst} "* ]]; then
    return 0
  fi
  DOCKER_TEST5_MLX5_VOLUMES+=" -v ${src}:${dst}:ro"
  _rccl_t5_dst_mounted+=" ${dst} "
}

_rccl_gin_gda_test5_mount_mlx5_from_host_dir() {
  local dir="$1" d cand real base _dirs="${RCCL_GIN_GDA_TEST2_HOST_SO_SEARCH_DIRS:-/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /lib64 /usr/lib64}"
  local ok_mr=0 ok_sys=0 check
  [[ -d "${dir}" ]] || {
    echo "error: RCCL_GIN_GDA_TEST5_HOST_MLX5_LIB_DIR is not a directory: ${dir}" >&2
    return 1
  }
  shopt -s nullglob
  for cand in "${dir}"/libmlx5*.so* "${dir}"/libmlx5dv*.so*; do
    [[ -f "${cand}" ]] || continue
    check=$(readlink -f "${cand}" 2>/dev/null || true)
    [[ -z "${check}" || ! -e "${check}" ]] && check="${cand}"
    objdump -T "${check}" 2>/dev/null | grep -q mlx5dv_reg_dmabuf_mr && ok_mr=1
    objdump -T "${check}" 2>/dev/null | grep -q mlx5dv_get_data_direct_sysfs_path && ok_sys=1
  done
  shopt -u nullglob
  if [[ "${ok_mr}" != 1 ]] || [[ "${ok_sys}" != 1 ]]; then
    echo "error: RCCL_GIN_GDA_TEST5_HOST_MLX5_LIB_DIR=${dir}: no libmlx5*.so* / libmlx5dv*.so* export both mlx5dv_reg_dmabuf_mr and mlx5dv_get_data_direct_sysfs_path." >&2
    echo "error: Install Mellanox OFED or newer rdma-core on the host, then set this variable to that lib directory." >&2
    return 1
  fi
  _rccl_t5_dst_mounted=""
  shopt -s nullglob
  for cand in "${dir}"/libmlx5*.so* "${dir}"/libmlx5dv*.so*; do
    [[ -f "${cand}" ]] || continue
    real=$(readlink -f "${cand}" 2>/dev/null || true)
    [[ -z "${real}" || ! -e "${real}" ]] && real="${cand}"
    base=$(basename "${cand}")
    for d in ${_dirs}; do
      _rccl_gin_gda_test5_so_add_bind "${real}" "${d}/${base}"
    done
    _rccl_gin_gda_test5_so_add_bind "${real}" "${real}"
  done
  shopt -u nullglob
  unset _rccl_t5_dst_mounted
  echo "=== RCCL_GIN_GDA: Test#5 bind-mounts from RCCL_GIN_GDA_TEST5_HOST_MLX5_LIB_DIR=${dir} (MLX5 DMA-BUF symbols verified on host). ===" >&2
}

if [[ -n "${RCCL_GIN_GDA_TEST5_HOST_MLX5_LIB_DIR:-}" ]]; then
  _rccl_gin_gda_test5_mount_mlx5_from_host_dir "${RCCL_GIN_GDA_TEST5_HOST_MLX5_LIB_DIR}" || exit 1
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

# for ((NP = 2; NP <= 8; NP <<= 1)); do
if [ 1 -eq 1 ]; then
set -x
  echo "=== Test#1: A2A, ${NP} gpus, Host Initiated ==="
  ${DOCKER_CMD} run ${DOCKER_GPU} ${DOCKER_IMAGE} \
    mpirun -n ${NP} ${MPI_OPT} \
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
fi

if [ 0 -eq 1 ]; then
set -x
  echo "=== Test#2: A2A, ${NP} gpus, GIN Host Proxy (Ib proxy; GinAlltoAllKernel; -D 3) ==="
  if [[ "${RCCL_GIN_GDA_DOCKER_UVERBS:-1}" != 0 ]] && [[ "${RCCL_GIN_GDA_UVERBS_ADDED:-0}" -eq 0 ]]; then
    if [[ "${RCCL_GIN_GDA_TEST2_DEV_INF_MOUNTED:-0}" -eq 1 ]]; then
      echo "=== RCCL_GIN_GDA: Test#2 bind-mounts host /dev/infiniband (no uverbs --device from this shell; cgroup/Slurm)." >&2
    else
      echo "=== RCCL_GIN_GDA: WARNING: Test#2 has no uverbs --device and no /dev/infiniband bind; IB GIN will likely fail (see RCCL_GIN_GDA_TEST2_BIND_HOST_DEV_INFINIBAND)." >&2
    fi
  fi
  ${DOCKER_CMD} run ${DOCKER_GPU}${DOCKER_TEST2_VOLUMES}  ${DOCKER_IMAGE} \
    mpirun -n ${NP} ${MPI_OPT} \
    -x OMPI_ALLOW_RUN_AS_ROOT=1 \
    -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    "${GIN_PLUGIN_X[@]}" \
    -x NCCL_NET_PLUGIN=none \
    -x NCCL_ENV_PLUGIN=none \
    -x RCCL_ROCSHMEM_ENABLE=0 \
    -x ROCSHMEM_BACKEND=ipc \
    -x ROCSHMEM_DISABLE_MIXED_IPC=1 \
    -x ROCSHMEM_SDMA_ENABLED=0 \
    -x ROCSHMEM_DEBUG_LEVEL=info:noversion \
    -x RCCL_ROCSHMEM_THRESHOLD=$((128*1024*1024)) \
    -x NCCL_DEBUG=INFO \
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
fi

set -x
  echo "=== Test#3: A2A, ${NP} gpus, GIN ROCSHMEM+SDMA ==="
  ${DOCKER_CMD} run ${DOCKER_GPU} ${DOCKER_IMAGE} \
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

if [[ "${RCCL_GIN_GDA_TEST5_MODE:-run}" != "skip" ]]; then
set -x
  echo "=== Test#5: A2A, ${NP} gpus, GIN Anvil SDMA (direct; NCCL_GIN_TYPE=5) ==="
  ${DOCKER_CMD} run ${DOCKER_GPU}${DOCKER_TEST2_VOLUMES}${DOCKER_TEST5_MLX5_VOLUMES} "${DOCKER_IMAGE}" \
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
    -x NCCL_DEBUG=VERSION \
    -x NCCL_GIN_ENABLE=1 \
    -x NCCL_GIN_TYPE=5 \
    -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS="${RCCL_GIN_SDMA_TEST5_NUM_CHANNELS:-1}" \
    -x NCCL_DEBUG_SUBSYS=INIT,NET \
    -x NCCL_CUMEM_ENABLE=1 \
    -x RCCL_ENABLE_INTRANET=1 \
    -x NCCL_DMABUF_ENABLE=1 \
    -x NCCL_MSCCL_ENABLE=0 \
    -x HSA_NO_SCRATCH_RECLAIM=1 \
    -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    rccl-tests/alltoall_perf -b 128 -e "${MAX_BYTES}" -f 2 -g 1 -R 2 -D 3 -A 1 -V 1
set +x
else
  echo "=== Test#5: A2A, ${NP} gpus, GIN Anvil SDMA (skipped; RCCL_GIN_GDA_TEST5_MODE=skip) ===" >&2
fi

if [ 0 -eq 1 ]; then
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
  ${DOCKER_CMD} run ${DOCKER_GPU} ${DOCKER_IMAGE} \
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
fi
fi
# done

