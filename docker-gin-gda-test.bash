#! /usr/bin/env bash
#
# Single-node GIN / alltoall_perf harness (non-Ruby nodes: uses `docker` not `sudo docker`).
# See docs/docker-gin-gda-ruby-gin-backends-and-tests.md for GIN design (types vs -D).
#
# Optional env (same semantics as docker-gin-gda-ruby-test.bash where noted):
#   RCCL_GIN_GDA_DOCKER_IT=1          → add docker -it (interactive TTY)
#   RCCL_GIN_GDA_DOCKER_ULIMIT_MEMLOCK=0 → omit --ulimit memlock=-1:-1
#   RCCL_GIN_GDA_DOCKER_UVERBS=0      → omit per-uverbs --device (Ib proxy / Test#2; default: readlink -f + nullglob on /dev/infiniband/uverbs* and /dev/uverbs*, numeric fallback)
#   RCCL_GIN_GDA_DOCKER_RDMA_GROUP=0  → omit --group-add rdma
#   RCCL_GIN_GDA_DOCKER_EXTRA         → extra docker run flags
#   RCCL_GIN_GDA_MPI_MCA_EXTRA        → extra mpirun -mca tokens
#   RCCL_GIN_GDA_TEST4_MODE=auto|run|skip → GIN GDA (Test#4): auto-skip if no bnxt_en or fw < min
#   RCCL_GIN_GDA_MIN_BNXT_FW_FOR_GDA  → BNXT firmware floor for auto (default 233.2.104.0)
#   RCCL_GIN_USE_EXTERNAL_PLUGIN=1    → do NOT pass NCCL_GIN_PLUGIN=none (external libnccl-gin.so)
#   RCCL_GIN_GDA_TEST2_BIND_HOST_RDMA_SO=1 (default) → Test#2: bind-mount individual host RDMA .so files
#       (same path in container) for verbs/rdmacm without replacing libc (see ddai-gin-perf.log).
#   RCCL_GIN_GDA_TEST2_BIND_HOST_RDMA_BASE → explicit list of basenames (plus libnl defaults); does not change mlx5 policy below.
#   RCCL_GIN_GDA_TEST2_BIND_HOST_MLX5_SO=0|1|adjacent (default: adjacent) → Test#2 host libmlx5* policy:
#       adjacent (default): bind libmlx5* from the same directory as resolved host libibverbs.so.1 (pairs rdma-core with verbs; ddai-gin-perf.log).
#       1: legacy — search RCCL_GIN_GDA_TEST2_HOST_SO_SEARCH_DIRS for libmlx5* (can pick a mismatched older .so).
#       0: do not bind host libmlx5* (image libmlx5 only).
#   RCCL_GIN_GDA_TEST2_HOST_MLX5_MIN_SO1_MINOR (default 25) → adjacent policy only: skip host libmlx5* when
#       resolved libmlx5.so.1.N.* has N < min (host mlx5 too old for RCCL MLX5_1.25 dlvsym; ddai-gin-perf.log).
#   RCCL_GIN_GDA_TEST2_BIND_HOST_RDMA_EXTRA → more basenames before default libnl mounts.
#   RCCL_GIN_GDA_TEST2_BIND_HOST_RDMA_SO=0 → disable per-.so mounts.
#   RCCL_GIN_GDA_TEST2_BIND_HOST_GNU_DIRS=1 → dangerous: whole /lib/… and /usr/lib/…-gnu dirs (breaks GLIBC if host older).
#   RCCL_GIN_GDA_TEST2_BIND_HOST_IB_SYSFS=1 (default) → Test#2: -v host /sys/class/infiniband and /etc/libibverbs.d (ro) so
#       ibverbs can enumerate HCAs inside Docker (fixes GIN off when libs are correct; ddai-gin-perf.log).
#   RCCL_GIN_GDA_TEST2_BIND_HOST_DEV_INFINIBAND=auto|on|off (default auto) → Test#2: if no uverbs --device, -v
#       /dev/infiniband:/dev/infiniband when /dev/infiniband or /sys/class/infiniband exists (no sysfs glob; ddai-gin-perf.log).
#   RCCL_GIN_GDA_TEST2_HOST_SO_SEARCH_DIRS → dirs to resolve RDMA .so basenames (default includes lib64 paths).
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

# Without per-uverbs --device, ibverbs sees no HCAs and GIN IB proxy stays off (ddai-gin-perf.log).
# Use nullglob; resolve with readlink -f before [[ -c ]] — /dev/infiniband/uverbs* are often symlinks
# and bash -c is false on the symlink path even when the target is a char dev (ddai-gin-perf.log).
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

MPI_CORE_MCA="-mca pml ob1 -mca btl self,vader,tcp -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none ${RCCL_GIN_GDA_MPI_MCA_EXTRA:-}"
MPI_OPT="--allow-run-as-root ${MPI_CORE_MCA}"

GIN_PLUGIN_X=()
if [[ "${RCCL_GIN_USE_EXTERNAL_PLUGIN:-0}" != 1 ]]; then
  GIN_PLUGIN_X=(-x NCCL_GIN_PLUGIN=none)
fi

# Test#2 (GIN IB proxy): bind host *resolved* ELF onto each logical path (ddai-gin-perf.log: libmlx5.so is often a symlink;
# -v symlink:symlink leaves wrong mlx5dv_*; use -v readlink -f(libmlx5.so):libmlx5.so).
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

# Bind host libmlx5* from the directory of resolved libibverbs.so.1 (same rdma-core install). RCCL dlopens
# "libmlx5.so" and dlvsym's MLX5_1.25 — image libmlx5 can be too old while host libibverbs is new (ddai-gin-perf.log).
# Host adjacent libmlx5.so.1.N.* with small N can also be too old vs RCCL while the image mlx5 is new (ddai-gin-perf.log).
_rccl_gin_gda_host_so_mount_mlx5_adjacent_to_ibverbs() {
  local _dirs="${RCCL_GIN_GDA_TEST2_HOST_SO_SEARCH_DIRS:-/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /lib64 /usr/lib64}"
  local d cand verbs_real vbdir mlx cand2 real check cand5 bn min_impl impl_n
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
    if [[ "${bn}" =~ ^libmlx5.*\.so\.1\.([0-9]+) ]]; then
      impl_n="${BASH_REMATCH[1]}"
      if (( 10#${impl_n} < 10#${min_impl} )); then
        echo "=== RCCL_GIN_GDA: Test#2 skipping adjacent host libmlx5* bind (${bn}; impl ${impl_n} < ${min_impl}; RCCL MLX5_1.25 dlvsym — using image libmlx5; ddai-gin-perf.log). ===" >&2
        return 0
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
  echo "warning: RCCL_GIN_GDA_TEST2_BIND_HOST_GNU_DIRS=1 bind-mounts full GNU lib dirs; host libc older than the image breaks librccl (ddai-gin-perf.log)." >&2
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
    # MLX5_SO=1: legacy multi-dir search for libmlx5* (old behavior). Default / adjacent: pair mlx5 with libibverbs below.
    if [[ "${RCCL_GIN_GDA_TEST2_BIND_HOST_MLX5_SO:-}" == 1 ]]; then
      _rccl_t2_mlx5_part="libmlx5.so libmlx5.so.1 libmlx5-infiniband.so.1 libmlx5dv.so libmlx5dv.so.1 "
    fi
    _rccl_t2_bases="${_rccl_t2_mlx5_part}libibverbs.so libibverbs.so.1 librdmacm.so librdmacm.so.1 libibumad.so.3 ${RCCL_GIN_GDA_TEST2_BIND_HOST_RDMA_EXTRA:-} libnl-3.so.200 libnl-route-3.so.200"
  fi
  for _rccl_t2_base in ${_rccl_t2_bases}; do
    [[ -n "${_rccl_t2_base// }" ]] || continue
    _rccl_gin_gda_host_so_mount_from_base "${_rccl_t2_base}" || true
  done
  # Adjacent mlx5: same rdma-core tree as host libibverbs (fixes RCCL dlvsym MLX5_1.25 on image-old libmlx5; ddai-gin-perf.log).
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
    # Slurm/cgroups often hide /dev/infiniband and /sys/class/infiniband/* from globs; dockerd still
    # has the real tree — bind when either path exists (ddai-gin-perf.log: compgen on sysfs was false).
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
  ${DOCKER_CMD} run ${DOCKER_GPU}${DOCKER_TEST2_VOLUMES} "${DOCKER_IMAGE}" \
    mpirun -n "${NP}" ${MPI_OPT} \
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

if [ 0 -eq 1 ]; then
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
fi

# done
