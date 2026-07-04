#! /usr/bin/env bash
# Preflight checks before docker-gin-gda-sdma*.bash build. Source from repo root.

_preflight_fail=0

_preflight_require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing ${path}" >&2
    _preflight_fail=1
  fi
}

# Required after GIN-rocSHMEM-API removal (type 4): shared init for GDA + Anvil plugins.
_preflight_require_file projects/rccl/src/include/gin/gin_host_rocshmem_common.h
_preflight_require_file projects/rccl/src/gin/gin_host_rocshmem_common.cc

# Removed with type 4; must not reappear in the tree.
for stale in \
  projects/rccl/src/gin/gin_plugin_rocshmem_api.cc \
  projects/rccl/src/include/gin/gin_host_rocshmem_api.h \
  projects/rccl/src/include/nccl_device/gin/rocshmem_api/gin_rocshmem_api.h; do
  if [[ -f "${stale}" ]]; then
    echo "ERROR: stale file should be removed: ${stale}" >&2
    _preflight_fail=1
  fi
done

# Catch merged/corrupt gin_anvil_sdma.h (duplicate helper definitions broke rccl-tests).
_anvil_header=projects/rccl/src/include/nccl_device/gin/anvil_sdma/gin_anvil_sdma.h
if [[ -f "${_anvil_header}" ]]; then
  _dup_count=$(grep -c 'NCCL_DEVICE_INLINE void markSdmaDirty' "${_anvil_header}" || true)
  if [[ "${_dup_count}" -gt 1 ]]; then
    echo "ERROR: ${_anvil_header} has ${_dup_count} markSdmaDirty definitions (expected 1)" >&2
    _preflight_fail=1
  fi
fi

# Host Open MPI dev packages (libopenmpi-dev).  Docker image builds carry their own MPI;
# the host only needs these for bare-metal mpirun or optional rocSHMEM unit suite F.
# GIN_ANVIL_PREFLIGHT_MPI: skip | warn (default from gin-anvil-smci355-test.bash) | require
_preflight_find_mpi_h() {
  local cand inc
  if command -v mpicc >/dev/null 2>&1; then
    inc="$(mpicc -showme:incdirs 2>/dev/null | awk '{print $1}')"
    if [[ -n "${inc}" && -f "${inc}/mpi.h" ]]; then
      echo "${inc}/mpi.h"
      return 0
    fi
  fi
  for cand in \
    /usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h \
    /usr/lib64/openmpi/include/mpi.h \
    /opt/openmpi/include/mpi.h \
    /usr/include/mpi.h \
    /usr/local/include/mpi.h; do
    if [[ -f "${cand}" ]]; then
      echo "${cand}"
      return 0
    fi
  done
  return 1
}

_preflight_check_openmpi_dev() {
  local mode="${GIN_ANVIL_PREFLIGHT_MPI:-skip}"
  local _mpi_h=""
  case "${mode}" in
    skip|0|off|false) return 0 ;;
    warn|warning|1) ;;
    require|required|error|fail) ;;
    *)
      echo "ERROR: GIN_ANVIL_PREFLIGHT_MPI must be skip, warn, or require (got ${mode})" >&2
      _preflight_fail=1
      return 0
      ;;
  esac

  if _mpi_h="$(_preflight_find_mpi_h)"; then
    echo "OK: Open MPI dev headers found (${_mpi_h})"
    return 0
  fi

  if command -v dpkg >/dev/null 2>&1 && dpkg -s libopenmpi-dev >/dev/null 2>&1; then
    echo "OK: libopenmpi-dev package installed (mpi.h path not in default search list)"
    return 0
  fi

  local _mpi_msg="Open MPI dev packages not found on host (need mpi.h from libopenmpi-dev).
  Install on Ubuntu 22.04:
    sudo apt-get install -y openmpi-bin libopenmpi-dev
  Suite F (rocSHMEM factory unit tests) is off by default; docker integration does not need host MPI.
  Set GIN_ANVIL_BUILD_SUITE_F=1 or GIN_ANVIL_LAYOUT=bare-metal to require host MPI."

  case "${mode}" in
    warn|warning|1)
      echo "WARNING: ${_mpi_msg}" >&2
      ;;
    require|required|error|fail)
      echo "ERROR: ${_mpi_msg}" >&2
      _preflight_fail=1
      ;;
  esac
}

_preflight_check_openmpi_dev

if [[ "${_preflight_fail}" -ne 0 ]]; then
  echo "Preflight failed; fix the above before running docker build." >&2
  return 1 2>/dev/null || exit 1
fi
