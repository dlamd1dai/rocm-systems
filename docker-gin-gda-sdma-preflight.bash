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

if [[ "${_preflight_fail}" -ne 0 ]]; then
  echo "Preflight failed; fix the above before running docker build." >&2
  return 1 2>/dev/null || exit 1
fi
