#! /usr/bin/env bash
# Baremetal 1-process-per-GPU / 4-GPU functional sweep of GIN Anvil-SDMA AllToAll
# (NCCL_GIN_TYPE=6, GinAlltoAllKernel -D 3) from 128 B to 4 GiB.
#
# Intended SUT: ctheliosp-1b112-a43-1.mnb.dcgpu (ssh alias mi455-sut).
# This script launches host mpirun against /dev/kfd (no docker run). By default
# it stages alltoall_perf, librccl, OpenMPI, and the image ROCm userland from
# DOCKER_IMAGE so the binary matches the PR build. Override ALLTOALL_PERF and
# RCCL_LIBDIR to use a local tree instead.
#
# Usage:
#   bash gin-sdma-a2a-mi455-baremetal-func.bash
#   MAX_BYTES=64M WARMUP=0 ITERS=1 bash gin-sdma-a2a-mi455-baremetal-func.bash
#   ALLTOALL_PERF=/path/alltoall_perf RCCL_LIBDIR=/path/lib bash ...
#   ROCM_FROM_HOST=1 bash ...        # skip the ~9 GB image ROCm copy, use /opt/rocm
#
# Exit 0 only if rccl-tests prints "Out of bounds values : 0 OK" and no GPU
# hang / NCCL error / Test failure lines.

set -euo pipefail

NP="${NP:-4}"
MIN_BYTES="${MIN_BYTES:-128}"
MAX_BYTES="${MAX_BYTES:-4G}"
WARMUP="${WARMUP:-${TEST5_WARMUP:-1}}"
ITERS="${ITERS:-${TEST5_ITERS:-2}}"
CTA_COUNT="${TEST5_D3_CTA_COUNT:-${TEST5_CTA_COUNT:-1}}"
# -D 3 is the GinAlltoAllKernel device-API launcher; D_MODE=0 runs the host path
# as a control when you want to check the harness rather than the GIN kernel.
D_MODE="${D_MODE:-3}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gin-sdma-a2a-mi455}"
STAGE="${GIN_A2A_STAGE:-${HOME}/gin-a2a-baremetal-stage}"
LOGDIR="${LOGDIR:-${HOME}/gin-a2a-runs/baremetal-$(date +%Y%m%d-%H%M%S)}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"

export HIP_VISIBLE_DEVICES
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_PLUGIN="${NCCL_NET_PLUGIN:-none}"
export NCCL_ENV_PLUGIN="${NCCL_ENV_PLUGIN:-none}"
export NCCL_GIN_PLUGIN="${NCCL_GIN_PLUGIN:-none}"
export NCCL_GIN_ENABLE="${NCCL_GIN_ENABLE:-1}"
export NCCL_GIN_TYPE="${NCCL_GIN_TYPE:-6}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export NCCL_CUMEM_SKIP_FREE="${NCCL_CUMEM_SKIP_FREE:-1}"
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-1}"
export NCCL_MSCCL_ENABLE="${NCCL_MSCCL_ENABLE:-0}"
export RCCL_ENABLE_INTRANET="${RCCL_ENABLE_INTRANET:-1}"
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"
export HSA_FORCE_FINE_GRAIN_PCIE="${HSA_FORCE_FINE_GRAIN_PCIE:-1}"
export NCCL_GIN_ANVIL_SDMA_THRESHOLD="${NCCL_GIN_ANVIL_SDMA_THRESHOLD:-128}"
export NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS="${NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS:-1}"
export RCCL_DDA_LL_THRESHOLD="${RCCL_DDA_LL_THRESHOLD:-65536}"
export RCCL_ROCSHMEM_ENABLE="${RCCL_ROCSHMEM_ENABLE:-0}"
export ROCSHMEM_SDMA_ENABLED="${ROCSHMEM_SDMA_ENABLED:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

mkdir -p "${LOGDIR}" "${STAGE}"
LOG="${LOGDIR}/alltoall-128B-4G.log"

_have_pr_bin() {
  local p="${1:-}"
  [[ -n "${p}" && -x "${p}" ]] || return 1
  command -v nm >/dev/null 2>&1 || return 0
  # grep -c rather than -q: -q exits early and SIGPIPEs nm, which pipefail reports as failure.
  nm -C "${p}" 2>/dev/null | grep -c "T gin_anvil_sdma_probe" >/dev/null
}

# Mirror the image paths under STAGE rather than flattening into one lib dir:
# mpirun is a symlink to orterun, OpenMPI needs its share/ tree via OPAL_PREFIX,
# and ROCm resolves rocm_sysdeps/lib relative to /opt/rocm/lib.
STAGE_PERF="${STAGE}/workspace/rccl-tests/alltoall_perf"
STAGE_MPI="${STAGE}/usr/lib64/openmpi"
STAGE_ROCM="${STAGE}/opt/rocm"
STAGE_RCCL_LIB="${STAGE}/workspace/rccl/lib"

# OpenMPI links PMIx/UCX/libfabric from the image /lib64, which a bare host may not
# have. Copy only the ones the host is missing, dereferencing each soname symlink so
# the staged copy is a real file plus a local link.
_stage_mpi_sysdeps() {
  local cid="$1" orig real base
  mkdir -p "${STAGE}.new/sysdeps"
  while read -r orig real; do
    [[ -n "${orig}" && -n "${real}" ]] || continue
    [[ -e "${orig}" ]] && continue
    base=$(basename "${real}")
    docker cp "${cid}:${real}" "${STAGE}.new/sysdeps/${base}" >/dev/null 2>&1 || continue
    [[ "${base}" == "$(basename "${orig}")" ]] || ln -sf "${base}" "${STAGE}.new/sysdeps/$(basename "${orig}")"
  done < <(docker run --rm --init "${DOCKER_IMAGE}" bash -lc '
    { ldd /usr/lib64/openmpi/bin/orterun
      for f in /usr/lib64/openmpi/lib/*.so* /usr/lib64/openmpi/lib/openmpi/*.so /usr/lib64/pmix/*.so; do
        ldd "$f" 2>/dev/null
      done
    } | awk "/=> \//{print \$3}" | sort -u | grep -v openmpi |
    while read -r p; do printf "%s %s\n" "$p" "$(readlink -f "$p")"; done' 2>/dev/null)
}

_stage_from_image() {
  local id stamp cid
  command -v docker >/dev/null 2>&1 || {
    echo "error: ALLTOALL_PERF not set and docker is missing; cannot stage ${DOCKER_IMAGE}" >&2
    return 1
  }
  docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1 || {
    echo "error: docker image '${DOCKER_IMAGE}' not found; load it or set ALLTOALL_PERF" >&2
    return 1
  }
  id=$(docker image inspect "${DOCKER_IMAGE}" --format '{{.Id}}')
  stamp="${STAGE}/.image-id"
  if [[ -x "${STAGE_PERF}" && -x "${STAGE_MPI}/bin/orterun" && -d "${STAGE}/sysdeps" \
        && -d "${STAGE}/usr/lib64/pmix" && -f "${stamp}" && "$(cat "${stamp}")" == "${id}" ]]; then
    echo "Using cached stage ${STAGE} for ${DOCKER_IMAGE}"
    return 0
  fi
  echo "Staging PR binaries from ${DOCKER_IMAGE} -> ${STAGE} (ROCm copy is ~9 GB on first run)"
  rm -rf "${STAGE}.new"
  mkdir -p "${STAGE}.new/workspace/rccl-tests" "${STAGE}.new/workspace/rccl" \
           "${STAGE}.new/usr/lib64" "${STAGE}.new/opt"
  cid=$(docker create "${DOCKER_IMAGE}")
  docker cp "${cid}:/workspace/rccl-tests/alltoall_perf" "${STAGE}.new/workspace/rccl-tests/alltoall_perf"
  docker cp "${cid}:/workspace/rccl/lib" "${STAGE}.new/workspace/rccl/"
  # Whole directories: symlinks such as bin/mpirun -> orterun resolve inside the copy.
  docker cp "${cid}:/usr/lib64/openmpi" "${STAGE}.new/usr/lib64/"
  # PMIx plugins and help text, found via PMIX_INSTALL_PREFIX at run time.
  mkdir -p "${STAGE}.new/usr/share"
  docker cp "${cid}:/usr/lib64/pmix" "${STAGE}.new/usr/lib64/"
  docker cp "${cid}:/usr/share/pmix" "${STAGE}.new/usr/share/" 2>/dev/null || true
  _stage_mpi_sysdeps "${cid}"
  if [[ "${ROCM_FROM_HOST:-0}" == 1 ]]; then
    echo "ROCM_FROM_HOST=1: using host ${HOST_ROCM:-/opt/rocm} instead of the image ROCm"
  else
    docker cp "${cid}:/opt/rocm" "${STAGE}.new/opt/"
  fi
  docker rm -f "${cid}" >/dev/null
  chmod +x "${STAGE}.new/workspace/rccl-tests/alltoall_perf"
  echo "${id}" > "${STAGE}.new/.image-id"
  rm -rf "${STAGE}.old"
  if [[ -d "${STAGE}" ]]; then mv "${STAGE}" "${STAGE}.old"; fi
  mv "${STAGE}.new" "${STAGE}"
  rm -rf "${STAGE}.old"
}

if [[ -z "${ALLTOALL_PERF:-}" ]]; then
  # _stage_from_image is a no-op when the stage already matches the image id.
  _stage_from_image
  ALLTOALL_PERF="${STAGE_PERF}"
fi
_have_pr_bin "${ALLTOALL_PERF}" || {
  echo "error: ${ALLTOALL_PERF} is not a GIN Anvil-SDMA alltoall_perf (missing gin_anvil_sdma_probe)" >&2
  exit 1
}

MPIRUN="${MPIRUN:-}"
if [[ -z "${MPIRUN}" ]]; then
  if [[ -x "${STAGE_MPI}/bin/orterun" ]]; then
    MPIRUN="${STAGE_MPI}/bin/orterun"
    # OpenMPI and PMIx relocated out of /usr find their plugins only via these prefixes.
    export OPAL_PREFIX="${STAGE_MPI}"
    export PMIX_PREFIX="${STAGE}/usr"
    export PMIX_EXEC_PREFIX="${STAGE}/usr"
    export PMIX_LIBDIR="${STAGE}/usr/lib64"
    export PMIX_DATADIR="${STAGE}/usr/share"
    export PMIX_DATAROOTDIR="${STAGE}/usr/share"
  else
    MPIRUN="$(command -v mpirun || true)"
  fi
fi
[[ -n "${MPIRUN}" && -x "${MPIRUN}" ]] || {
  echo "error: mpirun not found; set MPIRUN or stage from ${DOCKER_IMAGE}" >&2
  exit 1
}

ROCM_LIBDIR="${ROCM_LIBDIR:-}"
if [[ -z "${ROCM_LIBDIR}" ]]; then
  if [[ -d "${STAGE_ROCM}/lib" ]]; then
    ROCM_LIBDIR="${STAGE_ROCM}/lib"
    export ROCM_PATH="${ROCM_PATH:-${STAGE_ROCM}}"
    export HIP_PATH="${HIP_PATH:-${STAGE_ROCM}}"
  else
    ROCM_LIBDIR="${HOST_ROCM:-/opt/rocm}/lib"
  fi
fi

RCCL_LIBDIR="${RCCL_LIBDIR:-${STAGE_RCCL_LIB}}"
MPI_LIBDIR="${MPI_LIBDIR:-${STAGE_MPI}/lib}"
# sysdeps last: host copies of these libraries win when they exist.
export LD_LIBRARY_PATH="${RCCL_LIBDIR}:${MPI_LIBDIR}:${ROCM_LIBDIR}:${ROCM_LIBDIR}/rocm_sysdeps/lib:${STAGE}/sysdeps:${LD_LIBRARY_PATH:-}"
export PATH="$(dirname "${MPIRUN}"):${PATH}"

if command -v ldd >/dev/null 2>&1 && ldd "${ALLTOALL_PERF}" 2>/dev/null | grep -c "not found" >/dev/null; then
  echo "error: unresolved shared libraries for ${ALLTOALL_PERF}:" >&2
  ldd "${ALLTOALL_PERF}" 2>/dev/null | grep "not found" >&2
  exit 1
fi

[[ -e /dev/kfd ]] || { echo "error: /dev/kfd missing; this is not a GPU node" >&2; exit 1; }

# Used by the 2p8g launcher to reuse this script's image staging without running.
if [[ "${GIN_A2A_STAGE_ONLY:-0}" == 1 ]]; then
  echo "Staged ALLTOALL_PERF=${ALLTOALL_PERF}"
  echo "Staged MPIRUN=${MPIRUN}"
  echo "Staged STAGE=${STAGE}"
  exit 0
fi

MPI_OPT=(--allow-run-as-root -n "${NP}"
  -mca pml ob1 -mca btl self,vader,tcp
  -mca btl_vader_single_copy_mechanism none
  -mca hwloc_base_binding_policy none)

echo "=== GIN-SDMA A2A baremetal functional ==="
echo "SUT=$(hostname) NP=${NP} GPUs=${HIP_VISIBLE_DEVICES}"
echo "bin=${ALLTOALL_PERF}"
echo "mpirun=${MPIRUN}"
echo "range=${MIN_BYTES}..${MAX_BYTES} factor=2 warmup=${WARMUP} iters=${ITERS} -D ${D_MODE} -V ${CTA_COUNT}"
echo "log=${LOG}"

set +e
"${MPIRUN}" "${MPI_OPT[@]}" \
  -x HIP_VISIBLE_DEVICES \
  -x LD_LIBRARY_PATH \
  -x PATH \
  ${OPAL_PREFIX:+-x OPAL_PREFIX} \
  ${PMIX_PREFIX:+-x PMIX_PREFIX -x PMIX_EXEC_PREFIX -x PMIX_LIBDIR -x PMIX_DATADIR -x PMIX_DATAROOTDIR} \
  ${ROCM_PATH:+-x ROCM_PATH} \
  ${HIP_PATH:+-x HIP_PATH} \
  -x OMPI_ALLOW_RUN_AS_ROOT \
  -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM \
  -x NCCL_IB_DISABLE \
  -x NCCL_NET_PLUGIN \
  -x NCCL_ENV_PLUGIN \
  -x NCCL_GIN_PLUGIN \
  -x NCCL_GIN_ENABLE \
  -x NCCL_GIN_TYPE \
  -x NCCL_CUMEM_ENABLE \
  -x NCCL_CUMEM_SKIP_FREE \
  -x NCCL_MNNVL_ENABLE \
  -x NCCL_DMABUF_ENABLE \
  -x NCCL_MSCCL_ENABLE \
  -x RCCL_ENABLE_INTRANET \
  -x HSA_NO_SCRATCH_RECLAIM \
  -x HSA_FORCE_FINE_GRAIN_PCIE \
  -x NCCL_GIN_ANVIL_SDMA_THRESHOLD \
  -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS \
  -x RCCL_DDA_LL_THRESHOLD \
  -x RCCL_ROCSHMEM_ENABLE \
  -x ROCSHMEM_SDMA_ENABLED \
  -x NCCL_DEBUG \
  -x NCCL_DEBUG_SUBSYS \
  ${RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL:+-x RCCL_GIN_FABRIC_LL_THRESHOLD_ALLTOALL} \
  ${RCCL_GIN_FABRIC_LL_A2A_MAX_BPP:+-x RCCL_GIN_FABRIC_LL_A2A_MAX_BPP} \
  "${ALLTOALL_PERF}" \
  -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f 2 -g 1 -R 2 -D "${D_MODE}" -A 1 -V "${CTA_COUNT}" \
  -w "${WARMUP}" -n "${ITERS}" \
  >"${LOG}" 2>&1
rc=$?
set -e

tail -n 40 "${LOG}"

if grep -qE "Memory access fault|GPU HANG|HSA_STATUS_ERROR|Test failure|NCCL error" "${LOG}"; then
  echo "FAIL: GPU/NCCL error in ${LOG}" >&2
  exit 1
fi
if ! grep -q "Out of bounds values : 0 OK" "${LOG}"; then
  echo "FAIL: missing 'Out of bounds values : 0 OK' (mpirun exit ${rc}) log=${LOG}" >&2
  exit 1
fi
if [[ "${rc}" -ne 0 ]]; then
  echo "FAIL: mpirun exit ${rc} even though bounds line matched; see ${LOG}" >&2
  exit 1
fi
echo "PASS: AllToAll -D ${D_MODE} ${MIN_BYTES}..${MAX_BYTES} 1p${NP}g #wrong=0"
exit 0
