#! /usr/bin/env bash
# Bare-metal 2-node / 8-GPU (2p8g) GIN Anvil-SDMA AllToAll on MI455.
#
# This is the 1p4g script's counterpart: host mpirun against /dev/kfd on both
# nodes (no docker run of alltoall_perf). Optional docker is used only to stage
# PR binaries into ${GIN_A2A_STAGE}, same as gin-sdma-a2a-mi455-baremetal-func.bash.
# 8 MPI ranks: 4 per node, 1 GPU each. NCCL_GIN_TYPE=6, GinAlltoAllKernel -D 3.
#
# Launch from node 1. Binaries must sit at the same STAGE path on both nodes
# (shared NFS, or this script rsyncs the stage to HOST2).
#
# Usage:
#   HOST2=ctheliosp-rck-g02-j07-02.rck.dcgpu bash gin-sdma-a2a-mi455-baremetal-2p8g.bash
#   HOSTS=host1:4,host2:4 MAX_BYTES=64M WARMUP=0 ITERS=1 bash gin-sdma-a2a-mi455-baremetal-2p8g.bash
#   HOSTFILE=/path/hostfile bash gin-sdma-a2a-mi455-baremetal-2p8g.bash
#   NCCL_IB_DISABLE=0 bash gin-sdma-a2a-mi455-baremetal-2p8g.bash   # if verbs/RoCE is healthy
#
# Fabric LL arms only when the 8 GPUs are one MNNVL clique. Otherwise RCCL
# still runs AllToAll via gin.put/SDMA (or host path if D_MODE=0).
#
# Exit 0 only if rccl-tests prints "Out of bounds values : 0 OK" and no GPU
# hang / NCCL error / Test failure lines.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE1P="${HERE}/gin-sdma-a2a-mi455-baremetal-func.bash"

NP="${NP:-8}"
PPN="${PPN:-4}"
MIN_BYTES="${MIN_BYTES:-128}"
MAX_BYTES="${MAX_BYTES:-4G}"
WARMUP="${WARMUP:-${TEST5_WARMUP:-1}}"
ITERS="${ITERS:-${TEST5_ITERS:-2}}"
CTA_COUNT="${TEST5_D3_CTA_COUNT:-${TEST5_CTA_COUNT:-1}}"
D_MODE="${D_MODE:-3}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gin-sdma-a2a-mi455}"
STAGE="${GIN_A2A_STAGE:-${HOME}/gin-a2a-baremetal-stage}"
LOGDIR="${LOGDIR:-${HOME}/gin-a2a-runs/2p8g-$(date +%Y%m%d-%H%M%S)}"
HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
HOST1="${HOST1:-$(hostname -s)}"
HOST2="${HOST2:-}"
HOSTS="${HOSTS:-}"
HOSTFILE="${HOSTFILE:-}"
SYNC_STAGE="${SYNC_STAGE:-1}"
SSH="${SSH:-ssh}"
RSYNC="${RSYNC:-rsync}"

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

mkdir -p "${LOGDIR}"
LOG="${LOGDIR}/alltoall-2p8g.log"

if [[ -z "${HOSTS}" && -z "${HOSTFILE}" ]]; then
  if [[ -z "${HOST2}" ]]; then
    echo "error: set HOST2=<peer>, HOSTS=h1:${PPN},h2:${PPN}, or HOSTFILE=..." >&2
    exit 1
  fi
  HOSTS="${HOST1}:${PPN},${HOST2}:${PPN}"
fi

if [[ "${NP}" != "8" ]]; then
  echo "WARN: 2p8g default is 8 ranks; NP=${NP}" >&2
fi

# Same staging as 1p4g (image -> ${STAGE}) so both nodes run the PR binary.
if [[ -z "${ALLTOALL_PERF:-}" ]]; then
  [[ -x "${STAGE1P}" ]] || {
    echo "error: missing ${STAGE1P}; set ALLTOALL_PERF" >&2
    exit 1
  }
  GIN_A2A_STAGE_ONLY=1 GIN_A2A_STAGE="${STAGE}" DOCKER_IMAGE="${DOCKER_IMAGE}" \
    bash "${STAGE1P}"
fi

STAGE_PERF="${STAGE}/workspace/rccl-tests/alltoall_perf"
STAGE_MPI="${STAGE}/usr/lib64/openmpi"
ALLTOALL_PERF="${ALLTOALL_PERF:-${STAGE_PERF}}"
[[ -x "${ALLTOALL_PERF}" ]] || {
  echo "error: ${ALLTOALL_PERF} is not executable" >&2
  exit 1
}

MPIRUN="${MPIRUN:-}"
if [[ -z "${MPIRUN}" ]]; then
  if [[ -x "${STAGE_MPI}/bin/orterun" ]]; then
    MPIRUN="${STAGE_MPI}/bin/orterun"
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

if [[ -d "${STAGE}/opt/rocm/lib" ]]; then
  ROCM_LIBDIR="${ROCM_LIBDIR:-${STAGE}/opt/rocm/lib}"
  export ROCM_PATH="${ROCM_PATH:-${STAGE}/opt/rocm}"
  export HIP_PATH="${HIP_PATH:-${STAGE}/opt/rocm}"
else
  ROCM_LIBDIR="${ROCM_LIBDIR:-${HOST_ROCM:-/opt/rocm}/lib}"
fi
RCCL_LIBDIR="${RCCL_LIBDIR:-${STAGE}/workspace/rccl/lib}"
MPI_LIBDIR="${MPI_LIBDIR:-${STAGE_MPI}/lib}"
export LD_LIBRARY_PATH="${RCCL_LIBDIR}:${MPI_LIBDIR}:${ROCM_LIBDIR}:${ROCM_LIBDIR}/rocm_sysdeps/lib:${STAGE}/sysdeps:${LD_LIBRARY_PATH:-}"
export PATH="$(dirname "${MPIRUN}"):${PATH}"

_host_list() {
  if [[ -n "${HOSTFILE}" ]]; then
    awk 'NF && $1 !~ /^#/ { print $1 }' "${HOSTFILE}"
    return
  fi
  local tok host
  IFS=',' read -ra tok <<<"${HOSTS}"
  for host in "${tok[@]}"; do
    echo "${host%%:*}"
  done
}

_is_local() {
  local h="$1" me short fq
  me="$(hostname)"
  short="$(hostname -s)"
  fq="$(hostname -f 2>/dev/null || true)"
  [[ "${h}" == "${me}" || "${h}" == "${short}" || "${h}" == "${fq}" || "${h}" == "localhost" ]]
}

if [[ "${SYNC_STAGE}" == 1 && -d "${STAGE}" ]]; then
  while read -r h; do
    [[ -n "${h}" ]] || continue
    _is_local "${h}" && continue
    echo "Syncing ${STAGE} -> ${h}:${STAGE}"
    "${SSH}" "${h}" "mkdir -p '${STAGE}'"
    "${RSYNC}" -a --delete "${STAGE}/" "${h}:${STAGE}/"
  done < <(_host_list)
fi

while read -r h; do
  [[ -n "${h}" ]] || continue
  if _is_local "${h}"; then
    [[ -e /dev/kfd ]] || { echo "error: /dev/kfd missing on $(hostname)" >&2; exit 1; }
  else
    "${SSH}" "${h}" "test -e /dev/kfd" || {
      echo "error: /dev/kfd missing on ${h}" >&2
      exit 1
    }
  fi
done < <(_host_list)

if [[ "${NCCL_IB_DISABLE}" == 1 ]]; then
  echo "NOTE: NCCL_IB_DISABLE=1 (MI455 verbs often SEGV). Inter-node AllToAll then needs MNNVL/fabric. Set NCCL_IB_DISABLE=0 if RoCE/IB is healthy."
fi

MPI_OPT=(-n "${NP}"
  --map-by "ppr:${PPN}:node"
  --bind-to none
  -mca pml ob1 -mca btl self,vader,tcp
  -mca btl_vader_single_copy_mechanism none
  -mca hwloc_base_binding_policy none)
if [[ -n "${HOSTFILE}" ]]; then
  MPI_OPT+=(--hostfile "${HOSTFILE}")
else
  MPI_OPT+=(-H "${HOSTS}")
fi

echo "=== GIN-SDMA A2A 2p8g bare-metal functional ==="
echo "hosts=${HOSTS:-hostfile:${HOSTFILE}} NP=${NP} PPN=${PPN} GPUs=${HIP_VISIBLE_DEVICES}"
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
echo "PASS: AllToAll -D ${D_MODE} ${MIN_BYTES}..${MAX_BYTES} 2p8g #wrong=0"
exit 0
