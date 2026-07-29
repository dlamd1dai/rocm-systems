#!/usr/bin/env bash
# One-command verify for the GIN-SDMA LL (low-latency) tiny-message design on the
# SUT, parameterized by collective. Primary SUT: smci355-ccs-aus-m03-17
# (8x MI355X, ~/rocm-systems, ~/rt-build).
#
# Collective is selected with the COLL env var (default sendrecv):
#   COLL=sendrecv  bash gin-sdma-ll-verify.bash [NP]   # ring send/recv LL
#   COLL=scatter   bash gin-sdma-ll-verify.bash [NP]   # root fan-out LL
#
# Runs, in order:
#   1. BUILD  <coll>_perf + gin_sdma_policy_test (incremental, no GPU)
#   2. UT     gin_sdma_policy_test  (host-only policy tiering tests, no GPU)
#   3. GATE   gin_sdma_gpu_functional.sh <coll>  (-c 1 correctness on NP GPUs;
#             exercises LL-default, GIN-forced, and LL-disabled passes)
#   4. AB     tiny-size LL-on vs LL-off latency sweep (NP GPUs)
#   5. BIG    (opt-in) forced-GIN -c 1 across the 1 GiB ginPutChunked segment
#             boundary -- proves the 30-bit SDMA copy-count fix. For SendRecv the
#             per-put size is the message; for Scatter it is the per-rank chunk
#             (= total/NP), so the BIG size defaults differ per collective.
#
# Every GPU container is given a unique --name and killed on EXIT, so a hung/
# timed-out run cannot leave zombie *_perf ranks holding GPU contexts (which
# otherwise cause the next run to fail with HIP 'invalid argument').
#
# Toggle steps:   RUN_BUILD=0 RUN_UT=0 RUN_GATE=0 RUN_AB=0 RUN_BIG=1
# Sizes/iters:    GATE_MAX_BYTES (default 1M), AB_MIN_BYTES/AB_MAX_BYTES,
#                 AB_ITERS (default 50), AB_WARMUP (default 20),
#                 BIG_MIN_BYTES/BIG_MAX_BYTES (per-collective defaults below)
# Knobs:          DEVICE_CTA_COUNT (-V, default 32), NUM_CHANNELS (default 1)
# Bare metal:     USE_DOCKER=0  (run binaries directly, no docker)
#
# Note: since ginPutChunked splits every GIN-tier put into <=1 GiB segments (the
# HW 30-bit copy-count max), the -D 3 GIN path is size-safe past 1 GiB. RUN_BIG
# is opt-in only because it needs large HBM + longer wall time, not because it is
# unsafe.

set -euo pipefail

NP="${1:-8}"
COLL="${COLL:-sendrecv}"

# ---- per-collective parameters ----------------------------------------------
# Each collective differs in: perf binary, UT filter, its LL-cap env var, its
# LSA<->GIN threshold env var, the A/B sweep range + extra perf flags, and the
# BIG chunk-boundary sweep range. SendRecv's LL cap is on the full message;
# Scatter's is on the per-rank chunk (= total/NP), so its A/B range is scaled by
# NP and it needs a root (-r 0). BIG likewise crosses the 1 GiB *per-put* edge:
# for Scatter that put is the per-rank chunk, so the total must reach NP GiB.
case "$COLL" in
  sendrecv)
    BIN="sendrecv_perf"
    UT_FILTER='*SendRecv*:*Move*'
    LL_ENV="NCCL_GIN_ANVIL_SENDRECV_LL_MAX_BYTES"
    THR_ENV="NCCL_GIN_ANVIL_SDMA_THRESHOLD_SENDRECV"
    AB_MIN_DEFAULT="8"; AB_MAX_DEFAULT="2048"     # LL cap 2 KiB on the message
    AB_EXTRA="-z 0"
    BIG_MIN_DEFAULT="512M"; BIG_MAX_DEFAULT="2G"  # 1G = one max seg, 2G = two segs
    BIG_EXTRA="-z 0"
    ;;
  scatter)
    BIN="scatter_perf"
    UT_FILTER='*Scatter*:*Move*'
    LL_ENV="NCCL_GIN_ANVIL_SCATTER_LL_MAX_BYTES"
    THR_ENV="NCCL_GIN_ANVIL_SDMA_THRESHOLD_SCATTER"
    AB_MIN_DEFAULT="64"; AB_MAX_DEFAULT="32K"     # LL cap 2 KiB/chunk => total <= NP*2 KiB
    AB_EXTRA="-r 0 -z 0"
    BIG_MIN_DEFAULT="4G"; BIG_MAX_DEFAULT="16G"   # chunk=total/NP; 8G=>1 GiB chunk, 16G=>2 GiB
    BIG_EXTRA="-r 0 -z 0"
    ;;
  *)
    echo "unsupported COLL='$COLL' (want: sendrecv|scatter)" >&2
    exit 2
    ;;
esac

# ---- paths (host side; mounted into the container) ----
SRC_HOST="${SRC_HOST:-$HOME/rocm-systems/projects/rccl-tests}"
RT_BUILD_HOST="${RT_BUILD_HOST:-$HOME/rt-build}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gin-gda-sdma-713}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
USE_DOCKER="${USE_DOCKER:-1}"

# ---- step toggles ----
RUN_BUILD="${RUN_BUILD:-1}"
RUN_UT="${RUN_UT:-1}"
RUN_GATE="${RUN_GATE:-1}"
RUN_AB="${RUN_AB:-1}"
RUN_BIG="${RUN_BIG:-0}"        # opt-in >1 GiB chunk-boundary regression

# ---- sizes / iters ----
GATE_MAX_BYTES="${GATE_MAX_BYTES:-1M}"
GATE_ITERS="${GATE_ITERS:-10}"
GATE_WARMUP="${GATE_WARMUP:-3}"
AB_MIN_BYTES="${AB_MIN_BYTES:-$AB_MIN_DEFAULT}"
AB_MAX_BYTES="${AB_MAX_BYTES:-$AB_MAX_DEFAULT}"
AB_ITERS="${AB_ITERS:-50}"
AB_WARMUP="${AB_WARMUP:-20}"
BIG_MIN_BYTES="${BIG_MIN_BYTES:-$BIG_MIN_DEFAULT}"
BIG_MAX_BYTES="${BIG_MAX_BYTES:-$BIG_MAX_DEFAULT}"
BIG_ITERS="${BIG_ITERS:-2}"
BIG_WARMUP="${BIG_WARMUP:-1}"
FACTOR="${FACTOR:-2}"

# ---- device (-D 3) knobs (tuned 2026-07-28, 8x MI355X) ----
DEVICE_CTA_COUNT="${DEVICE_CTA_COUNT:-32}"   # -V; 32 near-optimal
NUM_CHANNELS="${NUM_CHANNELS:-1}"            # 1 is optimal; >=2 is correctness-safe but
                                             # perf-neutral (each peer already gets its own
                                             # per-peer SDMA queue, so the extra per-peer
                                             # channels are unused by a 1-chunk-per-peer
                                             # scatter fan-out). Measured 2026-07-28 on 8x
                                             # MI355X: NC=1/2/4 -> 178.6/180.4/176.8 GB/s avg.
GIN_RANKS="${GIN_RANKS:-2}"                  # -R symmetric register mode (required for -D 3)

# ---- per-step timeouts (host-side wall clock, seconds) ----
BUILD_TIMEOUT="${BUILD_TIMEOUT:-600}"
UT_TIMEOUT="${UT_TIMEOUT:-120}"
GATE_TIMEOUT="${GATE_TIMEOUT:-600}"
AB_TIMEOUT="${AB_TIMEOUT:-300}"
BIG_TIMEOUT="${BIG_TIMEOUT:-1200}"

_HOST="$(hostname -s 2>/dev/null || hostname)"
RUN_TAG="${COLL}llverify-$$"

DOCKER_GPU="--rm --init --ulimit memlock=-1:-1 --shm-size 64G --network host --device /dev/dri --device /dev/kfd --ipc host --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged"
if [[ -e /dev/infiniband ]]; then DOCKER_GPU+=" --device /dev/infiniband"; fi
if getent group rdma >/dev/null 2>&1; then DOCKER_GPU+=" --group-add rdma"; fi

MPI_OPT="${MPI_OPT:---allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none}"

# GIN Anvil-SDMA runtime env (NCCL_GIN_TYPE=6), matching the gate scripts.
MPI_BASE=(
  -x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  -x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc
  -x ROCSHMEM_DISABLE_MIXED_IPC=1 -x ROCSHMEM_DEBUG_LEVEL=info:noversion
  -x NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1 -x NCCL_MSCCL_ENABLE=0
  -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_GIN_PLUGIN=none -x NCCL_CUMEM_ENABLE=1
  -x NCCL_NET_PLUGIN=none -x ROCSHMEM_SDMA_ENABLED=0
  -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
  -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS="${NUM_CHANNELS}"
  -x HSA_FORCE_FINE_GRAIN_PCIE=1
)

# ---- container lifecycle: kill only OUR containers on exit ----
_cleanup() {
  local rc=$?
  if [[ "${USE_DOCKER}" == "1" ]]; then
    local ids
    ids="$(${DOCKER_CMD} ps -q --filter "name=${RUN_TAG}" 2>/dev/null || true)"
    if [[ -n "${ids}" ]]; then
      echo ">>> cleanup: killing stray ${RUN_TAG} containers"
      # shellcheck disable=SC2086
      ${DOCKER_CMD} kill ${ids} >/dev/null 2>&1 || true
    fi
  fi
  exit "${rc}"
}
trap _cleanup EXIT INT TERM

# _cpu_run <name> <timeout> -- <cmd...>   (no GPU devices; source+build mounts)
_cpu_run() {
  local name="$1" tmo="$2"; shift 2; [[ "$1" == "--" ]] && shift
  if [[ "${USE_DOCKER}" == "1" ]]; then
    timeout "${tmo}" ${DOCKER_CMD} run --rm --name "${name}" \
      -v "${SRC_HOST}:/src-tests" -v "${RT_BUILD_HOST}:/rt-build" \
      "${DOCKER_IMAGE}" bash -lc "$*"
  else
    timeout "${tmo}" bash -lc "$*"
  fi
}

# _gpu_run <name> <timeout> -- <cmd...>   (GPU devices; source+build mounts)
_gpu_run() {
  local name="$1" tmo="$2"; shift 2; [[ "$1" == "--" ]] && shift
  if [[ "${USE_DOCKER}" == "1" ]]; then
    timeout "${tmo}" ${DOCKER_CMD} run ${DOCKER_GPU} --name "${name}" \
      -v "${SRC_HOST}:/src-tests" -v "${RT_BUILD_HOST}:/rt-build" \
      "${DOCKER_IMAGE}" bash -lc "$*"
  else
    timeout "${tmo}" bash -lc "$*"
  fi
}

echo "=================================================================="
echo "GIN-SDMA LL verify: COLL=${COLL} NP=${NP} host=${_HOST} tag=${RUN_TAG}"
echo "  src=${SRC_HOST}  build=${RT_BUILD_HOST}  image=${DOCKER_IMAGE}"
echo "=================================================================="

# ---- 1. BUILD ----------------------------------------------------------------
if [[ "${RUN_BUILD}" != "0" ]]; then
  echo ">>> [1/4] BUILD ${BIN} + gin_sdma_policy_test"
  _cpu_run "${RUN_TAG}-build" "${BUILD_TIMEOUT}" -- \
    "cmake --build /rt-build --target ${BIN} gin_sdma_policy_test -j 8 2>&1 | grep -viE 'ANVIL_ENABLED'"
  echo ">>> [1/4] BUILD ok"
fi

# ---- 2. POLICY UT ------------------------------------------------------------
if [[ "${RUN_UT}" != "0" ]]; then
  echo ">>> [2/4] POLICY UT (${COLL} + Move tiers)"
  _cpu_run "${RUN_TAG}-ut" "${UT_TIMEOUT}" -- \
    "/rt-build/gin_sdma_policy_test --gtest_filter=\"${UT_FILTER}\""
  echo ">>> [2/4] POLICY UT ok"
fi

# ---- 3. FUNCTIONAL GATE (-c 1) ----------------------------------------------
if [[ "${RUN_GATE}" != "0" ]]; then
  echo ">>> [3/4] FUNCTIONAL GATE ${COLL} (-c 1), max=${GATE_MAX_BYTES}"
  _gpu_run "${RUN_TAG}-gate" "${GATE_TIMEOUT}" -- \
    "GIN_SDMA_MAX_BYTES=${GATE_MAX_BYTES} GIN_SDMA_ITERS=${GATE_ITERS} GIN_SDMA_WARMUP=${GATE_WARMUP} \
     GIN_SDMA_CTA=${DEVICE_CTA_COUNT} NUM_CHANNELS=${NUM_CHANNELS} \
     bash /src-tests/test/gin_sdma_gpu_functional.sh ${COLL} /rt-build ${NP} mpirun"
  echo ">>> [3/4] FUNCTIONAL GATE ok"
fi

# ---- 4. LL A/B LATENCY -------------------------------------------------------
if [[ "${RUN_AB}" != "0" ]]; then
  echo ">>> [4/4] LL A/B latency sweep ${AB_MIN_BYTES}..${AB_MAX_BYTES} (${AB_ITERS} iters)"
  SR="/rt-build/${BIN}"
  PERF_ARGS="-b ${AB_MIN_BYTES} -e ${AB_MAX_BYTES} -f ${FACTOR} -g 1 -R ${GIN_RANKS} -V ${DEVICE_CTA_COUNT} -D 3 -c 1 -n ${AB_ITERS} -w ${AB_WARMUP} ${AB_EXTRA}"
  _gpu_run "${RUN_TAG}-ab" "${AB_TIMEOUT}" -- "
    echo '==== ${COLL} LL ON (default cutover) ===='
    mpirun -n ${NP} ${MPI_OPT} ${MPI_BASE[*]} ${SR} ${PERF_ARGS}
    echo
    echo '==== ${COLL} LL OFF (${LL_ENV}=0) ===='
    mpirun -n ${NP} ${MPI_OPT} ${MPI_BASE[*]} -x ${LL_ENV}=0 ${SR} ${PERF_ARGS}
  "
  echo ">>> [4/4] LL A/B ok  (compare the 'time' (us) column: LL should be lower at the tiny sizes)"
fi

# ---- 5. BIG chunk-boundary regression (opt-in) ------------------------------
# Forces the GIN tier (${THR_ENV}=0) at sizes that cross the 1 GiB ginPutChunked
# segment edge, so the previously-corrupting >1 GiB per-put case is validated
# end-to-end (-c 1). Expect "# Out of bounds values : 0 OK" on every row.
if [[ "${RUN_BIG}" != "0" ]]; then
  echo ">>> [5/5] BIG forced-GIN regression ${BIG_MIN_BYTES}..${BIG_MAX_BYTES} (-c 1)"
  SR="/rt-build/${BIN}"
  BIG_ARGS="-b ${BIG_MIN_BYTES} -e ${BIG_MAX_BYTES} -f ${FACTOR} -g 1 -R ${GIN_RANKS} -V ${DEVICE_CTA_COUNT} -D 3 -c 1 -n ${BIG_ITERS} -w ${BIG_WARMUP} ${BIG_EXTRA}"
  _gpu_run "${RUN_TAG}-big" "${BIG_TIMEOUT}" -- "
    echo '==== ${COLL} forced-GIN, crossing 1 GiB per-put (ginPutChunked segments) ===='
    mpirun -n ${NP} ${MPI_OPT} ${MPI_BASE[*]} -x ${THR_ENV}=0 ${SR} ${BIG_ARGS}
  "
  echo ">>> [5/5] BIG ok  (every row must show '# Out of bounds values : 0 OK')"
fi

echo "=================================================================="
echo "PASS: gin-sdma-ll-verify  COLL=${COLL} NP=${NP}  steps: build=${RUN_BUILD} ut=${RUN_UT} gate=${RUN_GATE} ab=${RUN_AB} big=${RUN_BIG}"
echo "=================================================================="
