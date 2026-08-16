#! /usr/bin/env bash
# Single-node GIN-SDMA AllReduce perf + correctness harness (docker).
# Companion to gin-sdma-a2a-test.bash, trimmed to the AllReduce paths.
#
# Usage: ./gin-sdma-ar-test.bash [NP] [MAX_BYTES]
#   NP         ranks / local GPUs (default 8)
#   MAX_BYTES  max message size for the -b..-e sweep (default 128M; up to 4G ok)
#   MIN_BYTES  env: min message size for the sweep (default 128)
#   GIN_CONN_RETRIES env: re-launch the GIN test up to N times on the intermittent
#              gfx950 cuMem-VMM connectivity-gate abort (default 5; 0 disables)
#
# Tests (RCCL_GIN_RUN_TESTS, comma list; default "0,5"):
#   0 = host baseline  : all_reduce_perf -D 0 (ncclAllReduce) reference busbw + datacheck
#   5 = GIN Anvil-SDMA : all_reduce_perf -D 5 single-launch (default) or -D 6 two-launch
#                        (AR_MODE=d6). NCCL_GIN_TYPE=5, single node, intra-node LSA + SDMA
#                        copy engines; no RDMA/NIC involved.
#
# The GIN-SDMA AllReduce is single-node (ReduceScatter -> device-wide barrier ->
# AllGather over LSA/SDMA), so unlike the GDA A2A tests this harness needs no
# InfiniBand / mlx5 / uverbs plumbing -- just the GPU devices.
#
# Correctness: all_reduce_perf runs datacheck (-c 1); the run FAILS loudly unless
# the trailing "# Out of bounds values : 0 OK" line is present (mirrors the image
# build hardening in docker-gin-gda-sdma-build.bash).

set -o pipefail

NP=${1:-8}
MAX_BYTES="${2:-128M}"
MIN_BYTES="${MIN_BYTES:-128}"
DOCKER_CMD="${DOCKER_CMD:-docker}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gin-gda-sdma-713}"
RCCL_GIN_RUN_TESTS="${RCCL_GIN_RUN_TESTS:-${RUN_TESTS:-0,5}}"
ROCSHMEM_THRESHOLD=$((128 * 1024 * 1024))

# Shared warmup(skip)/timed(loop) counts applied to every test so the host and
# GIN paths are compared over the SAME iteration window. Also feed the -D 5
# in-kernel wall_clock64 skip/loop when device timing is on.
AR_WARMUP="${AR_WARMUP:-5}"
AR_ITERS="${AR_ITERS:-20}"

# AllReduce op / datatype (rccl-tests defaults: sum / float). Override to sweep.
AR_OP="${AR_OP:-sum}"
AR_DTYPE="${AR_DTYPE:-float}"

# -V grid CTA count for the GIN device kernel (size-adaptive up to kArMaxCtas=64
# inside the kernel; -V sets the requested/registered grid). 8 matches the A2A
# default; 0 lets the harness omit -V.
AR_CTA_COUNT="${AR_CTA_COUNT:-8}"

# d5 = single-launch (ReduceScatter -> in-kernel arGridBarrier -> AllGather), the
# default GIN-SDMA AllReduce. d6 = two-launch (RS kernel then AG kernel on the
# same stream; boundary = launch boundary, cannot deadlock on grid size).
AR_MODE="${AR_MODE:-d5}"

# [CONN-GATE-RETRY] The RCCL GIN anvil-sdma plugin runs an LSA signal
# connectivity gate at communicator init. On gfx950 (MI300/350/355) a cuMem-VMM
# peer mapping can intermittently come up broken; the plugin detects this and
# aborts (instead of hanging) with "LSA signal connectivity gate failed ...
# Re-launch the job". The fault is far more frequent when the registered
# symmetric window is large (multi-GiB -e), so re-launch the GIN test up to
# GIN_CONN_RETRIES times when we see that specific gate abort. Set to 0 to
# disable retries.
GIN_CONN_RETRIES="${GIN_CONN_RETRIES:-5}"

_run_test() { [[ ",${RCCL_GIN_RUN_TESTS}," == *",$1,"* ]]; }
_trace_on()  { [[ "${RCCL_GIN_ECHO:-1}" == 1 ]] && set -x; return 0; }
_trace_off() { [[ "${RCCL_GIN_ECHO:-1}" == 1 ]] && set +x; return 0; }

if [[ "${DOCKER_ULIMIT_MEMLOCK:-1}" != 0 ]]; then
  D_MEMLOCK=(--ulimit memlock=-1:-1)
else
  D_MEMLOCK=()
fi

DOCKER_GPU_COMMON="${D_MEMLOCK[*]} --shm-size 64G --network host --device /dev/dri --device /dev/kfd --ipc host --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged ${DOCKER_EXTRA:-}"

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
  -x ROCSHMEM_BACKEND=ipc
  -x ROCSHMEM_DISABLE_MIXED_IPC=1
  -x ROCSHMEM_DEBUG_LEVEL=info:noversion
  -x RCCL_ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD}"
  -x NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"
  -x NCCL_DEBUG_SUBSYS=INIT,NET
  -x RCCL_ENABLE_INTRANET=1
  -x NCCL_DMABUF_ENABLE=1
  -x NCCL_MSCCL_ENABLE=0
  -x HSA_NO_SCRATCH_RECLAIM=1
)

# [CUMEM-SKIP-FREE][WORKAROUND] gfx950 hipMemUnmap teardown deadlock: forward
# NCCL_CUMEM_SKIP_FREE=1 so ncclCuMemFree skips the hanging cuMem unmap at
# ncclCommDestroy (leaks the aperture; OS reclaims at process exit).
[[ -n "${NCCL_CUMEM_SKIP_FREE:-}" ]] && \
  MPI_BASE+=(-x "NCCL_CUMEM_SKIP_FREE=${NCCL_CUMEM_SKIP_FREE}")

# Fail unless rccl-tests reports a clean datacheck for every size in the sweep.
_check_datacheck() {  # $1=logfile $2=label
  if grep -qE "Out of bounds values : 0 OK" "$1"; then
    echo "=== ${2}: datacheck OK (Out of bounds values : 0) ==="
    return 0
  fi
  echo "ERROR: ${2}: datacheck did NOT report 'Out of bounds values : 0 OK'." >&2
  echo "------------------------------ log tail ------------------------------" >&2
  tail -n 40 "$1" >&2
  echo "----------------------------------------------------------------------" >&2
  return 1
}

# True if the log shows the intermittent gfx950 cuMem-VMM peer-map fault that the
# GIN plugin's connectivity gate aborts on (safe/expected to re-launch).
_is_conn_gate_fault() {  # $1=logfile
  grep -qE "LSA signal connectivity gate failed|cuMem-VMM peer-map fault|cuMem VMM peer mapping broken" "$1"
}

_common_perf_args() {  # $1=deviceImpl
  local v=()
  [[ "${AR_CTA_COUNT}" != 0 ]] && v=(-V "${AR_CTA_COUNT}")
  printf '%s\n' -b "${MIN_BYTES}" -e "${MAX_BYTES}" -f 2 -g 1 -R 2 -D "$1" \
    -o "${AR_OP}" -d "${AR_DTYPE}" -c 1 -w "${AR_WARMUP}" -n "${AR_ITERS}" "${v[@]}"
}

RC=0

# --------------------------------------------------------------------------
# Test#0: host baseline (ncclAllReduce, -D 0). Reference busbw + correctness.
# --------------------------------------------------------------------------
if _run_test 0; then
  _trace_on
  echo "=== Test#0: AllReduce, ${NP} gpus, Host Initiated (ncclAllReduce, -D 0, op=${AR_OP}, ${AR_DTYPE}) ==="
  mapfile -t _args < <(_common_perf_args 0)
  _log0="$(mktemp)"
  ${DOCKER_CMD} run ${DOCKER_GPU} "${DOCKER_IMAGE}" \
    mpirun -n "${NP}" ${MPI_OPT} \
    "${MPI_BASE[@]}" \
    -x NCCL_CUMEM_ENABLE=1 \
    -x ROCSHMEM_SDMA_ENABLED=0 \
    -x NCCL_GIN_ENABLE=0 \
    -x NCCL_GIN_TYPE=0 \
    rccl-tests/all_reduce_perf "${_args[@]}" 2>&1 | tee "${_log0}"
  _check_datacheck "${_log0}" "Test#0 host" || RC=1
  rm -f "${_log0}"
  _trace_off
fi

# --------------------------------------------------------------------------
# Test#5: GIN Anvil-SDMA AllReduce (NCCL_GIN_TYPE=5), -D 5 (single) / -D 6 (two).
# ReduceScatter -> device-wide barrier -> AllGather. arThr selects the AllGather
# tier (LSA peer-store below, SDMA put at/above); arOneShot enables the tiny
# out-of-place one-shot LSA read-reduce for the smallest sizes (-D 5 only).
# --------------------------------------------------------------------------
if _run_test 5; then
  _trace_on
  # Backend gin.put SDMA control: 0 => every GIN put uses the copy engine.
  NCCL_GIN_ANVIL_SDMA_THRESHOLD="${NCCL_GIN_ANVIL_SDMA_THRESHOLD:-0}"
  NCCL_GIN_ANVIL_SDMA_MAX_COPY_CHUNK="${NCCL_GIN_ANVIL_SDMA_MAX_COPY_CHUNK:-8192}"

  AR_MPI_EXTRA=(
    -x "NCCL_GIN_ANVIL_SDMA_THRESHOLD=${NCCL_GIN_ANVIL_SDMA_THRESHOLD}"
    -x "NCCL_GIN_ANVIL_SDMA_MAX_COPY_CHUNK=${NCCL_GIN_ANVIL_SDMA_MAX_COPY_CHUNK}"
  )
  # AllGather-tier LSA<->SDMA crossover (per-message bytes). Unset => kernel
  # built-in default (gin_sdma::kAllReduceSdmaThresholdDefault, 16 MiB). Set to 0
  # to force all-SDMA, or a huge value to force all-LSA.
  [[ -n "${AR_SDMA_THRESHOLD:-}" ]] && \
    AR_MPI_EXTRA+=(-x "NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLREDUCE=${AR_SDMA_THRESHOLD}")
  # Tiny one-shot LSA read-reduce cutoff (-D 5 only). Unset => built-in default
  # (gin_sdma::kAllReduceOneShotThresholdDefault, 256 KiB). Set 0 to disable.
  [[ -n "${AR_ONESHOT_THRESHOLD:-}" ]] && \
    AR_MPI_EXTRA+=(-x "NCCL_GIN_ANVIL_ONESHOT_THRESHOLD_ALLREDUCE=${AR_ONESHOT_THRESHOLD}")

  # Device-side (in-kernel wall_clock64) timing. 0=off (default), 1=augment (extra
  # "#[ar-devtime]" line), 2=device-time-only (report device latency AS the metric;
  # out-of-place pass only; warmup + datacheck still run). LOOP/SKIP default 10/10.
  AR_DEVICE_TIMING="${AR_DEVICE_TIMING:-0}"
  AR_DEVTIME_LOOP="${AR_DEVTIME_LOOP:-${AR_ITERS}}"
  AR_DEVTIME_SKIP="${AR_DEVTIME_SKIP:-${AR_WARMUP}}"
  if [[ "${AR_DEVICE_TIMING}" != 0 ]]; then
    AR_MPI_EXTRA+=(
      -x "NCCL_GIN_ANVIL_DEVICE_TIMING=${AR_DEVICE_TIMING}"
      -x "NCCL_GIN_ANVIL_AR_DEVTIME_LOOP=${AR_DEVTIME_LOOP}"
      -x "NCCL_GIN_ANVIL_AR_DEVTIME_SKIP=${AR_DEVTIME_SKIP}"
    )
  fi

  case "${AR_MODE}" in
    d5) _ar_impl=5 ;;
    d6) _ar_impl=6 ;;
    *) echo "error: AR_MODE must be d5 or d6" >&2; exit 1 ;;
  esac

  echo "=== Test#5: AllReduce, ${NP} gpus, GIN Anvil-SDMA -D ${_ar_impl} (${AR_MODE}, op=${AR_OP}, ${AR_DTYPE}, V=${AR_CTA_COUNT}, NCCL_GIN_TYPE=5, devtime=${AR_DEVICE_TIMING}, w=${AR_WARMUP}, n=${AR_ITERS}) ==="
  mapfile -t _args < <(_common_perf_args "${_ar_impl}")
  _log5="$(mktemp)"
  _attempt=1
  _max_attempts=$(( GIN_CONN_RETRIES + 1 ))
  while :; do
    ${DOCKER_CMD} run ${DOCKER_GPU} "${DOCKER_IMAGE}" \
      mpirun -n "${NP}" ${MPI_OPT} \
      "${MPI_BASE[@]}" \
      "${GIN_PLUGIN_X[@]}" \
      -x NCCL_CUMEM_ENABLE=1 \
      -x NCCL_NET_PLUGIN=none \
      -x NCCL_ENV_PLUGIN=none \
      -x ROCSHMEM_SDMA_ENABLED=0 \
      -x NCCL_DEBUG="${NCCL_DEBUG:-VERSION}" \
      -x NCCL_GIN_ENABLE=1 \
      -x NCCL_GIN_TYPE=5 \
      -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS="${AR_NUM_CHANNELS:-1}" \
      -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
      "${AR_MPI_EXTRA[@]}" \
      rccl-tests/all_reduce_perf "${_args[@]}" 2>&1 | tee "${_log5}"
    # Re-launch only for the intermittent gfx950 cuMem-VMM connectivity-gate
    # abort; a genuine datacheck mismatch is a real failure and must not retry.
    if ! grep -qE "Out of bounds values : 0 OK" "${_log5}" \
       && _is_conn_gate_fault "${_log5}" \
       && [[ "${_attempt}" -lt "${_max_attempts}" ]]; then
      echo "=== Test#5 GIN -D ${_ar_impl}: gfx950 cuMem-VMM connectivity-gate fault on attempt ${_attempt}/${_max_attempts}; re-launching ===" >&2
      _attempt=$(( _attempt + 1 ))
      sleep 3
      continue
    fi
    break
  done
  _check_datacheck "${_log5}" "Test#5 GIN -D ${_ar_impl}" || RC=1
  rm -f "${_log5}"
  _trace_off
fi

if [[ "${RC}" -ne 0 ]]; then
  echo "=== gin-sdma-ar-test: FAILED (see errors above) ===" >&2
else
  echo "=== gin-sdma-ar-test: all selected tests passed datacheck ==="
fi
exit "${RC}"
