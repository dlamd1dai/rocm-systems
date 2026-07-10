# gin-anvil-test-common.bash — shared GIN Anvil SDMA test runner (sourced by node wrappers)
#
# Required before source (set by gin-anvil-smci355-test.bash or gin-anvil-ruby-test.bash):
#   GIN_ANVIL_RUNNER_ID, GIN_ANVIL_HOST_DEFAULT, GIN_ANVIL_HOST_SHORT_DEFAULT,
#   GIN_ANVIL_GPU_ARCH_DEFAULT, GIN_ANVIL_BM_ROOT_SUFFIX, GIN_ANVIL_DOCKER_BUILD_SCRIPT,
#   GIN_ANVIL_DOCKER_TEST_SCRIPT, GIN_ANVIL_DOCKER_CMD_DEFAULT, GIN_ANVIL_LAYOUT_DOC

set -euo pipefail

: "${GIN_ANVIL_RUNNER_ID:?GIN_ANVIL_RUNNER_ID must be set before sourcing gin-anvil-test-common.bash}"
: "${GIN_ANVIL_HOST_DEFAULT:?}"
: "${GIN_ANVIL_HOST_SHORT_DEFAULT:?}"
: "${GIN_ANVIL_GPU_ARCH_DEFAULT:?}"
: "${GIN_ANVIL_BM_ROOT_SUFFIX:?}"
: "${GIN_ANVIL_DOCKER_BUILD_SCRIPT:?}"
: "${GIN_ANVIL_DOCKER_TEST_SCRIPT:?}"
: "${GIN_ANVIL_DOCKER_CMD_DEFAULT:?}"
: "${GIN_ANVIL_LAYOUT_DOC:?}"

GIN_ANVIL_HOST="${GIN_ANVIL_HOST:-${GIN_ANVIL_HOST_DEFAULT}}"
GIN_ANVIL_GPU_ARCH="${GIN_ANVIL_GPU_ARCH:-${GIN_ANVIL_GPU_ARCH_DEFAULT}}"
GIN_ANVIL_LAYOUT="${GIN_ANVIL_LAYOUT:-docker}"
GIN_ANVIL_NP="${GIN_ANVIL_NP:-8}"
GIN_ANVIL_MAX_BYTES="${GIN_ANVIL_MAX_BYTES:-128M}"
GIN_ANVIL_PHASE="${1:-all}"

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${_COMMON_DIR}}"
GIN_ANVIL_BM_ROOT="${GIN_ANVIL_BM_ROOT:-${REPO_ROOT}/${GIN_ANVIL_BM_ROOT_SUFFIX}}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
MPI_PREFIX="${MPI_PREFIX:-}"
ROCSHMEM_INSTALL_DIR="${ROCSHMEM_INSTALL_DIR:-${GIN_ANVIL_BM_ROOT}/install/rocshmem}"
GIN_ANVIL_BUILD_SUITE_F="${GIN_ANVIL_BUILD_SUITE_F:-0}"
GIN_ANVIL_SKIP_DOCKER_REBUILD="${GIN_ANVIL_SKIP_DOCKER_REBUILD:-0}"
RCCL_INSTALL_PREFIX="${RCCL_INSTALL_PREFIX:-${GIN_ANVIL_BM_ROOT}/install/rccl}"
DOCKER_IMAGE="${DOCKER_IMAGE:-rccl-gin-gda-sdma-713}"
DOCKER_CMD="${DOCKER_CMD:-${GIN_ANVIL_DOCKER_CMD_DEFAULT}}"

ROCSHMEM_SRC="${REPO_ROOT}/projects/rocshmem"
RCCL_SRC="${REPO_ROOT}/projects/rccl"
RCCL_TESTS_SRC="${REPO_ROOT}/projects/rccl-tests"

BM_BUILD_ROCSHMEM="${GIN_ANVIL_BM_ROOT}/build/rocshmem"
BM_BUILD_RCCL_UNIT="${GIN_ANVIL_BM_ROOT}/build/rccl-unit"
BM_BUILD_RCCL_LIB="${GIN_ANVIL_BM_ROOT}/build/rccl-lib"
BM_BUILD_RCCL_TESTS="${GIN_ANVIL_BM_ROOT}/build/rccl-tests"
BM_LOG_DIR="${GIN_ANVIL_BM_ROOT}/logs"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${BM_LOG_DIR}/gin-anvil-${GIN_ANVIL_RUNNER_ID}-${TIMESTAMP}.log"

RCCL_GIN_RUN_TESTS="${RCCL_GIN_RUN_TESTS:-${RUN_TESTS:-1,5}}"
TEST5_NUM_CHANNELS="${TEST5_NUM_CHANNELS:-1}"
TEST5_NUM_CTAS="${TEST5_NUM_CTAS:-${GIN_ANVIL_NP:-8}}"
TEST5_MODE="${TEST5_MODE:-run}"
TEST5_MLX5_PREFLIGHT="${TEST5_MLX5_PREFLIGHT:-0}"

MPI_OPT="-mca pml ob1 -mca btl self,vader,tcp -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none ${MPI_MCA_EXTRA:-}"
ROCSHMEM_THRESHOLD=$((128 * 1024 * 1024))

_LOG_TAG="gin-anvil-${GIN_ANVIL_RUNNER_ID}"

_log() {
  echo "[${_LOG_TAG}] $*"
}

_die() {
  echo "[${_LOG_TAG}] ERROR: $*" >&2
  exit 1
}

_host_check() {
  local host
  host="$(hostname -f 2>/dev/null || hostname)"
  if [[ "${host}" != "${GIN_ANVIL_HOST}" && "${host}" != *"${GIN_ANVIL_HOST_SHORT_DEFAULT}"* ]]; then
    _log "warning: expected ${GIN_ANVIL_HOST}; running on ${host}"
  fi
}

_gpu_check() {
  if ! command -v rocm-smi >/dev/null 2>&1; then
    _log "warning: rocm-smi not found; GPU checks skipped"
    return 0
  fi
  local n
  n="$(rocm-smi --showid 2>/dev/null | grep -c 'GPU' || true)"
  _log "detected ${n:-?} GPU(s) via rocm-smi"
}

_setup_log() {
  mkdir -p "${BM_LOG_DIR}"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  _log "log file: ${LOG_FILE}"
  _log "repo: ${REPO_ROOT}  layout: ${GIN_ANVIL_LAYOUT}  phase: ${GIN_ANVIL_PHASE}"
  _log "GPU arch: ${GIN_ANVIL_GPU_ARCH}  NP: ${GIN_ANVIL_NP}  MAX_BYTES: ${GIN_ANVIL_MAX_BYTES}"
}

_bm_runtime_env() {
  export PATH="${ROCSHMEM_INSTALL_DIR}/bin:${RCCL_INSTALL_PREFIX}/bin:${ROCM_PATH}/bin:${MPI_PREFIX:+$MPI_PREFIX/bin:}${PATH}"
  export LD_LIBRARY_PATH="${RCCL_INSTALL_PREFIX}/lib:${ROCSHMEM_INSTALL_DIR}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
  export ROCM_PATH
}

_docker_gpu_run() {
  ${DOCKER_CMD} run --rm --init \
    --device /dev/dri --device /dev/kfd --ipc host \
    --group-add video --group-add render \
    ${DOCKER_EXTRA:-} \
    "$@"
}

_resolve_mpi() {
  local cand inc

  if [[ -n "${MPI_PREFIX}" && -f "${MPI_PREFIX}/include/mpi.h" ]]; then
    return 0
  fi

  if command -v mpicc >/dev/null 2>&1; then
    inc="$(mpicc -showme:incdirs 2>/dev/null | awk '{print $1}')"
    if [[ -n "${inc}" && -f "${inc}/mpi.h" ]]; then
      MPI_PREFIX="$(dirname "${inc}")"
      return 0
    fi
  fi

  for cand in \
    /usr/lib/x86_64-linux-gnu/openmpi \
    /usr/lib64/openmpi \
    /opt/openmpi \
    /opt/ompi \
    /usr/local \
    /usr; do
    if [[ -f "${cand}/include/mpi.h" ]]; then
      MPI_PREFIX="${cand}"
      return 0
    fi
  done

  return 1
}

_mpi_dev_ready() {
  _resolve_mpi
}

_rocshmem_want_suite_f() {
  case "${GIN_ANVIL_BUILD_SUITE_F}" in
    0|off|no|false|skip) return 1 ;;
    1|on|yes|true|force) return 0 ;;
    auto) _mpi_dev_ready ;;
    *)
      _log "warning: unknown GIN_ANVIL_BUILD_SUITE_F=${GIN_ANVIL_BUILD_SUITE_F}; suite F disabled"
      return 1
      ;;
  esac
}

_mpi_missing_hint() {
  cat >&2 <<EOF
[${_LOG_TAG}] Open MPI development headers are required when GIN_ANVIL_BUILD_SUITE_F=1
or GIN_ANVIL_LAYOUT=bare-metal.  On Ubuntu 22.04/24.04:

  sudo apt-get install -y openmpi-bin libopenmpi-dev

Then re-run with a clean rocSHMEM build dir:

  GIN_ANVIL_CLEAN=1 $0 unit

Default workflow skips suite F (suites A–H and G on host; factory coverage via docker Test#5).
EOF
}

_clean_bm_build() {
  if [[ "${GIN_ANVIL_CLEAN:-0}" == 1 ]]; then
    _log "GIN_ANVIL_CLEAN=1: removing ${GIN_ANVIL_BM_ROOT}/build"
    rm -rf "${GIN_ANVIL_BM_ROOT}/build"
  fi
}

_preflight() {
  _log "=== Phase: preflight (B1) ==="
  cd "${REPO_ROOT}"

  if [[ -z "${GIN_ANVIL_PREFLIGHT_MPI:-}" ]]; then
    if [[ "${GIN_ANVIL_LAYOUT}" == bare-metal ]] || _rocshmem_want_suite_f; then
      GIN_ANVIL_PREFLIGHT_MPI=require
    else
      GIN_ANVIL_PREFLIGHT_MPI="${GIN_ANVIL_PREFLIGHT_MPI_DEFAULT:-warn}"
    fi
  fi
  export GIN_ANVIL_PREFLIGHT_MPI

  # shellcheck source=docker-gin-gda-sdma-preflight.bash
  source ./docker-gin-gda-sdma-preflight.bash
  _log "preflight OK (GIN_ANVIL_PREFLIGHT_MPI=${GIN_ANVIL_PREFLIGHT_MPI})"
}

_build_docker() {
  _log "=== Phase: build (docker, ${GIN_ANVIL_GPU_ARCH}) ==="
  cd "${REPO_ROOT}"
  if [[ "${GIN_ANVIL_SKIP_DOCKER_REBUILD}" == 1 ]] \
     && ${DOCKER_CMD} image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
    _log "GIN_ANVIL_SKIP_DOCKER_REBUILD=1: reusing existing image ${DOCKER_IMAGE}"
    return 0
  fi
  mkdir -p extra-rdma-debs
  GPU_TARGETS="${GIN_ANVIL_GPU_ARCH}" \
  RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS="${RCCL_IMAGE_REQUIRE_MLX5_DMABUF_SYMBOLS:-0}" \
    "./${GIN_ANVIL_DOCKER_BUILD_SCRIPT}"
  _log "docker image: ${DOCKER_IMAGE}"
}

_build_bare_metal_rocshmem() {
  local -a rocmshmem_extra=()
  local build_unit_tests=OFF

  if _rocshmem_want_suite_f; then
    if ! _mpi_dev_ready; then
      _mpi_missing_hint
      _die "MPI dev headers not found (set GIN_ANVIL_BUILD_SUITE_F=0 to skip suite F)"
    fi
    build_unit_tests=ON
    _log "MPI prefix: ${MPI_PREFIX} (suite F enabled)"
    rocmshmem_extra+=(
      -DUSE_EXTERNAL_MPI=ON
      -DMPI_ROOT="${MPI_PREFIX}"
      -DCMAKE_PREFIX_PATH="${MPI_PREFIX}"
    )
  else
    _log "building rocSHMEM without external MPI (suite F skipped; matches docker image flags)"
    rocmshmem_extra+=(-DUSE_EXTERNAL_MPI=OFF)
  fi

  _log "building rocSHMEM (USE_SDMA=ON, BUILD_UNIT_TESTS=${build_unit_tests})..."
  (
    cd "${BM_BUILD_ROCSHMEM}"
    "${ROCSHMEM_SRC}/scripts/build_configs/all_backends" \
      -DUSE_SDMA=ON \
      -DUSE_IPC=ON \
      -DGPU_TARGETS="${GIN_ANVIL_GPU_ARCH}" \
      -DCMAKE_INSTALL_PREFIX="${ROCSHMEM_INSTALL_DIR}" \
      -DBUILD_FUNCTIONAL_TESTS=OFF \
      -DBUILD_UNIT_TESTS="${build_unit_tests}" \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_CTESTS=OFF \
      -DBUILD_PYTHON_TESTS=OFF \
      "${rocmshmem_extra[@]}"
  )
}

_apply_rccl_wrap_patch() {
  local rccl_wrap="${RCCL_SRC}/src/rccl_wrap.cc"
  local rccl_wrap_bak="${RCCL_SRC}/src/rccl_wrap.cc.gin-anvil-bak"
  if [[ ! -f "${rccl_wrap_bak}" ]]; then
    cp -a "${rccl_wrap}" "${rccl_wrap_bak}"
  fi
  sed -i \
    's/if (comm->enableRocshmem && comm->nNodes > 1 && (comm->nRanks\/comm->nNodes == 8) && comm->rocshmemThreshold <= 1048576)/if (comm->enableRocshmem)/' \
    "${rccl_wrap}"
}

_build_bare_metal_rccl_unit() {
  _log "building RCCL unit tests (suites A–H, G) in ${BM_BUILD_RCCL_UNIT}..."
  cmake -S "${RCCL_SRC}" -B "${BM_BUILD_RCCL_UNIT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_ROCSHMEM_GIN=ON \
    -DENABLE_ROCSHMEM=OFF \
    -DGIN_ANVIL_UNIT_TESTS=ON \
    -DENABLE_DEVICE_LINKER=OFF \
    -DONLY_FUNCS='SendRecv|AlltoAllPivot|AlltoAllGda|AlltoAllvGda' \
    -DBUILD_TESTS=ON \
    -DGPU_TARGETS="${GIN_ANVIL_GPU_ARCH}" \
    -DROCSHMEM_INSTALL_DIR="${ROCSHMEM_INSTALL_DIR}" \
    -DROCSHMEM_SOURCE_DIR="${ROCSHMEM_SRC}" \
    -DROCSHMEM_BUILD_DIR="${BM_BUILD_ROCSHMEM}/include/rocshmem"
  cmake --build "${BM_BUILD_RCCL_UNIT}" \
    --target rccl-UnitTestsFixtures rccl-UnitTestsGinAnvilPlugin -j"$(nproc)"
}

_build_bare_metal_rccl_install() {
  _log "building RCCL library (install.sh → ${RCCL_INSTALL_PREFIX})..."
  _apply_rccl_wrap_patch
  (
    cd "${RCCL_SRC}"
    ./install.sh \
      --amdgpu_targets="${GIN_ANVIL_GPU_ARCH}" \
      --prefix="${RCCL_INSTALL_PREFIX}" \
      --no-device-linker \
      --no_clean \
      --cmake-options \
        "-DENABLE_ROCSHMEM_GIN=ON \
         -DROCSHMEM_INSTALL_DIR=${ROCSHMEM_INSTALL_DIR} \
         -DROCSHMEM_SOURCE_DIR=${ROCSHMEM_SRC} \
         -DROCSHMEM_BUILD_DIR=${BM_BUILD_ROCSHMEM}/include/rocshmem"
  )
}

_build_bare_metal_rccl_tests() {
  _log "building rccl-tests (alltoall_perf)..."
  local rccl_tests_prefix="${RCCL_INSTALL_PREFIX}"
  if _mpi_dev_ready; then
    rccl_tests_prefix="${RCCL_INSTALL_PREFIX};${MPI_PREFIX}"
  fi
  cmake -S "${RCCL_TESTS_SRC}" -B "${BM_BUILD_RCCL_TESTS}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_MPI=ON \
    -DENABLE_DEVICE_API=ON \
    -DENABLE_ROCSHMEM=ON \
    -DGPU_TARGETS="${GIN_ANVIL_GPU_ARCH}" \
    -DROCSHMEM_INSTALL_DIR="${ROCSHMEM_INSTALL_DIR}" \
    -DROCSHMEM_SOURCE_DIR="${ROCSHMEM_SRC}" \
    -DROCSHMEM_BUILD_DIR="${BM_BUILD_ROCSHMEM}/include/rocshmem" \
    -DRCCL_SOURCE_DIR="${RCCL_SRC}" \
    -DCMAKE_PREFIX_PATH="${rccl_tests_prefix}"
  cmake --build "${BM_BUILD_RCCL_TESTS}" -j"$(nproc)"
}

_build_bare_metal_unit_only() {
  _log "=== Phase: build (bare-metal unit → ${GIN_ANVIL_BM_ROOT}) ==="
  _clean_bm_build
  mkdir -p "${GIN_ANVIL_BM_ROOT}/install" "${BM_BUILD_ROCSHMEM}" "${BM_BUILD_RCCL_UNIT}"
  _build_bare_metal_rocshmem
  _build_bare_metal_rccl_unit
  _log "bare-metal unit build complete"
}

_build_bare_metal_full() {
  _log "=== Phase: build (bare-metal full → ${GIN_ANVIL_BM_ROOT}) ==="
  _clean_bm_build
  mkdir -p "${GIN_ANVIL_BM_ROOT}/install" \
    "${BM_BUILD_ROCSHMEM}" "${BM_BUILD_RCCL_UNIT}" "${BM_BUILD_RCCL_TESTS}"
  _build_bare_metal_rocshmem
  _build_bare_metal_rccl_install
  _build_bare_metal_rccl_tests
  _build_bare_metal_rccl_unit
  _log "bare-metal full build complete"
}

_build_bare_metal() {
  _build_bare_metal_full
}

_build() {
  case "${GIN_ANVIL_LAYOUT}" in
    docker) _build_docker ;;
    bare-metal) _build_bare_metal ;;
    *)
      _die "GIN_ANVIL_LAYOUT must be docker or bare-metal (got ${GIN_ANVIL_LAYOUT})"
      ;;
  esac
}

_verify() {
  _log "=== Phase: verify (B3/B4) ==="
  case "${GIN_ANVIL_LAYOUT}" in
    docker)
      _docker_gpu_run "${DOCKER_IMAGE}" rocshmem/bin/rocshmem_info | head -40 || true
      _docker_gpu_run "${DOCKER_IMAGE}" sh -lc \
        'f=/lib/x86_64-linux-gnu/libmlx5.so.1; test -e "$f" || f=/usr/lib/x86_64-linux-gnu/libmlx5.so.1; \
         rf=$(readlink -f "$f"); objdump -T "$rf" 2>/dev/null | grep mlx5dv_reg_dmabuf_mr || echo "MLX5 DMA-BUF symbol missing"' \
        || true
      ;;
    bare-metal)
      _bm_runtime_env
      "${ROCSHMEM_INSTALL_DIR}/bin/rocshmem_info" | head -40 || _die "rocshmem_info failed"
      local mlx5 f
      for f in /lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so.1; do
        [[ -e "${f}" ]] || continue
        mlx5="$(readlink -f "${f}")"
        if objdump -T "${mlx5}" 2>/dev/null | grep -q mlx5dv_reg_dmabuf_mr; then
          _log "MLX5 DMA-BUF symbol OK: ${mlx5}"
          return 0
        fi
      done
      _log "warning: mlx5dv_reg_dmabuf_mr not found in host libmlx5 (Test#5 may fail NET init)"
      ;;
  esac
}

_unit_build_if_missing() {
  local fixtures="${BM_BUILD_RCCL_UNIT}/test/rccl-UnitTestsFixtures"
  local plugin="${BM_BUILD_RCCL_UNIT}/test/rccl-UnitTestsGinAnvilPlugin"
  local factory="${BM_BUILD_ROCSHMEM}/tests/unit_tests/rocshmem_unit_tests"
  if [[ -x "${fixtures}" && -x "${plugin}" ]]; then
    if [[ -x "${factory}" ]] || ! _rocshmem_want_suite_f; then
      return 0
    fi
  fi
  _log "unit test binaries missing; running bare-metal unit build..."
  _build_bare_metal_unit_only
}

_unit() {
  _log "=== Phase: unit tests (suites A–H, F, G) — host GPU ==="
  _unit_build_if_missing
  _bm_runtime_env

  local fixtures="${BM_BUILD_RCCL_UNIT}/test/rccl-UnitTestsFixtures"
  local plugin="${BM_BUILD_RCCL_UNIT}/test/rccl-UnitTestsGinAnvilPlugin"
  local factory="${BM_BUILD_ROCSHMEM}/tests/unit_tests/rocshmem_unit_tests"

  [[ -x "${fixtures}" ]] || _die "missing ${fixtures}"
  [[ -x "${plugin}" ]] || _die "missing ${plugin}"

  local rc=0

  _log "--- Suite A–E + H: rccl-UnitTestsFixtures (GinAnvil*) ---"
  "${fixtures}" --gtest_filter='GinAnvil*' || rc=$?

  _log "--- Suite H only: GinAnvilSdmaTemplateTest.* ---"
  "${fixtures}" --gtest_filter='GinAnvilSdmaTemplateTest.*' || rc=$?

  _log "--- Suite G: rccl-UnitTestsGinAnvilPlugin ---"
  "${plugin}" --gtest_filter='GinAnvilPluginTest.*' || rc=$?

  if [[ -x "${factory}" ]]; then
    _log "--- Suite F: rocshmem_unit_tests (factory) ---"
    "${factory}" --gtest_filter='GinAnvilSdmaFactoryTest.*' || rc=$?
  else
    _log "note: suite F skipped (default; factory covered by docker Test#5). Opt in: GIN_ANVIL_BUILD_SUITE_F=1 + libopenmpi-dev"
  fi

  if [[ "${rc}" -ne 0 ]]; then
    _die "unit tests failed (exit ${rc})"
  fi
  _log "unit tests passed"
}

_integration_docker() {
  _log "=== Phase: integration (docker) RCCL_GIN_RUN_TESTS=${RCCL_GIN_RUN_TESTS} ==="
  cd "${REPO_ROOT}"
  DOCKER_CMD="${DOCKER_CMD}" \
  RCCL_GIN_RUN_TESTS="${RCCL_GIN_RUN_TESTS}" \
  TEST5_NUM_CHANNELS="${TEST5_NUM_CHANNELS}" \
  TEST5_NUM_CTAS="${TEST5_NUM_CTAS}" \
  TEST5_MODE="${TEST5_MODE}" \
  TEST5_MLX5_PREFLIGHT="${TEST5_MLX5_PREFLIGHT}" \
  DOCKER_IMAGE="${DOCKER_IMAGE}" \
    "./${GIN_ANVIL_DOCKER_TEST_SCRIPT}" "${GIN_ANVIL_NP}" "${GIN_ANVIL_MAX_BYTES}"
}

_integration_bare_metal() {
  _log "=== Phase: integration (bare-metal) RCCL_GIN_RUN_TESTS=${RCCL_GIN_RUN_TESTS} ==="
  local perf="${BM_BUILD_RCCL_TESTS}/alltoall_perf"
  [[ -x "${perf}" ]] || _die "missing ${perf}; run: GIN_ANVIL_LAYOUT=bare-metal $0 build"

  if ! command -v mpirun >/dev/null 2>&1; then
    _mpi_missing_hint
    _die "mpirun not found on PATH"
  fi

  _bm_runtime_env

  local -a mpi_base=(
    -x OMPI_ALLOW_RUN_AS_ROOT=1
    -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
    -x RCCL_ROCSHMEM_ENABLE=0
    -x ROCSHMEM_BACKEND=ipc
    -x ROCSHMEM_DISABLE_MIXED_IPC=1
    -x ROCSHMEM_DEBUG_LEVEL=info:noversion
    -x RCCL_ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD}"
    -x NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"
    -x NCCL_DEBUG_SUBSYS=INIT,NET
    -x NCCL_CUMEM_ENABLE=1
    -x RCCL_ENABLE_INTRANET=1
    -x NCCL_DMABUF_ENABLE=1
    -x NCCL_MSCCL_ENABLE=0
    -x HSA_NO_SCRATCH_RECLAIM=1
    -x NCCL_GIN_PLUGIN=none
    -x LD_LIBRARY_PATH
    -x PATH
  )

  _run_bm_test() {
    local d_mode="$1"
    shift
    local -a extra_env=("$@")
    mpirun -n "${GIN_ANVIL_NP}" --allow-run-as-root ${MPI_OPT} \
      "${mpi_base[@]}" \
      "${extra_env[@]}" \
      "${perf}" -b 128 -e "${GIN_ANVIL_MAX_BYTES}" -f 2 -g 1 -R 2 -D "${d_mode}" -A 1 -V 1
  }

  if [[ ",${RCCL_GIN_RUN_TESTS}," == *",1,"* ]]; then
    _log "--- C1 / Test#1: host baseline (-D 0) ---"
    _run_bm_test 0 \
      -x ROCSHMEM_SDMA_ENABLED=0 \
      -x NCCL_GIN_ENABLE=0 \
      -x NCCL_GIN_TYPE=0
  fi

  if [[ ",${RCCL_GIN_RUN_TESTS}," == *",5,"* && "${TEST5_MODE}" != skip ]]; then
    _log "--- C2 / Test#5: GIN Anvil SDMA (NCCL_GIN_TYPE=5, -D 3) ---"
    local -a test5_extra=(
      -x NCCL_NET_PLUGIN=none
      -x ROCSHMEM_SDMA_ENABLED=0
      -x NCCL_GIN_ENABLE=1
      -x NCCL_GIN_TYPE=5
      -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS="${TEST5_NUM_CHANNELS}"
      -x HSA_FORCE_FINE_GRAIN_PCIE=1
    )
    if [[ -n "${NCCL_GIN_ANVIL_SDMA_THRESHOLD:-}" ]]; then
      test5_extra+=(-x "NCCL_GIN_ANVIL_SDMA_THRESHOLD=${NCCL_GIN_ANVIL_SDMA_THRESHOLD}")
    fi
    if [[ -n "${NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL:-}" ]]; then
      test5_extra+=(-x "NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL=${NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL}")
    fi
    _run_bm_test 3 "${test5_extra[@]}"
  fi
}

_integration() {
  case "${GIN_ANVIL_LAYOUT}" in
    docker) _integration_docker ;;
    bare-metal) _integration_bare_metal ;;
    *) _die "unknown layout ${GIN_ANVIL_LAYOUT}" ;;
  esac
}

_isolation() {
  _log "=== Phase: isolation (D4/D5 threshold sweeps) ==="
  local saved_tests="${RCCL_GIN_RUN_TESTS}"
  RCCL_GIN_RUN_TESTS=5

  _log "--- D4: force all SDMA (THRESHOLD=0) ---"
  NCCL_GIN_ANVIL_SDMA_THRESHOLD=0 _integration

  _log "--- D5: force all IPC (THRESHOLD=65536) ---"
  NCCL_GIN_ANVIL_SDMA_THRESHOLD=65536 _integration

  RCCL_GIN_RUN_TESTS="${saved_tests}"
  unset NCCL_GIN_ANVIL_SDMA_THRESHOLD
  _log "isolation complete"
}

_usage() {
  cat <<EOF
GIN Anvil SDMA test runner for ${GIN_ANVIL_HOST_DEFAULT} (${GIN_ANVIL_RUNNER_LABEL})

Usage: $(basename "$0") [PHASE]

Phases:
  all           preflight + build + unit + integration (default)
  preflight     source docker-gin-gda-sdma-preflight.bash
  build         docker image OR bare-metal tree (GIN_ANVIL_LAYOUT)
  unit          GTest suites A–H + G on host; suite F off by default (opt-in)
  integration   Test#1 + Test#5 (docker or bare-metal)
  isolation     Test#5 with THRESHOLD=0 and THRESHOLD=65536
  verify        rocshmem_info + MLX5 symbol check
  help          this message

Key environment variables:
  GIN_ANVIL_LAYOUT=docker|bare-metal   default: docker
  GIN_ANVIL_NP=8                       mpirun rank count
  GIN_ANVIL_MAX_BYTES=128M             alltoall_perf -e
  GIN_ANVIL_BM_ROOT=${REPO_ROOT}/${GIN_ANVIL_BM_ROOT_SUFFIX}
  GIN_ANVIL_BUILD_SUITE_F=0|1|auto       default 0; 1=host rocSHMEM factory tests (suite F)
  GIN_ANVIL_PREFLIGHT_MPI=skip|warn|require
  GIN_ANVIL_CLEAN=1                    wipe gin-anvil-bm/build before bare-metal build
  GIN_ANVIL_SKIP_DOCKER_REBUILD=1      reuse existing docker image if present
  DOCKER_CMD=${GIN_ANVIL_DOCKER_CMD_DEFAULT}
  RCCL_GIN_RUN_TESTS=1,5               harness test selection
  TEST5_NUM_CHANNELS=1                 NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS
  NCCL_GIN_ANVIL_SDMA_THRESHOLD        IPC vs SDMA boundary (optional)
  NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL       OSS7 fused signal (D6, optional)

Bare-metal layout: docs/${GIN_ANVIL_LAYOUT_DOC}

Examples:
  $(basename "$0") all
  GIN_ANVIL_LAYOUT=bare-metal $(basename "$0") build unit integration
  RCCL_GIN_RUN_TESTS=5 $(basename "$0") integration
EOF
}

_run_phase() {
  case "${GIN_ANVIL_PHASE}" in
    help|-h|--help) _usage; exit 0 ;;
    preflight) _preflight ;;
    build) _preflight; _build; _verify ;;
    unit) _preflight; _unit ;;
    integration) _preflight; _integration ;;
    isolation) _preflight; _isolation ;;
    verify) _verify ;;
    all)
      _preflight
      _build
      _verify
      _unit
      _integration
      ;;
    *)
      _die "unknown phase '${GIN_ANVIL_PHASE}'; run '$0 help'"
      ;;
  esac
}

_gin_anvil_test_main() {
  _host_check
  _gpu_check
  _setup_log
  _run_phase
  _log "done (phase=${GIN_ANVIL_PHASE}, layout=${GIN_ANVIL_LAYOUT})"
}
