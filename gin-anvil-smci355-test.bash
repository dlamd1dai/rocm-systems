#!/usr/bin/env bash
# gin-anvil-smci355-test.bash — GIN Anvil SDMA test runner for smci355-ccs-aus-m03-17 (MI355X / gfx950)
#
# Runs preflight, build, GTest unit suites A–H + G (suite F opt-in), and integration (C1/C2) per
# docs/gin-anvil-sdma-unit-test-plan.md and docs/gin-anvil-sdma-backend-design.md.
#
# Usage:
#   ./gin-anvil-smci355-test.bash [PHASE]
#
# Phases:
#   all           preflight + build + unit + integration (default)
#   preflight     tree validation only
#   build         docker image OR bare-metal install (see GIN_ANVIL_LAYOUT)
#   unit          GTest suites A–H + F + G (always on host; builds if needed)
#   integration   Test#1 + Test#5 via docker or bare-metal mpirun
#   isolation     Test#5 threshold sweeps (SDMA-only + IPC-only)
#   verify        post-build checks (rocshmem_info, MLX5 symbol)
#   help          print usage
#
# Layout (GIN_ANVIL_LAYOUT):
#   docker      build/run integration via docker-gin-gda-sdma-*.bash (default)
#   bare-metal  install under gin-anvil-bm/; see docs/gin-anvil-smci355-bare-metal-layout.md
#
# Examples:
#   ./gin-anvil-smci355-test.bash all
#   GIN_ANVIL_LAYOUT=bare-metal ./gin-anvil-smci355-test.bash build unit integration
#   RCCL_GIN_RUN_TESTS=5 NCCL_GIN_ANVIL_SDMA_THRESHOLD=0 ./gin-anvil-smci355-test.bash integration
#   ./gin-anvil-smci355-test.bash isolation 2>&1 | tee gin-anvil-bm/logs/isolation.log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GIN_ANVIL_RUNNER_ID=smci355
export GIN_ANVIL_RUNNER_LABEL=MI355X
export GIN_ANVIL_HOST_DEFAULT="${GIN_ANVIL_HOST_DEFAULT:-smci355-ccs-aus-m03-17.cs-aus.dcgpu}"
export GIN_ANVIL_HOST_SHORT_DEFAULT="${GIN_ANVIL_HOST_SHORT_DEFAULT:-smci355-ccs-aus-m03-17}"
export GIN_ANVIL_GPU_ARCH_DEFAULT="${GIN_ANVIL_GPU_ARCH_DEFAULT:-gfx950}"
export GIN_ANVIL_BM_ROOT_SUFFIX="${GIN_ANVIL_BM_ROOT_SUFFIX:-gin-anvil-bm}"
export GIN_ANVIL_DOCKER_BUILD_SCRIPT="${GIN_ANVIL_DOCKER_BUILD_SCRIPT:-docker-gin-gda-sdma-build.bash}"
export GIN_ANVIL_DOCKER_TEST_SCRIPT="${GIN_ANVIL_DOCKER_TEST_SCRIPT:-docker-gin-gda-sdma-test.bash}"
export GIN_ANVIL_DOCKER_CMD_DEFAULT="${GIN_ANVIL_DOCKER_CMD_DEFAULT:-docker}"
export GIN_ANVIL_PREFLIGHT_MPI_DEFAULT="${GIN_ANVIL_PREFLIGHT_MPI_DEFAULT:-warn}"
export GIN_ANVIL_LAYOUT_DOC="${GIN_ANVIL_LAYOUT_DOC:-gin-anvil-smci355-bare-metal-layout.md}"

# shellcheck source=gin-anvil-test-common.bash
source "${SCRIPT_DIR}/gin-anvil-test-common.bash"
_gin_anvil_test_main "$@"
