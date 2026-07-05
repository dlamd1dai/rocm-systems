#!/usr/bin/env bash
# gin-anvil-ruby-test.bash — GIN Anvil SDMA test runner for Ruby cluster MI350X nodes (cv350-rck-* / gfx950)
#
# Equivalent to gin-anvil-smci355-test.bash with Ruby-cluster defaults:
#   - sudo docker + docker-gin-gda-sdma-ruby-*.bash (BuildKit --network=host, preflight in build)
#   - separate bare-metal tree under gin-anvil-bm-ruby/
#   - host name pattern cv350-rck-*.rck.dcgpu
#
# Usage:
#   ./gin-anvil-ruby-test.bash [PHASE]
#
# Phases: same as gin-anvil-smci355-test.bash (all, preflight, build, unit, integration, isolation, verify, help)
#
# Layout (GIN_ANVIL_LAYOUT):
#   docker      build/run via docker-gin-gda-sdma-ruby-*.bash (default)
#   bare-metal  install under gin-anvil-bm-ruby/; see docs/gin-anvil-ruby-bare-metal-layout.md
#
# Examples:
#   ./gin-anvil-ruby-test.bash all
#   GIN_ANVIL_SKIP_DOCKER_REBUILD=1 ./gin-anvil-ruby-test.bash unit
#   RCCL_GIN_RUN_TESTS=5 ./gin-anvil-ruby-test.bash integration 2>&1 | tee gin-anvil-bm-ruby/logs/run.log
#   ./gin-anvil-ruby-test.bash isolation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GIN_ANVIL_RUNNER_ID=ruby
export GIN_ANVIL_RUNNER_LABEL=MI350X
export GIN_ANVIL_HOST_DEFAULT="${GIN_ANVIL_HOST_DEFAULT:-cv350-rck-g03-e09-08.rck.dcgpu}"
export GIN_ANVIL_HOST_SHORT_DEFAULT="${GIN_ANVIL_HOST_SHORT_DEFAULT:-cv350-rck}"
export GIN_ANVIL_GPU_ARCH_DEFAULT="${GIN_ANVIL_GPU_ARCH_DEFAULT:-gfx950}"
export GIN_ANVIL_BM_ROOT_SUFFIX="${GIN_ANVIL_BM_ROOT_SUFFIX:-gin-anvil-bm-ruby}"
export GIN_ANVIL_DOCKER_BUILD_SCRIPT="${GIN_ANVIL_DOCKER_BUILD_SCRIPT:-docker-gin-gda-sdma-ruby-build.bash}"
export GIN_ANVIL_DOCKER_TEST_SCRIPT="${GIN_ANVIL_DOCKER_TEST_SCRIPT:-docker-gin-gda-sdma-ruby-test.bash}"
export GIN_ANVIL_DOCKER_CMD_DEFAULT="${GIN_ANVIL_DOCKER_CMD_DEFAULT:-sudo docker}"
export GIN_ANVIL_PREFLIGHT_MPI_DEFAULT="${GIN_ANVIL_PREFLIGHT_MPI_DEFAULT:-warn}"
export GIN_ANVIL_LAYOUT_DOC="${GIN_ANVIL_LAYOUT_DOC:-gin-anvil-ruby-bare-metal-layout.md}"

# shellcheck source=gin-anvil-test-common.bash
source "${SCRIPT_DIR}/gin-anvil-test-common.bash"
_gin_anvil_test_main "$@"
