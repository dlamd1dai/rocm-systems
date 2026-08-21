#!/usr/bin/env bash
# Run rccl-UnitTestsFixtures GIN device-template suites (GDA + Anvil-SDMA shadow).
# Requires a build with: -t --rocshmem-gin -DGIN_ANVIL_UNIT_TESTS=ON
#
# Usage:
#   FIXTURES_BIN=path/to/rccl-UnitTestsFixtures bash gin-fixtures-test.bash
#   # or from projects/rccl after install.sh -t:
#   bash ddai-artifacts/scripts/gin-fixtures-test.bash
#
# When librccl.so was built with rocSHMEM device bitcode only (no host librocshmem),
# preload the host-symbol stub:
#   LD_PRELOAD=ddai-artifacts/c/ddai-rocshmem-hoststub.so bash gin-fixtures-test.bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTI_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ARTI_DIR}/.." && pwd)"

FIXTURES_BIN="${FIXTURES_BIN:-${REPO_ROOT}/projects/rccl/build/release/test/rccl-UnitTestsFixtures}"
GTEST_FILTER="${GTEST_FILTER:-GinRocshmemGdaTemplateTest.*:GinAnvilSdmaTemplateTest.*:GinAnvilIpcDeviceTest.*}"

if [[ ! -x "${FIXTURES_BIN}" ]]; then
  echo "ERROR: rccl-UnitTestsFixtures not found at '${FIXTURES_BIN}'" >&2
  echo "Build with: cd projects/rccl && ./install.sh -t --rocshmem-gin -l --no_clean \\" >&2
  echo "  --cmake-options \"-DGIN_ANVIL_UNIT_TESTS=ON\"" >&2
  exit 1
fi

HOSTSTUB_SO="${ARTI_DIR}/c/ddai-rocshmem-hoststub.so"
if [[ -z "${LD_PRELOAD:-}" ]] && [[ -f "${ARTI_DIR}/c/ddai-rocshmem-hoststub.c" ]] && [[ ! -f "${HOSTSTUB_SO}" ]]; then
  echo "=== Building rocSHMEM host-symbol preload stub ==="
  hipcc -shared -fPIC -o "${HOSTSTUB_SO}" "${ARTI_DIR}/c/ddai-rocshmem-hoststub.c"
fi
if [[ -f "${HOSTSTUB_SO}" ]] && [[ -z "${LD_PRELOAD:-}" ]]; then
  export LD_PRELOAD="${HOSTSTUB_SO}${LD_PRELOAD:+:${LD_PRELOAD}}"
  echo "NOTE: LD_PRELOAD=${LD_PRELOAD}"
fi

echo "=== rccl-UnitTestsFixtures: ${GTEST_FILTER} ==="
"${FIXTURES_BIN}" --gtest_filter="${GTEST_FILTER}"
echo "=== Fixtures GIN suites PASS ==="
