#!/usr/bin/env bash
# Backward-compatible wrapper: the SendRecv LL verify pipeline is now the
# COLL=sendrecv case of the collective-parameterized gin-sdma-ll-verify.bash.
# All flags/env (RUN_*, GATE_MAX_BYTES, AB_*, BIG_*, ...) pass straight through.
#
#   bash gin-sdma-sendrecv-ll-verify.bash [NP]
# is exactly:
#   COLL=sendrecv bash gin-sdma-ll-verify.bash [NP]
#
# For Scatter use:  COLL=scatter bash gin-sdma-ll-verify.bash [NP]
set -euo pipefail
exec env COLL=sendrecv bash "$(dirname "$0")/gin-sdma-ll-verify.bash" "$@"
