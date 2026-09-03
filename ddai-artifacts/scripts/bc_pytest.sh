#!/bin/bash
# Validate the GIN-SDMA Broadcast pytest cases (test_Broadcast.py) against real
# hardware output: rebuild broadcast_perf from the branch sources mounted at
# /fix, then run the GIN cases so the new #wrong / plugin-bind assertions are
# exercised on a real results table.
#
# NCCL_GIN_TYPE=5 because this SUT image is on the NCCL 2.30.7 line, where the
# ANVIL_SDMA enum is 5; develop (and the pytest default) uses 6.
set -e
cd /workspace/rccl-tests
SRC=$(grep -m1 CMAKE_HOME_DIRECTORY CMakeCache.txt | cut -d= -f2)

cp /fix/broadcast.cu "$SRC/src/broadcast.cu"
cp /fix/gin_sdma_broadcast_policy.h "$SRC/src/gin_sdma_broadcast_policy.h"
cp /fix/gin_sdma_devtime.h "$SRC/src/gin_sdma_devtime.h"
cp /fix/test_Broadcast.py "$SRC/test/test_Broadcast.py"

cmake --build . --target broadcast_perf -j 32 > /tmp/b.log 2>&1 || { tail -30 /tmp/b.log; exit 1; }
echo "BUILD=OK"

python3 -m pytest --version >/dev/null 2>&1 || \
  pip install --quiet --break-system-packages pytest >/dev/null 2>&1
echo "pytest: $(python3 -m pytest --version 2>&1)"
echo "rocminfo GPU count: $(rocminfo 2>/dev/null | grep -c 'Device Type:.*GPU')"

cd "$SRC/test"
export RCCL_TESTS_GIN_SDMA_BCAST=1
export RCCL_TESTS_BCAST_GIN_TYPE=${BC_TYPE:-5}
export RCCL_TESTS_BCAST_NP=8
export RCCL_TESTS_BCAST_EXE=/workspace/rccl-tests/broadcast_perf
export RCCL_TESTS_BCAST_TIMEOUT_S=900
export RCCL_TESTS_MPI_OPTS="--allow-run-as-root -mca pml ob1 -mca btl ^openib"
export RCCL_TESTS_BCAST_XENV="NCCL_CUMEM_ENABLE=1 RCCL_ENABLE_INTRANET=1 NCCL_DMABUF_ENABLE=1 HSA_NO_SCRATCH_RECLAIM=1 NCCL_MSCCL_ENABLE=0"

python3 -m pytest test_Broadcast.py -v -p no:cacheprovider -k "${BC_K:-GinSdma}"
