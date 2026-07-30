#!/usr/bin/env bash
# Host (-D 0) vs GIN-SDMA (-D 3) ReduceScatter sweep, 128B..2GB, 8 ranks.
# Run INSIDE the rccl-gin-gda-sdma-713-rs image (host RS device kernel present)
# with -v ~/rt-build:/rt-build so /rt-build/reduce_scatter_perf (new GIN kernel)
# is used for BOTH modes for an apples-to-apples binary.
set -uo pipefail

EXE=/rt-build/reduce_scatter_perf
B=128; E=2G; F=2; N=20; W=5

MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)

# Common runtime (symmetric registration works for both host and GIN).
COMMON=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        -x NCCL_DEBUG=WARN -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_CUMEM_ENABLE=1)

# GIN Anvil-SDMA env for -D 3.
GINENV=(-x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1
        -x RCCL_ROCSHMEM_THRESHOLD=134217728 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
        -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_PLUGIN=none -x NCCL_NET_PLUGIN=none
        -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
        -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1)

echo "########## HOST-INITIATED ReduceScatter (-D 0), float/sum, OOP ##########"
mpirun -n 8 "${MPI_OPT[@]}" "${COMMON[@]}" \
  "$EXE" -b "$B" -e "$E" -f "$F" -g 1 -R 2 -D 0 -c 1 -n "$N" -w "$W" -o sum -d float -z 0 \
  2>&1 | grep -E "^ +[0-9]+ +[0-9]+ +float|Avg bus|Out of bounds" || echo "HOST_RUN_RC=$?"

echo ""
echo "########## GIN-SDMA ReduceScatter (-D 3), float/sum, OOP ##########"
mpirun -n 8 "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" \
  "$EXE" -b "$B" -e "$E" -f "$F" -g 1 -R 2 -V 32 -D 3 -c 1 -n "$N" -w "$W" -o sum -d float -z 0 \
  2>&1 | grep -E "^ +[0-9]+ +[0-9]+ +float|Avg bus|Out of bounds" || echo "GIN_RUN_RC=$?"

echo "SWEEP_DONE"
