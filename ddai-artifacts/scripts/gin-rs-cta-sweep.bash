#!/usr/bin/env bash
# GIN-SDMA (-D 3) ReduceScatter CTA-count (-V) sensitivity sweep, mid..large.
# Run INSIDE rccl-gin-gda-sdma-713-rs with -v ~/rt-build:/rt-build.
set -uo pipefail
EXE=/rt-build/reduce_scatter_perf
MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
GINENV=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 -x NCCL_DEBUG=WARN
        -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_CUMEM_ENABLE=1
        -x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1
        -x RCCL_ROCSHMEM_THRESHOLD=134217728 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
        -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_PLUGIN=none -x NCCL_NET_PLUGIN=none
        -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
        -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1)
for V in 32 64 96 128; do
  echo "########## -V $V ##########"
  mpirun -n 8 "${MPI_OPT[@]}" "${GINENV[@]}" \
    "$EXE" -b 1048576 -e 2147483648 -f 4 -g 1 -R 2 -V "$V" -D 3 -c 1 -n 20 -w 5 -o sum -d float -z 0 \
    2>&1 | grep -E "^ +[0-9]+ +[0-9]+ +float" || echo "RC=$?"
done
echo "CTA_SWEEP_DONE"
