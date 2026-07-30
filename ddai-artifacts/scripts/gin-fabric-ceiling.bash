#!/usr/bin/env bash
# Measure the per-GPU xGMI fabric ceiling for GIN-SDMA (-D 3) collectives at large
# sizes, to bound how much headroom (if any) exists above the CU LSA ReduceScatter
# result (~377 GB/s). alltoall = SDMA egress-driven all-to-all; all_gather = SDMA;
# reduce_scatter = CU LSA read-reduce (our kernel). Same fabric, same 8 ranks.
# Run INSIDE rccl-gin-gda-sdma-713-rs with -v ~/rt-build:/rt-build.
set -uo pipefail
MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
GINENV=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 -x NCCL_DEBUG=WARN
        -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_CUMEM_ENABLE=1
        -x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1
        -x RCCL_ROCSHMEM_THRESHOLD=134217728 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
        -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_PLUGIN=none -x NCCL_NET_PLUGIN=none
        -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
        -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1)
run() { echo "########## $1 ##########"; shift; mpirun -n 8 "${MPI_OPT[@]}" "${GINENV[@]}" "$@" \
  2>&1 | grep -E "^ +[0-9]+ +[0-9]+ +float"; }

run "alltoall -D3 (SDMA egress ceiling)" /rt-build/alltoall_perf \
  -b 67108864 -e 2147483648 -f 4 -g 1 -R 2 -A 1 -V 32 -D 3 -c 0 -n 20 -w 5 -d float -z 0
run "all_gather -D3 (SDMA)" /rt-build/all_gather_perf \
  -b 67108864 -e 2147483648 -f 4 -g 1 -R 2 -V 32 -D 3 -c 0 -n 20 -w 5 -d float -z 0
run "reduce_scatter -D3 (CU LSA read-reduce, our kernel)" /rt-build/reduce_scatter_perf \
  -b 67108864 -e 2147483648 -f 4 -g 1 -R 2 -V 32 -D 3 -c 0 -n 20 -w 5 -o sum -d float -z 0
echo "CEILING_DONE"
