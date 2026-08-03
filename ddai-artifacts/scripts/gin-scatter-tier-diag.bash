#!/usr/bin/env bash
# Diagnose the Scatter 2 MiB busbw cliff (M0 board: gin/host 57.6% @2 MiB total).
# Runs the SAME /rt-build/scatter_perf -D 3 across the 2 MiB..32 MiB band under
# three tier configs to localize the fix:
#   A default   : tuned 128 KiB/rank LSA<->GIN threshold (current behavior).
#   B all-LSA   : force LSA everywhere (THRESHOLD=2G) -> is LSA faster than GIN here?
#   C all-GIN   : force GIN everywhere (THRESHOLD=0)  -> pure GIN fan-out curve.
#   D GIN + interleave-off : isolate the LSA interleave knob (only affects LSA; sanity).
# Run INSIDE the rccl-gin-gda-sdma-713 container with -v ~/rt-build:/rt-build.
set -uo pipefail
EXE=/rt-build/scatter_perf
B="${B:-1048576}"; E="${E:-33554432}"; F=2; N=20; W=5; CTA=32
MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
GINENV=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 -x NCCL_DEBUG=WARN
        -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_CUMEM_ENABLE=1
        -x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1
        -x RCCL_ROCSHMEM_THRESHOLD=134217728 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
        -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_PLUGIN=none -x NCCL_NET_PLUGIN=none
        -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
        -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1)
run() { echo "########## $1 ##########"; shift
  mpirun -n 8 "${MPI_OPT[@]}" "${GINENV[@]}" "$@" "$EXE" \
    -b "$B" -e "$E" -f "$F" -g 1 -R 2 -V "$CTA" -D 3 -c 0 -n "$N" -w "$W" -r 0 -z 0 -d float \
    2>&1 | grep -E "^[[:space:]]*[0-9]+[[:space:]]+[0-9]+"; }

run "A default (tuned 128KiB/rank threshold)"
run "B all-LSA (THRESHOLD_SCATTER=2G)"   -x NCCL_GIN_ANVIL_SDMA_THRESHOLD_SCATTER=2147483648
run "C all-GIN (THRESHOLD_SCATTER=0)"    -x NCCL_GIN_ANVIL_SDMA_THRESHOLD_SCATTER=0
run "D all-LSA interleave-off"           -x NCCL_GIN_ANVIL_SDMA_THRESHOLD_SCATTER=2147483648 -x NCCL_GIN_ANVIL_SCATTER_LSA_INTERLEAVE=0
echo "SCATTER_DIAG_DONE"
