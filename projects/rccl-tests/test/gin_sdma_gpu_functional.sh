#!/usr/bin/env bash
# GPU functional exercise for the GIN-SDMA collective designs (deviceImpl == 3).
#
# Purpose: drive the real device kernels (GinHybridBroadcastKernel /
# GinScatterAllgatherBroadcastKernel / GinHybridAllGatherKernel /
# GinHybridAlltoAllKernel) across a size/root/in-place matrix that crosses every
# tier boundary, with the built-in correctness check (-c 1). rccl-tests' main
# returns non-zero when any element is wrong, so a clean exit == all exercised
# branches produced correct data. This is the "device branch checklist" half of
# the coverage plan (the host tier-decision logic is covered to ~100% by the
# gin_sdma_policy_test gtest binary).
#
# The MPI env / mca flags and the required -D 3 perf flags (-R 2 symmetric
# register mode, -g 1, -V <CTAs>, NCCL_GIN_TYPE=6 SDMA backend) mirror the
# canonical gate scripts (ddai-artifacts/scripts/gin-sdma-*-test.bash) so this
# runs identically to the measured perf gates, just with the size sweep and the
# correctness assertion foregrounded.
#
# Tier boundaries exercised (see gin_sdma_collective_policy.h):
#   Broadcast : LL (<=2 KiB) -> LSA (<=256 KiB) -> flat GIN (<2 MiB) -> scatter+AG (>=2 MiB)
#   AllGather : LL (<=4 KiB/rank) -> LSA single-CTA (<=8 KiB) -> LSA multi-CTA (<=256 KiB) -> GIN
#   AllToAll  : LSA (<=256 KiB/peer) -> GIN ; LL tier is opt-in and exercised via env
#
# Usage: gin_sdma_gpu_functional.sh <broadcast|all_gather|alltoall> <bin_dir> <np> [launcher]
set -euo pipefail

COLL="${1:?collective (broadcast|all_gather|alltoall)}"
BIN_DIR="${2:?build dir containing *_perf}"
NP="${3:-8}"
LAUNCHER="${4:-mpirun}"

case "$COLL" in
  broadcast)  BIN="broadcast_perf" ;;
  all_gather) BIN="all_gather_perf" ;;
  alltoall)   BIN="alltoall_perf" ;;
  *) echo "unknown collective: $COLL" >&2; exit 2 ;;
esac

EXE="${BIN_DIR}/${BIN}"
if [[ ! -x "$EXE" ]]; then
  echo "perf binary not found/executable: $EXE" >&2
  exit 2
fi

MIN_BYTES="${GIN_SDMA_MIN_BYTES:-8}"
MAX_BYTES="${GIN_SDMA_MAX_BYTES:-64M}"
FACTOR="${GIN_SDMA_FACTOR:-2}"
CTA="${GIN_SDMA_CTA:-32}"       # -V device CTA count (matches gate scripts)
ITERS="${GIN_SDMA_ITERS:-5}"
WARMUP="${GIN_SDMA_WARMUP:-2}"
ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD:-$((128 * 1024 * 1024))}"

# MPI transport flags (single-node xGMI/IPC; same as gin-sdma-*-test.bash).
MPI_OPT=(--allow-run-as-root
         -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none
         -mca hwloc_base_binding_policy none)

# GIN Anvil-SDMA runtime env (NCCL_GIN_TYPE=6). Mirrors MPI_BASE + the -D 3
# block in the gate scripts.
MPI_ENV=(-x OMPI_ALLOW_RUN_AS_ROOT=1
         -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
         -x RCCL_ROCSHMEM_ENABLE=0
         -x ROCSHMEM_BACKEND=ipc
         -x ROCSHMEM_DISABLE_MIXED_IPC=1
         -x ROCSHMEM_DEBUG_LEVEL=info:noversion
         -x RCCL_ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD}"
         -x NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
         -x RCCL_ENABLE_INTRANET=1
         -x NCCL_DMABUF_ENABLE=1
         -x NCCL_MSCCL_ENABLE=0
         -x HSA_NO_SCRATCH_RECLAIM=1
         -x NCCL_GIN_PLUGIN=none
         -x NCCL_CUMEM_ENABLE=1
         -x NCCL_NET_PLUGIN=none
         -x ROCSHMEM_SDMA_ENABLED=0
         -x NCCL_GIN_ENABLE=1
         -x NCCL_GIN_TYPE=6
         -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS="${NUM_CHANNELS:-1}"
         -x HSA_FORCE_FINE_GRAIN_PCIE=1)

# Extra site-specific launcher args (e.g. rankfile), and extra -x env passthrough.
read -r -a EXTRA_LAUNCH_ARGS <<< "${GIN_SDMA_LAUNCH_ARGS:-}"
read -r -a EXTRA_ENV <<< "${GIN_SDMA_EXTRA_ENV:-}"

run() {
  local desc="$1"; shift
  echo "=== [$COLL] $desc ==="
  set -x
  "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
    "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" \
    "$EXE" -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$FACTOR" -g 1 -R 2 -V "$CTA" \
    -D 3 -c 1 -n "$ITERS" -w "$WARMUP" "$@"
  set +x
}

case "$COLL" in
  broadcast)
    # Root 0 and last rank: rank==root, non-root, and last-rank tail slice in the
    # scatter+allgather kernel. Out-of-place and in-place.
    for root in 0 $((NP - 1)); do
      run "sweep root=$root out-of-place" -r "$root" -z 0
      run "sweep root=$root in-place"     -r "$root" -z 1
    done
    ;;
  all_gather)
    run "sweep out-of-place" -z 0
    run "sweep in-place"     -z 1
    ;;
  alltoall)
    # AllToAll does not support in-place; default LL tier is OFF.
    run "sweep (LL off)"
    # Opt in to the LL tier to exercise AlltoAllLLImpl / a2aLLEligible on device.
    echo "=== [$COLL] small sweep, LL opt-in (NCCL_GIN_ANVIL_A2A_LL_MAX_BYTES=4096) ==="
    set -x
    "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
      "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" -x NCCL_GIN_ANVIL_A2A_LL_MAX_BYTES=4096 \
      "$EXE" -b "$MIN_BYTES" -e 8K -f "$FACTOR" -g 1 -R 2 -V "$CTA" \
      -D 3 -c 1 -n "$ITERS" -w "$WARMUP"
    set +x
    ;;
esac

echo "PASS: $COLL GIN-SDMA GPU functional sweep"
