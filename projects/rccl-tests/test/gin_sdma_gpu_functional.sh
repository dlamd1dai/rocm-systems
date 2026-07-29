#!/usr/bin/env bash
# GPU functional exercise for the GIN-SDMA collective designs (deviceImpl == 3).
#
# Purpose: drive the real device kernels (GinHybridBroadcastKernel /
# GinScatterAllgatherBroadcastKernel / GinHybridAllGatherKernel /
# GinHybridAlltoAllKernel / GinScatterKernel / GinGatherKernel /
# GinSendRecvKernel) across a size/root/in-place matrix that crosses every
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
#   Scatter   : LL (<=2 KiB/rank chunk) -> LSA (<=128 KiB/rank chunk) -> GIN (root fan-out; OOP + in-place)
#   Gather    : LSA (<=256 KiB/rank chunk) -> GIN   (root fan-in;  OOP + in-place)
#   SendRecv  : LSA (<=256 KiB message)    -> GIN   (ring; OOP only)
#
# Usage: gin_sdma_gpu_functional.sh <broadcast|all_gather|alltoall|scatter|gather|sendrecv> <bin_dir> <np> [launcher]
set -euo pipefail

COLL="${1:?collective (broadcast|all_gather|alltoall|scatter|gather|sendrecv)}"
BIN_DIR="${2:?build dir containing *_perf}"
NP="${3:-8}"
LAUNCHER="${4:-mpirun}"

case "$COLL" in
  broadcast)  BIN="broadcast_perf" ;;
  all_gather) BIN="all_gather_perf" ;;
  alltoall)   BIN="alltoall_perf" ;;
  scatter)    BIN="scatter_perf" ;;
  gather)     BIN="gather_perf" ;;
  sendrecv)   BIN="sendrecv_perf" ;;
  *) echo "unknown collective: $COLL" >&2; exit 2 ;;
esac

EXE="${BIN_DIR}/${BIN}"
if [[ ! -x "$EXE" ]]; then
  echo "perf binary not found/executable: $EXE" >&2
  exit 2
fi

MIN_BYTES="${GIN_SDMA_MIN_BYTES:-8}"
# Default 64M keeps the gate fast; the GIN tier is size-safe above 1 GiB because
# every GIN put is split into <=1 GiB segments (common.h::ginPutChunked, the HW
# 30-bit copy-count max), so GIN_SDMA_MAX_BYTES may be raised past 1 GiB to
# exercise the chunk boundary without the old copy-count truncation.
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
  scatter|gather)
    # Root fan-out/fan-in: exercise rank==root and non-root paths on root 0 and
    # the last rank, across the tier ladder, out-of-place and in-place. For
    # Scatter this default sweep also crosses the LL tiny-chunk path (on by
    # default <=2 KiB/rank chunk); Gather has no LL tier.
    for root in 0 $((NP - 1)); do
      run "sweep root=$root out-of-place" -r "$root" -z 0
      run "sweep root=$root in-place"     -r "$root" -z 1
    done
    # Gather defaults to LSA at all sizes (tuned); force the GIN tier so its
    # kernel path stays covered. Scatter already crosses into GIN by 2 MiB, but a
    # forced pass is cheap and keeps both collectives symmetric here.
    THRENV="NCCL_GIN_ANVIL_SDMA_THRESHOLD_$(echo "$COLL" | tr a-z A-Z)"
    echo "=== [$COLL] GIN-tier forced (${THRENV}=0) ==="
    set -x
    "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
      "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" -x "${THRENV}=0" \
      "$EXE" -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$FACTOR" -g 1 -R 2 -V "$CTA" \
      -D 3 -c 1 -n "$ITERS" -w "$WARMUP" -r 0 -z 0
    set +x
    # Scatter LL is on by default; add an LL-disabled pass so the tiny-chunk LSA
    # store path stays covered too. (Gather has no LL tier, so skip it there.)
    if [[ "$COLL" == "scatter" ]]; then
      echo "=== [$COLL] LL disabled (NCCL_GIN_ANVIL_SCATTER_LL_MAX_BYTES=0) ==="
      set -x
      "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
        "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" -x NCCL_GIN_ANVIL_SCATTER_LL_MAX_BYTES=0 \
        "$EXE" -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$FACTOR" -g 1 -R 2 -V "$CTA" \
        -D 3 -c 1 -n "$ITERS" -w "$WARMUP" -r 0 -z 0
      set +x
      # LSA root fan-out defaults to peer-interleaved; add a sequential-layout
      # pass so the historical one-link-at-a-time store path stays covered too.
      echo "=== [$COLL] LSA sequential fan-out (NCCL_GIN_ANVIL_SCATTER_LSA_INTERLEAVE=0) ==="
      set -x
      "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
        "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" -x NCCL_GIN_ANVIL_SCATTER_LSA_INTERLEAVE=0 \
        "$EXE" -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$FACTOR" -g 1 -R 2 -V "$CTA" \
        -D 3 -c 1 -n "$ITERS" -w "$WARMUP" -r 0 -z 0
      set +x
    fi
    ;;
  sendrecv)
    # Ring send/recv; in-place is not validated for sendrecv (OOP only). The
    # default sweep exercises the LL tiny-message path (on by default <=2 KiB)
    # and the LSA path above it.
    run "sweep out-of-place (LL default on)" -z 0
    # Force the GIN tier so its kernel path stays covered (SendRecv defaults to
    # LSA at all sizes).
    echo "=== [$COLL] GIN-tier forced (NCCL_GIN_ANVIL_SDMA_THRESHOLD_SENDRECV=0) ==="
    set -x
    "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
      "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" -x NCCL_GIN_ANVIL_SDMA_THRESHOLD_SENDRECV=0 \
      "$EXE" -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$FACTOR" -g 1 -R 2 -V "$CTA" \
      -D 3 -c 1 -n "$ITERS" -w "$WARMUP" -z 0
    set +x
    # Disable LL (NCCL_GIN_ANVIL_SENDRECV_LL_MAX_BYTES=0) so the tiny-message LSA
    # store path stays covered too.
    echo "=== [$COLL] LL disabled (NCCL_GIN_ANVIL_SENDRECV_LL_MAX_BYTES=0) ==="
    set -x
    "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
      "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" -x NCCL_GIN_ANVIL_SENDRECV_LL_MAX_BYTES=0 \
      "$EXE" -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$FACTOR" -g 1 -R 2 -V "$CTA" \
      -D 3 -c 1 -n "$ITERS" -w "$WARMUP" -z 0
    set +x
    ;;
esac

echo "PASS: $COLL GIN-SDMA GPU functional sweep"
