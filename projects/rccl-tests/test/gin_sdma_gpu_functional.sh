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
# Usage: gin_sdma_gpu_functional.sh <broadcast|all_gather|alltoall|scatter|gather|sendrecv|reduce_scatter> <bin_dir> <np> [launcher]
set -euo pipefail

COLL="${1:?collective (broadcast|all_gather|alltoall|scatter|gather|sendrecv|reduce_scatter|reduce|all_reduce)}"
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
  reduce_scatter) BIN="reduce_scatter_perf" ;;
  reduce)     BIN="reduce_perf" ;;
  all_reduce) BIN="all_reduce_perf" ;;
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
    # the last rank, across the tier ladder, out-of-place and in-place. Both
    # Scatter and Gather cross the LL tiny-chunk path in this default sweep (on by
    # default <=2 KiB/rank chunk).
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
    # Scatter/Gather LL is on by default; add an LL-disabled pass so the tiny-chunk
    # LSA path stays covered too.
    LLENV="NCCL_GIN_ANVIL_$(echo "$COLL" | tr a-z A-Z)_LL_MAX_BYTES"
    echo "=== [$COLL] LL disabled (${LLENV}=0) ==="
    set -x
    "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
      "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" -x "${LLENV}=0" \
      "$EXE" -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$FACTOR" -g 1 -R 2 -V "$CTA" \
      -D 3 -c 1 -n "$ITERS" -w "$WARMUP" -r 0 -z 0
    set +x
    if [[ "$COLL" == "scatter" ]]; then
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
  reduce_scatter)
    # First reduction collective: two size tiers (LSA read-reduce small,
    # put-partials + SM reduce large) plus an op dimension. Start at 128 B so the
    # per-rank slice is a nonzero 16 B-aligned count (total/NP). Default op sweep
    # covers sum; the small-size op-matrix pass exercises Apply for every
    # supported op x type against the verifiable oracle.
    RS_MIN=$(( NP * 16 )); [[ "$MIN_BYTES" -gt "$RS_MIN" ]] && RS_MIN="$MIN_BYTES"
    for zp in 0 1; do
      echo "=== [$COLL] sweep op=sum $( [[ $zp == 0 ]] && echo out-of-place || echo in-place ) ==="
      set -x
      "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
        "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" \
        "$EXE" -b "$RS_MIN" -e "$MAX_BYTES" -f "$FACTOR" -g 1 -R 2 -V "$CTA" \
        -D 3 -c 1 -n "$ITERS" -w "$WARMUP" -o sum -z "$zp"
      set +x
    done
    # Force the large put-partials + SM-reduce tier (default cutover is 256 KiB).
    # Start at 1 MiB, not RS_MIN: forcing GIN at very small sizes hits a
    # pre-existing GIN/SDMA cold-start hang common to ALL GIN collectives (a
    # control run of alltoall_perf with NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLTOALL=0
    # -b 128 hangs at the same intermittent rate), so tiny forced-GIN is out of
    # scope here. The GIN large tier is stable at realistic slice sizes (>=16 KiB
    # in stress runs); 1 MiB gives ample margin while exercising the same kernel.
    RS_GIN_MIN="${GIN_SDMA_RS_GIN_MIN:-1048576}"
    [[ "$RS_GIN_MIN" -lt "$RS_MIN" ]] && RS_GIN_MIN="$RS_MIN"
    echo "=== [$COLL] GIN-tier forced (NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCESCATTER=0, -b ${RS_GIN_MIN}) ==="
    set -x
    "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
      "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" -x NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCESCATTER=0 \
      "$EXE" -b "$RS_GIN_MIN" -e "$MAX_BYTES" -f "$FACTOR" -g 1 -R 2 -V "$CTA" \
      -D 3 -c 1 -n "$ITERS" -w "$WARMUP" -o sum -z 0
    set +x
    # Op x type matrix on a small range (LSA tier): every supported op and every
    # type must match the verifiable oracle. fp8 prod/mulsum are skipped by the
    # driver; PreMulSum ("mulsum") is deferred (dispatch -> testNotImplemented).
    echo "=== [$COLL] op x type matrix, small range (-o all -d all) ==="
    set -x
    "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
      "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" \
      "$EXE" -b "$RS_MIN" -e 64K -f 4 -g 1 -R 2 -V "$CTA" \
      -D 3 -c 1 -n "$ITERS" -w "$WARMUP" -o all -d all -z 0
    set +x
    ;;
  reduce)
    # Reduce as reduce-scatter-to-root: each rank folds its owned slice from
    # every peer's sendbuff (ascending source order, verifiable oracle) and
    # writes the reduced slice into the root's recvbuff; non-root recvbuffs are
    # untouched. Exercise rank==root and non-root write paths on root 0 and the
    # last rank, both out-of-place and in-place, plus an op x type matrix. Start
    # at NP*16 B so the count aligns to the NP*VEC (16 B line) boundary.
    RED_MIN=$(( NP * 16 )); [[ "$MIN_BYTES" -gt "$RED_MIN" ]] && RED_MIN="$MIN_BYTES"
    for root in 0 $((NP - 1)); do
      for zp in 0 1; do
        echo "=== [$COLL] sweep root=$root op=sum $( [[ $zp == 0 ]] && echo out-of-place || echo in-place ) ==="
        set -x
        "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
          "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" \
          "$EXE" -b "$RED_MIN" -e "$MAX_BYTES" -f "$FACTOR" -g 1 -R 2 -V "$CTA" \
          -D 3 -c 1 -n "$ITERS" -w "$WARMUP" -r "$root" -o sum -z "$zp"
        set +x
      done
    done
    # Op x type matrix on a small range (root 0): every supported op and type
    # must match the verifiable oracle. fp8 prod and PreMulSum ("mulsum") are
    # skipped by the driver on the device path (SPECIALIZE_REDUCE_KERNEL nullptr).
    echo "=== [$COLL] op x type matrix, small range (-o all -d all, root 0) ==="
    set -x
    "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
      "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" \
      "$EXE" -b "$RED_MIN" -e 64K -f 4 -g 1 -R 2 -V "$CTA" \
      -D 3 -c 1 -n "$ITERS" -w "$WARMUP" -o all -d all -r 0 -z 0
    set +x
    ;;
  all_reduce)
    # P3 reduction collective; the GIN path is -D 5 (single-launch RS+AG compose),
    # NOT the -D 3 that run() hardcodes, so this case launches directly. Size-hybrid:
    # small out-of-place -> one-shot direct LSA read-reduce; large OR in-place ->
    # two-shot ReduceScatter+AllGather (grid self-caps to 16 CTAs regardless of -V).
    # all_reduce_perf runs in-place and out-of-place per size, so a full sweep covers
    # one-shot (small OOP), two-shot IP (small), and two-shot both (large).
    for zp in 0 1; do
      echo "=== [$COLL] sweep op=sum $( [[ $zp == 0 ]] && echo out-of-place || echo in-place ) (-D 5) ==="
      set -x
      "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
        "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" \
        "$EXE" -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$FACTOR" -g 1 -R 2 -V "$CTA" \
        -D 5 -c 1 -n "$ITERS" -w "$WARMUP" -o sum -z "$zp"
      set +x
    done
    # Force the all-GIN two-shot tier (threshold=0) so the RS+AG kernel path stays
    # covered even below the tuned cutover. Start at 1 MiB (like reduce_scatter): a
    # forced-GIN sweep from very small sizes hits the pre-existing GIN/SDMA
    # cold-start hang common to ALL GIN collectives, so tiny forced-GIN is out of
    # scope. The two-shot path is stable at realistic sizes.
    AR_GIN_MIN="${GIN_SDMA_AR_GIN_MIN:-1048576}"
    [[ "$AR_GIN_MIN" -lt "$MIN_BYTES" ]] && AR_GIN_MIN="$MIN_BYTES"
    echo "=== [$COLL] GIN two-shot forced (NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLREDUCE=0, -b ${AR_GIN_MIN}) ==="
    set -x
    "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
      "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" -x NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLREDUCE=0 \
      "$EXE" -b "$AR_GIN_MIN" -e "$MAX_BYTES" -f "$FACTOR" -g 1 -R 2 -V "$CTA" \
      -D 5 -c 1 -n "$ITERS" -w "$WARMUP" -o sum -z 0
    set +x
    # Op x type matrix on a small range: every supported op and type must match the
    # verifiable oracle. fp8 {prod,avg} and PreMulSum ("mulsum") are skipped by the
    # driver on the device path (SPECIALIZE_REDUCE_KERNEL nullptr).
    echo "=== [$COLL] op x type matrix, small range (-o all -d all) ==="
    set -x
    "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" \
      "${MPI_ENV[@]}" "${EXTRA_ENV[@]}" \
      "$EXE" -b "$MIN_BYTES" -e 64K -f 4 -g 1 -R 2 -V "$CTA" \
      -D 5 -c 1 -n "$ITERS" -w "$WARMUP" -o all -d all -z 0
    set +x
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
