#!/usr/bin/env bash
# AllToAllv (A2AV) perf eval: best HOST-initiated (-D 0) vs best GIN Anvil-SDMA
# (-D 3), single-node xGMI. Run INSIDE the rccl-gin-gda-sdma container (it drives
# mpirun directly, like test/gin_sdma_gpu_functional.sh); the MPI env mirrors
# ddai-artifacts/scripts/gin-sdma-a2a-test.bash so A2AV numbers are comparable to
# the A2A harness.
#
# What "best" means here (each path run in its own optimal config, then the
# per-size envelope is taken):
#   HOST  : RING (SM copy, pinned channels; best small/mid) and CE/SDMA
#           (NCCL_CUMEM + symmetric-reg -R 2 + NCCL_CTA_POLICY=ZERO; best large).
#   GIN   : our -D 3 GinHybridAlltoAllvKernel forced all-LSA (small) and forced
#           all-GIN/SDMA (large). The max of the two per size is what an ideally
#           tuned kAllToAllvSdmaThresholdDefault would deliver.
#
# Timing is launch-inclusive for BOTH paths (no HIP-graph capture), so the
# comparison is apples-to-apples. NOTE: -G graph replay is intentionally OFF by
# default (A2AV_CUDAGRAPH=0). rccl-tests' TimeTest wraps the per-size warm-up
# loop inside cudaStreamBeginCapture, and the A2AV -D 3 kernel lazily hipMalloc's
# its per-peer metadata buffer on the first RunColl of each size -- which lands
# inside that capture and is illegal ("operation not permitted when stream is
# capturing"). Enabling -G needs a capture-safe metadata path (pass the small
# per-peer arrays by value, or pre-allocate outside capture); until then leave
# A2AV_CUDAGRAPH=0.
#
# Usage: gin-sdma-a2av-test.bash <bin_dir> [np] [launcher]
#   bin_dir : dir containing a freshly-built alltoallv_perf (with the -D 3 kernel)
set -euo pipefail

BIN_DIR="${1:?bin dir containing alltoallv_perf}"
NP="${2:-8}"
LAUNCHER="${3:-mpirun}"
EXE="${BIN_DIR}/alltoallv_perf"
[[ -x "$EXE" ]] || { echo "alltoallv_perf not found/executable: $EXE" >&2; exit 2; }

MIN_BYTES="${A2AV_MIN_BYTES:-16K}"
MAX_BYTES="${A2AV_MAX_BYTES:-64M}"
FACTOR="${A2AV_FACTOR:-2}"
ITERS="${A2AV_ITERS:-20}"
WARMUP="${A2AV_WARMUP:-5}"
CTA="${A2AV_CTA:-32}"                 # -V device CTA count for the -D 3 kernel
# Forced-GIN cold-start floor: forcing SDMA at tiny sizes can hit the pre-existing
# GIN cold-start hang common to all GIN collectives, so start the GIN pass here.
GIN_MIN_BYTES="${A2AV_GIN_MIN_BYTES:-1M}"
# -G replay count for the GIN path. Default 0 (off): the A2AV -D 3 kernel is not
# yet HIP-graph-capture-safe (lazy hipMalloc of per-peer metadata during the
# captured warm-up). See header note before setting >0.
CUDAGRAPH="${A2AV_CUDAGRAPH:-0}"
HOST_NCHANNELS="${A2AV_HOST_NCHANNELS:-32}"
ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD:-$((128 * 1024 * 1024))}"
OUTDIR="${A2AV_OUTDIR:-/tmp/a2av-perf}"
mkdir -p "$OUTDIR"

MPI_OPT=(--allow-run-as-root
         -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none
         -mca hwloc_base_binding_policy none)

# Env common to every path.
BASE=(-x OMPI_ALLOW_RUN_AS_ROOT=1
      -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
      -x RCCL_ROCSHMEM_ENABLE=0
      -x RCCL_ROCSHMEM_THRESHOLD="${ROCSHMEM_THRESHOLD}"
      -x NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
      -x RCCL_ENABLE_INTRANET=1
      -x NCCL_DMABUF_ENABLE=1
      -x NCCL_MSCCL_ENABLE=0
      -x HSA_NO_SCRATCH_RECLAIM=1)

# GIN Anvil-SDMA base env (NCCL_GIN_TYPE=6), mirrors the A2A Test#5 block.
GIN_ENV=(-x ROCSHMEM_BACKEND=ipc
         -x ROCSHMEM_DISABLE_MIXED_IPC=1
         -x ROCSHMEM_DEBUG_LEVEL=info:noversion
         -x NCCL_GIN_PLUGIN=none
         -x NCCL_CUMEM_ENABLE=1
         -x NCCL_NET_PLUGIN=none
         -x ROCSHMEM_SDMA_ENABLED=0
         -x NCCL_GIN_ENABLE=1
         -x NCCL_GIN_TYPE=6
         -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS="${A2AV_NUM_CHANNELS:-1}"
         -x HSA_FORCE_FINE_GRAIN_PCIE=1
         # backend gin.put SDMA control: 0 = every GIN put drives the copy engine.
         -x NCCL_GIN_ANVIL_SDMA_THRESHOLD=0)

read -r -a EXTRA_LAUNCH_ARGS <<< "${A2AV_LAUNCH_ARGS:-}"
read -r -a EXTRA_ENV <<< "${A2AV_EXTRA_ENV:-}"

GRAPH_ARGS=()
[[ "$CUDAGRAPH" != 0 ]] && GRAPH_ARGS=(-G "$CUDAGRAPH")

# --- host RING: SM-copy ring, GIN off, pinned channels, no cumem, -R 0 ---
run_host_ring() {
  echo "=== [a2av] HOST RING (SM copy, channels=${HOST_NCHANNELS}, -D 0 -R 0) ==="
  set -x
  "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" "${BASE[@]}" "${EXTRA_ENV[@]}" \
    -x NCCL_CUMEM_ENABLE=0 -x NCCL_GIN_ENABLE=0 -x NCCL_GIN_TYPE=0 \
    -x NCCL_MIN_NCHANNELS="${HOST_NCHANNELS}" -x NCCL_MAX_NCHANNELS="${HOST_NCHANNELS}" \
    "$EXE" -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$FACTOR" -g 1 -R 0 -D 0 -A 1 -V 1 \
    -c 1 -w "$WARMUP" -n "$ITERS"
  set +x
}

# --- host CE/SDMA: copy engines, cumem + symmetric reg -R 2 + CTA_POLICY=ZERO ---
run_host_ce() {
  echo "=== [a2av] HOST CE/SDMA (cumem + -R 2 + CTA_POLICY=ZERO, -D 0) ==="
  set -x
  "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" "${BASE[@]}" "${EXTRA_ENV[@]}" \
    -x NCCL_CUMEM_ENABLE=1 -x NCCL_GIN_ENABLE=0 -x NCCL_GIN_TYPE=0 -x NCCL_CTA_POLICY=ZERO \
    "$EXE" -b "$MIN_BYTES" -e "$MAX_BYTES" -f "$FACTOR" -g 1 -R 2 -D 0 -A 1 -V 1 \
    -c 1 -w "$WARMUP" -n "$ITERS"
  set +x
}

# --- GIN -D 3 kernel, tier forced via the kernel A2AV threshold ---
#     $1 = NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLTOALLV (0=all-GIN, huge=all-LSA)
#     $2 = -b start bytes
run_gin() {
  local thr="$1" mn="$2"
  set -x
  "$LAUNCHER" -n "$NP" "${MPI_OPT[@]}" "${EXTRA_LAUNCH_ARGS[@]}" "${BASE[@]}" "${GIN_ENV[@]}" "${EXTRA_ENV[@]}" \
    -x NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLTOALLV="$thr" \
    "$EXE" -b "$mn" -e "$MAX_BYTES" -f "$FACTOR" -g 1 -R 2 -D 3 -A 1 -V "$CTA" \
    -c 1 -w "$WARMUP" -n "$ITERS" "${GRAPH_ARGS[@]}"
  set +x
}

# Extract "size<TAB>oop_busbw" rows (out-of-place busbw is column 8) into a tagged
# tsv for the summary. Skips the size==0 rows and any non-data lines.
extract() {  # $1 tag  $2 logfile
  awk -v tag="$1" '
    $3=="float" && $1 ~ /^[0-9]+$/ && $1+0>0 { printf "%s\t%s\t%s\n", tag, $1, $8 }
  ' "$2" >> "${OUTDIR}/all.tsv"
}

: > "${OUTDIR}/all.tsv"

echo "############ HOST-initiated A2AV ############"
run_host_ring 2>&1 | tee "${OUTDIR}/host-ring.log";  extract host-ring "${OUTDIR}/host-ring.log"
run_host_ce   2>&1 | tee "${OUTDIR}/host-ce.log";    extract host-ce   "${OUTDIR}/host-ce.log"

echo "############ GIN Anvil-SDMA A2AV (-D 3) ############"
echo "=== [a2av] GIN forced all-LSA (THRESHOLD_ALLTOALLV=huge, V=${CTA}, graph=${CUDAGRAPH}) ==="
run_gin 2147483647 "$MIN_BYTES" 2>&1 | tee "${OUTDIR}/gin-lsa.log"; extract gin-lsa "${OUTDIR}/gin-lsa.log"
echo "=== [a2av] GIN forced all-GIN/SDMA (THRESHOLD_ALLTOALLV=0, V=${CTA}, graph=${CUDAGRAPH}, -b ${GIN_MIN_BYTES}) ==="
run_gin 0 "$GIN_MIN_BYTES" 2>&1 | tee "${OUTDIR}/gin-gin.log";      extract gin-gin "${OUTDIR}/gin-gin.log"

echo
echo "################ A2AV best-of summary (out-of-place busbw, GB/s) ################"
echo "# HOST = max(host-ring, host-ce);  GIN = max(gin-lsa, gin-gin) [the -D 3 envelope]"
awk '
  { tag=$1; sz=$2+0; bw=$3+0; seen[sz]=1
    if (tag=="host-ring" || tag=="host-ce") { if (bw>host[sz]) { host[sz]=bw; hmode[sz]=tag } }
    else                                    { if (bw>gin[sz])  { gin[sz]=bw;  gmode[sz]=tag } }
  }
  END {
    n=0; for (s in seen) sizes[n++]=s
    for (i=0;i<n;i++) for (j=i+1;j<n;j++) if (sizes[i]+0>sizes[j]+0) { t=sizes[i]; sizes[i]=sizes[j]; sizes[j]=t }
    printf "%12s  %12s %-9s  %12s %-8s  %8s\n", "bytes", "best_host", "(mode)", "best_gin", "(mode)", "gin/host"
    for (i=0;i<n;i++) { s=sizes[i]; h=host[s]+0; g=gin[s]+0
      hm=(s in host)?hmode[s]:"-"; gm=(s in gin)?gmode[s]:"-"
      r=(h>0 && g>0)?sprintf("%.2fx", g/h):"-"
      printf "%12d  %12.2f %-9s  %12.2f %-8s  %8s\n", s, h, hm, g, gm, r
    }
  }
' "${OUTDIR}/all.tsv"
echo "# raw logs: ${OUTDIR}/{host-ring,host-ce,gin-lsa,gin-gin}.log"
