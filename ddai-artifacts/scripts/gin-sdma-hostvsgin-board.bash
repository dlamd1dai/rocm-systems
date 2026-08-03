#!/usr/bin/env bash
# M0 baseline/roofline board for the GIN-SDMA perf-optimization plan.
#
# For each collective, sweep the SAME binary at host (-D 0) and GIN-SDMA (-D 3),
# -c 0 (perf), out-of-place, and print a per-size gap table:
#   size   host_busbw   gin_busbw   gin/host%
# so M1 targeting is data-driven. Roofline is measured separately by
# gin-fabric-ceiling.bash (xGMI read/write ceiling at large sizes).
#
# Generalizes ddai-artifacts/scripts/gin-rs-hostvsgin-sweep.bash across all 7
# collectives in scope (A2A / AllReduce excluded per the plan). Run INSIDE the
# rccl-gin-gda-sdma-713 container with -v ~/rt-build:/rt-build so the same fresh
# binaries drive both modes for an apples-to-apples comparison.
#
# Usage: gin-sdma-hostvsgin-board.bash [bin_dir] [np] ["coll1 coll2 ..."]
#   bin_dir default /rt-build ; np default 8 ; colls default = all 7 in scope.
set -uo pipefail

BIN_DIR="${1:-/rt-build}"
NP="${2:-8}"
read -r -a COLLS <<< "${3:-broadcast all_gather scatter gather sendrecv reduce_scatter reduce}"

B="${BOARD_MIN:-128}"; E="${BOARD_MAX:-2G}"; F="${BOARD_FACTOR:-2}"
N="${BOARD_ITERS:-20}"; W="${BOARD_WARMUP:-5}"; CTA="${BOARD_CTA:-32}"
OUTDIR="${BOARD_OUTDIR:-/tmp/gin-sdma-board}"
mkdir -p "$OUTDIR"

MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
COMMON=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        -x NCCL_DEBUG=WARN -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_CUMEM_ENABLE=1)
GINENV=(-x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1
        -x RCCL_ROCSHMEM_THRESHOLD=134217728 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
        -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_PLUGIN=none -x NCCL_NET_PLUGIN=none
        -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
        -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1)

# Per-collective binary + extra flags (reduction ops, root).
bin_of()   { case "$1" in
  broadcast) echo broadcast_perf ;; all_gather) echo all_gather_perf ;;
  scatter) echo scatter_perf ;; gather) echo gather_perf ;;
  sendrecv) echo sendrecv_perf ;; reduce_scatter) echo reduce_scatter_perf ;;
  reduce) echo reduce_perf ;; all_reduce) echo all_reduce_perf ;; alltoall) echo alltoall_perf ;;
  *) echo "" ;; esac; }
flags_of() { case "$1" in
  reduce_scatter|reduce|all_reduce) echo "-o sum -d float" ;;
  *) echo "-d float" ;; esac; }
root_of()  { case "$1" in broadcast|scatter|gather|reduce) echo "-r 0" ;; *) echo "" ;; esac; }

# busbw = field immediately before the first "#wrong" value. With -c 0 rccl-tests
# prints "N/A" there, so this is robust to per-collective column layout (redop /
# root columns vary). NOTE: key ONLY off "N/A" -- a bare "0" would false-match the
# reduce/broadcast/scatter/gather "root" column (root 0). Emits "<size> <busbw>".
parse_busbw() {  # $1 = raw log
  awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ {
         for (i=4;i<=NF;i++) if ($i=="N/A") { print $1, $(i-1); break }
       }' "$1"
}

for c in "${COLLS[@]}"; do
  bin="$(bin_of "$c")"; exe="${BIN_DIR}/${bin}"
  if [[ ! -x "$exe" ]]; then echo "SKIP $c: $exe not found"; continue; fi
  extra="$(flags_of "$c") $(root_of "$c")"
  hlog="$OUTDIR/${c}.host.log"; glog="$OUTDIR/${c}.gin.log"

  echo "########## $c : HOST (-D 0) ##########"
  # shellcheck disable=SC2086
  mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" \
    "$exe" -b "$B" -e "$E" -f "$F" -g 1 -R 2 -D 0 -c 0 -n "$N" -w "$W" -z 0 $extra \
    2>&1 | tee "$hlog" | grep -E "^[[:space:]]*[0-9]+[[:space:]]+[0-9]+|Avg bus" || echo "HOST_RC=$?"

  echo "########## $c : GIN-SDMA (-D 3) ##########"
  # shellcheck disable=SC2086
  mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" \
    "$exe" -b "$B" -e "$E" -f "$F" -g 1 -R 2 -V "$CTA" -D 3 -c 0 -n "$N" -w "$W" -z 0 $extra \
    2>&1 | tee "$glog" | grep -E "^[[:space:]]*[0-9]+[[:space:]]+[0-9]+|Avg bus" || echo "GIN_RC=$?"

  echo ""
  echo "===== GAP BOARD: $c  (size  host_GBs  gin_GBs  gin/host%) ====="
  join -j1 <(parse_busbw "$hlog" | sort -k1,1) <(parse_busbw "$glog" | sort -k1,1) \
    | sort -k1,1n \
    | awk '{ pct=($2>0)?($3/$2*100):0; printf "%12s  %9.2f  %9.2f  %7.1f%%\n",$1,$2,$3,pct }'
  echo "==============================================================="
  echo ""
done

echo "BOARD_DONE  (raw logs in $OUTDIR)"
