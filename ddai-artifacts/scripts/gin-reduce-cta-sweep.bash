#!/usr/bin/env bash
# Reduce (-D 3) large-tier CTA (-V) sweep for the GIN-SDMA perf-optimization plan.
#
# The M0 board measured every collective at -V 32. The broadcast large-tier win
# turned out to be almost entirely CTA-count driven (238 -> 350 GB/s going
# -V 32 -> 128), because more CTAs are needed to saturate all of a GPU's xGMI
# links. Reduce is the root-INGRESS mirror (each rank writes its reduced slice
# into the root's recvbuff, spread over the root's incoming links), so the same
# lever may transfer for free. This script re-measures reduce large-tier:
#   * one warm host (-D 0) baseline, and
#   * GIN (-D 3) at several -V (CTA) counts,
# printing gin/host% per size so we can decide whether the CTA lift alone closes
# the ~75%-of-host plateau before designing a ring-reduce.
#
# Run INSIDE the rccl-gin-gda-sdma-713 container (same mounts as the board):
#   -v ~/rt-build:/rt-build -v ~/rocm-systems:/rocm-systems
#
# Usage: gin-reduce-cta-sweep.bash [bin_dir] [np] ["V1 V2 ..."]
set -uo pipefail

BIN_DIR="${1:-/rt-build}"
NP="${2:-8}"
read -r -a VLIST <<< "${3:-32 64 128}"

B="${SWEEP_MIN:-128M}"; E="${SWEEP_MAX:-2G}"; F="${SWEEP_FACTOR:-2}"
N="${SWEEP_ITERS:-20}"; W="${SWEEP_WARMUP:-5}"
OUTDIR="${SWEEP_OUTDIR:-/tmp/gin-reduce-cta}"
mkdir -p "$OUTDIR"

exe="${BIN_DIR}/reduce_perf"
if [[ ! -x "$exe" ]]; then echo "MISSING $exe"; exit 2; fi

MPI_OPT=(--allow-run-as-root -mca pml ob1 -mca btl self,vader,tcp
         -mca btl_vader_single_copy_mechanism none -mca hwloc_base_binding_policy none)
COMMON=(-x OMPI_ALLOW_RUN_AS_ROOT=1 -x OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        -x NCCL_DEBUG=WARN -x HSA_NO_SCRATCH_RECLAIM=1 -x NCCL_CUMEM_ENABLE=1)
GINENV=(-x RCCL_ROCSHMEM_ENABLE=0 -x ROCSHMEM_BACKEND=ipc -x ROCSHMEM_DISABLE_MIXED_IPC=1
        -x RCCL_ROCSHMEM_THRESHOLD=134217728 -x RCCL_ENABLE_INTRANET=1 -x NCCL_DMABUF_ENABLE=1
        -x NCCL_MSCCL_ENABLE=0 -x NCCL_GIN_PLUGIN=none -x NCCL_NET_PLUGIN=none
        -x ROCSHMEM_SDMA_ENABLED=0 -x NCCL_GIN_ENABLE=1 -x NCCL_GIN_TYPE=6
        -x NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS=1 -x HSA_FORCE_FINE_GRAIN_PCIE=1)
EXTRA=(-o sum -d float -r 0)

# busbw = field right before the first "N/A" (-c 0). Emits "<size> <busbw>".
parse_busbw() {
  awk '/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ {
         for (i=4;i<=NF;i++) if ($i=="N/A") { print $1, $(i-1); break }
       }' "$1"
}
# warm: discard first mpirun, keep the second (cold-start guard).
run_warm() { local log="$1"; shift; "$@" >/dev/null 2>&1 || true; "$@" 2>&1 | tee "$log" >/dev/null; }

hlog="$OUTDIR/reduce.host.log"
echo "########## reduce HOST (-D 0)  [warm] ##########"
run_warm "$hlog" mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" \
  "$exe" -b "$B" -e "$E" -f "$F" -g 1 -R 2 -D 0 -c 0 -n "$N" -w "$W" -z 0 "${EXTRA[@]}"

declare -A GBW
for V in "${VLIST[@]}"; do
  glog="$OUTDIR/reduce.gin.V${V}.log"
  echo "########## reduce GIN (-D 3) -V ${V}  [warm] ##########"
  run_warm "$glog" mpirun -n "$NP" "${MPI_OPT[@]}" "${COMMON[@]}" "${GINENV[@]}" \
    "$exe" -b "$B" -e "$E" -f "$F" -g 1 -R 2 -V "$V" -D 3 -c 0 -n "$N" -w "$W" -z 0 "${EXTRA[@]}"
done

echo ""
echo "===== REDUCE CTA SWEEP  (size  host  $(printf 'V%s(gin,%%) ' "${VLIST[@]}")) ====="
# Build a joined table: host, then each V's gin GB/s and gin/host%.
tmph="$OUTDIR/.host.tsv"; parse_busbw "$hlog" | sort -k1,1n > "$tmph"
for V in "${VLIST[@]}"; do
  parse_busbw "$OUTDIR/reduce.gin.V${V}.log" | sort -k1,1n > "$OUTDIR/.v${V}.tsv"
done
while read -r sz hbw; do
  line=$(printf "%12s  %9.2f" "$sz" "$hbw")
  for V in "${VLIST[@]}"; do
    gbw=$(awk -v s="$sz" '$1==s{print $2}' "$OUTDIR/.v${V}.tsv")
    if [[ -n "$gbw" ]]; then
      pct=$(awk -v g="$gbw" -v h="$hbw" 'BEGIN{printf (h>0)?g/h*100:0}')
      line+=$(printf "   %9.2f %6.1f%%" "$gbw" "$pct")
    else
      line+=$(printf "   %9s %6s" "-" "-")
    fi
  done
  echo "$line"
done < "$tmph"
echo "======================================================================="
echo "SWEEP_DONE (raw logs in $OUTDIR)"
