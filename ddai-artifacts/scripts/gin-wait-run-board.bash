#!/usr/bin/env bash
# Wait for the shared node's GPUs to go idle (no foreign tenant), then run the
# warm host-vs-GIN board for the given collectives inside the container. Written
# for the A2A / AllReduce measurement while the node was occupied by another
# user; it self-launches the moment the node frees so no manual polling is needed.
#
# Usage: gin-wait-run-board.bash "<coll1 coll2 ...>" [out_log]
set -uo pipefail

COLLS="${1:-alltoall all_reduce}"
OUT="${2:-/tmp/ar_a2a_board.log}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-240}"     # give up after this many minutes
POLL_S="${POLL_S:-30}"

me="$(whoami)"
idle_gpu() {  # 0 (true) if every GPU < 10% use
  rocm-smi --showuse 2>/dev/null | awk -F: '/GPU use/ { gsub(/[^0-9]/,"",$2); if ($2+0 >= 10) bad=1 } END { exit bad?1:0 }'
}
foreign_orted() {  # count orted procs NOT owned by me
  ps -o user --no-headers -C orted 2>/dev/null | grep -vxc "$me" || true
}

echo "WAIT_START $(date -u +%FT%TZ)  colls=[$COLLS]  me=$me"
deadline=$(( $(date +%s) + MAX_WAIT_MIN*60 ))
while :; do
  fo="$(foreign_orted)"; fo="${fo:-0}"
  if idle_gpu && [[ "$fo" -eq 0 ]]; then
    # require two consecutive idle samples to avoid a brief lull between jobs
    sleep 8
    if idle_gpu && [[ "$(foreign_orted)" -eq 0 ]]; then
      echo "NODE_IDLE $(date -u +%FT%TZ) -- launching board"
      break
    fi
  fi
  if (( $(date +%s) > deadline )); then
    echo "GAVE_UP $(date -u +%FT%TZ) -- node still busy after ${MAX_WAIT_MIN}min"; exit 3
  fi
  sleep "$POLL_S"
done

cd "$HOME/rocm-systems" || exit 2
docker run --rm --init --ulimit memlock=-1:-1 --shm-size 64G --network host \
  --device /dev/dri --device /dev/kfd --device /dev/infiniband --ipc host \
  --group-add video --group-add render --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged --group-add rdma \
  -v "$HOME/rt-build":/rt-build -v "$HOME/rocm-systems":/rocm-systems \
  rccl-gin-gda-sdma-713 \
  bash /rocm-systems/ddai-artifacts/scripts/gin-sdma-hostvsgin-board.bash /rt-build 8 "$COLLS" \
  > "$OUT" 2>&1
echo "BOARD_EXIT=$? $(date -u +%FT%TZ)  (raw board output in $OUT)"
