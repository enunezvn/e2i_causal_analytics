#!/usr/bin/env bash
# oom_run.sh — run a memory-bounded, single-threaded probe with a pre-flight floor.
# Usage: ./oom_run.sh <max_gib> <label> -- <command...>
set -euo pipefail
MAX_GIB="$1"; LABEL="$2"; shift 2
[ "${1:-}" = "--" ] && shift
FLOOR_MB=3000
AVAIL_MB=$(free -m | awk '/^Mem:/{print $7}')
if [ "$AVAIL_MB" -lt "$FLOOR_MB" ]; then
  echo "ABORT [$LABEL]: only ${AVAIL_MB}MB available (<${FLOOR_MB}MB floor)"; exit 99
fi
echo "RUN [$LABEL]: cap=${MAX_GIB}G avail=${AVAIL_MB}MB"
exec systemd-run --user --scope --quiet \
  -p MemoryMax="${MAX_GIB}G" -p MemorySwapMax=0 \
  env LOKY_MAX_CPU_COUNT=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$@"
