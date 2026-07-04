#!/usr/bin/env bash
# One-shot droplet memory audit — per-container RAM + swap + host totals.
# Safe/read-only: only runs `free`, `docker ps`, `docker stats --no-stream`, `docker inspect`.
# Usage:  bash scripts/mem_snapshot.sh
#         (or via Claude Code:  ! bash scripts/mem_snapshot.sh )
set -uo pipefail

KEY="${DEPLOY_KEY:-$HOME/.ssh/deploy_ed25519}"

# --- resolve droplet host from the usual places (env > .env > connect script) ---
HOST="${DEPLOY_HOST:-}"
if [ -z "$HOST" ] && [ -f .env ]; then
  HOST="$(grep -E '^(DEPLOY_HOST|DROPLET_HOST|DROPLET_IP)=' .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d '"'"'"' ' )"
fi
if [ -z "$HOST" ]; then
  HOST="$(grep -oE '[0-9]{1,3}(\.[0-9]{1,3}){3}' scripts/droplet-connect.sh 2>/dev/null | head -1)"
fi
if [ -z "$HOST" ]; then
  echo "!! Could not auto-detect droplet host. Re-run as:  DEPLOY_HOST=<ip> bash scripts/mem_snapshot.sh"
  exit 2
fi

USER_AT="${DEPLOY_USER:-enunez}@${HOST}"
echo "### Connecting to ${USER_AT} (key: ${KEY})"

ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=12 -o BatchMode=yes "$USER_AT" 'bash -s' <<'REMOTE'
echo "================ HOST MEMORY (MB) ================"
free -m
echo
echo "================ SWAP ================"
swapon --show 2>/dev/null || cat /proc/swaps
echo
echo "================ CONTAINER COUNT ================"
echo "running: $(docker ps -q | wc -l)    all(incl stopped): $(docker ps -aq | wc -l)"
echo
echo "================ PER-CONTAINER LIVE USAGE (sorted by RAM) ================"
# MemUsage  MemPerc  Name  — sorted descending by absolute usage
docker stats --no-stream --format '{{.MemUsage}}|{{.MemPerc}}|{{.Name}}' \
  | sort -t'|' -k1 -hr \
  | awk -F'|' '{printf "%-22s %-8s %s\n", $1, $2, $3}'
echo
echo "================ CONFIGURED MEM LIMITS (HostConfig.Memory, bytes; 0 = unlimited) ================"
for c in $(docker ps --format '{{.Names}}'); do
  lim=$(docker inspect -f '{{.HostConfig.Memory}}' "$c" 2>/dev/null)
  # human-ize
  if [ "${lim:-0}" = "0" ]; then hr="unlimited"; else hr="$(awk -v b="$lim" 'BEGIN{printf "%.2f GiB", b/1073741824}')"; fi
  printf "%-40s %s\n" "$c" "$hr"
done
echo
echo "================ STOPPED CONTAINERS (candidates already paused) ================"
docker ps -a --filter status=exited --format '{{.Names}}  ({{.Status}})'
REMOTE

echo
echo "### snapshot complete"
