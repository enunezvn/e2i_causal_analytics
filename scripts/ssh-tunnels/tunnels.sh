#!/usr/bin/env bash
# Quick tunnel launcher — run on your LOCAL machine.
# Usage:
#   bash tunnels.sh        # Start all tunnels (foreground, Ctrl+C to stop)
#   bash tunnels.sh stop   # Kill any running tunnel processes

set -euo pipefail

DROPLET="enunez@138.197.4.36"

start_tunnels() {
  echo "Starting SSH tunnels to ${DROPLET}..."
  echo ""
  echo "  Frontend (HTTPS): https://localhost:8443"
  echo "  Frontend (HTTP):  http://localhost:3002"
  echo "  MLflow:           http://localhost:5000"
  echo "  BentoML:          http://localhost:3000"
  echo "  Opik UI:          http://localhost:5173"
  echo "  Opik Backend:     http://localhost:8084"
  echo "  Opik Py Backend:  http://localhost:8001"
  echo "  Opik MinIO:       http://localhost:9090"
  echo "  FalkorDB Browser: http://localhost:3030"
  echo "  Supabase Studio:  http://localhost:3001"
  echo "  Prometheus:       http://localhost:9091"
  echo "  Grafana:          http://localhost:3200"
  echo "  Loki:             http://localhost:3101"
  echo "  Alertmanager:     http://localhost:9093"
  echo ""
  echo "Press Ctrl+C to stop."
  echo ""

  ssh -N \
    -L 8443:localhost:443 \
    -L 3002:localhost:3002 \
    -L 5000:localhost:5000 \
    -L 3000:localhost:3000 \
    -L 5173:localhost:5173 \
    -L 8084:localhost:8084 \
    -L 8001:localhost:8001 \
    -L 9090:localhost:9090 \
    -L 3030:localhost:3030 \
    -L 3001:localhost:3001 \
    -L 9091:localhost:9091 \
    -L 3200:localhost:3200 \
    -L 3101:localhost:3101 \
    -L 9093:localhost:9093 \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    "${DROPLET}"
}

stop_tunnels() {
  pkill -f "ssh.*-L.*8443:localhost:443.*${DROPLET}" 2>/dev/null && echo "Tunnels stopped." || echo "No running tunnels found."
}

case "${1:-start}" in
  start) start_tunnels ;;
  stop)  stop_tunnels ;;
  *)     echo "Usage: $0 [start|stop]"; exit 1 ;;
esac
