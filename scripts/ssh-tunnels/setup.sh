#!/usr/bin/env bash
# E2I Analytics — SSH Tunnel Setup
# Run this on your LOCAL machine (not the droplet).
# Sets up persistent autossh tunnels via systemd user services.
#
# Tunnels:
#   localhost:8443  → Droplet nginx (frontend + API via HTTPS)
#   localhost:3002  → E2I Frontend (HTTP)
#   localhost:5000  → MLflow
#   localhost:3000  → BentoML
#   localhost:5173  → Opik UI
#   localhost:8084  → Opik Backend
#   localhost:8001  → Opik Python Backend
#   localhost:9090  → Opik MinIO Console
#   localhost:3030  → FalkorDB Browser
#   localhost:3001  → Supabase Studio
#   localhost:9091  → Prometheus
#   localhost:3200  → Grafana
#   localhost:3101  → Loki
#   localhost:9093  → Alertmanager
#
# Prerequisites:
#   - autossh installed (brew install autossh / apt install autossh)
#   - SSH key auth to enunez@138.197.4.36 working
#
# Usage:
#   bash setup.sh          # Install and start all tunnel services
#   bash setup.sh remove   # Stop and remove all tunnel services

set -euo pipefail

DROPLET_HOST="138.197.4.36"
DROPLET_USER="enunez"
SERVICE_DIR="${HOME}/.config/systemd/user"

declare -A TUNNELS=(
  ["e2i-frontend-https"]="8443:localhost:443"
  ["e2i-frontend"]="3002:localhost:3002"
  ["e2i-mlflow"]="5000:localhost:5000"
  ["e2i-bentoml"]="3000:localhost:3000"
  ["e2i-opik-ui"]="5173:localhost:5173"
  ["e2i-opik-backend"]="8084:localhost:8084"
  ["e2i-opik-python"]="8001:localhost:8001"
  ["e2i-opik-minio"]="9090:localhost:9090"
  ["e2i-falkordb"]="3030:localhost:3030"
  ["e2i-supabase"]="3001:localhost:3001"
  ["e2i-prometheus"]="9091:localhost:9091"
  ["e2i-grafana"]="3200:localhost:3200"
  ["e2i-loki"]="3101:localhost:3101"
  ["e2i-alertmanager"]="9093:localhost:9093"
)

check_deps() {
  if ! command -v autossh &>/dev/null; then
    echo "ERROR: autossh not found."
    echo "  macOS:  brew install autossh"
    echo "  Ubuntu: sudo apt install autossh"
    echo "  Fedora: sudo dnf install autossh"
    exit 1
  fi
}

check_ssh() {
  echo "Verifying SSH connectivity to ${DROPLET_USER}@${DROPLET_HOST}..."
  if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${DROPLET_USER}@${DROPLET_HOST}" true 2>/dev/null; then
    echo "ERROR: Cannot SSH to ${DROPLET_USER}@${DROPLET_HOST}"
    echo "Ensure your SSH key is added and the host is reachable."
    exit 1
  fi
  echo "SSH connection OK."
}

install_services() {
  mkdir -p "${SERVICE_DIR}"

  for name in "${!TUNNELS[@]}"; do
    local tunnel="${TUNNELS[$name]}"
    local local_port="${tunnel%%:*}"
    local service_file="${SERVICE_DIR}/${name}-tunnel.service"

    cat > "${service_file}" <<UNIT
[Unit]
Description=SSH tunnel: ${name} (localhost:${local_port})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/autossh -M 0 -N \
  -o "ServerAliveInterval=30" \
  -o "ServerAliveCountMax=3" \
  -o "ExitOnForwardFailure=yes" \
  -o "StrictHostKeyChecking=accept-new" \
  -L ${tunnel} \
  ${DROPLET_USER}@${DROPLET_HOST}
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
UNIT

    echo "Created ${service_file}"
  done

  systemctl --user daemon-reload

  for name in "${!TUNNELS[@]}"; do
    systemctl --user enable --now "${name}-tunnel.service"
    echo "Started ${name}-tunnel.service"
  done

  echo ""
  echo "All tunnels active. Access:"
  echo "  Frontend (HTTPS): https://localhost:8443  (accept cert warning)"
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
  echo "Management:"
  echo "  Status:  systemctl --user status e2i-*-tunnel"
  echo "  Stop:    systemctl --user stop e2i-*-tunnel"
  echo "  Start:   systemctl --user start e2i-*-tunnel"
  echo "  Logs:    journalctl --user -u e2i-mlflow-tunnel -f"
}

remove_services() {
  for name in "${!TUNNELS[@]}"; do
    local svc="${name}-tunnel.service"
    if systemctl --user is-active "${svc}" &>/dev/null; then
      systemctl --user stop "${svc}"
    fi
    if systemctl --user is-enabled "${svc}" &>/dev/null; then
      systemctl --user disable "${svc}"
    fi
    rm -f "${SERVICE_DIR}/${svc}"
    echo "Removed ${svc}"
  done
  systemctl --user daemon-reload
  echo "All tunnel services removed."
}

# --- Main ---

if [[ "${1:-}" == "remove" ]]; then
  remove_services
  exit 0
fi

check_deps
check_ssh
install_services
