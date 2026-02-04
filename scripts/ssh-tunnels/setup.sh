#!/usr/bin/env bash
# E2I Analytics — SSH Tunnel Setup
# Run this on your LOCAL machine (not the droplet).
# Sets up persistent autossh tunnels via systemd user services.
#
# Tunnels:
#   localhost:8443  → Droplet nginx (frontend + API via HTTPS)
#   localhost:5000  → MLflow
#   localhost:5173  → Opik
#   localhost:3030  → FalkorDB Browser
#   localhost:3000  → Supabase Studio
#   localhost:3100  → Grafana
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
  ["e2i-frontend"]="8443:localhost:443"
  ["e2i-mlflow"]="5000:localhost:5000"
  ["e2i-opik"]="5173:localhost:5173"
  ["e2i-falkordb"]="3030:localhost:3030"
  ["e2i-supabase"]="3000:localhost:3000"
  ["e2i-grafana"]="3100:localhost:3100"
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
  echo "  Frontend:         https://localhost:8443  (accept cert warning)"
  echo "  MLflow:           http://localhost:5000"
  echo "  Opik:             http://localhost:5173"
  echo "  FalkorDB Browser: http://localhost:3030"
  echo "  Supabase Studio:  http://localhost:3000"
  echo "  Grafana:          http://localhost:3100"
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
