#!/bin/bash
# harden_ssh.sh - Idempotent SSH and fail2ban hardening
#
# Applies defense-in-depth settings to sshd_config and fail2ban.
# Safe to re-run — backs up config before changes, validates with
# sshd -t, and auto-rolls back if validation fails.
#
# sshd_config changes:
#   MaxAuthTries        6 (default) → 3
#   MaxStartups         10:30:100   → 3:50:10
#   LoginGraceTime      120s        → 30s
#   ClientAliveInterval 0           → 300
#   ClientAliveCountMax 3           → 2
#   X11Forwarding       yes         → no
#   AllowUsers          (not set)   → enunez
#
# fail2ban changes (jail.local):
#   bantime             600         → 3600
#   maxretry            5           → 3
#
# Usage:
#   sudo ./harden_ssh.sh            # Apply changes
#   sudo ./harden_ssh.sh --dry-run  # Show what would change

set -euo pipefail

SSHD_CONFIG="/etc/ssh/sshd_config"
BACKUP_DIR="/etc/ssh/backups"
FAIL2BAN_JAIL="/etc/fail2ban/jail.local"
DRY_RUN=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $1"; }

usage() {
    cat << 'USAGE'
Usage: harden_ssh.sh [OPTIONS]

Idempotent SSH and fail2ban hardening script.

Options:
  --dry-run    Show what would change without modifying anything
  -h, --help   Show this help message
USAGE
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        -h|--help)  usage ;;
        *) log_error "Unknown option: $1"; usage ;;
    esac
done

# Check root
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

echo "========================================"
echo "SSH Hardening Script"
echo "Dry run: $DRY_RUN"
echo "========================================"
echo ""

# ─── Helper: set or update an sshd_config directive ───
# Sets the value if it exists (commented or not), appends if missing.
set_sshd_option() {
    local key="$1"
    local value="$2"
    local file="$3"

    # Check current value
    current=$(grep -E "^${key}\s" "$file" 2>/dev/null | tail -1 | awk '{$1=""; print $0}' | xargs || echo "(not set)")

    if [[ "$current" == "$value" ]]; then
        log_info "$key already set to $value — skipping"
        return
    fi

    log_info "$key: $current → $value"

    if [[ "$DRY_RUN" == true ]]; then
        return
    fi

    # Remove all existing lines (commented or active) for this key
    sed -i "/^#*\s*${key}\s/d" "$file"

    # Append the new value
    echo "$key $value" >> "$file"
}

# ─── Step 1: Backup sshd_config ───
log_step "Backing up sshd_config..."
mkdir -p "$BACKUP_DIR"
backup_file="${BACKUP_DIR}/sshd_config.$(date +%Y%m%d_%H%M%S)"

if [[ "$DRY_RUN" == false ]]; then
    cp "$SSHD_CONFIG" "$backup_file"
    log_info "Backup saved to: $backup_file"
else
    log_info "[DRY RUN] Would back up to: ${BACKUP_DIR}/sshd_config.<timestamp>"
fi
echo ""

# ─── Step 2: Apply sshd_config changes ───
log_step "Applying sshd_config hardening..."

if [[ "$DRY_RUN" == true ]]; then
    # Work on a temp copy for dry-run display
    work_file=$(mktemp)
    cp "$SSHD_CONFIG" "$work_file"
else
    work_file="$SSHD_CONFIG"
fi

set_sshd_option "MaxAuthTries"        "3"        "$work_file"
set_sshd_option "MaxStartups"         "3:50:10"  "$work_file"
set_sshd_option "LoginGraceTime"      "30"       "$work_file"
set_sshd_option "ClientAliveInterval" "300"       "$work_file"
set_sshd_option "ClientAliveCountMax" "2"         "$work_file"
set_sshd_option "X11Forwarding"       "no"        "$work_file"
set_sshd_option "AllowUsers"          "enunez"    "$work_file"

if [[ "$DRY_RUN" == true ]]; then
    rm -f "$work_file"
fi
echo ""

# ─── Step 3: Validate sshd_config ───
log_step "Validating sshd_config..."
if [[ "$DRY_RUN" == false ]]; then
    if sshd -t 2>&1; then
        log_info "sshd_config validation passed"
    else
        log_error "sshd_config validation FAILED — rolling back!"
        cp "$backup_file" "$SSHD_CONFIG"
        log_info "Rolled back to: $backup_file"
        exit 1
    fi
else
    log_info "[DRY RUN] Would validate with: sshd -t"
fi
echo ""

# ─── Step 4: Reload sshd ───
log_step "Reloading sshd..."
if [[ "$DRY_RUN" == false ]]; then
    systemctl reload sshd
    log_info "sshd reloaded successfully"
else
    log_info "[DRY RUN] Would run: systemctl reload sshd"
fi
echo ""

# ─── Step 5: fail2ban hardening ───
log_step "Applying fail2ban hardening..."

if [[ "$DRY_RUN" == true ]]; then
    if [[ -f "$FAIL2BAN_JAIL" ]]; then
        log_info "[DRY RUN] Current jail.local:"
        grep -E "^(bantime|maxretry)" "$FAIL2BAN_JAIL" 2>/dev/null | while read -r line; do
            log_info "  $line"
        done
    fi
    log_info "[DRY RUN] Would set: bantime = 3600, maxretry = 3"
else
    # Create or update jail.local
    if [[ -f "$FAIL2BAN_JAIL" ]]; then
        # Update existing values
        if grep -q "^bantime" "$FAIL2BAN_JAIL"; then
            sed -i 's/^bantime\s*=.*/bantime = 3600/' "$FAIL2BAN_JAIL"
        else
            echo "bantime = 3600" >> "$FAIL2BAN_JAIL"
        fi

        if grep -q "^maxretry" "$FAIL2BAN_JAIL"; then
            sed -i 's/^maxretry\s*=.*/maxretry = 3/' "$FAIL2BAN_JAIL"
        else
            echo "maxretry = 3" >> "$FAIL2BAN_JAIL"
        fi
        log_info "Updated existing $FAIL2BAN_JAIL"
    else
        # Create new jail.local
        cat > "$FAIL2BAN_JAIL" << 'JAIL'
[DEFAULT]
bantime = 3600
maxretry = 3

[sshd]
enabled = true
JAIL
        log_info "Created $FAIL2BAN_JAIL"
    fi

    # Reload fail2ban
    if systemctl is-active --quiet fail2ban; then
        systemctl reload fail2ban
        log_info "fail2ban reloaded"
    else
        log_warn "fail2ban is not running — changes will apply on next start"
    fi
fi
echo ""

# ─── Step 6: Show summary ───
log_step "Current effective settings:"
if [[ "$DRY_RUN" == false ]]; then
    sshd -T 2>/dev/null | grep -iE "maxauthtries|maxstartups|logingracetime|clientaliveinterval|clientalivecountmax|x11forwarding|allowusers" | sort
else
    log_info "[DRY RUN] Would show: sshd -T | grep <settings>"
fi
echo ""

echo "========================================"
echo "SSH hardening complete!"
echo "========================================"
echo ""
log_warn "IMPORTANT: Test SSH access from another terminal before"
log_warn "closing this session to verify you can still connect:"
log_warn "  ssh enunez@<server-ip>"
