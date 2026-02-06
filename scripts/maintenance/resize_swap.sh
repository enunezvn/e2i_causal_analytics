#!/bin/bash
# resize_swap.sh - Increase swap space and tune swappiness
#
# Resizes the swap file (default: 4G → 8G) and sets vm.swappiness=10
# so the kernel prefers RAM and only swaps when necessary.
#
# Usage:
#   sudo ./resize_swap.sh          # Resize to 8G (default)
#   sudo ./resize_swap.sh 12       # Resize to 12G
#
# NOTE: Must run AFTER Docker cleanup (Step 1) since it needs free disk.
#       The `swapoff` step may take several minutes when swap is fully used.

set -euo pipefail

SIZE_GB="${1:-8}"
SWAP_FILE="/swapfile"
SWAPPINESS=10

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

# Check root
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

# Validate SIZE_GB is a positive integer
if ! [[ "$SIZE_GB" =~ ^[0-9]+$ ]] || [[ "$SIZE_GB" -lt 1 ]]; then
    log_error "SIZE_GB must be a positive integer, got: $SIZE_GB"
    exit 1
fi

echo "========================================"
echo "Swap Resize Script"
echo "========================================"
echo ""

# Show current state
log_step "Current swap status:"
swapon --show
echo ""
free -h | head -3
echo ""

# Check available disk space
available_gb=$(df -BG / | awk 'NR==2 {gsub(/G/,""); print $4}')
current_swap_gb=$(swapon --show --bytes --noheadings 2>/dev/null | awk '{sum+=$3} END {printf "%.0f", sum/1073741824}')
current_swap_gb=${current_swap_gb:-0}
needed_gb=$((SIZE_GB - current_swap_gb))

log_info "Current swap: ${current_swap_gb}G, Target: ${SIZE_GB}G, Additional needed: ${needed_gb}G"
log_info "Available disk space: ${available_gb}G"

if [[ $needed_gb -gt 0 ]] && [[ $available_gb -lt $((needed_gb + 2)) ]]; then
    log_error "Not enough disk space. Need at least $((needed_gb + 2))G free, have ${available_gb}G"
    log_error "Run docker_cleanup.sh first to free disk space."
    exit 1
fi

if [[ $current_swap_gb -ge $SIZE_GB ]]; then
    log_info "Swap is already ${current_swap_gb}G (>= target ${SIZE_GB}G). Skipping resize."
else
    # Disable current swap
    log_step "Disabling current swap (this may take several minutes if swap is heavily used)..."
    log_warn "System may be sluggish during swapoff — this is normal."
    swapoff -a
    log_info "Swap disabled."

    # Resize swap file
    log_step "Allocating ${SIZE_GB}G swap file..."
    fallocate -l "${SIZE_GB}G" "$SWAP_FILE"
    chmod 600 "$SWAP_FILE"
    log_info "Swap file allocated."

    # Format and enable
    log_step "Formatting swap file..."
    mkswap "$SWAP_FILE"

    log_step "Enabling swap..."
    swapon "$SWAP_FILE"
    log_info "Swap enabled."
fi

# Set swappiness
log_step "Setting vm.swappiness=$SWAPPINESS..."
sysctl vm.swappiness=$SWAPPINESS

# Persist swappiness in sysctl.conf
if grep -q "^vm.swappiness" /etc/sysctl.conf; then
    sed -i "s/^vm.swappiness=.*/vm.swappiness=$SWAPPINESS/" /etc/sysctl.conf
    log_info "Updated vm.swappiness in /etc/sysctl.conf"
else
    echo "vm.swappiness=$SWAPPINESS" >> /etc/sysctl.conf
    log_info "Added vm.swappiness=$SWAPPINESS to /etc/sysctl.conf"
fi

# Verify fstab entry
log_step "Verifying /etc/fstab entry..."
if grep -q "$SWAP_FILE" /etc/fstab; then
    log_info "Swap entry already exists in /etc/fstab"
else
    echo "$SWAP_FILE none swap sw 0 0" >> /etc/fstab
    log_info "Added swap entry to /etc/fstab"
fi

echo ""
log_step "Final swap status:"
swapon --show
echo ""
free -h | head -3
echo ""
log_info "vm.swappiness=$(cat /proc/sys/vm/swappiness)"

echo ""
echo "========================================"
echo "Swap resize complete!"
echo "========================================"
