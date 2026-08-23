#!/bin/bash
# setup_cron.sh - Install cron jobs for orphan cleanup and memory monitoring
#
# This script sets up:
# 1. Orphan process cleanup every 15 minutes
# 2. Memory monitoring every 5 minutes
# 3. Session start check (run on login)
#
# Usage: sudo ./setup_cron.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/var/log/e2i"
CRON_FILE="/etc/cron.d/e2i-maintenance"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check root
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

log_info "Setting up E2I maintenance cron jobs..."

# Create log directory
log_info "Creating log directory: $LOG_DIR"
mkdir -p "$LOG_DIR"
chmod 755 "$LOG_DIR"

# Make scripts executable
log_info "Making maintenance scripts executable..."
chmod +x "$SCRIPT_DIR/cleanup_orphans.sh"
chmod +x "$SCRIPT_DIR/memory_monitor.sh"
chmod +x "$SCRIPT_DIR/docker_cleanup.sh"

# Create cron file
log_info "Installing cron jobs to $CRON_FILE"
cat > "$CRON_FILE" << 'EOF'
# E2I Causal Analytics - Maintenance Cron Jobs
# Installed by: scripts/maintenance/setup_cron.sh
#
# Format: minute hour day month weekday user command

SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin

# Orphan process cleanup - every 15 minutes
*/15 * * * * root /home/enunez/Projects/e2i_causal_analytics/scripts/maintenance/cleanup_orphans.sh >> /var/log/e2i/orphan_cleanup.log 2>&1

# Memory monitoring - every 5 minutes with auto-cleanup enabled
*/5 * * * * root /home/enunez/Projects/e2i_causal_analytics/scripts/maintenance/memory_monitor.sh --auto-cleanup >> /var/log/e2i/memory_monitor.log 2>&1

# Daily log rotation at 2am
0 2 * * * root find /var/log/e2i -name "*.log" -size +10M -exec truncate -s 1M {} \;

# Docker disk cleanup - every Sunday at 3am
0 3 * * 0 root /home/enunez/Projects/e2i_causal_analytics/scripts/maintenance/docker_cleanup.sh >> /var/log/e2i/docker_cleanup.log 2>&1
EOF

chmod 644 "$CRON_FILE"

# Setup login check script in /etc/profile.d
log_info "Setting up session start orphan check..."
cat > /etc/profile.d/e2i-session-check.sh << 'EOF'
# E2I Session Start Check
# Automatically check for orphan processes at the start of each session

if [[ -x /home/enunez/Projects/e2i_causal_analytics/scripts/maintenance/cleanup_orphans.sh ]]; then
    # Run quick orphan check (non-blocking, background)
    orphan_count=$(ps aux | grep -E 'exec\(eval' | grep -v grep | wc -l)
    if [[ $orphan_count -gt 0 ]]; then
        echo -e "\033[1;33m[E2I] Warning: Found $orphan_count orphan process(es). Run cleanup_orphans.sh to remove.\033[0m"
    fi
fi
EOF
chmod 644 /etc/profile.d/e2i-session-check.sh

# Create convenience aliases
log_info "Setting up convenience aliases..."
cat >> /home/enunez/.bash_aliases 2>/dev/null << 'EOF' || true

# E2I Maintenance aliases
alias e2i-cleanup='/home/enunez/Projects/e2i_causal_analytics/scripts/maintenance/cleanup_orphans.sh'
alias e2i-cleanup-dry='/home/enunez/Projects/e2i_causal_analytics/scripts/maintenance/cleanup_orphans.sh --dry-run --verbose'
alias e2i-memcheck='/home/enunez/Projects/e2i_causal_analytics/scripts/maintenance/memory_monitor.sh'
alias e2i-logs='tail -f /var/log/e2i/*.log'
alias e2i-orphans='ps aux | grep -E "exec\(eval|esbuild.*service" | grep -v grep'
alias e2i-docker-cleanup='/home/enunez/Projects/e2i_causal_analytics/scripts/maintenance/docker_cleanup.sh'
alias e2i-docker-cleanup-dry='/home/enunez/Projects/e2i_causal_analytics/scripts/maintenance/docker_cleanup.sh --dry-run'
EOF
chown enunez:enunez /home/enunez/.bash_aliases 2>/dev/null || true

# Verify cron is running
if ! systemctl is-active --quiet cron; then
    log_warn "Cron service not running, starting it..."
    systemctl enable cron
    systemctl start cron
fi

# Run initial cleanup
log_info "Running initial orphan cleanup..."
"$SCRIPT_DIR/cleanup_orphans.sh" --verbose || true

log_info "Running initial memory check..."
"$SCRIPT_DIR/memory_monitor.sh" || true

echo ""
log_info "=== Setup Complete ==="
echo ""
echo "Cron jobs installed:"
echo "  - Orphan cleanup:    Every 15 minutes"
echo "  - Memory monitoring: Every 5 minutes (with auto-cleanup)"
echo "  - Log rotation:      Daily at 2am"
echo "  - Docker cleanup:    Weekly, Sundays at 3am"
echo ""
echo "Convenience aliases (after re-login):"
echo "  - e2i-cleanup      Run orphan cleanup"
echo "  - e2i-cleanup-dry  Dry run (show what would be killed)"
echo "  - e2i-memcheck     Run memory check"
echo "  - e2i-docker-cleanup  Run Docker disk cleanup"
echo "  - e2i-docker-cleanup-dry  Dry run Docker cleanup"
echo "  - e2i-logs         Tail all maintenance logs"
echo "  - e2i-orphans      List current orphan processes"
echo ""
echo "Session start check:"
echo "  - Orphan warning displayed on login if orphans detected"
echo ""
echo "Log files:"
echo "  - /var/log/e2i/orphan_cleanup.log"
echo "  - /var/log/e2i/memory_monitor.log"
echo "  - /var/log/e2i/docker_cleanup.log"
