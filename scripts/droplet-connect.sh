#!/bin/bash
#
# droplet-connect.sh - Connect to E2I DigitalOcean droplet
#
# This script:
#   1. Checks droplet connectivity
#   2. Verifies Docker containers are healthy
#   3. Checks key dependencies
#   4. Starts SSH tunnel for web UI access
#
# Usage: ./scripts/droplet-connect.sh [--check-only] [--kill-tunnel]

set -e

# Configuration
DROPLET_IP="159.89.180.27"
SSH_KEY="$HOME/.ssh/replit"
SSH_USER="root"
SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Services to forward
declare -A SERVICES=(
    [5000]="MLflow"
    [5173]="Opik UI"
    [8080]="Opik Backend"
    [8001]="E2I API"
)

# Required healthy containers (core services)
REQUIRED_CONTAINERS=(
    "e2i_mlflow"
    "e2i_redis"
    "e2i_falkordb"
    "opik-frontend-1"
    "opik-backend-1"
    "opik-python-backend-1"
    "opik-clickhouse-1"
    "opik-redis-1"
    "opik-mysql-1"
)

print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}\n"
}

print_status() {
    local status=$1
    local message=$2
    if [ "$status" = "ok" ]; then
        echo -e "  ${GREEN}✓${NC} $message"
    elif [ "$status" = "warn" ]; then
        echo -e "  ${YELLOW}⚠${NC} $message"
    elif [ "$status" = "error" ]; then
        echo -e "  ${RED}✗${NC} $message"
    else
        echo -e "  ${BLUE}→${NC} $message"
    fi
}

check_ssh_key() {
    if [ ! -f "$SSH_KEY" ]; then
        print_status "error" "SSH key not found: $SSH_KEY"
        echo -e "\n  Please ensure your SSH key exists at $SSH_KEY"
        exit 1
    fi
    print_status "ok" "SSH key found: $SSH_KEY"
}

check_connectivity() {
    print_header "Checking Droplet Connectivity"

    check_ssh_key

    print_status "info" "Connecting to $DROPLET_IP..."

    if ssh -i "$SSH_KEY" $SSH_OPTS "$SSH_USER@$DROPLET_IP" "echo 'connected'" &>/dev/null; then
        print_status "ok" "SSH connection successful"

        # Get uptime
        local uptime=$(ssh -i "$SSH_KEY" $SSH_OPTS "$SSH_USER@$DROPLET_IP" "uptime -p" 2>/dev/null)
        print_status "ok" "Droplet uptime: $uptime"
        return 0
    else
        print_status "error" "Cannot connect to droplet"
        exit 1
    fi
}

check_docker_health() {
    print_header "Checking Docker Container Health"

    local unhealthy_count=0
    local healthy_count=0

    # Get container status
    local container_status=$(ssh -i "$SSH_KEY" $SSH_OPTS "$SSH_USER@$DROPLET_IP" \
        "docker ps -a --format '{{.Names}}|{{.Status}}'" 2>/dev/null)

    for container in "${REQUIRED_CONTAINERS[@]}"; do
        local status=$(echo "$container_status" | grep "^$container|" | cut -d'|' -f2)

        if [ -z "$status" ]; then
            print_status "error" "$container: NOT FOUND"
            ((unhealthy_count++))
        elif echo "$status" | grep -q "(healthy)"; then
            print_status "ok" "$container: $status"
            ((healthy_count++))
        elif echo "$status" | grep -q "Up"; then
            print_status "warn" "$container: $status (no health check)"
            ((healthy_count++))
        elif echo "$status" | grep -q "Exited (0)"; then
            # Init containers that exit with 0 are OK
            if echo "$container" | grep -qE "(init|generator|mc)"; then
                print_status "ok" "$container: $status (init container)"
                ((healthy_count++))
            else
                print_status "error" "$container: $status"
                ((unhealthy_count++))
            fi
        else
            print_status "error" "$container: $status"
            ((unhealthy_count++))
        fi
    done

    echo ""
    print_status "info" "Summary: $healthy_count healthy, $unhealthy_count unhealthy"

    if [ $unhealthy_count -gt 0 ]; then
        print_status "warn" "Some containers are not healthy"
        return 1
    fi
    return 0
}

check_dependencies() {
    print_header "Checking Dependencies"

    local deps_ok=true

    # Check Python
    local python_version=$(ssh -i "$SSH_KEY" $SSH_OPTS "$SSH_USER@$DROPLET_IP" \
        "python3 --version 2>/dev/null" | awk '{print $2}')
    if [ -n "$python_version" ]; then
        print_status "ok" "Python: $python_version"
    else
        print_status "error" "Python: NOT FOUND"
        deps_ok=false
    fi

    # Check Node
    local node_version=$(ssh -i "$SSH_KEY" $SSH_OPTS "$SSH_USER@$DROPLET_IP" \
        "node --version 2>/dev/null")
    if [ -n "$node_version" ]; then
        print_status "ok" "Node.js: $node_version"
    else
        print_status "error" "Node.js: NOT FOUND"
        deps_ok=false
    fi

    # Check Docker
    local docker_version=$(ssh -i "$SSH_KEY" $SSH_OPTS "$SSH_USER@$DROPLET_IP" \
        "docker --version 2>/dev/null" | awk '{print $3}' | tr -d ',')
    if [ -n "$docker_version" ]; then
        print_status "ok" "Docker: $docker_version"
    else
        print_status "error" "Docker: NOT FOUND"
        deps_ok=false
    fi

    # Check Docker Compose
    local compose_version=$(ssh -i "$SSH_KEY" $SSH_OPTS "$SSH_USER@$DROPLET_IP" \
        "docker compose version 2>/dev/null" | awk '{print $4}')
    if [ -n "$compose_version" ]; then
        print_status "ok" "Docker Compose: $compose_version"
    else
        print_status "error" "Docker Compose: NOT FOUND"
        deps_ok=false
    fi

    if [ "$deps_ok" = false ]; then
        return 1
    fi
    return 0
}

check_services() {
    print_header "Checking Service Accessibility"

    for port in "${!SERVICES[@]}"; do
        local service="${SERVICES[$port]}"
        local status=$(ssh -i "$SSH_KEY" $SSH_OPTS "$SSH_USER@$DROPLET_IP" \
            "curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 http://localhost:$port/ 2>/dev/null")

        if [ "$status" = "200" ] || [ "$status" = "302" ] || [ "$status" = "301" ]; then
            print_status "ok" "$service (port $port): HTTP $status"
        elif [ "$status" = "404" ]; then
            print_status "warn" "$service (port $port): HTTP $status (service up, no root handler)"
        else
            print_status "error" "$service (port $port): HTTP $status or unreachable"
        fi
    done
}

kill_existing_tunnels() {
    print_header "Stopping Existing SSH Tunnels"

    local killed=false
    for port in "${!SERVICES[@]}"; do
        if pkill -f "ssh.*-L $port:localhost:$port.*$DROPLET_IP" 2>/dev/null; then
            print_status "ok" "Killed tunnel for port $port"
            killed=true
        fi
    done

    if [ "$killed" = false ]; then
        print_status "info" "No existing tunnels found"
    fi
}

start_tunnel() {
    print_header "Starting SSH Tunnel"

    # Check for existing tunnels
    local existing_tunnel=$(pgrep -f "ssh.*-L 5000:localhost:5000.*$DROPLET_IP" 2>/dev/null)
    if [ -n "$existing_tunnel" ]; then
        print_status "warn" "SSH tunnel already running (PID: $existing_tunnel)"
        echo -e "\n  Use ${YELLOW}--kill-tunnel${NC} to stop existing tunnels first"
        return 0
    fi

    # Build tunnel command
    local tunnel_args=""
    for port in "${!SERVICES[@]}"; do
        tunnel_args="$tunnel_args -L $port:localhost:$port"
    done

    print_status "info" "Starting tunnel for ports: ${!SERVICES[*]}"

    ssh -i "$SSH_KEY" $SSH_OPTS $tunnel_args -N -f "$SSH_USER@$DROPLET_IP"

    if [ $? -eq 0 ]; then
        print_status "ok" "SSH tunnel started successfully"
        echo ""
        echo -e "  ${GREEN}Access URLs:${NC}"
        for port in "${!SERVICES[@]}"; do
            echo -e "    ${SERVICES[$port]}: ${BLUE}http://localhost:$port${NC}"
        done
        echo ""
        print_status "info" "To stop tunnel: $0 --kill-tunnel"
    else
        print_status "error" "Failed to start SSH tunnel"
        return 1
    fi
}

verify_tunnel() {
    print_header "Verifying Local Tunnel Access"

    sleep 1  # Give tunnel a moment to establish

    for port in "${!SERVICES[@]}"; do
        local service="${SERVICES[$port]}"
        local status=$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 3 "http://localhost:$port/" 2>/dev/null)

        if [ "$status" = "200" ] || [ "$status" = "302" ] || [ "$status" = "301" ]; then
            print_status "ok" "$service (localhost:$port): HTTP $status"
        elif [ "$status" = "404" ]; then
            print_status "warn" "$service (localhost:$port): HTTP $status (accessible)"
        else
            print_status "error" "$service (localhost:$port): Cannot connect"
        fi
    done
}

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Connect to E2I DigitalOcean droplet and start SSH tunnel"
    echo ""
    echo "Options:"
    echo "  --check-only    Only run health checks, don't start tunnel"
    echo "  --kill-tunnel   Stop existing SSH tunnels"
    echo "  --tunnel-only   Skip health checks, just start tunnel"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                  # Full check + start tunnel"
    echo "  $0 --check-only     # Just check droplet health"
    echo "  $0 --kill-tunnel    # Stop running tunnels"
}

# Main
main() {
    local check_only=false
    local kill_tunnel=false
    local tunnel_only=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check-only)
                check_only=true
                shift
                ;;
            --kill-tunnel)
                kill_tunnel=true
                shift
                ;;
            --tunnel-only)
                tunnel_only=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    echo -e "${BLUE}"
    echo "  ╔═══════════════════════════════════════════════════════╗"
    echo "  ║     E2I Droplet Connection Manager                    ║"
    echo "  ║     Droplet: $DROPLET_IP                       ║"
    echo "  ╚═══════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Handle kill tunnel
    if [ "$kill_tunnel" = true ]; then
        kill_existing_tunnels
        exit 0
    fi

    # Handle tunnel only
    if [ "$tunnel_only" = true ]; then
        check_connectivity
        start_tunnel
        verify_tunnel
        exit 0
    fi

    # Full flow
    check_connectivity
    check_docker_health
    check_dependencies
    check_services

    if [ "$check_only" = true ]; then
        print_header "Health Check Complete"
        exit 0
    fi

    start_tunnel
    verify_tunnel

    print_header "Connection Complete"
    echo -e "  ${GREEN}All services are accessible via localhost${NC}"
    echo ""
}

main "$@"
