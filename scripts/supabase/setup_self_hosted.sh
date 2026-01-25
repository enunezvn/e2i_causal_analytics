#!/bin/bash
# =============================================================================
# Setup Self-Hosted Supabase
# =============================================================================
# This script sets up a self-hosted Supabase instance using Docker Compose
# on the DigitalOcean droplet.
#
# Prerequisites:
#   - Docker and Docker Compose installed
#   - SSH access to the droplet
#   - Sufficient disk space (10GB+)
#
# Usage:
#   ./setup_self_hosted.sh [--generate-keys] [--start]
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
SUPABASE_DIR="/opt/supabase"
E2I_SUPABASE_CONFIG="$PROJECT_ROOT/docker/supabase"

# Default values
GENERATE_KEYS=false
START_SERVICES=false
POSTGRES_PASSWORD=""
JWT_SECRET=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --generate-keys)
            GENERATE_KEYS=true
            shift
            ;;
        --start)
            START_SERVICES=true
            shift
            ;;
        --postgres-password)
            POSTGRES_PASSWORD="$2"
            shift 2
            ;;
        --jwt-secret)
            JWT_SECRET="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --generate-keys       Generate new secure keys"
            echo "  --start               Start services after setup"
            echo "  --postgres-password   Set PostgreSQL password"
            echo "  --jwt-secret          Set JWT secret"
            echo "  --help, -h            Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== Self-Hosted Supabase Setup ===${NC}"
echo "Supabase directory: $SUPABASE_DIR"
echo ""

# Function to generate secure secrets
generate_secrets() {
    echo -e "${YELLOW}Generating secure secrets...${NC}"

    if [[ -z "$POSTGRES_PASSWORD" ]]; then
        POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9' | head -c 32)
        echo "Generated POSTGRES_PASSWORD"
    fi

    if [[ -z "$JWT_SECRET" ]]; then
        JWT_SECRET=$(openssl rand -base64 48 | tr -dc 'a-zA-Z0-9' | head -c 48)
        echo "Generated JWT_SECRET"
    fi

    # Generate ANON_KEY and SERVICE_ROLE_KEY
    # These need to be valid JWTs signed with the JWT_SECRET
    echo -e "${YELLOW}Generating API keys...${NC}"

    # Payload for anon key
    ANON_PAYLOAD='{"role":"anon","iss":"supabase","iat":1640000000,"exp":1893456000}'
    SERVICE_PAYLOAD='{"role":"service_role","iss":"supabase","iat":1640000000,"exp":1893456000}'

    # Base64 encode (URL-safe)
    encode_base64url() {
        echo -n "$1" | base64 -w 0 | tr '+/' '-_' | tr -d '='
    }

    # Create JWT header
    JWT_HEADER='{"alg":"HS256","typ":"JWT"}'
    HEADER_B64=$(encode_base64url "$JWT_HEADER")

    # Create JWTs (simplified - for production use proper JWT library)
    ANON_PAYLOAD_B64=$(encode_base64url "$ANON_PAYLOAD")
    SERVICE_PAYLOAD_B64=$(encode_base64url "$SERVICE_PAYLOAD")

    # Note: For production, use the Supabase key generator
    # https://supabase.com/docs/guides/self-hosting/docker#generate-api-keys

    echo ""
    echo -e "${BLUE}Generated Secrets (SAVE THESE SECURELY):${NC}"
    echo "============================================"
    echo "POSTGRES_PASSWORD=$POSTGRES_PASSWORD"
    echo "JWT_SECRET=$JWT_SECRET"
    echo ""
    echo -e "${YELLOW}IMPORTANT: Generate ANON_KEY and SERVICE_ROLE_KEY using:${NC}"
    echo "https://supabase.com/docs/guides/self-hosting/docker#generate-api-keys"
    echo "Use the JWT_SECRET above when generating keys."
    echo "============================================"

    # Save to a secure file
    cat > "$E2I_SUPABASE_CONFIG/.secrets" << EOF
# Supabase Self-Hosted Secrets
# Generated: $(date -Iseconds)
# WARNING: Keep this file secure and never commit to git!

POSTGRES_PASSWORD=$POSTGRES_PASSWORD
JWT_SECRET=$JWT_SECRET

# Generate ANON_KEY and SERVICE_ROLE_KEY at:
# https://supabase.com/docs/guides/self-hosting/docker#generate-api-keys
ANON_KEY=<generate-this>
SERVICE_ROLE_KEY=<generate-this>
EOF
    chmod 600 "$E2I_SUPABASE_CONFIG/.secrets"
    echo -e "${GREEN}Secrets saved to: $E2I_SUPABASE_CONFIG/.secrets${NC}"
}

# Check Docker
check_docker() {
    echo "Checking Docker installation..."

    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker not installed${NC}"
        exit 1
    fi

    if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Error: Docker Compose not installed${NC}"
        exit 1
    fi

    echo -e "${GREEN}Docker is installed${NC}"
}

# Clone Supabase repository
setup_supabase_repo() {
    echo "Setting up Supabase repository..."

    if [[ -d "$SUPABASE_DIR" ]]; then
        echo -e "${YELLOW}Supabase directory exists. Updating...${NC}"
        cd "$SUPABASE_DIR"
        git pull origin master || true
    else
        echo "Cloning Supabase repository..."
        sudo git clone --depth 1 https://github.com/supabase/supabase "$SUPABASE_DIR"
        sudo chown -R "$(whoami):$(whoami)" "$SUPABASE_DIR"
    fi

    cd "$SUPABASE_DIR/docker"

    if [[ ! -f ".env" ]]; then
        cp .env.example .env
        echo -e "${GREEN}Created .env from template${NC}"
    fi
}

# Configure for E2I integration
configure_e2i_integration() {
    echo "Configuring E2I integration..."

    mkdir -p "$E2I_SUPABASE_CONFIG"

    # Create Docker Compose override for E2I network integration
    cat > "$E2I_SUPABASE_CONFIG/docker-compose.override.yml" << 'EOF'
# Docker Compose override for E2I integration
# This file extends the official Supabase docker-compose.yml
# to integrate with E2I's existing network

version: '3.8'

services:
  # Ensure PostgreSQL is accessible from E2I backend
  db:
    networks:
      - default
      - e2i-backend-network
    ports:
      - "5433:5432"  # Use 5433 to avoid conflict with any existing postgres

  # Kong API Gateway accessible externally
  kong:
    networks:
      - default
      - e2i-backend-network
    ports:
      - "54321:8000"  # HTTP - external 54321 maps to internal 8000
      - "8443:8443"  # HTTPS

  # Studio for database management
  studio:
    ports:
      - "3001:3000"  # Use 3001 to avoid conflict with frontend

networks:
  e2i-backend-network:
    external: true
    name: e2i-backend-network
EOF

    echo -e "${GREEN}Created Docker Compose override${NC}"

    # Create E2I-specific environment template
    cat > "$E2I_SUPABASE_CONFIG/.env.e2i.template" << 'EOF'
############
# E2I Causal Analytics - Supabase Self-Hosted Configuration
# Copy to /opt/supabase/docker/.env and fill in secrets
############

############
# Secrets - CHANGE ALL OF THESE!
############
POSTGRES_PASSWORD=your-super-secure-password-here
JWT_SECRET=your-super-secret-jwt-token-with-at-least-32-characters

# Generate these at: https://supabase.com/docs/guides/self-hosting/docker#generate-api-keys
ANON_KEY=your-generated-anon-key
SERVICE_ROLE_KEY=your-generated-service-role-key

############
# Database Configuration
############
POSTGRES_HOST=db
POSTGRES_DB=postgres
POSTGRES_PORT=5432

############
# API Proxy - Kong Configuration
############
KONG_HTTP_PORT=54321
KONG_HTTPS_PORT=8443

############
# PostgREST Configuration
############
PGRST_DB_SCHEMAS=public,storage,graphql_public

############
# Auth - GoTrue Configuration
############
SITE_URL=http://138.197.4.36
ADDITIONAL_REDIRECT_URLS=http://localhost:5174,http://localhost:3000
JWT_EXPIRY=3600
DISABLE_SIGNUP=false
API_EXTERNAL_URL=http://138.197.4.36:54321

# Email settings (optional)
GOTRUE_SMTP_HOST=
GOTRUE_SMTP_PORT=587
GOTRUE_SMTP_USER=
GOTRUE_SMTP_PASS=
GOTRUE_SMTP_SENDER_NAME=E2I Analytics

############
# Studio Configuration
############
STUDIO_DEFAULT_ORGANIZATION=E2I Causal Analytics
STUDIO_DEFAULT_PROJECT=E2I Production
STUDIO_PORT=3000
SUPABASE_PUBLIC_URL=http://138.197.4.36:54321

############
# Storage Configuration
############
STORAGE_BACKEND=file

############
# Feature Flags (disable unused features to save resources)
############
ENABLE_PHONE_SIGNUP=false
ENABLE_PHONE_AUTOCONFIRM=false

############
# Logging
############
LOGFLARE_LOGGER_BACKEND_API_KEY=
LOGFLARE_API_KEY=

############
# Analytics (optional)
############
ENABLE_ANALYTICS=false
EOF

    echo -e "${GREEN}Created E2I environment template${NC}"
}

# Create startup script
create_startup_script() {
    cat > "$E2I_SUPABASE_CONFIG/start.sh" << 'EOF'
#!/bin/bash
# Start Supabase self-hosted services

set -e

SUPABASE_DIR="/opt/supabase/docker"
E2I_CONFIG_DIR="$(dirname "$0")"

cd "$SUPABASE_DIR"

# Ensure E2I backend network exists
docker network create e2i-backend-network 2>/dev/null || true

# Start with E2I override
docker compose \
    -f docker-compose.yml \
    -f "$E2I_CONFIG_DIR/docker-compose.override.yml" \
    up -d

echo "Supabase services starting..."
echo "Studio will be available at: http://localhost:3001"
echo "API Gateway at: http://localhost:54321"
EOF
    chmod +x "$E2I_SUPABASE_CONFIG/start.sh"

    cat > "$E2I_SUPABASE_CONFIG/stop.sh" << 'EOF'
#!/bin/bash
# Stop Supabase self-hosted services

SUPABASE_DIR="/opt/supabase/docker"
E2I_CONFIG_DIR="$(dirname "$0")"

cd "$SUPABASE_DIR"

docker compose \
    -f docker-compose.yml \
    -f "$E2I_CONFIG_DIR/docker-compose.override.yml" \
    down
EOF
    chmod +x "$E2I_SUPABASE_CONFIG/stop.sh"

    echo -e "${GREEN}Created startup scripts${NC}"
}

# Main execution
check_docker

mkdir -p "$E2I_SUPABASE_CONFIG"

if [[ "$GENERATE_KEYS" == "true" ]]; then
    generate_secrets
fi

setup_supabase_repo
configure_e2i_integration
create_startup_script

echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Next steps:"
echo "1. Generate API keys at: https://supabase.com/docs/guides/self-hosting/docker#generate-api-keys"
echo "2. Update /opt/supabase/docker/.env with your secrets"
echo "3. Start Supabase: $E2I_SUPABASE_CONFIG/start.sh"
echo "4. Import your data: ./import_data.sh"
echo ""

if [[ "$START_SERVICES" == "true" ]]; then
    echo -e "${YELLOW}Starting Supabase services...${NC}"
    "$E2I_SUPABASE_CONFIG/start.sh"
fi
