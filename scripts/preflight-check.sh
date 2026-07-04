#!/bin/bash
# ==============================================================================
# E2I Causal Analytics - Pre-Flight Check Script
# ==============================================================================
# Validates environment setup before starting Docker Compose
# ==============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
ERRORS=0
WARNINGS=0

echo "=========================================="
echo "E2I Causal Analytics - Pre-Flight Check"
echo "=========================================="
echo ""

# ------------------------------------------------------------------------------
# Check 1: Docker and Docker Compose installed
# ------------------------------------------------------------------------------
echo -e "${BLUE}[1/7] Checking Docker installation...${NC}"

if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version)
    echo -e "${GREEN}✓ Docker installed: $DOCKER_VERSION${NC}"
else
    echo -e "${RED}✗ Docker not found. Please install Docker first.${NC}"
    ERRORS=$((ERRORS + 1))
fi

if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version)
    echo -e "${GREEN}✓ Docker Compose installed: $COMPOSE_VERSION${NC}"
else
    echo -e "${RED}✗ Docker Compose not found. Please install Docker Compose.${NC}"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# ------------------------------------------------------------------------------
# Check 2: Environment file exists
# ------------------------------------------------------------------------------
echo -e "${BLUE}[2/7] Checking environment file...${NC}"

if [ -f ".env.dev" ]; then
    echo -e "${GREEN}✓ .env.dev file found${NC}"
else
    echo -e "${RED}✗ .env.dev file not found${NC}"
    echo -e "${YELLOW}  Run: cp .env.template .env.dev${NC}"
    ERRORS=$((ERRORS + 1))
fi

if [ -f ".env" ]; then
    echo -e "${YELLOW}⚠ .env file found (will be used by docker-compose)${NC}"
    WARNINGS=$((WARNINGS + 1))
fi

echo ""

# ------------------------------------------------------------------------------
# Check 3: Required environment variables
# ------------------------------------------------------------------------------
echo -e "${BLUE}[3/7] Checking required environment variables...${NC}"

# Source the .env.dev file if it exists
if [ -f ".env.dev" ]; then
    export $(grep -v '^#' .env.dev | xargs)
fi

# Required variables
REQUIRED_VARS=(
    "DATABASE_URL"
    "CLAUDE_API_KEY"
    "SUPABASE_URL"
    "SUPABASE_ANON_KEY"
)

MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ] || [ "${!var}" = "xxxxx" ] || [[ "${!var}" == *"TODO"* ]]; then
        echo -e "${RED}✗ $var not set or contains placeholder${NC}"
        MISSING_VARS+=("$var")
        ERRORS=$((ERRORS + 1))
    else
        # Mask sensitive values
        MASKED_VALUE=$(echo "${!var}" | sed 's/\(.\{10\}\).*/\1.../')
        echo -e "${GREEN}✓ $var set: $MASKED_VALUE${NC}"
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Missing variables need to be set in .env.dev:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo -e "  - $var"
    done
fi

echo ""

# ------------------------------------------------------------------------------
# Check 4: Docker daemon running
# ------------------------------------------------------------------------------
echo -e "${BLUE}[4/7] Checking Docker daemon...${NC}"

if docker info &> /dev/null; then
    echo -e "${GREEN}✓ Docker daemon is running${NC}"
else
    echo -e "${RED}✗ Docker daemon not running${NC}"
    echo -e "${YELLOW}  Start Docker Desktop or run: sudo systemctl start docker${NC}"
    ERRORS=$((ERRORS + 1))
fi

echo ""

# ------------------------------------------------------------------------------
# Check 5: Port availability
# ------------------------------------------------------------------------------
echo -e "${BLUE}[5/7] Checking port availability...${NC}"

PORTS=(8000 3000 5000 6382)
PORT_NAMES=("FastAPI" "Frontend" "MLflow" "Redis")

for i in "${!PORTS[@]}"; do
    PORT=${PORTS[$i]}
    NAME=${PORT_NAMES[$i]}

    if lsof -Pi :$PORT -sTCP:LISTEN -t &>/dev/null; then
        echo -e "${YELLOW}⚠ Port $PORT ($NAME) is already in use${NC}"
        WARNINGS=$((WARNINGS + 1))
    else
        echo -e "${GREEN}✓ Port $PORT ($NAME) is available${NC}"
    fi
done

echo ""

# ------------------------------------------------------------------------------
# Check 6: Required directories and files
# ------------------------------------------------------------------------------
echo -e "${BLUE}[6/7] Checking project structure...${NC}"

REQUIRED_PATHS=(
    "docker-compose.yml"
    "docker-compose.dev.yml"
    "docker/fastapi/Dockerfile"
    "docker/frontend/Dockerfile"
    "docker/mlflow/Dockerfile"
    "src"
    "config"
)

for path in "${REQUIRED_PATHS[@]}"; do
    if [ -e "$path" ]; then
        echo -e "${GREEN}✓ $path exists${NC}"
    else
        echo -e "${RED}✗ $path not found${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done

echo ""

# ------------------------------------------------------------------------------
# Check 7: Docker Compose configuration validity
# ------------------------------------------------------------------------------
echo -e "${BLUE}[7/7] Validating Docker Compose configuration...${NC}"

if docker compose -f docker-compose.yml -f docker-compose.dev.yml config --quiet 2>&1 | grep -q "error"; then
    echo -e "${RED}✗ Docker Compose configuration has errors${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✓ Docker Compose configuration is valid${NC}"
fi

echo ""

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo "=========================================="
echo "Pre-Flight Check Summary"
echo "=========================================="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed! Ready to start.${NC}"
    echo ""
    echo "Run the following command to start services:"
    echo -e "${BLUE}docker compose -f docker-compose.yml -f docker-compose.dev.yml up${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ $WARNINGS warning(s) found, but you can proceed.${NC}"
    echo ""
    echo "Run the following command to start services:"
    echo -e "${BLUE}docker compose -f docker-compose.yml -f docker-compose.dev.yml up${NC}"
    exit 0
else
    echo -e "${RED}✗ $ERRORS error(s) and $WARNINGS warning(s) found.${NC}"
    echo -e "${RED}Please fix the errors above before starting.${NC}"
    exit 1
fi
