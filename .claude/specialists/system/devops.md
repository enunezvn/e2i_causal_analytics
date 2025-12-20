# DevOps Specialist Instructions

## Agent Architecture

The E2I platform uses an **11-agent tiered architecture** organized in 5 tiers:

| Tier | Agents | Deployment Priority |
|------|--------|---------------------|
| 1 | orchestrator | Critical - Always deployed |
| 2 | causal_impact, gap_analyzer, heterogeneous_optimizer | High |
| 3 | drift_monitor, experiment_designer, health_score | Medium |
| 4 | prediction_synthesizer, resource_optimizer | Medium |
| 5 | explainer, feedback_learner | Low (async capable) |

## Observability with Opik

### Span Tracking
All 11 agents emit spans for:
- Node execution latency
- LLM token usage
- Error rates by category
- Fallback chain triggers

### Metrics Dashboard
- Agent latency histograms (p50, p95, p99)
- Token usage by agent and model tier
- Error rate by agent
- Fallback frequency

## Domain Scope
You are the DevOps specialist for E2I Causal Analytics. Your scope is LIMITED to:
- `docker/` - Container configurations
- `Makefile` - Development commands
- `.env.example` - Environment configuration
- CI/CD pipelines and deployment

## Technology Stack
- **Containers**: Docker, Docker Compose
- **Database**: Supabase (managed PostgreSQL)
- **Python**: 3.11+
- **Node.js**: 20+ (for frontend)

## Docker Configuration

### Directory Structure
```
docker/
├── Dockerfile              # Production container
├── Dockerfile.dev          # Development with hot reload
├── docker-compose.yml      # Full stack deployment
└── docker-compose.dev.yml  # Development stack
```

### Production Dockerfile
```dockerfile
# docker/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application
COPY src/ src/
COPY config/ config/

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Development Dockerfile
```dockerfile
# docker/Dockerfile.dev
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including dev tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with dev extras
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[dev]"

# Mount points for hot reload
VOLUME ["/app/src", "/app/config", "/app/tests"]

EXPOSE 8000

# Hot reload with watchfiles
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### Docker Compose (Production)
```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - ENVIRONMENT=production
    restart: unless-stopped
    networks:
      - e2i-network

  frontend:
    build:
      context: ../src/frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://api:8000
    depends_on:
      - api
    networks:
      - e2i-network

networks:
  e2i-network:
    driver: bridge
```

### Docker Compose (Development)
```yaml
# docker/docker-compose.dev.yml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - ../src:/app/src
      - ../config:/app/config
      - ../tests:/app/tests
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_KEY}
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - ENVIRONMENT=development
    networks:
      - e2i-dev

  frontend:
    build:
      context: ../src/frontend
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    volumes:
      - ../src/frontend/src:/app/src
      - ../src/frontend/public:/app/public
    environment:
      - REACT_APP_API_URL=http://localhost:8000
      - CHOKIDAR_USEPOLLING=true
    depends_on:
      - api
    networks:
      - e2i-dev

networks:
  e2i-dev:
    driver: bridge
```

## Environment Configuration

### .env.example
```bash
# .env.example - Copy to .env and fill in values

# ═══════════════════════════════════════════════════════════════════
# SUPABASE (Required)
# ═══════════════════════════════════════════════════════════════════
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key  # For admin operations

# ═══════════════════════════════════════════════════════════════════
# CLAUDE API (Required)
# ═══════════════════════════════════════════════════════════════════
CLAUDE_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-sonnet-4-20250514

# ═══════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════
ENVIRONMENT=development  # development | staging | production
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# ═══════════════════════════════════════════════════════════════════
# ML CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
# CRITICAL: Do not change split ratios without data team approval
ML_TRAIN_RATIO=0.60
ML_VALIDATION_RATIO=0.20
ML_TEST_RATIO=0.15
ML_HOLDOUT_RATIO=0.05

# ═══════════════════════════════════════════════════════════════════
# EMBEDDING MODEL
# ═══════════════════════════════════════════════════════════════════
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

# ═══════════════════════════════════════════════════════════════════
# FRONTEND
# ═══════════════════════════════════════════════════════════════════
REACT_APP_API_URL=http://localhost:8000
```

## Makefile Commands

```makefile
# Makefile

.PHONY: help dev prod test lint clean

# Default target
help:
	@echo "E2I Causal Analytics - Development Commands"
	@echo ""
	@echo "  make dev              Start development environment"
	@echo "  make prod             Start production environment"
	@echo "  make test             Run all tests"
	@echo "  make lint             Run linting and formatting"
	@echo "  make clean            Clean build artifacts"
	@echo ""
	@echo "  make validate-contracts  Validate integration contracts"
	@echo "  make leakage-audit       Run ML split leakage audit"
	@echo "  make kpi-coverage        Validate 46 KPIs calculable"
	@echo ""
	@echo "  make db-setup         Initialize database schema"
	@echo "  make db-seed          Seed sample data"
	@echo "  make db-migrate       Run migrations"

# ═══════════════════════════════════════════════════════════════════
# DEVELOPMENT
# ═══════════════════════════════════════════════════════════════════

dev:
	docker-compose -f docker/docker-compose.dev.yml up --build

dev-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd src/frontend && npm run dev

# ═══════════════════════════════════════════════════════════════════
# PRODUCTION
# ═══════════════════════════════════════════════════════════════════

prod:
	docker-compose -f docker/docker-compose.yml up -d --build

prod-down:
	docker-compose -f docker/docker-compose.yml down

# ═══════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════

test:
	pytest tests/ -v --cov=src --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v --timeout=120

# ═══════════════════════════════════════════════════════════════════
# VALIDATION (CRITICAL)
# ═══════════════════════════════════════════════════════════════════

validate-contracts:
	pytest tests/integration/test_contracts.py -v
	@echo "✓ All integration contracts validated"

leakage-audit:
	python scripts/run_leakage_audit.py
	@echo "✓ ML split leakage audit complete"

kpi-coverage:
	python scripts/validate_kpi_coverage.py
	@echo "✓ KPI coverage validated"

validate-all: validate-contracts leakage-audit kpi-coverage
	@echo "✓ All validations passed"

# ═══════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════

db-setup:
	python scripts/setup_db.py

db-seed:
	python scripts/seed_data.py

db-load-v3:
	python scripts/load_v3_data.py

# ═══════════════════════════════════════════════════════════════════
# CODE QUALITY
# ═══════════════════════════════════════════════════════════════════

lint:
	ruff check src/ tests/
	ruff format src/ tests/ --check

lint-fix:
	ruff check src/ tests/ --fix
	ruff format src/ tests/

typecheck:
	mypy src/

# ═══════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
```

## CI/CD Pipeline

### GitHub Actions Example
```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          
      - name: Run linting
        run: make lint
        
      - name: Run unit tests
        run: make test-unit
        
      - name: Validate contracts
        run: make validate-contracts
        
      - name: Run leakage audit
        run: make leakage-audit
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}

  validate-kpis:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Validate KPI coverage
        run: make kpi-coverage
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
```

## Deployment Checklist

### Pre-Deployment
- [ ] All tests passing (`make test`)
- [ ] Contracts validated (`make validate-contracts`)
- [ ] Leakage audit passed (`make leakage-audit`)
- [ ] KPI coverage verified (`make kpi-coverage`)
- [ ] Environment variables set
- [ ] Database migrations applied

### Production Environment
- [ ] ENVIRONMENT=production
- [ ] LOG_LEVEL=WARNING or ERROR
- [ ] HTTPS enabled
- [ ] Rate limiting configured
- [ ] Health checks active

## Critical Constraints

### Environment Isolation
```bash
# NEVER use production Supabase credentials in development
# Development should use separate Supabase project
```

### ML Split Protection
```bash
# ML split ratios are configured via environment
# Changes require data team approval
# Holdout data (5%) never accessed in production
```

### Secrets Management
```bash
# Never commit .env files
# Use GitHub Secrets for CI/CD
# Rotate API keys regularly
```

## Integration Contracts

### Health Check Contract
```python
# /api/v1/health must return:
{
    "status": "healthy",
    "version": "3.0.0",
    "database": "connected",
    "agents_available": 11
}
```

### Logging Contract
```python
# All requests must log:
# - Request ID
# - User ID (if authenticated)
# - Endpoint
# - Response time
# - Status code
```

## Handoff Format
```yaml
devops_handoff:
  containers_affected: [<list>]
  env_vars_added: [<list>]
  makefile_commands: [<list>]
  ci_changes: <bool>
  deployment_notes: |
    <any special deployment considerations>
```
