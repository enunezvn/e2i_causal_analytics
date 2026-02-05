# =============================================================================
# E2I Causal Analytics - Makefile
# =============================================================================
# Convenience commands for Docker development and deployment
#
# Usage:
#   make help      - Show all available commands
#   make dev       - Start development environment
#   make prod      - Start production environment
#   make logs      - View logs
#   make down      - Stop all containers
# =============================================================================

.PHONY: help dev prod down logs shell test clean build push

# Default target
.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# VARIABLES
# -----------------------------------------------------------------------------
COMPOSE := docker compose
COMPOSE_DEV := $(COMPOSE) -f docker-compose.yml -f docker-compose.dev.yml
PROJECT_NAME := e2i-causal-analytics

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

# -----------------------------------------------------------------------------
# HELP
# -----------------------------------------------------------------------------
help: ## Show this help message
	@echo ""
	@echo "$(CYAN)E2I Causal Analytics - Docker Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)Development:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'dev|test|shell|logs' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Production:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'prod|build|push' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Utilities:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E 'down|clean|status|volumes' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# -----------------------------------------------------------------------------
# DEVELOPMENT
# -----------------------------------------------------------------------------
dev: ## Start development environment with hot-reloading
	@echo "$(GREEN)Starting E2I development environment...$(RESET)"
	$(COMPOSE_DEV) up --build

dev-d: ## Start development environment in detached mode
	@echo "$(GREEN)Starting E2I development environment (detached)...$(RESET)"
	$(COMPOSE_DEV) up --build -d

dev-tools: ## Start dev environment with Flower and Redis Commander
	@echo "$(GREEN)Starting E2I with dev tools...$(RESET)"
	$(COMPOSE_DEV) --profile dev-tools up --build

logs: ## View logs from all containers
	$(COMPOSE_DEV) logs -f

logs-api: ## View API container logs
	$(COMPOSE_DEV) logs -f api

logs-worker: ## View Worker container logs
	$(COMPOSE_DEV) logs -f worker

shell-api: ## Open shell in API container
	$(COMPOSE_DEV) exec api /bin/bash

shell-worker: ## Open shell in Worker container
	$(COMPOSE_DEV) exec worker /bin/bash

# -----------------------------------------------------------------------------
# TESTING
# -----------------------------------------------------------------------------
test: ## Run tests in container
	@echo "$(GREEN)Running tests...$(RESET)"
	$(COMPOSE_DEV) --profile testing run --rm test

test-unit: ## Run unit tests only
	$(COMPOSE_DEV) --profile testing run --rm test pytest tests/unit -v

test-integration: ## Run integration tests only
	$(COMPOSE_DEV) --profile testing run --rm test pytest tests/integration -v

test-cov: ## Run tests with coverage report
	$(COMPOSE_DEV) --profile testing run --rm test pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# -----------------------------------------------------------------------------
# PRODUCTION
# -----------------------------------------------------------------------------
prod: ## Start production environment
	@echo "$(GREEN)Starting E2I production environment...$(RESET)"
	$(COMPOSE) up -d

prod-build: ## Build production images
	@echo "$(GREEN)Building production images...$(RESET)"
	$(COMPOSE) build --no-cache

prod-logs: ## View production logs
	$(COMPOSE) logs -f

# -----------------------------------------------------------------------------
# BUILD & PUSH
# -----------------------------------------------------------------------------
build: ## Build all Docker images
	@echo "$(GREEN)Building all images...$(RESET)"
	$(COMPOSE) build

build-api: ## Build API image only
	$(COMPOSE) build api

build-frontend: ## Build Frontend image only
	$(COMPOSE) build frontend

push: ## Push images to registry (requires REGISTRY env var)
	@echo "$(YELLOW)Pushing images to $(REGISTRY)...$(RESET)"
	docker push $(REGISTRY)/e2i-api:latest
	docker push $(REGISTRY)/e2i-frontend:latest
	docker push $(REGISTRY)/e2i-worker:latest

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
down: ## Stop all containers
	@echo "$(YELLOW)Stopping all containers...$(RESET)"
	$(COMPOSE_DEV) down

down-v: ## Stop all containers and remove volumes
	@echo "$(YELLOW)Stopping containers and removing volumes...$(RESET)"
	$(COMPOSE_DEV) down -v

status: ## Show status of all containers
	$(COMPOSE_DEV) ps

volumes: ## List all project volumes
	@echo "$(CYAN)E2I Volumes:$(RESET)"
	docker volume ls | grep e2i

clean: ## Remove all containers, networks, and volumes
	@echo "$(YELLOW)Cleaning up E2I Docker resources...$(RESET)"
	$(COMPOSE_DEV) down -v --remove-orphans
	docker system prune -f

clean-all: ## Full cleanup including images
	@echo "$(YELLOW)Full cleanup (including images)...$(RESET)"
	$(COMPOSE_DEV) down -v --remove-orphans --rmi all
	docker system prune -af

# -----------------------------------------------------------------------------
# INDIVIDUAL SERVICES
# -----------------------------------------------------------------------------
up-infra: ## Start only infrastructure (Redis, FalkorDB)
	$(COMPOSE_DEV) up -d redis falkordb

up-mlops: ## Start MLOps services (MLflow, BentoML, Feast, Opik)
	$(COMPOSE_DEV) up -d mlflow bentoml feast opik

restart-api: ## Restart API container
	$(COMPOSE_DEV) restart api

restart-worker: ## Restart Worker container
	$(COMPOSE_DEV) restart worker

# -----------------------------------------------------------------------------
# DATABASE & MIGRATIONS
# -----------------------------------------------------------------------------
migrate: ## Run database migrations (via API container)
	$(COMPOSE_DEV) exec api python scripts/setup_db.py

seed: ## Seed database with sample data
	$(COMPOSE_DEV) exec api python scripts/seed_data.py

# -----------------------------------------------------------------------------
# MLOPS OPERATIONS
# -----------------------------------------------------------------------------
mlflow-ui: ## Open MLflow UI in browser
	@echo "$(CYAN)Opening MLflow UI at http://localhost:5000$(RESET)"
	open http://localhost:5000 || xdg-open http://localhost:5000

opik-ui: ## Open Opik UI in browser
	@echo "$(CYAN)Opening Opik UI at http://localhost:5173$(RESET)"
	open http://localhost:5173 || xdg-open http://localhost:5173

flower-ui: ## Open Flower (Celery monitoring) in browser
	@echo "$(CYAN)Opening Flower at http://localhost:5555$(RESET)"
	open http://localhost:5555 || xdg-open http://localhost:5555

# -----------------------------------------------------------------------------
# HEALTH CHECKS
# -----------------------------------------------------------------------------
health: ## Check health of all services
	@echo "$(CYAN)Checking service health...$(RESET)"
	@echo ""
	@echo "API:      $$(curl -s http://localhost:8000/health || echo 'DOWN')"
	@echo "MLflow:   $$(curl -s http://localhost:5000/health || echo 'DOWN')"
	@echo "BentoML:  $$(curl -s http://localhost:3000/healthz || echo 'DOWN')"
	@echo "Feast:    $$(curl -s http://localhost:6566/health || echo 'DOWN')"
	@echo "Opik:     $$(curl -s http://localhost:5173/health || echo 'DOWN')"
	@echo "Redis:    $$(docker exec e2i_redis_dev redis-cli ping 2>/dev/null || echo 'DOWN')"
	@echo "FalkorDB: $$(docker exec e2i_falkordb_dev redis-cli ping 2>/dev/null || echo 'DOWN')"
	@echo "Frontend: $$(curl -s http://localhost:3002/health || echo 'DOWN')"
