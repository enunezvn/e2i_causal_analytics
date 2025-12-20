.PHONY: help install dev-install test lint format clean docker-up docker-down db-init data-generate

help:
	@echo "E2I Causal Analytics - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install production dependencies"
	@echo "  make dev-install   Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run tests"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with black"
	@echo "  make clean         Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up     Start Docker services (Redis, FalkorDB)"
	@echo "  make docker-down   Stop Docker services"
	@echo ""
	@echo "Database:"
	@echo "  make db-init       Initialize database schemas"
	@echo "  make data-generate Generate synthetic data"
	@echo ""

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .ruff_cache/

docker-up:
	docker run -d --name redis-working-memory -p 6379:6379 redis:latest || true
	docker run -d --name falkordb -p 6380:6379 falkordb/falkordb:latest || true
	@echo "Docker services started: Redis (6379), FalkorDB (6380)"

docker-down:
	docker stop redis-working-memory falkordb 2>/dev/null || true
	docker rm redis-working-memory falkordb 2>/dev/null || true
	@echo "Docker services stopped"

db-init:
	@echo "Initializing database schemas..."
	@echo "Run your database initialization scripts here"

data-generate:
	python src/ml/data_generator.py
	python src/ml/data_loader.py
