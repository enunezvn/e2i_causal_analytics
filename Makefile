.PHONY: help install dev-install test test-fast test-seq test-cov lint format clean docker-up docker-down db-init data-generate

help:
	@echo "E2I Causal Analytics - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install production dependencies"
	@echo "  make dev-install   Install development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test          Run tests (4 workers, memory-safe, terminal report)"
	@echo "  make test-fast     Run tests (no coverage, faster)"
	@echo "  make test-seq      Run tests sequentially (low memory systems)"
	@echo "  make test-cov      Run tests with full coverage (HTML + XML reports)"
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

# Memory-safe test run: 4 workers, scope-based distribution, coverage
# System: 7.5GB RAM, heavy ML imports (~500MB/worker)
# Coverage config from pyproject.toml [tool.coverage.*]
test:
	pytest tests/ --cov --cov-report=term-missing

# Fast test run without coverage (lower memory footprint)
test-fast:
	pytest tests/

# Sequential run for very low memory systems (--cov adds overhead)
test-seq:
	pytest tests/ -n 0 --cov --cov-report=term-missing

# Full coverage run with all reports (HTML + XML for CI/CD)
# Creates: htmlcov/ directory and coverage.xml
test-cov:
	pytest tests/ --cov --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo ""
	@echo "Coverage reports generated:"
	@echo "  - Terminal: above"
	@echo "  - HTML: htmlcov/index.html"
	@echo "  - XML: coverage.xml (for CI/CD)"

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
	docker run -d --name e2i_redis -p 6382:6379 redis:latest || true
	docker run -d --name e2i_falkordb -p 6381:6379 falkordb/falkordb:latest || true
	@echo "Docker services started: Redis (6382), FalkorDB (6381)"

docker-down:
	docker stop e2i_redis e2i_falkordb 2>/dev/null || true
	docker rm e2i_redis e2i_falkordb 2>/dev/null || true
	@echo "Docker services stopped"

db-init:
	@echo "Initializing database schemas..."
	@echo "Run your database initialization scripts here"

data-generate:
	python src/ml/data_generator.py
	python src/ml/data_loader.py
