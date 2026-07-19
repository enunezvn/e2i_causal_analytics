.PHONY: help install dev-install test test-fast test-seq test-cov tier1-5-test lint format clean docker-up docker-down docker-logs deploy deploy-build db-init data-generate api-docs generate-types curate-candidates check-compile-backlog

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
	@echo "  make tier1-5-test  Run Tier 1-5 agent harness against cached tier0 outputs"
	@echo "  make lint          Run linting checks"
	@echo "  make format        Format code with black"
	@echo "  make api-docs      Generate OpenAPI spec + Redoc HTML"
	@echo "  make generate-types Regenerate committed frontend/src/types/generated/api.ts"
	@echo "  make clean         Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up     Start all services (compose + dev overlay)"
	@echo "  make docker-down   Stop all services"
	@echo "  make docker-logs   Tail logs for API + frontend"
	@echo "  make deploy        Deploy: git pull + restart workers"
	@echo "  make deploy-build  Deploy with image rebuild"
	@echo ""
	@echo "Database:"
	@echo "  make db-init       Initialize database schemas"
	@echo "  make data-generate Generate synthetic data"
	@echo ""
	@echo "Layer-4 audit curation:"
	@echo "  make curate-candidates  Generate compile-set candidate report from"
	@echo "                          the Layer-4 audit trail (REQUIRED before"
	@echo "                          running scripts/compile_causal_role_classifier.py)"
	@echo "  make check-compile-backlog  Count accepted candidates vs threshold"
	@echo "                              (issue #236; suitable for cron/Slack)"
	@echo ""

# Layer-4 audit-signal curation entry point. Plan:
# .claude/plans/layer4_evaluator_audit_consumer.md (Task 10, Codex Gate-2 MED-5
# forcing function).
curate-candidates:
	@if [ -z "$$ADAPTIVE_VALIDITY_ARTIFACTS_DIR" ]; then \
	    echo "ERROR: ADAPTIVE_VALIDITY_ARTIFACTS_DIR is not set."; \
	    echo "       See .env.example or docker/env.example for the var."; \
	    echo "       Inside docker-compose this defaults to /app/data/audit_artifacts."; \
	    exit 1; \
	fi
	mkdir -p ./candidates
	.venv/bin/python scripts/curate_compile_set_candidates.py \
	    --artifacts-dir "$$ADAPTIVE_VALIDITY_ARTIFACTS_DIR" \
	    --output-dir ./candidates

# Phase 4.5 auto-trigger surface (issue #236). Counts accepted compile-set
# candidates (those whose 4 required fill-ins are non-null) under
# ./candidates/ since the compiled artifact's mtime, and prints a
# grep-friendly "READY" signal when the backlog crosses the threshold
# (default 5). Wire this into a weekly cron / GitHub Action to nudge
# operators when the classifier has enough new evidence to warrant a
# recompile, without auto-running the 5-15min compile job itself.
check-compile-backlog:
	@python scripts/check_compile_set_candidate_backlog.py \
	    --candidates-dir ./candidates \
	    --artifact artifacts/dspy/causal_role_classifier.json

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

# Tier 1-5 agent integration harness. Requires a cached Tier 0 state at
# scripts/tier0_output_cache/latest.pkl. A small, sanitized fixture is committed
# there (issue #600) so CI runs the 13 agents on every PR; refresh it with:
#   python scripts/generate_tier0_fixture.py
# For a full real local cache instead, use:
#   python scripts/run_tier1_5_test.py --run-tier0-first
# CI uses --skip-observability so Opik isn't required.
tier1-5-test:
	@mkdir -p docs/results
	@if [ ! -f scripts/tier0_output_cache/latest.pkl ]; then \
	    echo "⚠️  scripts/tier0_output_cache/latest.pkl not found — skipping tier1-5 harness."; \
	    echo "    Generate it with: python scripts/run_tier1_5_test.py --run-tier0-first"; \
	else \
	    python scripts/run_tier1_5_test.py \
	        --tier0-cache scripts/tier0_output_cache/latest.pkl \
	        --skip-observability \
	        --output docs/results/tier1_5_pipeline_latest.json; \
	fi

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
	docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml down

docker-logs:
	docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml logs -f api frontend

deploy:
	./scripts/deploy.sh

deploy-build:
	./scripts/deploy.sh --build

db-init:
	@echo "Initializing database schemas..."
	@echo "Run your database initialization scripts here"

data-generate:
	python src/ml/data_generator.py
	python src/ml/data_loader.py

api-docs:
	./scripts/generate_api_docs.sh

# Regenerate the committed frontend contract baseline the same way CI's
# verify-types drift gate does (static spec export — no running server).
generate-types:
	python -m scripts.export_openapi --output openapi.json
	cd frontend && npx openapi-typescript ../openapi.json -o src/types/generated/api.ts
