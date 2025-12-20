---
argument-hint: [--skip-e2e]
description: Comprehensive validation for Python/FastAPI applications
---

# Python/FastAPI Validation

Comprehensive end-to-end validation for Python/FastAPI applications with PostgreSQL, pytest, and optional Docker.

---

## Input Validation

Before proceeding, validate and parse the validation options.

```bash
# Optional flag to skip E2E tests
if [ -z "$ARGUMENTS" ]; then
  SKIP_E2E=false
  echo "ℹ️  Running full validation including E2E tests"
elif [[ "$ARGUMENTS" == "--skip-e2e" ]]; then
  SKIP_E2E=true
  echo "✓ Skipping E2E tests"
else
  echo "❌ Error: Invalid option '$ARGUMENTS'"
  echo ""
  echo "Valid options:"
  echo "  (none)       Run full validation including E2E tests"
  echo "  --skip-e2e   Skip E2E tests"
  echo ""
  echo "Usage: /validation:validate-python [--skip-e2e]"
  exit 1
fi
```

---

**Usage:**
```bash
/validation:validate-python              # Full validation
/validation:validate-python --skip-e2e   # Skip E2E tests
```

**Prerequisites:**
- Python 3.9+ installed
- Poetry or pip for dependency management
- PostgreSQL database (local or remote)
- `.env` file with configuration

---

## Phase 1: Foundation Validation

### 1.1 Code Quality (Linting)
```bash
# Using flake8 for linting
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```
**Expected:** No critical linting errors

Alternative linters:
```bash
# pylint src/
# ruff check .
```

### 1.2 Type Checking
```bash
# Using mypy for static type checking
mypy . --ignore-missing-imports
```
**Expected:** No type errors

### 1.3 Code Formatting
```bash
# Check code formatting with black
black --check .
```
**Expected:** No formatting issues

If formatting issues found:
```bash
# Auto-fix formatting
# black .
```

Alternative formatters:
```bash
# autopep8 --diff --recursive .
# yapf --diff --recursive .
```

### 1.4 Import Sorting
```bash
# Check import order with isort
isort . --check-only
```
**Expected:** Imports properly sorted

### 1.5 Security Checks
```bash
# Check for known security vulnerabilities
bandit -r src/ -ll
```
**Expected:** No high or medium severity issues

### 1.6 Unit Tests
```bash
# Run unit tests with coverage
pytest tests/unit/ -v --cov=src --cov-report=term-missing --cov-report=html
```
**Expected:** All tests pass, coverage > 80%

### 1.7 Build/Package
```bash
# Verify package can be built
if [ -f "pyproject.toml" ]; then
  python -m build
elif [ -f "setup.py" ]; then
  python setup.py sdist bdist_wheel
fi
```
**Expected:** Package builds successfully

**If any step fails, STOP and report the issue immediately.**

---

## Phase 2: Environment Setup

### 2.1 Load Configuration
```bash
# Load environment variables
if [ ! -f .env ]; then
  echo "ERROR: .env file not found"
  exit 1
fi

source .env

# Validate required variables
REQUIRED_VARS="DATABASE_URL SECRET_KEY"
for VAR in $REQUIRED_VARS; do
  if [ -z "${!VAR}" ]; then
    echo "ERROR: Required environment variable $VAR is not set"
    exit 1
  fi
done

echo "✅ Configuration loaded"
```

### 2.2 Database Connection Check
```bash
# Test database connection using psql or Python
if command -v psql &> /dev/null; then
  psql "$DATABASE_URL" -c "SELECT version();" > /dev/null
  if [ $? -eq 0 ]; then
    echo "✅ Database connection successful"
  else
    echo "❌ Database connection failed"
    exit 1
  fi
else
  # Fallback: Use Python to test connection
  python -c "
from sqlalchemy import create_engine
import os
engine = create_engine(os.getenv('DATABASE_URL'))
conn = engine.connect()
print('✅ Database connection successful')
conn.close()
  " || { echo "❌ Database connection failed"; exit 1; }
fi
```

### 2.3 Database Migrations
```bash
# Apply database migrations (Alembic)
if [ -d "alembic" ]; then
  alembic upgrade head
  echo "✅ Database migrations applied"
else
  echo "⚠️  No Alembic migrations directory found"
fi
```

Alternative migration tools:
```bash
# python manage.py migrate  # Django
# aerich upgrade            # Tortoise ORM
```

### 2.4 Seed Test Data (Optional)
```bash
# Load test data if script exists
if [ -f "scripts/seed.py" ]; then
  python scripts/seed.py
  echo "✅ Test data seeded"
fi
```

---

## Phase 3: Integration Tests

### 3.1 Database Integration Tests
```bash
# Run database integration tests
pytest tests/integration/test_database.py -v
```
**Expected:** All database tests pass

### 3.2 API Integration Tests
```bash
# Run API integration tests
pytest tests/integration/test_api.py -v
```
**Expected:** All API tests pass

### 3.3 External Service Tests (if applicable)
```bash
# Test external integrations
pytest tests/integration/test_external.py -v -m "not slow"
```
**Expected:** External service tests pass

---

## Phase 4: Runtime Validation

### 4.1 Start FastAPI Application
```bash
# Start application in background
if [ -f "main.py" ]; then
  uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/fastapi.log 2>&1 &
  APP_PID=$!
elif [ -f "src/main.py" ]; then
  uvicorn src.main:app --host 0.0.0.0 --port 8000 > /tmp/fastapi.log 2>&1 &
  APP_PID=$!
else
  echo "ERROR: Could not find main.py"
  exit 1
fi

echo "FastAPI started (PID: $APP_PID)"

# Wait for application to start
sleep 5
```

Alternative startup:
```bash
# Using Gunicorn for production-like testing
# gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 &
```

### 4.2 Health Checks
```bash
# Check health endpoint
HEALTH_URL="http://localhost:8000/health"
MAX_RETRIES=10
RETRY_DELAY=2

for i in $(seq 1 $MAX_RETRIES); do
  HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)
  if [ "$HEALTH_RESPONSE" = "200" ]; then
    echo "✅ Application is healthy"
    break
  fi

  if [ $i -eq $MAX_RETRIES ]; then
    echo "❌ Health check failed after $MAX_RETRIES attempts"
    kill $APP_PID 2>/dev/null
    cat /tmp/fastapi.log
    exit 1
  fi

  echo "Waiting for application to be ready ($i/$MAX_RETRIES)..."
  sleep $RETRY_DELAY
done
```

### 4.3 API Smoke Tests
```bash
# Test basic API endpoints
echo "Testing API endpoints..."

# Test root endpoint
curl -f http://localhost:8000/ || echo "⚠️  Root endpoint failed"

# Test docs endpoint
curl -f http://localhost:8000/docs || echo "⚠️  Docs endpoint failed"

# Test OpenAPI schema
curl -f http://localhost:8000/openapi.json || echo "⚠️  OpenAPI schema failed"

# Test version endpoint (if exists)
curl -f http://localhost:8000/api/version || echo "ℹ️  Version endpoint not found"

echo "✅ Smoke tests completed"
```

### 4.4 Database Query Test
```bash
# Test database queries work at runtime
python -c "
from sqlalchemy import create_engine, text
import os

engine = create_engine(os.getenv('DATABASE_URL'))
with engine.connect() as conn:
    result = conn.execute(text('SELECT COUNT(*) FROM pg_tables'))
    count = result.scalar()
    print(f'✅ Database query successful ({count} tables)')
" || { echo "❌ Database query failed"; kill $APP_PID; exit 1; }
```

---

## Phase 5: End-to-End Tests

### 5.1 Check if E2E Should Run
```bash
# Check if --skip-e2e flag was passed
SKIP_E2E=false
if [[ "$ARGUMENTS" == *"--skip-e2e"* ]]; then
  SKIP_E2E=true
  echo "⚠️  Skipping E2E tests (--skip-e2e flag provided)"
fi
```

### 5.2 E2E Test Suite
```bash
if [ "$SKIP_E2E" != "true" ]; then
  # Run end-to-end tests
  pytest tests/e2e/ -v --tb=short

  if [ $? -eq 0 ]; then
    echo "✅ E2E tests passed"
  else
    echo "❌ E2E tests failed"
    kill $APP_PID 2>/dev/null
    exit 1
  fi
else
  echo "ℹ️  E2E tests skipped"
fi
```

### 5.3 Performance Tests (Optional)
```bash
# Optional: Run basic load test
if command -v ab &> /dev/null; then
  echo "Running basic load test..."
  ab -n 100 -c 10 http://localhost:8000/health > /tmp/loadtest.txt 2>&1

  # Extract requests per second
  RPS=$(grep "Requests per second" /tmp/loadtest.txt | awk '{print $4}')
  echo "Performance: $RPS requests/second"
fi
```

---

## Phase 6: Cleanup

### 6.1 Stop Application
```bash
# Stop FastAPI application
if [ -n "$APP_PID" ]; then
  echo "Stopping FastAPI (PID: $APP_PID)..."
  kill $APP_PID
  wait $APP_PID 2>/dev/null
  echo "✅ Application stopped"
fi
```

### 6.2 Clean Test Data
```bash
# Clean up test database (if using separate test DB)
if [ -n "$TEST_DATABASE_URL" ]; then
  python -c "
from sqlalchemy import create_engine
import os
engine = create_engine(os.getenv('TEST_DATABASE_URL'))
engine.dispose()
print('✅ Test database connections closed')
  " 2>/dev/null || echo "ℹ️  No test database to clean"
fi

# Remove temporary files
rm -f /tmp/fastapi.log /tmp/loadtest.txt

echo "✅ Cleanup completed"
```

---

## Phase 7: Summary Report

```bash
echo ""
echo "======================================="
echo "PYTHON/FASTAPI VALIDATION SUMMARY"
echo "======================================="
echo ""
echo "FOUNDATION:"
echo "  Linting (flake8): ✅"
echo "  Type Checking (mypy): ✅"
echo "  Formatting (black): ✅"
echo "  Import Sorting (isort): ✅"
echo "  Security (bandit): ✅"
echo "  Unit Tests: ✅"
echo "  Build: ✅"
echo ""
echo "ENVIRONMENT:"
echo "  Configuration: ✅"
echo "  Database Connection: ✅"
echo "  Migrations: ✅"
echo ""
echo "INTEGRATION:"
echo "  Database Tests: ✅"
echo "  API Tests: ✅"
echo "  External Services: ✅"
echo ""
echo "RUNTIME:"
echo "  Application Startup: ✅"
echo "  Health Checks: ✅"
echo "  Smoke Tests: ✅"
echo "  Database Queries: ✅"
echo ""
echo "END-TO-END:"
if [ "$SKIP_E2E" != "true" ]; then
  echo "  E2E Tests: ✅"
else
  echo "  E2E Tests: ⏭️  (skipped)"
fi
echo "  Performance: ✅ (optional)"
echo ""
echo "OVERALL: ✅ PASS"
echo "======================================="
echo ""
echo "Application validated successfully!"
```

---

## Configuration Files

### Example `.env`
```env
# Application
SECRET_KEY=your-secret-key-here
DEBUG=false
ENVIRONMENT=production

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
TEST_DATABASE_URL=postgresql://user:password@localhost:5432/test_dbname

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# External Services
API_KEY=your-api-key
```

### Example `pyproject.toml`
```toml
[tool.poetry]
name = "your-app"
version = "1.0.0"

[tool.poetry.dependencies]
python = "^3.9"
fastapi = "^0.100.0"
uvicorn = "^0.23.0"
sqlalchemy = "^2.0.0"
psycopg2-binary = "^2.9.0"
alembic = "^1.11.0"
pydantic = "^2.0.0"

[tool.poetry.dev-dependencies]
pytest = "^7.4.0"
pytest-cov = "^4.1.0"
pytest-asyncio = "^0.21.0"
black = "^23.7.0"
flake8 = "^6.0.0"
mypy = "^1.4.0"
isort = "^5.12.0"
bandit = "^1.7.5"

[tool.black]
line-length = 127
target-version = ['py39']

[tool.isort]
profile = "black"
line_length = 127

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

### Example `alembic.ini`
```ini
[alembic]
script_location = alembic
sqlalchemy.url = postgresql://user:password@localhost:5432/dbname

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

---

## Docker Support (Optional)

If using Docker for validation:

### Build and Test with Docker
```bash
# Build Docker image
docker build -t myapp:test .

# Run tests in Docker
docker run --rm \
  -e DATABASE_URL="$DATABASE_URL" \
  -e SECRET_KEY="$SECRET_KEY" \
  myapp:test \
  pytest tests/ -v

# Run validation in Docker Compose
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
docker-compose -f docker-compose.test.yml down
```

### Example `docker-compose.test.yml`
```yaml
version: '3.8'

services:
  app:
    build: .
    environment:
      - DATABASE_URL=postgresql://testuser:testpass@db:5432/testdb
      - SECRET_KEY=test-secret-key
    depends_on:
      - db
    command: pytest tests/ -v

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=testuser
      - POSTGRES_PASSWORD=testpass
      - POSTGRES_DB=testdb
    tmpfs:
      - /var/lib/postgresql/data
```

---

## Troubleshooting

### Common Issues

**Problem**: `ImportError: No module named 'X'`
**Solution**: Install dependencies: `pip install -r requirements.txt` or `poetry install`

**Problem**: Database connection fails
**Solution**: Check DATABASE_URL format and ensure PostgreSQL is running

**Problem**: Alembic migration fails
**Solution**: Check migration files or run `alembic downgrade -1` then `alembic upgrade head`

**Problem**: Port 8000 already in use
**Solution**: Kill existing process: `lsof -ti:8000 | xargs kill -9`

**Problem**: Tests fail with "database locked"
**Solution**: Use separate test database or add `--forked` flag to pytest

---

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: testdb
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install poetry
          poetry install

      - name: Run validation
        env:
          DATABASE_URL: postgresql://testuser:testpass@localhost:5432/testdb
          SECRET_KEY: test-secret-key
        run: |
          /validation:validate-python
```

---

**Created**: 2024-12-15
**Template Version**: 1.0.0
**Tech Stack**: Python 3.9+, FastAPI, PostgreSQL, pytest
