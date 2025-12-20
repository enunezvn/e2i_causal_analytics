---
argument-hint: [custom-args]
description: Language-agnostic validation template - customize for your tech stack
---

# Validation Template

This is a **language-agnostic template** for creating comprehensive end-to-end validation commands. Use this as a reference to create validation workflows for your specific tech stack.

## Purpose

End-to-end validation ensures your application works correctly across all layers:
- Code quality (linting, type checking, formatting)
- Tests (unit, integration, end-to-end)
- Build process
- Runtime behavior
- External integrations

## Template Structure

All validation commands should follow this phase-based approach:

---

## Phase 1: Foundation Validation

Validate basic code quality and correctness before testing runtime behavior.

### 1.1 Code Quality Checks
```bash
# Replace with your linter
# Examples:
# - npm run lint              # ESLint
# - flake8 .                  # Python
# - golangci-lint run         # Go
# - rubocop                   # Ruby
# - cargo clippy              # Rust

<YOUR_LINTER_COMMAND>
```
**Expected:** No linting errors or warnings

### 1.2 Type Checking (if applicable)
```bash
# Replace with your type checker
# Examples:
# - npm run type-check        # TypeScript
# - mypy .                    # Python
# - go vet ./...              # Go
# - flow check                # Flow

<YOUR_TYPE_CHECKER_COMMAND>
```
**Expected:** No type errors

### 1.3 Code Formatting (optional but recommended)
```bash
# Replace with your formatter check
# Examples:
# - npm run format:check      # Prettier
# - black --check .           # Black (Python)
# - gofmt -l .                # Go
# - cargo fmt -- --check      # Rust

<YOUR_FORMATTER_CHECK_COMMAND>
```
**Expected:** No formatting issues

### 1.4 Unit Tests
```bash
# Replace with your test runner
# Examples:
# - npm test                  # Jest/Vitest/Mocha
# - pytest tests/unit/        # Python
# - go test ./...             # Go
# - cargo test                # Rust
# - bundle exec rspec         # Ruby (RSpec)

<YOUR_UNIT_TEST_COMMAND>
```
**Expected:** All tests pass

### 1.5 Build
```bash
# Replace with your build command
# Examples:
# - npm run build             # Node.js/TS
# - python -m build           # Python
# - go build ./...            # Go
# - cargo build               # Rust
# - mvn clean package         # Java (Maven)
# - gradle build              # Java (Gradle)

<YOUR_BUILD_COMMAND>
```
**Expected:** Clean build with no errors

**If any step fails, STOP and report the issue immediately.**

---

## Phase 2: Environment Setup

Prepare the environment for runtime testing.

### 2.1 Configuration Validation
```bash
# Verify required configuration files exist
# Examples:
# - .env file with required variables
# - config.json or config.yaml
# - Database connection strings

if [ ! -f .env ]; then
  echo "ERROR: .env file not found"
  exit 1
fi

# Load environment variables
source .env

# Validate required variables
REQUIRED_VARS="<LIST_YOUR_REQUIRED_ENV_VARS>"
for VAR in $REQUIRED_VARS; do
  if [ -z "${!VAR}" ]; then
    echo "ERROR: Required environment variable $VAR is not set"
    exit 1
  fi
done

echo "✅ Configuration validated"
```

### 2.2 Dependency Check
```bash
# Verify required tools/services are available
# Examples:
# - Database (PostgreSQL, MySQL, MongoDB)
# - Cache (Redis, Memcached)
# - Message queue (RabbitMQ, Kafka)
# - External APIs

# Check database connection
<DATABASE_CONNECTION_CHECK_COMMAND>

# Check other dependencies
<DEPENDENCY_CHECK_COMMANDS>

echo "✅ Dependencies available"
```

### 2.3 Database Migrations (if applicable)
```bash
# Apply database migrations
# Examples:
# - npm run migrate           # Node.js (Prisma, TypeORM)
# - alembic upgrade head      # Python (SQLAlchemy)
# - go run migrations/*.go    # Go (custom)
# - flyway migrate            # Java (Flyway)

<YOUR_MIGRATION_COMMAND>

echo "✅ Database schema up to date"
```

### 2.4 Seed Data (optional)
```bash
# Load test data
# Examples:
# - npm run seed              # Node.js
# - python scripts/seed.py    # Python
# - go run scripts/seed.go    # Go

<YOUR_SEED_COMMAND>

echo "✅ Test data loaded"
```

---

## Phase 3: Integration Tests

Test component interactions and external integrations.

### 3.1 Database Integration Tests
```bash
# Test database operations
# Examples:
# - pytest tests/integration/test_db.py
# - npm run test:integration
# - go test ./tests/integration/...

<YOUR_INTEGRATION_TEST_COMMAND>
```
**Expected:** All integration tests pass

### 3.2 API Integration Tests (if applicable)
```bash
# Test API endpoints
# Examples:
# - npm run test:api
# - pytest tests/api/
# - go test ./tests/api/...
# - newman run postman_collection.json  # Postman/Newman

<YOUR_API_TEST_COMMAND>
```
**Expected:** All API tests pass

### 3.3 External Service Tests (if applicable)
```bash
# Test integrations with external services
# Examples:
# - Payment gateways (Stripe, PayPal)
# - Email services (SendGrid, AWS SES)
# - Cloud storage (S3, GCS, Azure Blob)
# - Third-party APIs

<YOUR_EXTERNAL_SERVICE_TEST_COMMAND>
```
**Expected:** External integrations working

---

## Phase 4: Runtime Validation

Test the application in a running state.

### 4.1 Start Application
```bash
# Start the application
# Examples:
# - npm start &               # Node.js
# - python main.py &          # Python
# - go run main.go &          # Go
# - docker-compose up -d      # Docker
# - ./gradlew bootRun &       # Spring Boot

<YOUR_START_COMMAND> &
APP_PID=$!

# Wait for application to start
sleep <STARTUP_TIME_SECONDS>
```

### 4.2 Health Checks
```bash
# Verify application is healthy
# Examples:
# - curl http://localhost:PORT/health
# - wget -qO- http://localhost:PORT/healthz
# - nc -zv localhost PORT

HEALTH_CHECK_URL="<YOUR_HEALTH_ENDPOINT>"
curl -f $HEALTH_CHECK_URL || { echo "Health check failed"; exit 1; }

echo "✅ Application is healthy"
```

### 4.3 Smoke Tests
```bash
# Run basic functionality tests
# Examples:
# - curl http://localhost:PORT/api/version
# - Test login endpoint
# - Test main user flow

<YOUR_SMOKE_TEST_COMMANDS>

echo "✅ Smoke tests passed"
```

---

## Phase 5: End-to-End Tests

Test complete user workflows.

### 5.1 E2E Test Suite
```bash
# Run end-to-end tests
# Examples:
# - npm run test:e2e          # Playwright/Cypress/Puppeteer
# - pytest tests/e2e/         # Python (Selenium)
# - go test ./tests/e2e/...   # Go

<YOUR_E2E_TEST_COMMAND>
```
**Expected:** All E2E tests pass

### 5.2 Performance Tests (optional)
```bash
# Run performance/load tests
# Examples:
# - k6 run load-test.js       # k6
# - ab -n 1000 -c 10 URL      # Apache Bench
# - wrk -t12 -c400 -d30s URL  # wrk

<YOUR_PERFORMANCE_TEST_COMMAND>
```
**Expected:** Performance within acceptable limits

---

## Phase 6: Cleanup

Clean up test environment and resources.

### 6.1 Stop Application
```bash
# Stop the application
if [ -n "$APP_PID" ]; then
  kill $APP_PID
  wait $APP_PID 2>/dev/null
fi

# Or stop Docker containers
# docker-compose down

<YOUR_STOP_COMMAND>

echo "✅ Application stopped"
```

### 6.2 Clean Test Data
```bash
# Remove test data
# Examples:
# - npm run db:reset          # Reset database
# - rm -rf test-data/         # Remove test files

<YOUR_CLEANUP_COMMAND>

echo "✅ Test environment cleaned"
```

---

## Phase 7: Summary Report

Provide a comprehensive validation summary.

```bash
echo ""
echo "======================================="
echo "VALIDATION SUMMARY"
echo "======================================="
echo ""
echo "FOUNDATION:"
echo "  Code Quality: ✅/❌"
echo "  Type Checking: ✅/❌"
echo "  Code Formatting: ✅/❌"
echo "  Unit Tests: ✅/❌"
echo "  Build: ✅/❌"
echo ""
echo "INTEGRATION:"
echo "  Database Tests: ✅/❌"
echo "  API Tests: ✅/❌"
echo "  External Services: ✅/❌"
echo ""
echo "RUNTIME:"
echo "  Application Startup: ✅/❌"
echo "  Health Checks: ✅/❌"
echo "  Smoke Tests: ✅/❌"
echo ""
echo "END-TO-END:"
echo "  E2E Tests: ✅/❌"
echo "  Performance Tests: ✅/❌"
echo ""
echo "OVERALL: ✅ PASS / ❌ FAIL"
echo "======================================="
```

---

## Customization Guide

### For Python Projects
See `.claude/commands/validation/validate-python.md`

### For TypeScript/Node.js Projects
See `.claude/commands/validation/validate-typescript.md`

### For Go Projects
See `.claude/commands/validation/validate-go.md`

### Creating Your Own

1. **Copy this template** to `validate.md` in your project
2. **Replace placeholders** with your actual commands:
   - `<YOUR_LINTER_COMMAND>` → Your linter
   - `<YOUR_TYPE_CHECKER_COMMAND>` → Your type checker
   - `<YOUR_UNIT_TEST_COMMAND>` → Your test runner
   - `<YOUR_BUILD_COMMAND>` → Your build command
   - etc.
3. **Remove phases** that don't apply to your project
4. **Add phases** for project-specific requirements
5. **Adjust timeouts** based on your application's startup time
6. **Update health check URLs** to match your endpoints

### Best Practices

1. **Fail fast**: Stop validation on first failure
2. **Clear output**: Use ✅/❌ symbols for status
3. **Detailed errors**: Show relevant logs on failure
4. **Idempotent**: Can be run multiple times safely
5. **Environment-aware**: Support different environments (dev, staging, prod)
6. **Documented**: Comment complex steps
7. **Maintainable**: Keep commands simple and readable

### Common Patterns

#### Environment Variable Validation
```bash
REQUIRED_VARS="DATABASE_URL API_KEY SECRET_KEY"
for VAR in $REQUIRED_VARS; do
  if [ -z "${!VAR}" ]; then
    echo "ERROR: $VAR not set"
    exit 1
  fi
done
```

#### Retry Logic for Flaky Tests
```bash
MAX_RETRIES=3
for i in $(seq 1 $MAX_RETRIES); do
  npm run test:flaky && break
  [ $i -lt $MAX_RETRIES ] && echo "Retry $i/$MAX_RETRIES..." && sleep 5
done
```

#### Parallel Test Execution
```bash
# Run tests in parallel for faster validation
npm run test:unit &
UNIT_PID=$!

npm run test:integration &
INTEGRATION_PID=$!

wait $UNIT_PID $INTEGRATION_PID
```

#### Conditional Validation
```bash
# Skip E2E tests in CI if not needed
if [ "$SKIP_E2E" != "true" ]; then
  npm run test:e2e
fi
```

---

## Integration with Development Workflow

### Pre-commit Validation
```bash
# Run quick validation before commit
/validation:validate --fast  # Skip slow tests
```

### Pre-push Validation
```bash
# Run full validation before push
/validation:validate
```

### CI/CD Pipeline
```bash
# Run in continuous integration
/validation:validate --ci  # With CI-specific flags
```

### Pre-release Validation
```bash
# Comprehensive validation before release
/validation:validate --full --performance
```

---

## Troubleshooting

### Common Issues

**Problem**: Tests timeout
**Solution**: Increase sleep times or add retry logic

**Problem**: Flaky tests
**Solution**: Add retry logic or fix test isolation

**Problem**: Environment variables missing
**Solution**: Check .env file and add validation

**Problem**: Database connection fails
**Solution**: Verify database is running and credentials are correct

**Problem**: Port already in use
**Solution**: Stop existing processes or use different port

---

## Examples

For complete, working examples in different languages:
- Python/FastAPI: `.claude/commands/validation/validate-python.md`
- TypeScript/Node.js: `.claude/commands/validation/validate-typescript.md`
- Go: `.claude/commands/validation/validate-go.md`

For customization guidance:
- See `CUSTOMIZATION_GUIDE.md` in the project root

---

**Created**: 2024-12-15
**Template Version**: 1.0.0
