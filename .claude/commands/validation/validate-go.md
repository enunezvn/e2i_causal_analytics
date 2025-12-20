---
argument-hint: [--skip-e2e]
description: Comprehensive validation for Go applications
---

# Go Validation

Comprehensive end-to-end validation for Go applications with PostgreSQL, Go's testing framework, and optional Docker.

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
  echo "Usage: /validation:validate-go [--skip-e2e]"
  exit 1
fi
```

---

**Usage:**
```bash
/validation:validate-go              # Full validation
/validation:validate-go --skip-e2e   # Skip E2E tests
```

**Prerequisites:**
- Go 1.21+ installed
- PostgreSQL database (local or remote)
- `.env` file with configuration (optional)

---

## Phase 1: Foundation Validation

### 1.1 Dependency Check
```bash
# Download and verify dependencies
go mod download
go mod verify

echo "✅ Dependencies verified"
```

### 1.2 Code Formatting
```bash
# Check code formatting with gofmt
UNFORMATTED=$(gofmt -l .)

if [ -n "$UNFORMATTED" ]; then
  echo "❌ The following files are not formatted:"
  echo "$UNFORMATTED"
  echo ""
  echo "Run 'gofmt -w .' to fix"
  exit 1
else
  echo "✅ All files are properly formatted"
fi
```

### 1.3 Go Vet (Static Analysis)
```bash
# Run go vet for static analysis
go vet ./...

if [ $? -eq 0 ]; then
  echo "✅ Go vet passed"
else
  echo "❌ Go vet found issues"
  exit 1
fi
```

### 1.4 Linting (golangci-lint)
```bash
# Run golangci-lint for comprehensive linting
if command -v golangci-lint &> /dev/null; then
  golangci-lint run ./...

  if [ $? -eq 0 ]; then
    echo "✅ Linting passed"
  else
    echo "❌ Linting failed"
    exit 1
  fi
else
  echo "⚠️  golangci-lint not installed, skipping"
  echo "Install with: curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin"
fi
```

### 1.5 Security Checks (gosec)
```bash
# Run security checks with gosec
if command -v gosec &> /dev/null; then
  gosec ./...

  if [ $? -eq 0 ]; then
    echo "✅ Security checks passed"
  else
    echo "⚠️  Security issues detected"
  fi
else
  echo "ℹ️  gosec not installed, skipping security checks"
  echo "Install with: go install github.com/securego/gosec/v2/cmd/gosec@latest"
fi
```

### 1.6 Unit Tests
```bash
# Run unit tests with coverage
go test ./... -v -race -coverprofile=coverage.out -covermode=atomic

if [ $? -eq 0 ]; then
  echo "✅ All tests passed"

  # Display coverage summary
  go tool cover -func=coverage.out | grep total:

  # Generate HTML coverage report
  go tool cover -html=coverage.out -o coverage.html
  echo "📊 Coverage report generated: coverage.html"
else
  echo "❌ Tests failed"
  exit 1
fi
```

### 1.7 Build
```bash
# Build the application
go build -v -o ./bin/app ./cmd/...

if [ $? -eq 0 ]; then
  echo "✅ Build successful"
  ls -lh ./bin/app
else
  echo "❌ Build failed"
  exit 1
fi
```

Alternative build with version info:
```bash
# VERSION=$(git describe --tags --always --dirty)
# BUILD_TIME=$(date -u '+%Y-%m-%d_%H:%M:%S')
# go build -ldflags "-X main.Version=$VERSION -X main.BuildTime=$BUILD_TIME" -o ./bin/app ./cmd/...
```

**If any step fails, STOP and report the issue immediately.**

---

## Phase 2: Environment Setup

### 2.1 Load Configuration
```bash
# Load environment variables from .env if exists
if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
  echo "✅ Configuration loaded from .env"
else
  echo "⚠️  No .env file found, using environment variables"
fi

# Validate required variables
REQUIRED_VARS="DATABASE_URL"
for VAR in $REQUIRED_VARS; do
  if [ -z "${!VAR}" ]; then
    echo "ERROR: Required environment variable $VAR is not set"
    exit 1
  fi
done

echo "✅ Configuration validated"
```

### 2.2 Database Connection Check
```bash
# Test database connection using psql
if command -v psql &> /dev/null; then
  psql "$DATABASE_URL" -c "SELECT version();" > /dev/null
  if [ $? -eq 0 ]; then
    echo "✅ Database connection successful"
  else
    echo "❌ Database connection failed"
    exit 1
  fi
else
  echo "⚠️  psql not installed, skipping direct connection test"
  echo "Database connection will be tested during application startup"
fi
```

### 2.3 Database Migrations
```bash
# Apply database migrations
if [ -f "./bin/migrate" ] || [ -f "./scripts/migrate.sh" ]; then
  # Using custom migration script
  if [ -f "./scripts/migrate.sh" ]; then
    ./scripts/migrate.sh
  elif [ -f "./bin/migrate" ]; then
    ./bin/migrate up
  fi
  echo "✅ Database migrations applied"
elif command -v migrate &> /dev/null; then
  # Using golang-migrate
  migrate -database "$DATABASE_URL" -path ./migrations up
  echo "✅ Database migrations applied"
else
  echo "ℹ️  No migration tool found"
fi
```

Common migration tools:
```bash
# golang-migrate
# migrate -database "$DATABASE_URL" -path ./migrations up

# goose
# goose -dir ./migrations postgres "$DATABASE_URL" up

# Custom Go migration
# go run cmd/migrate/main.go up
```

### 2.4 Seed Test Data (Optional)
```bash
# Load test data if script exists
if [ -f "./scripts/seed.sh" ]; then
  ./scripts/seed.sh
  echo "✅ Test data seeded"
elif [ -f "./cmd/seed/main.go" ]; then
  go run ./cmd/seed/main.go
  echo "✅ Test data seeded"
fi
```

---

## Phase 3: Integration Tests

### 3.1 Integration Tests
```bash
# Run integration tests (tagged with integration build tag)
go test -v -tags=integration ./...

if [ $? -eq 0 ]; then
  echo "✅ Integration tests passed"
else
  echo "❌ Integration tests failed"
  exit 1
fi
```

Alternative:
```bash
# Run tests in specific directory
# go test -v ./tests/integration/...

# Run with short flag to skip integration tests normally
# go test -v -short ./...
```

### 3.2 Database Tests
```bash
# Run database-specific integration tests
if [ -d "./tests/database" ]; then
  go test -v ./tests/database/...

  if [ $? -eq 0 ]; then
    echo "✅ Database tests passed"
  else
    echo "❌ Database tests failed"
    exit 1
  fi
fi
```

---

## Phase 4: Runtime Validation

### 4.1 Start Application
```bash
# Start application in background
if [ -f "./bin/app" ]; then
  ./bin/app > /tmp/go-app.log 2>&1 &
  APP_PID=$!
else
  echo "ERROR: Application binary not found. Run build first."
  exit 1
fi

echo "Application started (PID: $APP_PID)"

# Wait for application to start
sleep 3
```

Alternative startup:
```bash
# Run from source
# go run ./cmd/server/main.go > /tmp/go-app.log 2>&1 &
# APP_PID=$!
```

### 4.2 Health Checks
```bash
# Determine port from environment or default
PORT=${PORT:-8080}
HEALTH_URL="http://localhost:${PORT}/health"
MAX_RETRIES=10
RETRY_DELAY=2

echo "Checking health endpoint: $HEALTH_URL"

for i in $(seq 1 $MAX_RETRIES); do
  HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL 2>/dev/null)

  if [ "$HEALTH_RESPONSE" = "200" ]; then
    echo "✅ Application is healthy"
    break
  fi

  if [ $i -eq $MAX_RETRIES ]; then
    echo "❌ Health check failed after $MAX_RETRIES attempts"
    echo "Application logs:"
    cat /tmp/go-app.log
    kill $APP_PID 2>/dev/null
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

PORT=${PORT:-8080}

# Test root endpoint
curl -f http://localhost:${PORT}/ || echo "ℹ️  Root endpoint not available"

# Test version endpoint (common in Go apps)
curl -f http://localhost:${PORT}/version || echo "ℹ️  Version endpoint not found"

# Test metrics endpoint (Prometheus)
curl -f http://localhost:${PORT}/metrics || echo "ℹ️  Metrics endpoint not found"

# Test readiness endpoint
curl -f http://localhost:${PORT}/ready || echo "ℹ️  Readiness endpoint not found"

echo "✅ Smoke tests completed"
```

### 4.4 Database Query Test
```bash
# Test database connection from Go application
go run -tags=dbtest ./scripts/test-db-connection.go 2>&1

if [ $? -eq 0 ]; then
  echo "✅ Database query successful"
else
  echo "❌ Database query failed"
  kill $APP_PID 2>/dev/null
  exit 1
fi
```

Simple test script example (`scripts/test-db-connection.go`):
```go
//go:build dbtest

package main

import (
    "database/sql"
    "fmt"
    "os"
    _ "github.com/lib/pq"
)

func main() {
    db, err := sql.Open("postgres", os.Getenv("DATABASE_URL"))
    if err != nil {
        fmt.Printf("Failed to connect: %v\n", err)
        os.Exit(1)
    }
    defer db.Close()

    var count int
    err = db.QueryRow("SELECT COUNT(*) FROM information_schema.tables").Scan(&count)
    if err != nil {
        fmt.Printf("Query failed: %v\n", err)
        os.Exit(1)
    }

    fmt.Printf("✅ Database query successful (%d tables)\n", count)
}
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
  if [ -d "./tests/e2e" ]; then
    go test -v -tags=e2e ./tests/e2e/...

    if [ $? -eq 0 ]; then
      echo "✅ E2E tests passed"
    else
      echo "❌ E2E tests failed"
      kill $APP_PID 2>/dev/null
      exit 1
    fi
  else
    echo "ℹ️  No E2E tests found"
  fi
else
  echo "ℹ️  E2E tests skipped"
fi
```

### 5.3 Performance Tests (Optional)
```bash
# Optional: Run benchmark tests
if [ -d "./benchmarks" ]; then
  echo "Running benchmarks..."
  go test -bench=. -benchmem ./benchmarks/... | tee /tmp/benchmarks.txt

  echo "📊 Benchmark results saved to /tmp/benchmarks.txt"
fi

# Load testing with hey (if installed)
if command -v hey &> /dev/null; then
  echo "Running load test..."
  PORT=${PORT:-8080}
  hey -n 1000 -c 10 http://localhost:${PORT}/health > /tmp/loadtest.txt

  # Display results
  cat /tmp/loadtest.txt | grep -E "Requests/sec|Latency"
fi
```

---

## Phase 6: Cleanup

### 6.1 Stop Application
```bash
# Stop application
if [ -n "$APP_PID" ]; then
  echo "Stopping application (PID: $APP_PID)..."
  kill $APP_PID
  wait $APP_PID 2>/dev/null
  echo "✅ Application stopped"
fi
```

### 6.2 Clean Test Data
```bash
# Remove temporary files
rm -f /tmp/go-app.log /tmp/benchmarks.txt /tmp/loadtest.txt

# Clean build artifacts (optional)
# rm -rf ./bin/ coverage.out coverage.html

echo "✅ Cleanup completed"
```

---

## Phase 7: Summary Report

```bash
echo ""
echo "======================================="
echo "GO VALIDATION SUMMARY"
echo "======================================="
echo ""
echo "FOUNDATION:"
echo "  Dependencies: ✅"
echo "  Formatting (gofmt): ✅"
echo "  Static Analysis (go vet): ✅"
echo "  Linting (golangci-lint): ✅"
echo "  Security (gosec): ✅"
echo "  Unit Tests: ✅"
echo "  Build: ✅"
echo ""
echo "ENVIRONMENT:"
echo "  Configuration: ✅"
echo "  Database Connection: ✅"
echo "  Migrations: ✅"
echo ""
echo "INTEGRATION:"
echo "  Integration Tests: ✅"
echo "  Database Tests: ✅"
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
PORT=8080
ENVIRONMENT=production

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
TEST_DATABASE_URL=postgresql://user:password@localhost:5432/test_dbname

# Logging
LOG_LEVEL=info

# External Services
API_KEY=your-api-key
```

### Example `go.mod`
```go
module github.com/yourorg/yourapp

go 1.21

require (
    github.com/gorilla/mux v1.8.0
    github.com/lib/pq v1.10.9
    github.com/joho/godotenv v1.5.1
)
```

### Example `.golangci.yml`
```yaml
linters:
  enable:
    - gofmt
    - govet
    - errcheck
    - staticcheck
    - unused
    - gosimple
    - ineffassign
    - typecheck
    - gosec
    - gocritic
    - revive

linters-settings:
  govet:
    check-shadowing: true
  gocyclo:
    min-complexity: 15
  dupl:
    threshold: 100

issues:
  exclude-use-default: false
  max-issues-per-linter: 0
  max-same-issues: 0

run:
  timeout: 5m
  tests: true
```

### Example Project Structure
```
.
├── cmd/
│   ├── server/
│   │   └── main.go          # Application entry point
│   ├── migrate/
│   │   └── main.go          # Migration tool
│   └── seed/
│       └── main.go          # Seed data
├── internal/
│   ├── api/                 # HTTP handlers
│   ├── service/             # Business logic
│   ├── repository/          # Data access
│   └── models/              # Data models
├── pkg/                     # Public packages
├── migrations/              # Database migrations
├── tests/
│   ├── integration/
│   ├── e2e/
│   └── database/
├── scripts/                 # Utility scripts
├── .env                     # Environment variables
├── .golangci.yml            # Linter configuration
├── go.mod
├── go.sum
└── Makefile
```

### Example `Makefile`
```makefile
.PHONY: build test lint fmt clean

build:
	go build -v -o ./bin/app ./cmd/server

test:
	go test -v -race -coverprofile=coverage.out ./...

test-integration:
	go test -v -tags=integration ./...

lint:
	golangci-lint run ./...

fmt:
	gofmt -w .
	goimports -w .

vet:
	go vet ./...

clean:
	rm -rf ./bin coverage.out coverage.html

migrate-up:
	migrate -database "$(DATABASE_URL)" -path ./migrations up

migrate-down:
	migrate -database "$(DATABASE_URL)" -path ./migrations down 1

run:
	go run ./cmd/server/main.go

validate:
	/validation:validate-go
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
  myapp:test \
  go test ./... -v

# Run validation in Docker Compose
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
docker-compose -f docker-compose.test.yml down
```

### Example `Dockerfile`
```dockerfile
# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o /app/bin/app ./cmd/server

# Runtime stage
FROM alpine:latest

RUN apk --no-cache add ca-certificates

WORKDIR /root/

COPY --from=builder /app/bin/app .

EXPOSE 8080

CMD ["./app"]
```

### Example `docker-compose.test.yml`
```yaml
version: '3.8'

services:
  app:
    build: .
    environment:
      - DATABASE_URL=postgresql://testuser:testpass@db:5432/testdb
      - PORT=8080
    depends_on:
      - db
    command: go test ./... -v

  db:
    image: postgres:15-alpine
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

**Problem**: `cannot find package`
**Solution**: Run `go mod download` or `go mod tidy`

**Problem**: Database connection fails
**Solution**: Check DATABASE_URL format: `postgresql://user:password@host:port/dbname`

**Problem**: Tests fail with race detector
**Solution**: Fix race conditions or run without `-race` flag (not recommended)

**Problem**: Build fails with missing dependencies
**Solution**: Run `go mod tidy` to sync dependencies

**Problem**: Port already in use
**Solution**: Kill existing process: `lsof -ti:8080 | xargs kill -9`

**Problem**: golangci-lint too slow
**Solution**: Use `--fast` flag or disable slow linters

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
        image: postgres:15-alpine
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

      - name: Set up Go
        uses: actions/setup-go@v4
        with:
          go-version: '1.21'
          cache: true

      - name: Install golangci-lint
        run: |
          curl -sSfL https://raw.githubusercontent.com/golangci/golangci-lint/master/install.sh | sh -s -- -b $(go env GOPATH)/bin

      - name: Run validation
        env:
          DATABASE_URL: postgresql://testuser:testpass@localhost:5432/testdb
        run: |
          /validation:validate-go
```

---

**Created**: 2024-12-15
**Template Version**: 1.0.0
**Tech Stack**: Go 1.21+, PostgreSQL, Go testing framework
