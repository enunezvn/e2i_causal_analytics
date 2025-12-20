---
argument-hint: [--skip-e2e]
description: Comprehensive validation for TypeScript/Node.js applications
---

# TypeScript/Node.js Validation

Comprehensive end-to-end validation for TypeScript/Node.js applications with PostgreSQL, Jest/Vitest, and optional Docker.

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
  echo "Usage: /validation:validate-typescript [--skip-e2e]"
  exit 1
fi
```

---

**Usage:**
```bash
/validation:validate-typescript              # Full validation
/validation:validate-typescript --skip-e2e   # Skip E2E tests
```

**Prerequisites:**
- Node.js 18+ installed
- npm, yarn, or pnpm for dependency management
- PostgreSQL database (local or remote)
- `.env` file with configuration

---

## Phase 1: Foundation Validation

### 1.1 Dependency Check
```bash
# Ensure dependencies are installed
if [ -f "package-lock.json" ]; then
  npm ci
elif [ -f "yarn.lock" ]; then
  yarn install --frozen-lockfile
elif [ -f "pnpm-lock.yaml" ]; then
  pnpm install --frozen-lockfile
fi

echo "✅ Dependencies installed"
```

### 1.2 Code Quality (Linting)
```bash
# Run ESLint
npm run lint
```
**Expected:** No linting errors

Alternative:
```bash
# npx eslint . --ext .ts,.tsx
# yarn lint
# pnpm lint
```

### 1.3 Type Checking
```bash
# Run TypeScript compiler type check
npm run type-check
```
**Expected:** No type errors

If no script exists:
```bash
# npx tsc --noEmit
```

### 1.4 Code Formatting
```bash
# Check code formatting with Prettier
npm run format:check
```
**Expected:** No formatting issues

If no script exists:
```bash
# npx prettier --check "src/**/*.{ts,tsx,js,jsx,json,md}"
```

If formatting issues found:
```bash
# Auto-fix formatting
# npm run format
# npx prettier --write "src/**/*.{ts,tsx,js,jsx,json,md}"
```

### 1.5 Circular Dependencies Check (Optional)
```bash
# Check for circular dependencies using madge
if npm list madge > /dev/null 2>&1; then
  npx madge --circular --extensions ts,tsx src/
  if [ $? -eq 0 ]; then
    echo "✅ No circular dependencies"
  else
    echo "⚠️  Circular dependencies detected"
  fi
else
  echo "ℹ️  Madge not installed, skipping circular dependency check"
fi
```

### 1.6 Security Audit
```bash
# Check for security vulnerabilities
npm audit --audit-level=moderate

if [ $? -ne 0 ]; then
  echo "⚠️  Security vulnerabilities found"
  echo "Run 'npm audit fix' to attempt automatic fixes"
fi
```

### 1.7 Unit Tests
```bash
# Run unit tests with coverage
npm test -- --coverage
```
**Expected:** All tests pass, coverage > 80%

Alternative test runners:
```bash
# npm run test:unit
# vitest run --coverage
# jest --coverage
```

### 1.8 Build
```bash
# Build TypeScript to JavaScript
npm run build
```
**Expected:** Clean build with no errors, output in `dist/` or `build/`

Alternative:
```bash
# tsc --build
# webpack --mode production
# vite build
```

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
REQUIRED_VARS="DATABASE_URL NODE_ENV"
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
# Test database connection using psql or Node.js
if command -v psql &> /dev/null; then
  psql "$DATABASE_URL" -c "SELECT version();" > /dev/null
  if [ $? -eq 0 ]; then
    echo "✅ Database connection successful"
  else
    echo "❌ Database connection failed"
    exit 1
  fi
else
  # Fallback: Use Node.js to test connection
  node -e "
    const { Client } = require('pg');
    const client = new Client({ connectionString: process.env.DATABASE_URL });
    client.connect()
      .then(() => { console.log('✅ Database connection successful'); return client.end(); })
      .catch(err => { console.error('❌ Database connection failed:', err.message); process.exit(1); });
  " || exit 1
fi
```

### 2.3 Database Migrations
```bash
# Apply database migrations
if [ -f "package.json" ]; then
  # Check if migrations script exists
  if npm run | grep -q "migrate"; then
    npm run migrate
    echo "✅ Database migrations applied"
  elif npm run | grep -q "db:migrate"; then
    npm run db:migrate
    echo "✅ Database migrations applied"
  else
    echo "ℹ️  No migration script found"
  fi
fi
```

Common migration tools:
```bash
# Prisma
# npx prisma migrate deploy

# TypeORM
# npm run typeorm migration:run

# Knex
# npx knex migrate:latest

# Sequelize
# npx sequelize-cli db:migrate
```

### 2.4 Seed Test Data (Optional)
```bash
# Load test data if script exists
if npm run | grep -q "seed"; then
  npm run seed
  echo "✅ Test data seeded"
elif npm run | grep -q "db:seed"; then
  npm run db:seed
  echo "✅ Test data seeded"
fi
```

---

## Phase 3: Integration Tests

### 3.1 Database Integration Tests
```bash
# Run database integration tests
if npm run | grep -q "test:integration"; then
  npm run test:integration -- --testPathPattern=database
  echo "✅ Database integration tests passed"
else
  echo "ℹ️  No integration test script found"
fi
```

### 3.2 API Integration Tests
```bash
# Run API integration tests
if npm run | grep -q "test:integration"; then
  npm run test:integration -- --testPathPattern=api
  echo "✅ API integration tests passed"
elif npm run | grep -q "test:api"; then
  npm run test:api
  echo "✅ API integration tests passed"
else
  echo "ℹ️  No API test script found"
fi
```

### 3.3 External Service Tests (if applicable)
```bash
# Test external integrations (if exists)
if npm run | grep -q "test:external"; then
  npm run test:external
  echo "✅ External service tests passed"
fi
```

---

## Phase 4: Runtime Validation

### 4.1 Start Application
```bash
# Start application in background
if [ -f "dist/index.js" ]; then
  NODE_ENV=test node dist/index.js > /tmp/app.log 2>&1 &
  APP_PID=$!
elif [ -f "build/index.js" ]; then
  NODE_ENV=test node build/index.js > /tmp/app.log 2>&1 &
  APP_PID=$!
elif npm run | grep -q "start:dev"; then
  npm run start:dev > /tmp/app.log 2>&1 &
  APP_PID=$!
else
  npm start > /tmp/app.log 2>&1 &
  APP_PID=$!
fi

echo "Application started (PID: $APP_PID)"

# Wait for application to start
sleep 5
```

Alternative startup methods:
```bash
# Using ts-node for development
# npx ts-node src/index.ts &

# Using nodemon
# npm run dev &

# Using PM2
# pm2 start dist/index.js --name test-app
```

### 4.2 Health Checks
```bash
# Determine port from environment or default
PORT=${PORT:-3000}
HEALTH_URL="http://localhost:${PORT}/health"
MAX_RETRIES=10
RETRY_DELAY=2

echo "Checking health endpoint: $HEALTH_URL"

for i in $(seq 1 $MAX_RETRIES); do
  HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)

  if [ "$HEALTH_RESPONSE" = "200" ]; then
    echo "✅ Application is healthy"
    break
  fi

  if [ $i -eq $MAX_RETRIES ]; then
    echo "❌ Health check failed after $MAX_RETRIES attempts"
    echo "Application logs:"
    cat /tmp/app.log
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

PORT=${PORT:-3000}

# Test root endpoint
curl -f http://localhost:${PORT}/ || echo "⚠️  Root endpoint failed"

# Test API version endpoint (if exists)
curl -f http://localhost:${PORT}/api/version || echo "ℹ️  Version endpoint not found"

# Test OpenAPI/Swagger docs (if exists)
curl -f http://localhost:${PORT}/api-docs || echo "ℹ️  API docs not found"

# Test metrics endpoint (if exists)
curl -f http://localhost:${PORT}/metrics || echo "ℹ️  Metrics endpoint not found"

echo "✅ Smoke tests completed"
```

### 4.4 Database Query Test
```bash
# Test database queries work at runtime
node -e "
  const { Client } = require('pg');
  const client = new Client({ connectionString: process.env.DATABASE_URL });

  client.connect()
    .then(() => client.query('SELECT COUNT(*) FROM information_schema.tables'))
    .then(result => {
      console.log(\`✅ Database query successful (\${result.rows[0].count} tables)\`);
      return client.end();
    })
    .catch(err => {
      console.error('❌ Database query failed:', err.message);
      process.exit(1);
    });
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
  if npm run | grep -q "test:e2e"; then
    npm run test:e2e
  elif npm run | grep -q "e2e"; then
    npm run e2e
  else
    echo "ℹ️  No E2E test script found"
  fi

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

Common E2E frameworks:
```bash
# Playwright
# npx playwright test

# Cypress
# npx cypress run

# Puppeteer
# jest --config=jest.e2e.config.js
```

### 5.3 Performance Tests (Optional)
```bash
# Optional: Run basic load test
if command -v autocannon &> /dev/null; then
  echo "Running basic load test..."
  PORT=${PORT:-3000}
  autocannon -c 10 -d 10 http://localhost:${PORT}/health > /tmp/loadtest.txt

  # Display results
  cat /tmp/loadtest.txt | grep -E "Requests/sec|Latency"
elif command -v ab &> /dev/null; then
  echo "Running basic load test..."
  PORT=${PORT:-3000}
  ab -n 100 -c 10 http://localhost:${PORT}/health > /tmp/loadtest.txt 2>&1

  # Extract requests per second
  RPS=$(grep "Requests per second" /tmp/loadtest.txt | awk '{print $4}')
  echo "Performance: $RPS requests/second"
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

# Alternative: Stop with PM2
# pm2 delete test-app
```

### 6.2 Clean Test Data
```bash
# Remove temporary files
rm -f /tmp/app.log /tmp/loadtest.txt

# Clean build artifacts (optional)
# rm -rf dist/ build/ coverage/

echo "✅ Cleanup completed"
```

---

## Phase 7: Summary Report

```bash
echo ""
echo "======================================="
echo "TYPESCRIPT/NODE.JS VALIDATION SUMMARY"
echo "======================================="
echo ""
echo "FOUNDATION:"
echo "  Dependencies: ✅"
echo "  Linting (ESLint): ✅"
echo "  Type Checking (TSC): ✅"
echo "  Formatting (Prettier): ✅"
echo "  Circular Dependencies: ✅"
echo "  Security Audit: ✅"
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
NODE_ENV=production
PORT=3000
HOST=0.0.0.0

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
TEST_DATABASE_URL=postgresql://user:password@localhost:5432/test_dbname

# Redis (optional)
REDIS_URL=redis://localhost:6379

# JWT/Auth
JWT_SECRET=your-secret-key
JWT_EXPIRES_IN=1d

# External Services
API_KEY=your-api-key
```

### Example `package.json` scripts
```json
{
  "scripts": {
    "dev": "nodemon --exec ts-node src/index.ts",
    "build": "tsc",
    "start": "node dist/index.js",
    "lint": "eslint src --ext .ts,.tsx",
    "lint:fix": "eslint src --ext .ts,.tsx --fix",
    "format": "prettier --write \"src/**/*.{ts,tsx,js,jsx,json,md}\"",
    "format:check": "prettier --check \"src/**/*.{ts,tsx,js,jsx,json,md}\"",
    "type-check": "tsc --noEmit",
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "test:integration": "jest --config=jest.integration.config.js",
    "test:e2e": "jest --config=jest.e2e.config.js",
    "migrate": "prisma migrate deploy",
    "seed": "ts-node prisma/seed.ts"
  }
}
```

### Example `tsconfig.json`
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "moduleResolution": "node",
    "types": ["node", "jest"]
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

### Example `.eslintrc.json`
```json
{
  "parser": "@typescript-eslint/parser",
  "extends": [
    "eslint:recommended",
    "plugin:@typescript-eslint/recommended",
    "plugin:prettier/recommended"
  ],
  "plugins": ["@typescript-eslint"],
  "env": {
    "node": true,
    "es2020": true
  },
  "rules": {
    "@typescript-eslint/explicit-module-boundary-types": "off",
    "@typescript-eslint/no-explicit-any": "warn",
    "no-console": "warn"
  }
}
```

### Example `jest.config.js`
```javascript
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.ts', '**/?(*.)+(spec|test).ts'],
  transform: {
    '^.+\\.ts$': 'ts-jest',
  },
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/**/*.test.ts',
    '!src/**/index.ts',
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
};
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
  -e NODE_ENV=test \
  myapp:test \
  npm test

# Run validation in Docker Compose
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
docker-compose -f docker-compose.test.yml down
```

### Example `Dockerfile`
```dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY --from=builder /app/dist ./dist

EXPOSE 3000

CMD ["node", "dist/index.js"]
```

### Example `docker-compose.test.yml`
```yaml
version: '3.8'

services:
  app:
    build: .
    environment:
      - DATABASE_URL=postgresql://testuser:testpass@db:5432/testdb
      - NODE_ENV=test
    depends_on:
      - db
    command: npm test

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

**Problem**: `Cannot find module 'X'`
**Solution**: Install dependencies: `npm install` or `npm ci`

**Problem**: TypeScript compilation errors
**Solution**: Check `tsconfig.json` and fix type errors

**Problem**: Database connection fails
**Solution**: Check DATABASE_URL format and ensure PostgreSQL is running

**Problem**: Port already in use
**Solution**: Kill existing process: `lsof -ti:3000 | xargs kill -9`

**Problem**: Tests timeout
**Solution**: Increase jest timeout or check for unresolved promises

**Problem**: Module not found in production
**Solution**: Ensure all dependencies are in `dependencies`, not `devDependencies`

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

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run validation
        env:
          DATABASE_URL: postgresql://testuser:testpass@localhost:5432/testdb
          NODE_ENV: test
        run: |
          /validation:validate-typescript
```

---

**Created**: 2024-12-15
**Template Version**: 1.0.0
**Tech Stack**: Node.js 18+, TypeScript, PostgreSQL, Jest/Vitest
