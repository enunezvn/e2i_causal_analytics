#!/bin/bash
#
# Script to run the new unit tests for digital_twin and workers modules
# Created as part of test coverage improvement initiative
#

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}E2I Causal Analytics - New Unit Tests${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "CLAUDE.md" ]; then
    echo -e "${RED}Error: Must run from project root directory${NC}"
    exit 1
fi

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest not found. Please install:${NC}"
    echo "  pip install pytest pytest-asyncio pytest-xdist pytest-cov"
    exit 1
fi

# Test discovery
echo -e "${YELLOW}Test Discovery:${NC}"
echo "Digital Twin tests:"
find tests/unit/test_digital_twin -name "test_*.py" -exec basename {} \; | sort
echo ""
echo "Workers tests:"
find tests/unit/test_workers -name "test_*.py" -exec basename {} \; | sort
echo ""

# Count tests
echo -e "${YELLOW}Counting tests...${NC}"
DIGITAL_TWIN_TESTS=$(pytest tests/unit/test_digital_twin/ --collect-only -q 2>/dev/null | tail -1 || echo "0 tests")
WORKERS_TESTS=$(pytest tests/unit/test_workers/ --collect-only -q 2>/dev/null | tail -1 || echo "0 tests")
echo "Digital Twin: $DIGITAL_TWIN_TESTS"
echo "Workers: $WORKERS_TESTS"
echo ""

# Run tests based on argument
case "${1:-all}" in
    "digital_twin"|"dt")
        echo -e "${GREEN}Running Digital Twin tests...${NC}"
        pytest tests/unit/test_digital_twin/ -v --no-header -n 4
        ;;

    "workers"|"w")
        echo -e "${GREEN}Running Workers tests...${NC}"
        pytest tests/unit/test_workers/ -v --no-header -n 4
        ;;

    "coverage"|"cov")
        echo -e "${GREEN}Running tests with coverage...${NC}"
        pytest tests/unit/test_digital_twin/ tests/unit/test_workers/ \
            --cov=src/digital_twin \
            --cov=src/workers \
            --cov-report=html \
            --cov-report=term-missing \
            -n 4
        echo ""
        echo -e "${BLUE}Coverage report generated in: htmlcov/index.html${NC}"
        ;;

    "quick"|"q")
        echo -e "${GREEN}Running quick test (no coverage, 4 workers)...${NC}"
        pytest tests/unit/test_digital_twin/ tests/unit/test_workers/ -v --no-header -n 4 -x
        ;;

    "sequential"|"seq")
        echo -e "${GREEN}Running tests sequentially (for debugging)...${NC}"
        pytest tests/unit/test_digital_twin/ tests/unit/test_workers/ -v --no-header
        ;;

    "all"|*)
        echo -e "${GREEN}Running all new tests...${NC}"
        pytest tests/unit/test_digital_twin/ tests/unit/test_workers/ -v --no-header -n 4
        ;;
esac

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed${NC}"
fi

echo ""
echo -e "${BLUE}Usage:${NC}"
echo "  $0 [all|digital_twin|workers|coverage|quick|sequential]"
echo ""
echo "  all (default)  - Run all new tests"
echo "  digital_twin   - Run only digital_twin tests"
echo "  workers        - Run only workers tests"
echo "  coverage       - Run with coverage report"
echo "  quick          - Fast run, stop on first failure"
echo "  sequential     - Run without parallelization (for debugging)"

exit $EXIT_CODE
