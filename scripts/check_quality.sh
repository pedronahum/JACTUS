#!/usr/bin/env bash
# Code quality check script for jactus

set -e  # Exit on error

echo "Running code quality checks for jactus..."
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track failures
FAILURES=0

# Function to run a check
run_check() {
    local name=$1
    local command=$2

    echo -e "${YELLOW}Running $name...${NC}"
    if eval $command; then
        echo -e "${GREEN}✓ $name passed${NC}"
        echo ""
    else
        echo -e "${RED}✗ $name failed${NC}"
        echo ""
        ((FAILURES++))
    fi
}

# Run checks
run_check "Black (code formatting)" "black --check src/ tests/ examples/"
run_check "Ruff (linting)" "ruff check src/ tests/"
run_check "Mypy (type checking)" "mypy src/jactus"

# Summary
echo "========================================"
if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}All quality checks passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAILURES check(s) failed${NC}"
    echo ""
    echo "To fix formatting issues, run:"
    echo "  make format"
    echo ""
    echo "To fix some linting issues automatically, run:"
    echo "  make lint-fix"
    exit 1
fi
