#!/usr/bin/env bash
# Test execution script for jactus

set -e  # Exit on error

# Default values
COVERAGE=false
PARALLEL=false
MARKER=""
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--coverage)
            COVERAGE=true
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -m|--marker)
            MARKER="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --coverage    Run with coverage report"
            echo "  -p, --parallel    Run tests in parallel"
            echo "  -m, --marker      Run tests with specific marker (unit, integration, slow)"
            echo "  -v, --verbose     Verbose output"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest tests/"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$PARALLEL" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

if [ -n "$MARKER" ]; then
    PYTEST_CMD="$PYTEST_CMD -m $MARKER"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=jactus --cov-report=html --cov-report=term-missing"
fi

# Run tests
echo "Running tests with command: $PYTEST_CMD"
echo ""
eval $PYTEST_CMD

# Show coverage report location if generated
if [ "$COVERAGE" = true ]; then
    echo ""
    echo "Coverage report generated at: htmlcov/index.html"
fi
