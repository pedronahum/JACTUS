#!/usr/bin/env bash
# Documentation build script for jactus

set -e  # Exit on error

# Default values
CLEAN=false
SERVE=false
PORT=8000

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -s|--serve)
            SERVE=true
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --clean     Clean build directory before building"
            echo "  -s, --serve     Serve documentation after building"
            echo "  -p, --port      Port for serving documentation (default: 8000)"
            echo "  -h, --help      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Navigate to docs directory
cd docs

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning documentation build directory..."
    make clean
fi

# Build documentation
echo "Building documentation..."
make html

echo ""
echo "Documentation built successfully!"
echo "HTML documentation: docs/_build/html/index.html"

# Serve if requested
if [ "$SERVE" = true ]; then
    echo ""
    echo "Serving documentation on http://localhost:$PORT"
    echo "Press Ctrl+C to stop"
    python -m http.server --directory _build/html $PORT
fi
