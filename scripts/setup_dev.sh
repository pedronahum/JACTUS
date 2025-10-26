#!/usr/bin/env bash
# Development environment setup script for jactus

set -e  # Exit on error

echo "Setting up jactus development environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

echo "Detected Python version: $PYTHON_VERSION"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "Error: Python 3.10 or higher is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install package in editable mode with all dependencies
echo "Installing jactus with development dependencies..."
pip install -e ".[dev,docs,viz]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

echo ""
echo "Development environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Available make commands:"
echo "  make help         - Show all available commands"
echo "  make test         - Run tests"
echo "  make format       - Format code with black"
echo "  make lint         - Run linter"
echo "  make typecheck    - Run type checker"
echo "  make docs         - Build documentation"
