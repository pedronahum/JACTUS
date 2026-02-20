"""Shared utilities for JACTUS MCP tools."""

from pathlib import Path


def get_jactus_root() -> Path:
    """Get the root directory of the JACTUS repository."""
    current = Path(__file__).parent
    for _ in range(10):  # Safety limit
        if (current / "src" / "jactus").exists():
            return current
        current = current.parent

    # Fallback: 5 levels up
    return Path(__file__).parent.parent.parent.parent.parent
