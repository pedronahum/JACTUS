"""Tests for system tools."""

import pytest
from jactus_mcp.tools import system


def test_get_version_info():
    """Test getting version information."""
    result = system.get_version_info()

    assert "jactus_mcp_version" in result
    assert "jactus_version" in result
    assert "jactus_installed" in result
    assert "jactus_importable" in result
    assert "compatible" in result
    assert "python_version" in result

    # JACTUS should be installed in test environment
    assert result["jactus_installed"] is True
    assert result["jactus_importable"] is True
    assert result["jactus_version"] is not None


def test_health_check():
    """Test health check functionality."""
    result = system.health_check()

    assert "status" in result
    assert "message" in result
    assert "checks" in result
    assert "versions" in result
    assert "recommendations" in result

    # Should have these checks
    checks = result["checks"]
    assert "jactus_installed" in checks
    assert "jactus_importable" in checks
    assert "examples_found" in checks
    assert "docs_found" in checks
    assert "contracts_accessible" in checks
    assert "root_directory_found" in checks


def test_health_check_status_healthy():
    """Test that health check returns healthy status when everything is OK."""
    result = system.health_check()

    # In test environment, should be healthy
    assert result["status"] in ["healthy", "degraded"]

    # JACTUS should be accessible
    assert result["checks"]["jactus_installed"] is True
    assert result["checks"]["jactus_importable"] is True


def test_health_check_versions():
    """Test that health check includes version information."""
    result = system.health_check()

    versions = result["versions"]
    assert "mcp_server" in versions
    assert "jactus" in versions
    assert "python" in versions

    # JACTUS version should be present
    assert versions["jactus"] is not None
    assert versions["python"] is not None


def test_health_check_recommendations():
    """Test that health check provides recommendations."""
    result = system.health_check()

    assert "recommendations" in result
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) > 0


def test_version_info_python_version():
    """Test that Python version is correctly formatted."""
    result = system.get_version_info()

    python_version = result["python_version"]
    assert isinstance(python_version, str)
    # Should be in format: X.Y.Z
    parts = python_version.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
