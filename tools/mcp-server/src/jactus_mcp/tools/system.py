"""System information and health check tools."""

from pathlib import Path
from typing import Any

from ._utils import get_jactus_root


def get_version_info() -> dict[str, Any]:
    """Get JACTUS and MCP server version information.

    Returns:
        Dictionary with version information and compatibility status.
    """
    info = {
        "jactus_mcp_version": None,
        "jactus_version": None,
        "jactus_installed": False,
        "jactus_importable": False,
        "compatible": False,
        "python_version": None,
    }

    # Get MCP server version
    try:
        import jactus_mcp
        info["jactus_mcp_version"] = jactus_mcp.__version__
    except Exception:
        info["jactus_mcp_version"] = "unknown"

    # Get Python version
    import sys
    info["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Check JACTUS installation
    try:
        import jactus
        info["jactus_installed"] = True
        info["jactus_importable"] = True
        info["jactus_version"] = getattr(jactus, "__version__", "unknown")
    except ImportError:
        info["jactus_installed"] = False
        info["jactus_importable"] = False
        info["jactus_version"] = None

    # Check compatibility (simple version for now)
    if info["jactus_version"] and info["jactus_mcp_version"]:
        # Both installed
        info["compatible"] = True
    else:
        info["compatible"] = False

    return info


def health_check() -> dict[str, Any]:
    """Verify MCP server and JACTUS are working correctly.

    Returns:
        Dictionary with health status and diagnostic information.
    """
    checks = {
        "jactus_installed": False,
        "jactus_importable": False,
        "examples_found": False,
        "docs_found": False,
        "contracts_accessible": False,
        "root_directory_found": False,
    }

    errors = []

    # Check JACTUS installation
    try:
        import jactus
        checks["jactus_installed"] = True
        checks["jactus_importable"] = True

        # Check if we can access contracts
        from jactus.contracts import CONTRACT_REGISTRY
        checks["contracts_accessible"] = len(CONTRACT_REGISTRY) > 0

    except ImportError as e:
        errors.append(f"JACTUS not installed: {str(e)}")
    except Exception as e:
        errors.append(f"Error accessing JACTUS: {str(e)}")

    # Check directory structure
    try:
        root = get_jactus_root()
        checks["root_directory_found"] = (root / "src" / "jactus").exists()

        # Check for examples
        examples_dir = root / "examples"
        checks["examples_found"] = examples_dir.exists() and any(examples_dir.glob("*.py"))

        # Check for docs
        docs_dir = root / "docs"
        checks["docs_found"] = docs_dir.exists() and any(docs_dir.glob("*.md"))

    except Exception as e:
        errors.append(f"Error checking directory structure: {str(e)}")

    # Determine overall status
    if all(checks.values()):
        status = "healthy"
        message = "All systems operational"
    elif checks["jactus_installed"] and checks["jactus_importable"]:
        status = "degraded"
        message = "JACTUS installed but some resources not found"
    else:
        status = "unhealthy"
        message = "JACTUS not properly installed"

    # Get version info
    version_info = get_version_info()

    return {
        "status": status,
        "message": message,
        "checks": checks,
        "errors": errors if errors else None,
        "versions": {
            "mcp_server": version_info["jactus_mcp_version"],
            "jactus": version_info["jactus_version"],
            "python": version_info["python_version"],
        },
        "recommendations": _get_recommendations(checks, errors),
    }


def _get_recommendations(checks: dict[str, bool], errors: list[str]) -> list[str]:
    """Get recommendations based on health check results.

    Args:
        checks: Dictionary of check results
        errors: List of errors encountered

    Returns:
        List of recommended actions
    """
    recommendations = []

    if not checks["jactus_installed"]:
        recommendations.append("Install JACTUS: cd /path/to/JACTUS && pip install .")

    if not checks["root_directory_found"]:
        recommendations.append("Ensure MCP server is run from JACTUS repository root")

    if not checks["examples_found"]:
        recommendations.append("Examples directory not found - verify JACTUS installation")

    if not checks["docs_found"]:
        recommendations.append("Documentation directory not found - verify JACTUS installation")

    if not checks["contracts_accessible"]:
        recommendations.append("Cannot access contracts - check JACTUS installation")

    if not recommendations:
        recommendations.append("All systems operational - no actions needed")

    return recommendations
