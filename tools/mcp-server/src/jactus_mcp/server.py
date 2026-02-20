"""JACTUS MCP Server.

This MCP (Model Context Protocol) server provides AI assistants like Claude Code
with direct access to JACTUS capabilities, enabling accurate code generation,
validation, and discovery without hallucination.
"""

import functools
import json
import logging
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from .tools import contracts, examples, validation, documentation, system
from .tools._utils import get_jactus_root

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastMCP server
mcp = FastMCP("jactus-mcp")


# ---- Structured Logging Decorator (Item 14) ----


def log_tool_call(func):
    """Decorator that logs tool calls with timing."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        tool_name = func.__name__
        args_summary = {k: str(v)[:100] for k, v in kwargs.items()} if kwargs else {}
        logger.info(f"Tool call: {tool_name} args={args_summary}")
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"Tool {tool_name} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"Tool {tool_name} failed after {duration:.3f}s: {e}")
            raise

    return wrapper


# ---- Contract Discovery Tools ----


@mcp.tool()
@log_tool_call
def jactus_list_contracts() -> dict[str, Any]:
    """List all available ACTUS contract types in JACTUS."""
    return contracts.list_contracts()


@mcp.tool()
@log_tool_call
def jactus_get_contract_info(contract_type: str) -> dict[str, Any]:
    """Get detailed information about a specific contract type.

    Args:
        contract_type: ACTUS contract type code (e.g., PAM, LAM, SWPPV)
    """
    return contracts.get_contract_info(contract_type)


@mcp.tool()
@log_tool_call
def jactus_get_contract_schema(contract_type: str) -> dict[str, Any]:
    """Get required and optional parameters for a contract type.

    Args:
        contract_type: ACTUS contract type code (e.g., PAM, LAM, SWPPV)
    """
    return contracts.get_contract_schema(contract_type)


@mcp.tool()
@log_tool_call
def jactus_get_event_types() -> dict[str, Any]:
    """List all ACTUS event types."""
    return contracts.get_event_types()


# ---- Simulation Tool ----


@mcp.tool()
@log_tool_call
def jactus_simulate_contract(
    attributes: dict[str, Any],
    risk_factors: dict[str, float] | None = None,
    constant_value: float | None = None,
    include_states: bool = False,
) -> dict[str, Any]:
    """Simulate an ACTUS contract and return structured cash flow events.

    Creates a contract from the provided attributes, runs the ACTUS simulation
    engine, and returns all generated events with payoffs and timing.

    Args:
        attributes: Contract attributes dict. Must include contract_type (e.g., "PAM"),
            status_date, contract_role ("RPA" or "RPL"), and type-specific required fields.
        risk_factors: Optional dict mapping risk factor identifiers to values.
            When provided, uses DictRiskFactorObserver for market-data-dependent contracts.
        constant_value: Constant risk factor value (default 0.0). Used only when
            risk_factors is not provided. Useful for simple fixed-rate scenarios.
        include_states: If True, include contract state before/after each event.
    """
    from .tools import simulate

    return simulate.simulate_contract(
        attributes=attributes,
        risk_factors=risk_factors,
        constant_value=constant_value,
        include_states=include_states,
    )


# ---- Example Tools ----


@mcp.tool()
@log_tool_call
def jactus_list_examples() -> dict[str, Any]:
    """List all available code examples in JACTUS."""
    return examples.list_examples()


@mcp.tool()
@log_tool_call
def jactus_get_example(example_name: str) -> dict[str, Any]:
    """Retrieve a specific code example.

    Args:
        example_name: Name of the example (e.g., pam_example, interest_rate_swap_example)
    """
    return examples.get_example(example_name)


@mcp.tool()
@log_tool_call
def jactus_get_quick_start() -> str:
    """Get a simple quick start example for JACTUS showing a basic PAM contract."""
    return examples.get_quick_start_example()


@mcp.tool()
@log_tool_call
def jactus_run_example(example_name: str) -> dict[str, Any]:
    """Execute a JACTUS example and return its output.

    Args:
        example_name: Name of the example (e.g., pam_example, lam_example)
    """
    return examples.run_example(example_name)


# ---- Validation Tools ----


@mcp.tool()
@log_tool_call
def jactus_validate_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Validate contract attributes for correctness.

    Args:
        attributes: Contract attributes dictionary to validate
    """
    return validation.validate_attributes(attributes)


# ---- Documentation Tools ----


@mcp.tool()
@log_tool_call
def jactus_search_docs(query: str) -> dict[str, Any]:
    """Search JACTUS documentation for specific topics.

    Args:
        query: Search query (e.g., 'day count convention', 'state transition')
    """
    return documentation.search_docs(query)


@mcp.tool()
@log_tool_call
def jactus_get_doc_structure() -> dict[str, Any]:
    """Get the structure of JACTUS documentation, listing all files with their section headers."""
    return documentation.get_doc_structure()


@mcp.tool()
@log_tool_call
def jactus_get_topic_guide(topic: str) -> dict[str, Any]:
    """Get a guide for a specific JACTUS topic.

    Args:
        topic: Topic name. Available: contracts, jax, events, attributes
    """
    return documentation.get_topic_guide(topic)


# ---- System Tools ----


@mcp.tool()
@log_tool_call
def jactus_health_check() -> dict[str, Any]:
    """Verify MCP server and JACTUS are working correctly."""
    return system.health_check()


@mcp.tool()
@log_tool_call
def jactus_get_version_info() -> dict[str, Any]:
    """Get JACTUS and MCP server version information."""
    return system.get_version_info()


# ---- Resources ----


@mcp.resource("jactus://docs/architecture")
def architecture_guide() -> str:
    """JACTUS Architecture Guide - Complete system architecture, design patterns, and implementation details."""
    root = get_jactus_root()
    path = root / "docs" / "ARCHITECTURE.md"
    return path.read_text() if path.exists() else "Resource file not found"


@mcp.resource("jactus://docs/pam")
def pam_walkthrough() -> str:
    """PAM Contract Walkthrough - Deep dive into JACTUS internals using Principal at Maturity contract."""
    root = get_jactus_root()
    path = root / "docs" / "PAM.md"
    return path.read_text() if path.exists() else "Resource file not found"


@mcp.resource("jactus://docs/derivatives")
def derivatives_guide() -> str:
    """Derivative Contracts Guide - Complete guide to all 8 derivative contract types."""
    root = get_jactus_root()
    path = root / "docs" / "derivatives.md"
    return path.read_text() if path.exists() else "Resource file not found"


@mcp.resource("jactus://docs/readme")
def readme() -> str:
    """JACTUS README - Project overview, quick start, and installation."""
    root = get_jactus_root()
    path = root / "README.md"
    return path.read_text() if path.exists() else "Resource file not found"


# ---- Dynamic Contract Resources (Item 13) ----


@mcp.resource("jactus://contract/{contract_type}")
def contract_resource(contract_type: str) -> str:
    """Dynamic resource for contract type information and schema."""
    info = contracts.get_contract_info(contract_type)
    schema = contracts.get_contract_schema(contract_type)
    return json.dumps({"info": info, "schema": schema}, indent=2)


# ---- Prompts ----


@mcp.prompt()
def create_contract(contract_type: str = "PAM") -> str:
    """Guide to create a new JACTUS contract."""
    schema = contracts.get_contract_schema(contract_type)
    return f"""I want to create a {contract_type} contract in JACTUS.

Required fields: {list(schema.get('required_fields', {}).keys())}
Optional fields: {list(schema.get('optional_fields', {}).keys())}

Please help me create a complete, working example with proper validation."""


@mcp.prompt()
def troubleshoot_error(error_message: str) -> str:
    """Help troubleshoot a JACTUS error."""
    health = system.health_check()
    return f"""I'm getting this error in JACTUS:
{error_message}

System health check:
{health}

Please help me understand what's wrong and how to fix it."""


@mcp.prompt()
def understand_contract(contract_type: str = "PAM") -> str:
    """Explain how a specific contract type works."""
    info = contracts.get_contract_info(contract_type)
    return f"""Please explain the {contract_type} contract type in JACTUS.

Contract info: {info}

I want to understand:
1. What is this contract used for?
2. What are the key parameters?
3. How do cash flows work?
4. What are common use cases?"""


@mcp.prompt()
def compare_contracts(contract_type_1: str = "PAM", contract_type_2: str = "LAM") -> str:
    """Compare two contract types."""
    info1 = contracts.get_contract_info(contract_type_1)
    info2 = contracts.get_contract_info(contract_type_2)
    return f"""Please compare {contract_type_1} and {contract_type_2} contracts:

{contract_type_1}: {info1}
{contract_type_2}: {info2}

What are the key differences? When should I use each one?"""


# ---- Entry point ----


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
