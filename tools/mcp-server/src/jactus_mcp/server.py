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
mcp = FastMCP(
    "jactus-mcp",
    instructions="""JACTUS MCP Server - Financial contract simulation powered by JAX.

This server provides tools for discovering, validating, and simulating ACTUS
(Algorithmic Contract Types Unified Standards) financial contracts. JACTUS
supports 18 contract types across 4 categories: principal instruments (PAM,
LAM, LAX, NAM, ANN, CLM), non-principal instruments (UMP, CSH, STK), exotic
instruments (COM), and derivatives (FXOUT, OPTNS, FUTUR, SWPPV, SWAPS, CAPFL,
CEG, CEC).

Typical workflow:
1. Use jactus_list_contracts to discover available contract types
2. Use jactus_get_contract_schema to understand required/optional parameters
3. Use jactus_validate_attributes to verify your parameters before simulation
4. Use jactus_simulate_contract to run the simulation and get cash flows
5. Use jactus_search_docs or jactus_get_topic_guide for deeper understanding

Key concepts:
- contract_role: RPA = receive principal (lender/asset), RPL = receive payoff (borrower/liability)
- Dates use ISO format: "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS"
- Cycles use notation like "1M" (monthly), "3M" (quarterly), "6M" (semi-annual), "1Y" (annual)
- Risk factors: use constant_value for fixed-rate contracts, risk_factors dict for static
  market data, or time_series for time-varying rates (e.g., floating-rate resets)
- Interpolation: step vs linear only differ when query dates fall BETWEEN data points.
  If rate reset dates align exactly with time series dates, results will be identical.
  To compare modes, use data points at different frequencies than the reset cycle.

For derivative contracts (SWPPV, OPTNS, FUTUR, CAPFL), risk_factors or time_series
is usually required. For simple principal contracts (PAM, LAM), constant_value
is often sufficient.
""",
)


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
    """List all 18 available ACTUS contract types organized by category.

    Returns contract types grouped into: principal (PAM, LAM, LAX, NAM, ANN, CLM),
    non-principal (UMP, CSH, STK), exotic (COM), and derivative (FXOUT, OPTNS,
    FUTUR, SWPPV, SWAPS, CAPFL, CEG, CEC).

    Start here to discover which contract type matches your financial instrument.
    Follow up with jactus_get_contract_info for details or jactus_get_contract_schema
    for the required parameters.
    """
    return contracts.list_contracts()


@mcp.tool()
@log_tool_call
def jactus_get_contract_info(contract_type: str) -> dict[str, Any]:
    """Get detailed information about a specific ACTUS contract type.

    Returns the contract description, category, implementation class, and module path.
    Use this to understand what a contract type represents before building one.

    Args:
        contract_type: ACTUS contract type code. Examples: PAM (bonds/loans),
            LAM (amortizing loans), ANN (mortgages), SWPPV (interest rate swaps),
            OPTNS (options), FXOUT (FX forwards).
    """
    return contracts.get_contract_info(contract_type)


@mcp.tool()
@log_tool_call
def jactus_get_contract_schema(contract_type: str) -> dict[str, Any]:
    """Get required and optional parameters for a contract type.

    Returns field names, types, and descriptions for both required and optional
    parameters, plus example Python code for creating the contract. Use this
    before calling jactus_simulate_contract to know which fields to provide.

    Args:
        contract_type: ACTUS contract type code (e.g., PAM, LAM, SWPPV).
    """
    return contracts.get_contract_schema(contract_type)


@mcp.tool()
@log_tool_call
def jactus_get_event_types() -> dict[str, Any]:
    """List all ACTUS event types with descriptions.

    Returns event type codes (IED, IP, PR, MD, RR, etc.) and their meanings.
    Events represent cash flows and state transitions during a contract's life.
    Use this to understand the events returned by jactus_simulate_contract.
    """
    return contracts.get_event_types()


@mcp.tool()
@log_tool_call
def jactus_list_risk_factor_observers() -> dict[str, Any]:
    """List all available risk factor observer types with usage guidance.

    Returns observer types organized by complexity, from simple constant values to
    advanced time-series and curve observers. Each entry includes a description,
    typical use case, and whether it's available via MCP or requires the Python API.

    Use this to determine which risk factor approach to use with jactus_simulate_contract.
    For MCP simulation, you can use: constant_value (default), risk_factors (dict), or
    time_series (time-varying). For advanced observers (curves, composites, callbacks,
    JAX), use the Python API directly.
    """
    return contracts.list_risk_factor_observers()


# ---- Simulation Tool ----


@mcp.tool()
@log_tool_call
def jactus_simulate_contract(
    attributes: dict[str, Any],
    risk_factors: dict[str, float] | None = None,
    time_series: dict[str, list[list]] | None = None,
    interpolation: str = "step",
    extrapolation: str = "flat",
    constant_value: float | None = None,
    include_states: bool = False,
    event_limit: int | None = None,
    event_offset: int = 0,
) -> dict[str, Any]:
    """Simulate an ACTUS contract and return structured cash flow events.

    Creates a contract from the provided attributes, runs the ACTUS simulation
    engine, and returns all generated events with payoffs, timing, and optional
    contract state snapshots.

    Common workflow:
    1. Use jactus_get_contract_schema to get required fields for your contract type
    2. Build the attributes dict with those fields
    3. Call this tool to simulate
    4. Examine the events and summary in the response

    Risk factor observer selection (in priority order):
    1. time_series - Time-varying market data with interpolation (for rate resets)
    2. risk_factors - Fixed per-identifier values (for static market data)
    3. constant_value - Single constant for all risk factors (default: 0.0)

    Output size management:
    - For contracts with many events, use event_limit and event_offset to paginate
    - If include_states=True produces output that is too large, events are
      auto-truncated to first 5 + last 5, with a pagination hint in the response

    Args:
        attributes: Contract attributes dict. Must include contract_type (e.g., "PAM"),
            status_date (ISO date), contract_role ("RPA" or "RPL"), and type-specific
            required fields. Use jactus_get_contract_schema to see required fields.
        risk_factors: Dict mapping risk factor identifiers to constant values.
            Example: {"LIBOR-3M": 0.05, "USD/EUR": 1.18}
        time_series: Dict mapping identifiers to time-value pairs for time-varying data.
            Each entry is [date_string, value]. Example:
            {"LIBOR-3M": [["2024-01-01", 0.04], ["2024-07-01", 0.045]]}
        interpolation: Interpolation method for time_series: "step" (default) or "linear".
            Step uses the most recent known value; linear interpolates between points.
            Note: both modes give identical results when query dates exactly match data
            points. To see differences, use data points at different dates than resets.
        extrapolation: Extrapolation method for time_series: "flat" (default) or "raise".
            Flat returns the nearest endpoint value; raise returns an error.
        constant_value: Constant risk factor value (default 0.0). Used only when
            neither risk_factors nor time_series is provided.
        include_states: If True, include contract state before/after each event.
            Warning: this significantly increases output size for contracts with many events.
        event_limit: Maximum number of events to return. Use with event_offset for
            pagination. The summary always covers all events regardless.
        event_offset: Number of events to skip from the beginning (default 0).

    Returns:
        Dict with: success, contract_type, num_events, events (list of event dicts),
        summary (total_inflows, total_outflows, net_cashflow, first/last_event),
        initial_state, final_state. If paginated: includes pagination dict with
        total_events, offset, limit, returned. On error: success=False, error,
        error_type, hint.
    """
    from .tools import simulate

    return simulate.simulate_contract(
        attributes=attributes,
        risk_factors=risk_factors,
        time_series=time_series,
        interpolation=interpolation,
        extrapolation=extrapolation,
        constant_value=constant_value,
        include_states=include_states,
        event_limit=event_limit,
        event_offset=event_offset,
    )


# ---- Example Tools ----


@mcp.tool()
@log_tool_call
def jactus_list_examples() -> dict[str, Any]:
    """List all available code examples in JACTUS.

    Returns Python scripts and Jupyter notebooks from the examples directory.
    Use jactus_get_example to retrieve the code or jactus_run_example to execute it.
    """
    return examples.list_examples()


@mcp.tool()
@log_tool_call
def jactus_get_example(example_name: str) -> dict[str, Any]:
    """Retrieve a specific code example's source code.

    Returns the full source code, docstring, and metadata for an example.
    Use jactus_list_examples first to see available examples.

    Args:
        example_name: Name of the example (e.g., pam_example, interest_rate_swap_example).
    """
    return examples.get_example(example_name)


@mcp.tool()
@log_tool_call
def jactus_get_quick_start() -> str:
    """Get a simple quick start example showing a basic PAM contract simulation.

    Returns ready-to-run Python code that creates a PAM (Principal at Maturity)
    contract and simulates it. Good starting point for learning the JACTUS API.
    """
    return examples.get_quick_start_example()


@mcp.tool()
@log_tool_call
def jactus_run_example(example_name: str) -> dict[str, Any]:
    """Execute a JACTUS example and return its output.

    Runs the example in a subprocess with a 30-second timeout and returns
    stdout, stderr, and return code. Use jactus_list_examples to see available examples.

    Args:
        example_name: Name of the example (e.g., pam_example, lam_example).
    """
    return examples.run_example(example_name)


# ---- Validation Tools ----


@mcp.tool()
@log_tool_call
def jactus_validate_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Validate contract attributes for correctness before simulation.

    Checks that all required fields are present, values are valid, and types are
    correct. Returns field-level error messages and warnings for unknown fields.
    Call this before jactus_simulate_contract to catch errors early.

    Args:
        attributes: Contract attributes dictionary to validate. Should include
            contract_type, status_date, contract_role, and type-specific fields.
    """
    return validation.validate_attributes(attributes)


# ---- Documentation Tools ----


@mcp.tool()
@log_tool_call
def jactus_search_docs(query: str) -> dict[str, Any]:
    """Search JACTUS documentation for specific topics.

    Searches across architecture docs, contract guides, and the README.
    Returns matching lines with context. Use jactus_get_topic_guide for
    structured guides on common topics.

    Args:
        query: Search query (e.g., 'day count convention', 'state transition',
            'rate reset', 'prepayment').
    """
    return documentation.search_docs(query)


@mcp.tool()
@log_tool_call
def jactus_get_doc_structure() -> dict[str, Any]:
    """Get the structure of JACTUS documentation, listing all files with their section headers.

    Returns available documentation files with their headers, useful for
    understanding what documentation is available before searching.
    """
    return documentation.get_doc_structure()


@mcp.tool()
@log_tool_call
def jactus_get_topic_guide(topic: str) -> dict[str, Any]:
    """Get a structured guide for a specific JACTUS topic.

    Returns a comprehensive markdown guide on the requested topic.
    More focused than jactus_search_docs for common areas.

    Args:
        topic: Topic name. Available: "contracts" (overview of all types),
            "jax" (JAX integration and autodiff), "events" (event types and lifecycle),
            "attributes" (contract parameters and conventions).
    """
    return documentation.get_topic_guide(topic)


# ---- System Tools ----


@mcp.tool()
@log_tool_call
def jactus_health_check() -> dict[str, Any]:
    """Verify MCP server and JACTUS are working correctly.

    Checks that JACTUS is installed and importable, examples and docs are
    accessible, and contracts are registered. Returns status ("healthy",
    "degraded", or "unhealthy") with specific check results.
    """
    return system.health_check()


@mcp.tool()
@log_tool_call
def jactus_get_version_info() -> dict[str, Any]:
    """Get JACTUS and MCP server version information.

    Returns versions for both the MCP server and the JACTUS library,
    plus Python version and compatibility status.
    """
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
