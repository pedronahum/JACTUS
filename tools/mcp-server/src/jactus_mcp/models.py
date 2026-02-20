"""Response models for structured output schemas.

These Pydantic models define the expected structure of tool responses.
They serve as documentation and can be used for validation. When FastMCP
adds native outputSchema support, these can be passed directly to
@mcp.tool(output_model=...).
"""

from pydantic import BaseModel


class ContractListResponse(BaseModel):
    """Response from jactus_list_contracts."""

    total_contracts: int
    categories: dict[str, list[str]]
    all_contracts: list[str]


class ContractInfoResponse(BaseModel):
    """Response from jactus_get_contract_info."""

    contract_type: str | None = None
    description: str | None = None
    category: str | None = None
    implemented: bool | None = None
    class_name: str | None = None
    module: str | None = None
    error: str | None = None


class SimulationEvent(BaseModel):
    """A single contract event from simulation."""

    event_type: str
    event_time: str
    payoff: float
    currency: str
    sequence: int


class SimulationSummary(BaseModel):
    """Summary statistics from a contract simulation."""

    total_cashflows: int
    total_inflows: float
    total_outflows: float
    net_cashflow: float
    first_event: str | None = None
    last_event: str | None = None


class SimulationResponse(BaseModel):
    """Response from jactus_simulate_contract."""

    success: bool
    contract_type: str | None = None
    num_events: int | None = None
    events: list[SimulationEvent] | None = None
    summary: SimulationSummary | None = None
    initial_state: dict | None = None
    final_state: dict | None = None
    error: str | None = None
    hint: str | None = None


class HealthCheckResponse(BaseModel):
    """Response from jactus_health_check."""

    status: str
    message: str
    checks: dict[str, bool]
    errors: list[str] | None = None
    versions: dict[str, str | None]
    recommendations: list[str]


class ValidationResponse(BaseModel):
    """Response from jactus_validate_attributes."""

    valid: bool
    contract_type: str | None = None
    attributes: dict | None = None
    warnings: list[str] | None = None
    errors: list[str] | None = None
    message: str | None = None
    hint: str | None = None
