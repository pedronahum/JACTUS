"""Contract simulation tools."""

import logging
from typing import Any

from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.core.types import DayCountConvention
from jactus.observers import ConstantRiskFactorObserver, DictRiskFactorObserver
from jactus_mcp.tools._utils import DATE_FIELDS, ARRAY_DATE_FIELDS
from jactus_mcp.tools.validation import _detect_unknown_fields

logger = logging.getLogger(__name__)


def _parse_datetime(value: str) -> ActusDateTime:
    """Parse an ISO date string to ActusDateTime.

    Supports: YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS
    """
    if "T" in value:
        return ActusDateTime.from_iso(value)
    parts = value.split("-")
    return ActusDateTime(int(parts[0]), int(parts[1]), int(parts[2]))


def _prepare_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Convert string values to proper JACTUS types.

    Handles contract_type, contract_role, day_count_convention, and date fields.
    """
    attrs = dict(attributes)

    # Convert contract_type string to enum
    if "contract_type" in attrs and isinstance(attrs["contract_type"], str):
        attrs["contract_type"] = ContractType[attrs["contract_type"]]

    # Convert contract_role string to enum
    if "contract_role" in attrs and isinstance(attrs["contract_role"], str):
        attrs["contract_role"] = ContractRole[attrs["contract_role"]]

    # Convert day_count_convention string to enum
    if "day_count_convention" in attrs and isinstance(attrs["day_count_convention"], str):
        dcc_map = {
            "AA": DayCountConvention.AA,
            "A360": DayCountConvention.A360,
            "A365": DayCountConvention.A365,
            "E30360ISDA": DayCountConvention.E30360ISDA,
            "E30360": DayCountConvention.E30360,
            "30E360": DayCountConvention.E30360,  # Common alias
            "B30360": DayCountConvention.B30360,
            "BUS252": DayCountConvention.BUS252,
        }
        dcc_value = attrs["day_count_convention"]
        if dcc_value in dcc_map:
            attrs["day_count_convention"] = dcc_map[dcc_value]

    # Convert single date strings to ActusDateTime
    for field in DATE_FIELDS:
        if field in attrs and isinstance(attrs[field], str):
            attrs[field] = _parse_datetime(attrs[field])

    # Convert array date fields (list of date strings → list of ActusDateTime)
    for field in ARRAY_DATE_FIELDS:
        if field in attrs and isinstance(attrs[field], list):
            attrs[field] = [
                _parse_datetime(v) if isinstance(v, str) else v
                for v in attrs[field]
            ]

    return attrs


def simulate_contract(
    attributes: dict[str, Any],
    risk_factors: dict[str, float] | None = None,
    constant_value: float | None = None,
    include_states: bool = False,
) -> dict[str, Any]:
    """Simulate an ACTUS contract and return structured event results.

    Args:
        attributes: Contract attributes dictionary. Must include contract_type,
            status_date, contract_role, and type-specific required fields.
        risk_factors: Optional dict of risk factor identifiers to values.
            If provided, uses DictRiskFactorObserver.
        constant_value: Optional constant risk factor value (default 0.0).
            Used only when risk_factors is not provided.
        include_states: If True, include pre/post state for each event.

    Returns:
        Dictionary with contract_type, num_events, events list, and summary.
    """
    try:
        # Detect unknown fields before preparation
        unknown_field_warnings = _detect_unknown_fields(attributes)

        # Prepare attributes (convert strings to enums/dates)
        prepared = _prepare_attributes(attributes)

        # Strip unknown keys so Pydantic doesn't silently ignore them
        valid_fields = set(ContractAttributes.model_fields.keys())
        prepared = {k: v for k, v in prepared.items() if k in valid_fields}

        contract_attrs = ContractAttributes(**prepared)

        # Create risk factor observer
        if risk_factors:
            rf_observer = DictRiskFactorObserver(risk_factors)
        else:
            rf_observer = ConstantRiskFactorObserver(
                constant_value=constant_value if constant_value is not None else 0.0
            )

        # Create and simulate
        contract = create_contract(contract_attrs, rf_observer)
        result = contract.simulate()

        # Serialize events using ContractEvent.to_dict()
        events = []
        for event in result.events:
            event_dict = event.to_dict()
            if include_states:
                event_dict["state_pre"] = (
                    event.state_pre.to_dict() if event.state_pre else None
                )
                event_dict["state_post"] = (
                    event.state_post.to_dict() if event.state_post else None
                )
            events.append(event_dict)

        # Build summary
        payoffs = [float(e.payoff) for e in result.events]
        non_zero_payoffs = [p for p in payoffs if abs(p) > 1e-10]

        response = {
            "success": True,
            "contract_type": contract_attrs.contract_type.name,
            "num_events": len(events),
            "events": events,
            "summary": {
                "total_cashflows": len(non_zero_payoffs),
                "total_inflows": sum(p for p in non_zero_payoffs if p > 0),
                "total_outflows": sum(p for p in non_zero_payoffs if p < 0),
                "net_cashflow": sum(non_zero_payoffs),
                "first_event": events[0]["event_time"] if events else None,
                "last_event": events[-1]["event_time"] if events else None,
            },
            "initial_state": result.initial_state.to_dict() if result.initial_state else None,
            "final_state": result.final_state.to_dict() if result.final_state else None,
        }
        if unknown_field_warnings:
            response["warnings"] = unknown_field_warnings
        return response

    except KeyError as e:
        return {
            "success": False,
            "error": f"Invalid attribute value: {e!s}",
            "hint": "Use jactus_get_contract_schema to see required fields.",
        }
    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "hint": "Use jactus_validate_attributes to check your attributes first.",
        }
