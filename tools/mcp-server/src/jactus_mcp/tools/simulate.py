"""Contract simulation tools."""

import json
import logging
from typing import Any

from pydantic import ValidationError

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


# Maximum output size in characters before auto-truncation kicks in
_MAX_OUTPUT_CHARS = 40_000


def simulate_contract(
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
    """Simulate an ACTUS contract and return structured event results.

    Args:
        attributes: Contract attributes dictionary. Must include contract_type,
            status_date, contract_role, and type-specific required fields.
        risk_factors: Optional dict of risk factor identifiers to values.
            If provided, uses DictRiskFactorObserver.
        time_series: Optional dict mapping identifiers to time-value pairs.
            Each entry is [date_string, value]. Takes priority over risk_factors.
        interpolation: Interpolation method for time_series: "step" or "linear".
        extrapolation: Extrapolation method for time_series: "flat" or "raise".
        constant_value: Optional constant risk factor value (default 0.0).
            Used only when neither risk_factors nor time_series is provided.
        include_states: If True, include pre/post state for each event.
        event_limit: Maximum number of events to return. If None, returns all
            events (subject to auto-truncation if output is too large).
        event_offset: Number of events to skip from the beginning (default 0).
            Use with event_limit for pagination.

    Returns:
        Dictionary with contract_type, num_events, events list, and summary.
        The summary always covers ALL events regardless of pagination.
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

        # Create risk factor observer (priority: time_series > risk_factors > constant)
        if time_series:
            from jactus.observers import TimeSeriesRiskFactorObserver

            parsed_ts: dict[str, list[tuple[ActusDateTime, float]]] = {}
            for identifier, series in time_series.items():
                parsed_series = []
                for entry in series:
                    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                        raise ValueError(
                            f"Each time series entry must be [date_string, value], "
                            f"got {entry!r} for '{identifier}'"
                        )
                    dt = _parse_datetime(str(entry[0]))
                    val = float(entry[1])
                    parsed_series.append((dt, val))
                parsed_ts[identifier] = parsed_series
            rf_observer = TimeSeriesRiskFactorObserver(
                parsed_ts,
                interpolation=interpolation,
                extrapolation=extrapolation,
            )
        elif risk_factors:
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

        # Build summary (always covers ALL events, before pagination)
        payoffs = [float(e.payoff) for e in result.events]
        non_zero_payoffs = [p for p in payoffs if abs(p) > 1e-10]
        total_events = len(events)

        summary = {
            "total_cashflows": len(non_zero_payoffs),
            "total_inflows": sum(p for p in non_zero_payoffs if p > 0),
            "total_outflows": sum(p for p in non_zero_payoffs if p < 0),
            "net_cashflow": sum(non_zero_payoffs),
            "first_event": events[0]["event_time"] if events else None,
            "last_event": events[-1]["event_time"] if events else None,
        }

        # Apply explicit pagination
        pagination = None
        if event_offset > 0 or event_limit is not None:
            events = events[event_offset:]
            if event_limit is not None:
                events = events[:event_limit]
            pagination = {
                "total_events": total_events,
                "offset": event_offset,
                "limit": event_limit,
                "returned": len(events),
            }

        # Auto-truncation: if output is still too large, reduce events
        truncated = False
        if not pagination:
            estimated_size = len(json.dumps(events[:1])) * len(events) if events else 0
            if estimated_size > _MAX_OUTPUT_CHARS:
                # Keep first 5 and last 5 events
                keep = min(10, len(events))
                head = events[:keep // 2]
                tail = events[-(keep - keep // 2):]
                omitted = total_events - len(head) - len(tail)
                events = head + tail
                truncated = True
                pagination = {
                    "total_events": total_events,
                    "returned": len(events),
                    "truncated": True,
                    "omitted": omitted,
                    "hint": (
                        f"Output was too large ({total_events} events with states). "
                        f"Showing first {len(head)} and last {len(tail)} events. "
                        f"Use event_limit and event_offset to paginate through all events."
                    ),
                }

        response = {
            "success": True,
            "contract_type": contract_attrs.contract_type.name,
            "num_events": total_events,
            "events": events,
            "summary": summary,
            "initial_state": result.initial_state.to_dict() if result.initial_state else None,
            "final_state": result.final_state.to_dict() if result.final_state else None,
        }
        if pagination:
            response["pagination"] = pagination
        if unknown_field_warnings:
            response["warnings"] = unknown_field_warnings
        return response

    except KeyError as e:
        return {
            "success": False,
            "error_type": "invalid_attribute",
            "error": f"Invalid attribute value: {e!s}",
            "hint": "Use jactus_get_contract_schema to see valid values for enums like contract_type and contract_role.",
        }
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            errors.append(f"{field}: {msg}")
        return {
            "success": False,
            "error_type": "validation_error",
            "error": "Contract attribute validation failed",
            "details": errors,
            "hint": "Use jactus_validate_attributes to check your attributes first.",
        }
    except TypeError as e:
        return {
            "success": False,
            "error_type": "type_error",
            "error": f"Type mismatch in attributes: {e!s}",
            "hint": "Check that numeric fields are numbers and date fields are ISO strings.",
        }
    except ValueError as e:
        return {
            "success": False,
            "error_type": "value_error",
            "error": str(e),
            "hint": "Check the format of your input values (dates, time series entries, etc.).",
        }
    except Exception as e:
        logger.error(f"Simulation error: {e}", exc_info=True)
        return {
            "success": False,
            "error_type": "simulation_error",
            "error": str(e),
            "hint": "Use jactus_validate_attributes to check your attributes first.",
        }
