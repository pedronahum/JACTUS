"""Contract simulation tools."""

import json
import logging
from typing import Any

from pydantic import ValidationError

from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver, DictRiskFactorObserver
from jactus_mcp.tools._utils import prepare_attributes, parse_datetime
from jactus_mcp.tools.validation import _detect_unknown_fields

logger = logging.getLogger(__name__)


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
    child_contracts: dict[str, dict[str, Any]] | None = None,
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
        child_contracts: Optional dict mapping child identifiers to attribute dicts.
            Required for composite contracts (SWAPS, CAPFL, CEG, CEC). Each child
            is simulated first and its results are fed into the parent contract.

    Returns:
        Dictionary with contract_type, num_events, events list, and summary.
        The summary always covers ALL events regardless of pagination.
    """
    # Contracts that require child contracts
    _CHILD_CONTRACT_REQUIRED = {"CAPFL", "SWAPS", "CEG", "CEC"}

    ct_str = attributes.get("contract_type", "")
    if isinstance(ct_str, str) and ct_str in _CHILD_CONTRACT_REQUIRED and not child_contracts:
        return {
            "success": False,
            "error_type": "missing_child_contracts",
            "error": (
                f"{ct_str} requires child contracts. Provide a child_contracts dict "
                f"mapping identifiers to contract attribute dicts. Each child is "
                f"simulated first, then its results are fed into the {ct_str} parent."
            ),
            "hint": (
                f"Call jactus_get_contract_schema('{ct_str}') to see the required "
                f"format and an example with child_contracts."
            ),
        }

    try:
        # Detect unknown fields before preparation
        unknown_field_warnings = _detect_unknown_fields(attributes)

        # Prepare attributes (convert strings to enums/dates)
        prepared = prepare_attributes(attributes)

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
                    dt = parse_datetime(str(entry[0]))
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

        # Create child contract observer if child_contracts provided
        child_observer = None
        child_results = {}
        if child_contracts:
            from jactus.observers.child_contract import SimulatedChildContractObserver

            child_observer = SimulatedChildContractObserver()

            for child_id, child_attrs_raw in child_contracts.items():
                try:
                    child_prepared = prepare_attributes(child_attrs_raw)
                    child_prepared = {
                        k: v for k, v in child_prepared.items() if k in valid_fields
                    }
                    child_contract_attrs = ContractAttributes(**child_prepared)
                    child_contract = create_contract(child_contract_attrs, rf_observer)
                    child_result = child_contract.simulate()

                    child_observer.register_simulation(
                        child_id,
                        child_result.events,
                        child_contract_attrs,
                        child_result.initial_state,
                    )

                    # Summarize child results
                    child_payoffs = [float(e.payoff) for e in child_result.events]
                    child_non_zero = [p for p in child_payoffs if abs(p) > 1e-10]
                    child_results[child_id] = {
                        "contract_type": child_contract_attrs.contract_type.name,
                        "num_events": len(child_result.events),
                        "net_cashflow": sum(child_non_zero),
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error_type": "child_simulation_error",
                        "error": (
                            f"Child contract '{child_id}' failed: {e!s}"
                        ),
                        "hint": (
                            "Check the child contract attributes. Each child must be "
                            "a valid, self-contained contract (e.g., PAM, LAM, ANN)."
                        ),
                    }

        # Create and simulate
        contract = create_contract(contract_attrs, rf_observer, child_observer)
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
        if child_results:
            response["child_results"] = child_results
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
