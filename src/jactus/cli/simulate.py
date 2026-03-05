"""``jactus simulate`` command."""

from __future__ import annotations

import json
import logging
from typing import Any

import typer

from jactus.cli.output import (
    OutputFormat,
    format_currency,
    print_csv_output,
    print_error,
    print_json,
    print_table,
)

logger = logging.getLogger(__name__)


def simulate_cmd(
    contract_type: str = typer.Option(..., "--type", help="ACTUS contract type (e.g., PAM, ANN)"),
    attrs: str | None = typer.Option(None, "--attrs", help="Inline JSON or path to JSON file"),  # noqa: UP007
    stdin: bool = typer.Option(False, "--stdin", help="Read attrs from stdin"),
    observer: str = typer.Option(
        "constant", "--observer", help="Observer type: constant, market, custom"
    ),
    observer_params: str | None = typer.Option(
        None, "--observer-params", help="JSON params for observer"
    ),  # noqa: UP007
    nonzero: bool = typer.Option(False, "--nonzero", help="Filter to non-zero payoff events"),
    from_date: str | None = typer.Option(None, "--from", help="Start date filter YYYY-MM-DD"),  # noqa: UP007
    to_date: str | None = typer.Option(None, "--to", help="End date filter YYYY-MM-DD"),  # noqa: UP007
    fields: str | None = typer.Option(
        None, "--fields", help="Comma-separated event fields to include"
    ),  # noqa: UP007
    child_contracts_file: str | None = typer.Option(
        None, "--child-contracts", help="JSON file with child contracts"
    ),  # noqa: UP007
) -> None:
    """Run a full contract simulation and return cash flow events."""
    from pydantic import ValidationError

    from jactus.cli import get_state, load_attrs, prepare_attributes
    from jactus.contracts import create_contract
    from jactus.core import ContractAttributes

    state = get_state()

    # Load attributes
    try:
        raw = load_attrs(attrs, stdin)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1) from None

    raw.setdefault("contract_type", contract_type.upper())

    # Load child contracts if provided
    child_contracts_raw: dict[str, dict[str, Any]] | None = None
    if child_contracts_file:
        try:
            from pathlib import Path

            child_contracts_raw = json.loads(Path(child_contracts_file).read_text())
        except Exception as e:
            print_error(f"Failed to load child contracts: {e}")
            raise typer.Exit(code=1) from None

    # Parse observer params
    obs_params: dict[str, Any] = {}
    if observer_params:
        try:
            obs_params = json.loads(observer_params)
        except json.JSONDecodeError as e:
            print_error(f"Invalid observer-params JSON: {e}")
            raise typer.Exit(code=1) from None

    try:
        # Prepare attributes
        prepared = prepare_attributes(raw)
        valid_fields = set(ContractAttributes.model_fields.keys())
        prepared = {k: v for k, v in prepared.items() if k in valid_fields}
        contract_attrs = ContractAttributes(**prepared)

        # Create risk factor observer
        rf_observer = _create_observer(observer, obs_params)

        # Handle child contracts for composite types
        child_observer = None
        child_results: dict[str, Any] = {}
        if child_contracts_raw:
            child_observer, child_results = _simulate_children(
                child_contracts_raw, rf_observer, valid_fields
            )

        # Simulate
        contract = create_contract(contract_attrs, rf_observer, child_observer)
        result = contract.simulate()

        # Build events list
        events = []
        for event in result.events:
            ev = event.to_dict()
            events.append(ev)

        # Apply filters
        if nonzero:
            events = [e for e in events if abs(e.get("payoff", 0)) > 1e-10]
        if from_date:
            events = [e for e in events if str(e.get("event_time", "")) >= from_date]
        if to_date:
            events = [e for e in events if str(e.get("event_time", "")) <= to_date]

        # Field filtering
        field_list: list[str] | None = None
        if fields:
            field_list = [f.strip() for f in fields.split(",")]
            events = [{k: v for k, v in e.items() if k in field_list} for e in events]

        # Summary (always computed on full result, before filtering)
        all_payoffs = [float(e.payoff) for e in result.events]
        non_zero_payoffs = [p for p in all_payoffs if abs(p) > 1e-10]
        summary = {
            "total_events": len(result.events),
            "total_cashflow": sum(non_zero_payoffs),
            "first_event": result.events[0].to_dict()["event_time"] if result.events else None,
            "last_event": result.events[-1].to_dict()["event_time"] if result.events else None,
        }

        # Output
        output_data: dict[str, Any] = {
            "contract_id": contract_attrs.contract_id,
            "contract_type": contract_attrs.contract_type.name,
            "status": "success",
            "events": events,
            "summary": summary,
        }
        if child_results:
            output_data["child_results"] = child_results

        if state.output == OutputFormat.JSON:
            print_json(output_data, state.pretty)
        elif state.output == OutputFormat.CSV:
            if events:
                cols = list(events[0].keys())
                rows = [[e.get(c, "") for c in cols] for e in events]
                print_csv_output(cols, rows)
        else:
            _print_text_output(contract_attrs, events, summary, state.no_color, result.events)

    except ValidationError as e:
        errors = [
            f"{' -> '.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in e.errors()
        ]
        if state.output == OutputFormat.JSON:
            print_json({"status": "error", "errors": errors}, state.pretty)
        else:
            print_error("Validation failed")
            for err in errors:
                print_error(f"  {err}")
        raise typer.Exit(code=3) from None
    except Exception as e:
        logger.debug("Simulation error: %s", e, exc_info=True)
        if state.output == OutputFormat.JSON:
            print_json({"status": "error", "error": str(e)}, state.pretty)
        else:
            print_error(f"Simulation error: {e}")
        raise typer.Exit(code=2) from None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _create_observer(observer_type: str, params: dict[str, Any]) -> Any:
    """Create a risk factor observer from CLI flags."""
    from jactus.cli import parse_datetime
    from jactus.observers import ConstantRiskFactorObserver, DictRiskFactorObserver

    if observer_type == "constant":
        rate = params.get("rate", params.get("constant_value", 0.0))
        return ConstantRiskFactorObserver(constant_value=float(rate))

    if observer_type == "market":
        # Check for time_series first
        if "time_series" in params:
            from jactus.observers import TimeSeriesRiskFactorObserver

            parsed_ts: dict[str, list[tuple[Any, float]]] = {}
            for identifier, series in params["time_series"].items():
                parsed_series = []
                for entry in series:
                    dt = parse_datetime(str(entry[0]))
                    val = float(entry[1])
                    parsed_series.append((dt, val))
                parsed_ts[identifier] = parsed_series
            return TimeSeriesRiskFactorObserver(
                parsed_ts,
                interpolation=params.get("interpolation", "step"),
                extrapolation=params.get("extrapolation", "flat"),
            )
        # Fall back to dict observer
        risk_factors = params.get("risk_factors", params)
        # Filter out non-numeric keys
        rf = {k: float(v) for k, v in risk_factors.items() if isinstance(v, (int, float))}
        return DictRiskFactorObserver(rf)

    # Default
    return ConstantRiskFactorObserver(constant_value=0.0)


def _simulate_children(
    child_contracts_raw: dict[str, dict[str, Any]],
    rf_observer: Any,
    valid_fields: set[str],
) -> tuple[Any, dict[str, Any]]:
    """Simulate child contracts and return (child_observer, child_results)."""
    from jactus.cli import prepare_attributes
    from jactus.contracts import create_contract
    from jactus.core import ContractAttributes
    from jactus.observers.child_contract import SimulatedChildContractObserver

    child_observer = SimulatedChildContractObserver()
    child_results: dict[str, Any] = {}

    for child_id, child_attrs_raw in child_contracts_raw.items():
        child_prepared = prepare_attributes(child_attrs_raw)
        child_prepared = {k: v for k, v in child_prepared.items() if k in valid_fields}
        child_contract_attrs = ContractAttributes(**child_prepared)
        child_contract = create_contract(child_contract_attrs, rf_observer)
        child_result = child_contract.simulate()

        child_observer.register_simulation(
            child_id, child_result.events, child_contract_attrs, child_result.initial_state
        )
        child_payoffs = [float(e.payoff) for e in child_result.events]
        child_non_zero = [p for p in child_payoffs if abs(p) > 1e-10]
        child_results[child_id] = {
            "contract_type": child_contract_attrs.contract_type.name,
            "num_events": len(child_result.events),
            "net_cashflow": sum(child_non_zero),
        }

    return child_observer, child_results


def _print_text_output(
    attrs: Any,
    events: list[dict[str, Any]],
    summary: dict[str, Any],
    no_color: bool,
    raw_events: list[Any] | None = None,
) -> None:
    """Print simulation results as a rich table."""
    rows = []
    for i, ev in enumerate(events):
        payoff = ev.get("payoff", 0)
        # Get notional from post-event state if available
        notional = ""
        if raw_events and i < len(raw_events):
            state_post = raw_events[i].state_post
            if state_post is not None:
                notional = f"${float(state_post.nt):,.0f}"
        rows.append(
            [
                str(ev.get("event_time", "")),
                str(ev.get("event_type", "")),
                format_currency(float(payoff)),
                notional,
            ]
        )

    print_table(
        f"CONTRACT: {attrs.contract_id} ({attrs.contract_type.name})",
        ["Date", "Event", "Payoff", "Notional"],
        rows,
        no_color,
    )

    from jactus.cli.output import console

    console.print(f"Net cash flow: {format_currency(summary['total_cashflow'])}")
