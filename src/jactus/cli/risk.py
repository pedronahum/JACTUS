"""``jactus risk`` subcommands: dv01, duration, convexity, sensitivities."""

from __future__ import annotations

import logging
from typing import Any

import typer

from jactus.cli.output import OutputFormat, print_error, print_json, print_table

logger = logging.getLogger(__name__)

risk_app = typer.Typer(no_args_is_help=True)


# ---------------------------------------------------------------------------
# Shared computation helpers
# ---------------------------------------------------------------------------


def _simulate_npv(
    raw_attrs: dict[str, Any], rate_override: float | None = None
) -> tuple[float, str]:
    """Simulate a contract and return (NPV, contract_id).

    NPV is the sum of payoffs discounted at the contract's nominal interest rate.
    If *rate_override* is provided, it replaces nominal_interest_rate.
    """
    from jactus.cli import prepare_attributes
    from jactus.contracts import create_contract
    from jactus.core import ContractAttributes
    from jactus.observers import ConstantRiskFactorObserver

    attrs = dict(raw_attrs)
    if rate_override is not None:
        attrs["nominal_interest_rate"] = rate_override

    prepared = prepare_attributes(attrs)
    valid_fields = set(ContractAttributes.model_fields.keys())
    prepared = {k: v for k, v in prepared.items() if k in valid_fields}
    contract_attrs = ContractAttributes(**prepared)

    rf_observer = ConstantRiskFactorObserver(constant_value=0.0)
    contract = create_contract(contract_attrs, rf_observer)
    result = contract.simulate()

    # Discount rate = nominal interest rate
    discount_rate = float(contract_attrs.nominal_interest_rate or 0.0)

    # Compute NPV as sum of discounted payoffs
    npv = 0.0
    if result.events and contract_attrs.status_date:
        for event in result.events:
            payoff = float(event.payoff)
            if abs(payoff) < 1e-10:
                continue
            # Years from status date to event time
            days = event.event_time.days_between(contract_attrs.status_date)
            years = days / 365.25
            if years > 0 and discount_rate > 0:
                npv += payoff / ((1 + discount_rate) ** years)
            else:
                npv += payoff

    return npv, contract_attrs.contract_id


def _compute_dv01(raw_attrs: dict[str, Any], bump: float = 0.0001) -> float:
    """Compute DV01 using central finite difference."""
    base_rate = float(raw_attrs.get("nominal_interest_rate", 0.0))
    npv_up, _ = _simulate_npv(raw_attrs, base_rate + bump / 2)
    npv_down, _ = _simulate_npv(raw_attrs, base_rate - bump / 2)
    return npv_up - npv_down


def _compute_duration(raw_attrs: dict[str, Any], bump: float = 0.0001) -> tuple[float, float]:
    """Compute (modified_duration, macaulay_duration)."""
    base_rate = float(raw_attrs.get("nominal_interest_rate", 0.0))
    npv_base, _ = _simulate_npv(raw_attrs)
    if abs(npv_base) < 1e-10:
        return 0.0, 0.0

    dv01 = _compute_dv01(raw_attrs, bump)
    modified = -dv01 / (npv_base * bump)
    macaulay = modified * (1 + base_rate)
    return modified, macaulay


def _compute_convexity(raw_attrs: dict[str, Any], bump: float = 0.0001) -> float:
    """Compute convexity using central finite difference."""
    base_rate = float(raw_attrs.get("nominal_interest_rate", 0.0))
    npv_base, _ = _simulate_npv(raw_attrs)
    if abs(npv_base) < 1e-10:
        return 0.0

    npv_up, _ = _simulate_npv(raw_attrs, base_rate + bump)
    npv_down, _ = _simulate_npv(raw_attrs, base_rate - bump)
    return (npv_up + npv_down - 2 * npv_base) / (npv_base * bump * bump)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@risk_app.command("dv01")
def dv01(
    contract_type: str = typer.Option(..., "--type", help="Contract type"),
    attrs: str | None = typer.Option(None, "--attrs", help="Inline JSON or file path"),  # noqa: UP007
    stdin: bool = typer.Option(False, "--stdin", help="Read attrs from stdin"),
    bump: float = typer.Option(0.0001, "--bump", help="Rate bump size (default: 1bp)"),
) -> None:
    """Compute DV01 (dollar value of a basis point)."""
    from jactus.cli import get_state, load_attrs

    state = get_state()
    try:
        raw = load_attrs(attrs, stdin)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1) from None

    raw.setdefault("contract_type", contract_type.upper())

    try:
        val = _compute_dv01(raw, bump)
        contract_id = raw.get("contract_id", "unknown")
        result = {
            "contract_id": contract_id,
            "metric": "dv01",
            "value": round(val, 6),
            "bump_size": bump,
            "units": "USD/bp",
        }
        if state.output == OutputFormat.JSON:
            print_json(result, state.pretty)
        else:
            from jactus.cli.output import console

            console.print(f"DV01: {val:.6f} USD/bp (bump={bump})")
    except Exception as e:
        print_error(f"Risk computation error: {e}")
        raise typer.Exit(code=2) from None


@risk_app.command("duration")
def duration(
    contract_type: str = typer.Option(..., "--type", help="Contract type"),
    attrs: str | None = typer.Option(None, "--attrs", help="Inline JSON or file path"),  # noqa: UP007
    stdin: bool = typer.Option(False, "--stdin", help="Read attrs from stdin"),
) -> None:
    """Compute modified and Macaulay duration."""
    from jactus.cli import get_state, load_attrs

    state = get_state()
    try:
        raw = load_attrs(attrs, stdin)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1) from None

    raw.setdefault("contract_type", contract_type.upper())

    try:
        modified, macaulay = _compute_duration(raw)
        result = {
            "contract_id": raw.get("contract_id", "unknown"),
            "metric": "duration",
            "modified_duration": round(modified, 6),
            "macaulay_duration": round(macaulay, 6),
            "units": "years",
        }
        if state.output == OutputFormat.JSON:
            print_json(result, state.pretty)
        else:
            from jactus.cli.output import console

            console.print(f"Modified Duration: {modified:.6f} years")
            console.print(f"Macaulay Duration: {macaulay:.6f} years")
    except Exception as e:
        print_error(f"Risk computation error: {e}")
        raise typer.Exit(code=2) from None


@risk_app.command("convexity")
def convexity(
    contract_type: str = typer.Option(..., "--type", help="Contract type"),
    attrs: str | None = typer.Option(None, "--attrs", help="Inline JSON or file path"),  # noqa: UP007
    stdin: bool = typer.Option(False, "--stdin", help="Read attrs from stdin"),
) -> None:
    """Compute convexity (second-order rate sensitivity)."""
    from jactus.cli import get_state, load_attrs

    state = get_state()
    try:
        raw = load_attrs(attrs, stdin)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1) from None

    raw.setdefault("contract_type", contract_type.upper())

    try:
        val = _compute_convexity(raw)
        result = {
            "contract_id": raw.get("contract_id", "unknown"),
            "metric": "convexity",
            "value": round(val, 6),
            "units": "years^2",
        }
        if state.output == OutputFormat.JSON:
            print_json(result, state.pretty)
        else:
            from jactus.cli.output import console

            console.print(f"Convexity: {val:.6f} years^2")
    except Exception as e:
        print_error(f"Risk computation error: {e}")
        raise typer.Exit(code=2) from None


@risk_app.command("sensitivities")
def sensitivities(
    contract_type: str = typer.Option(..., "--type", help="Contract type"),
    attrs: str | None = typer.Option(None, "--attrs", help="Inline JSON or file path"),  # noqa: UP007
    stdin: bool = typer.Option(False, "--stdin", help="Read attrs from stdin"),
    metrics: str | None = typer.Option(
        None, "--metrics", help="Comma-separated: dv01,duration,convexity"
    ),  # noqa: UP007
) -> None:
    """Compute all risk sensitivities in one call."""
    from jactus.cli import get_state, load_attrs

    state = get_state()
    try:
        raw = load_attrs(attrs, stdin)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1) from None

    raw.setdefault("contract_type", contract_type.upper())

    requested = {"dv01", "duration", "convexity"}
    if metrics:
        requested = {m.strip().lower() for m in metrics.split(",")}

    try:
        result: dict[str, Any] = {
            "contract_id": raw.get("contract_id", "unknown"),
            "contract_type": contract_type.upper(),
            "sensitivities": {},
        }

        if "dv01" in requested:
            val = _compute_dv01(raw)
            result["sensitivities"]["dv01"] = {"value": round(val, 6), "units": "USD/bp"}

        if "duration" in requested:
            mod, mac = _compute_duration(raw)
            result["sensitivities"]["modified_duration"] = {
                "value": round(mod, 6),
                "units": "years",
            }
            result["sensitivities"]["macaulay_duration"] = {
                "value": round(mac, 6),
                "units": "years",
            }

        if "convexity" in requested:
            val = _compute_convexity(raw)
            result["sensitivities"]["convexity"] = {"value": round(val, 6), "units": "years^2"}

        if state.output == OutputFormat.JSON:
            print_json(result, state.pretty)
        else:
            rows = [
                [name, str(data["value"]), data["units"]]
                for name, data in result["sensitivities"].items()
            ]
            print_table(
                f"SENSITIVITIES: {raw.get('contract_id', 'unknown')} ({contract_type.upper()})",
                ["Metric", "Value", "Units"],
                rows,
                state.no_color,
            )
    except Exception as e:
        print_error(f"Risk computation error: {e}")
        raise typer.Exit(code=2) from None
