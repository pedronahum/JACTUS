"""``jactus portfolio`` subcommands: simulate, aggregate."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import typer

from jactus.cli.output import (
    OutputFormat,
    format_currency,
    print_error,
    print_json,
    print_table,
)

logger = logging.getLogger(__name__)

portfolio_app = typer.Typer(no_args_is_help=True)


def _load_portfolio(file_path: str) -> dict[str, Any]:
    """Load and validate portfolio JSON file."""
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Portfolio file not found: {file_path}")
    data = json.loads(p.read_text())
    if "contracts" not in data:
        raise ValueError("Portfolio file must contain a 'contracts' array")
    return data  # type: ignore[no-any-return]


def _create_observer_from_config(obs_config: dict[str, Any] | None) -> Any:
    """Create a risk factor observer from portfolio config."""
    from jactus.observers import ConstantRiskFactorObserver, DictRiskFactorObserver

    if obs_config is None:
        return ConstantRiskFactorObserver(constant_value=0.0)

    obs_type = obs_config.get("type", "constant")
    params = obs_config.get("params", {})

    if obs_type == "constant":
        return ConstantRiskFactorObserver(constant_value=float(params.get("rate", 0.0)))
    if obs_type == "market":
        risk_factors = {k: float(v) for k, v in params.items() if isinstance(v, (int, float))}
        return DictRiskFactorObserver(risk_factors)
    return ConstantRiskFactorObserver(constant_value=0.0)


@portfolio_app.command("simulate")
def simulate_portfolio(
    file: str = typer.Option(..., "--file", help="Path to portfolio JSON file"),
) -> None:
    """Simulate multiple contracts from a portfolio file."""
    from jactus.cli import get_state, prepare_attributes
    from jactus.contracts import create_contract
    from jactus.core import ContractAttributes

    state = get_state()

    try:
        portfolio = _load_portfolio(file)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1) from None

    portfolio_id = portfolio.get("portfolio_id", "unknown")
    rf_observer = _create_observer_from_config(portfolio.get("observer"))
    valid_fields = set(ContractAttributes.model_fields.keys())
    contracts_output: list[dict[str, Any]] = []

    for entry in portfolio["contracts"]:
        ct = entry.get("type", "")
        raw_attrs = entry.get("attrs", {})
        raw_attrs.setdefault("contract_type", ct)

        try:
            prepared = prepare_attributes(raw_attrs)
            prepared = {k: v for k, v in prepared.items() if k in valid_fields}
            contract_attrs = ContractAttributes(**prepared)
            contract = create_contract(contract_attrs, rf_observer)
            result = contract.simulate()

            events = [e.to_dict() for e in result.events]
            payoffs = [float(e.payoff) for e in result.events]
            non_zero = [p for p in payoffs if abs(p) > 1e-10]

            contracts_output.append(
                {
                    "contract_id": contract_attrs.contract_id,
                    "contract_type": ct,
                    "events": events,
                    "summary": {
                        "total_events": len(events),
                        "total_cashflow": sum(non_zero),
                        "first_event": events[0]["event_time"] if events else None,
                        "last_event": events[-1]["event_time"] if events else None,
                    },
                }
            )
        except Exception as e:
            contracts_output.append(
                {
                    "contract_id": raw_attrs.get("contract_id", "unknown"),
                    "contract_type": ct,
                    "status": "error",
                    "error": str(e),
                }
            )

    output = {
        "portfolio_id": portfolio_id,
        "status": "success",
        "contracts": contracts_output,
    }

    if state.output == OutputFormat.JSON:
        print_json(output, state.pretty)
    else:
        rows = []
        for c in contracts_output:
            if "error" in c:
                rows.append([c["contract_id"], c["contract_type"], "ERROR", c["error"]])
            else:
                rows.append(
                    [
                        c["contract_id"],
                        c["contract_type"],
                        format_currency(c["summary"]["total_cashflow"]),
                        str(c["summary"]["total_events"]),
                    ]
                )
        print_table(
            f"PORTFOLIO: {portfolio_id}",
            ["Contract", "Type", "Net Cashflow", "Events"],
            rows,
            state.no_color,
        )


@portfolio_app.command("aggregate")
def aggregate(
    file: str = typer.Option(..., "--file", help="Path to portfolio JSON file"),
    frequency: str = typer.Option(
        "daily", "--frequency", help="Bucketing: daily, monthly, quarterly, annual"
    ),
    currency: str | None = typer.Option(None, "--currency", help="Display currency label"),  # noqa: UP007
) -> None:
    """Aggregate net cash flows across all contracts by date."""
    from jactus.cli import get_state, prepare_attributes
    from jactus.contracts import create_contract
    from jactus.core import ContractAttributes

    state = get_state()

    try:
        portfolio = _load_portfolio(file)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1) from None

    portfolio_id = portfolio.get("portfolio_id", "unknown")
    rf_observer = _create_observer_from_config(portfolio.get("observer"))
    valid_fields = set(ContractAttributes.model_fields.keys())

    # Collect all events across all contracts
    all_events: list[tuple[str, str, float]] = []  # (date_str, contract_id, payoff)

    for entry in portfolio["contracts"]:
        ct = entry.get("type", "")
        raw_attrs = entry.get("attrs", {})
        raw_attrs.setdefault("contract_type", ct)

        try:
            prepared = prepare_attributes(raw_attrs)
            prepared = {k: v for k, v in prepared.items() if k in valid_fields}
            contract_attrs = ContractAttributes(**prepared)
            contract = create_contract(contract_attrs, rf_observer)
            result = contract.simulate()

            for event in result.events:
                payoff = float(event.payoff)
                if abs(payoff) > 1e-10:
                    date_str = event.to_dict()["event_time"][:10]  # YYYY-MM-DD
                    all_events.append((date_str, contract_attrs.contract_id, payoff))
        except Exception:
            continue

    # Bucket by period — use separate dicts for type safety
    payoff_buckets: dict[str, float] = defaultdict(float)
    contract_buckets: dict[str, list[str]] = defaultdict(list)

    for date_str, contract_id, payoff in all_events:
        period = _to_period(date_str, frequency)
        payoff_buckets[period] += payoff
        if contract_id not in contract_buckets[period]:
            contract_buckets[period].append(contract_id)

    # Sort by period
    sorted_periods = sorted(payoff_buckets.keys())
    cashflows = [
        {
            "period": p,
            "net_payoff": round(payoff_buckets[p], 2),
            "contracts": contract_buckets[p],
        }
        for p in sorted_periods
    ]

    output: dict[str, Any] = {
        "portfolio_id": portfolio_id,
        "frequency": frequency,
        "cashflows": cashflows,
    }
    if currency:
        output["currency"] = currency

    if state.output == OutputFormat.JSON:
        print_json(output, state.pretty)
    else:
        table_rows: list[list[str]] = []
        for cf in cashflows:
            net = float(cf["net_payoff"])  # type: ignore[arg-type]
            contracts: list[str] = cf["contracts"]  # type: ignore[assignment]
            table_rows.append([str(cf["period"]), format_currency(net), ", ".join(contracts)])
        print_table(
            f"AGGREGATE: {portfolio_id} ({frequency})",
            ["Period", "Net Payoff", "Contracts"],
            table_rows,
            state.no_color,
        )


def _to_period(date_str: str, frequency: str) -> str:
    """Convert a YYYY-MM-DD date to a period string."""
    if frequency == "monthly":
        return date_str[:7]  # YYYY-MM
    if frequency == "quarterly":
        month = int(date_str[5:7])
        quarter = (month - 1) // 3 + 1
        return f"{date_str[:4]}-Q{quarter}"
    if frequency == "annual":
        return date_str[:4]
    return date_str  # daily
