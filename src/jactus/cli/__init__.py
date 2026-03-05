"""JACTUS command-line interface.

Built with Typer. Entry point: ``jactus = "jactus.cli:app"``
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import typer

from jactus.cli.output import OutputFormat

# ---------------------------------------------------------------------------
# Shared CLI state (populated by the root callback, read by subcommands)
# ---------------------------------------------------------------------------


@dataclass
class CLIState:
    """Mutable state shared across all CLI commands via typer.Context.obj."""

    output: OutputFormat = OutputFormat.TEXT
    pretty: bool = True
    no_color: bool = False
    log_level: str = "WARNING"


_state = CLIState()


def get_state() -> CLIState:
    """Return the global CLI state."""
    return _state


# ---------------------------------------------------------------------------
# Attribute loading helpers (shared by simulate, validate, risk, …)
# ---------------------------------------------------------------------------


def load_attrs(attrs: str | None, stdin: bool) -> dict[str, Any]:
    """Load contract attributes from inline JSON, a file path, or stdin."""
    if stdin:
        return json.load(sys.stdin)  # type: ignore[no-any-return]
    if attrs is not None:
        p = Path(attrs)
        if p.is_file():
            return json.loads(p.read_text())  # type: ignore[no-any-return]
        # Treat as inline JSON string
        return json.loads(attrs)  # type: ignore[no-any-return]
    raise typer.BadParameter("Provide --attrs (inline JSON or file path) or --stdin")


# ---------------------------------------------------------------------------
# Attribute preparation (string → enum / date conversion)
# Mirrors tools/mcp-server/src/jactus_mcp/tools/_utils.py
# ---------------------------------------------------------------------------

from jactus.core import ActusDateTime, ContractRole, ContractType  # noqa: E402
from jactus.core.types import (  # noqa: E402
    BusinessDayConvention,
    Calendar,
    ContractPerformance,
    DayCountConvention,
    EndOfMonthConvention,
    FeeBasis,
    InterestCalculationBase,
    PrepaymentEffect,
    ScalingEffect,
)

DATE_FIELDS = [
    "status_date",
    "contract_deal_date",
    "initial_exchange_date",
    "maturity_date",
    "purchase_date",
    "termination_date",
    "settlement_date",
    "option_exercise_end_date",
    "exercise_date",
    "dividend_anchor",
    "interest_payment_anchor",
    "interest_capitalization_end_date",
    "principal_redemption_anchor",
    "fee_payment_anchor",
    "rate_reset_anchor",
    "scaling_index_anchor",
    "interest_calculation_base_anchor",
    "amortization_date",
]

ARRAY_DATE_FIELDS = [
    "analysis_dates",
    "array_pr_anchor",
    "array_ip_anchor",
    "array_rr_anchor",
]

DCC_ALIASES: dict[str, DayCountConvention] = {
    "30E360": DayCountConvention.E30360,
    "30E360ISDA": DayCountConvention.E30360ISDA,
    "30360": DayCountConvention.B30360,
}

SCALING_ALIASES: dict[str, ScalingEffect] = {
    "000": ScalingEffect.S000,
    "0N0": ScalingEffect.S0N0,
    "00M": ScalingEffect.S00M,
    "0NM": ScalingEffect.S0NM,
}

ENUM_FIELDS: list[tuple[str, type, dict[str, Any] | None]] = [
    ("contract_type", ContractType, None),
    ("contract_role", ContractRole, None),
    ("day_count_convention", DayCountConvention, DCC_ALIASES),
    ("business_day_convention", BusinessDayConvention, None),
    ("end_of_month_convention", EndOfMonthConvention, None),
    ("calendar", Calendar, None),
    ("fee_basis", FeeBasis, None),
    ("prepayment_effect", PrepaymentEffect, None),
    ("scaling_effect", ScalingEffect, SCALING_ALIASES),
    ("contract_performance", ContractPerformance, None),
    ("credit_event_type", ContractPerformance, None),
    ("interest_calculation_base", InterestCalculationBase, None),
]


def _convert_enum(value: str, enum_class: Any, aliases: dict[str, Any] | None = None) -> Any:
    if aliases and value in aliases:
        return aliases[value]
    try:
        return enum_class[value]
    except KeyError:
        pass
    try:
        return enum_class(value)
    except ValueError:
        pass
    return value


def parse_datetime(value: str) -> ActusDateTime:
    """Parse an ISO date string to ActusDateTime."""
    if "T" in value:
        return ActusDateTime.from_iso(value)
    parts = value.split("-")
    return ActusDateTime(int(parts[0]), int(parts[1]), int(parts[2]))


def prepare_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Convert string values in *attributes* to proper JACTUS types."""
    attrs = dict(attributes)
    for field_name, enum_class, aliases in ENUM_FIELDS:
        if field_name in attrs and isinstance(attrs[field_name], str):
            attrs[field_name] = _convert_enum(attrs[field_name], enum_class, aliases)
    for f in DATE_FIELDS:
        if f in attrs and isinstance(attrs[f], str):
            attrs[f] = parse_datetime(attrs[f])
    for f in ARRAY_DATE_FIELDS:
        if f in attrs and isinstance(attrs[f], list):
            attrs[f] = [parse_datetime(v) if isinstance(v, str) else v for v in attrs[f]]
    return attrs


# ---------------------------------------------------------------------------
# Version helper
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    if value:
        from jactus import __version__

        print(f"jactus {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Root Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="jactus",
    help="JACTUS — JAX ACTUS Financial Contract Simulation",
    invoke_without_command=True,
    no_args_is_help=True,
)


@app.callback()
def main(
    ctx: typer.Context,
    output: str | None = typer.Option(None, "--output", help="Output format: text, json, csv"),  # noqa: UP007
    pretty: bool = typer.Option(True, "--pretty/--no-pretty", help="Pretty-print JSON"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable ANSI color"),
    log_level: str = typer.Option("WARNING", "--log-level", help="Log verbosity"),
    version: bool | None = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True
    ),  # noqa: UP007
) -> None:
    """JACTUS — JAX ACTUS Financial Contract Simulation."""
    # Resolve output format
    if output is not None:
        try:
            _state.output = OutputFormat(output)
        except ValueError:
            raise typer.BadParameter(
                f"Invalid output format: {output}. Use text, json, or csv"
            ) from None
    else:
        _state.output = OutputFormat.TEXT if sys.stdout.isatty() else OutputFormat.JSON

    _state.pretty = pretty
    _state.no_color = no_color
    _state.log_level = log_level

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.WARNING), stream=sys.stderr
    )


# ---------------------------------------------------------------------------
# Register subcommand groups (imported lazily to avoid circular deps)
# ---------------------------------------------------------------------------

from jactus.cli.contract import contract_app  # noqa: E402
from jactus.cli.docs import docs_app  # noqa: E402
from jactus.cli.observer import observer_app  # noqa: E402
from jactus.cli.portfolio import portfolio_app  # noqa: E402
from jactus.cli.risk import risk_app  # noqa: E402
from jactus.cli.simulate import simulate_cmd  # noqa: E402

app.add_typer(contract_app, name="contract", help="Contract type discovery and validation")
app.add_typer(risk_app, name="risk", help="Risk sensitivity metrics (DV01, duration, convexity)")
app.add_typer(portfolio_app, name="portfolio", help="Multi-contract portfolio simulation")
app.add_typer(observer_app, name="observer", help="Risk factor observer utilities")
app.add_typer(docs_app, name="docs", help="Documentation search")
app.command(name="simulate", help="Run a full contract simulation")(simulate_cmd)
