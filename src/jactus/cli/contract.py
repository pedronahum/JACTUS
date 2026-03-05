"""``jactus contract`` subcommands: list, schema, validate."""

from __future__ import annotations

import difflib
from typing import Any

import typer

from jactus.cli.output import OutputFormat, print_csv_output, print_error, print_json, print_table
from jactus.contracts import CONTRACT_REGISTRY
from jactus.core import ContractAttributes
from jactus.core.attributes import ATTRIBUTE_MAP

contract_app = typer.Typer(no_args_is_help=True)

# ---------------------------------------------------------------------------
# Contract metadata (mirrored from MCP tools/contracts.py)
# ---------------------------------------------------------------------------

DESCRIPTIONS: dict[str, str] = {
    "PAM": "Principal at Maturity",
    "LAM": "Linear Amortizer",
    "LAX": "Exotic Linear Amortizer",
    "NAM": "Negative Amortizer",
    "ANN": "Annuity",
    "CLM": "Call Money",
    "UMP": "Undefined Maturity Profile",
    "CSH": "Cash",
    "STK": "Stock",
    "COM": "Commodity",
    "FXOUT": "Foreign Exchange Outright",
    "OPTNS": "Options",
    "FUTUR": "Futures",
    "SWPPV": "Plain Vanilla Swap",
    "SWAPS": "Generic Swap",
    "CAPFL": "Cap/Floor",
    "CEG": "Credit Enhancement Guarantee",
    "CEC": "Credit Enhancement Collateral",
}

CATEGORIES: dict[str, str] = {
    "PAM": "principal",
    "LAM": "principal",
    "LAX": "principal",
    "NAM": "principal",
    "ANN": "principal",
    "CLM": "principal",
    "UMP": "non_principal",
    "CSH": "non_principal",
    "STK": "non_principal",
    "COM": "exotic",
    "FXOUT": "derivative",
    "OPTNS": "derivative",
    "FUTUR": "derivative",
    "SWPPV": "derivative",
    "SWAPS": "derivative",
    "CAPFL": "derivative",
    "CEG": "derivative",
    "CEC": "derivative",
}

SHORT_DESCRIPTIONS: dict[str, str] = {
    "PAM": "Interest-only loans and bonds. Principal repaid at maturity.",
    "LAM": "Fixed principal amortization loans.",
    "LAX": "Variable amortization schedules.",
    "NAM": "Loans with increasing principal balance.",
    "ANN": "Equal periodic payments covering principal and interest (mortgages).",
    "CLM": "Variable principal with on-demand repayment.",
    "UMP": "Revolving credit lines.",
    "CSH": "Money market accounts and escrow.",
    "STK": "Equity positions.",
    "COM": "Physical commodities and futures underliers.",
    "FXOUT": "FX forwards and swaps.",
    "OPTNS": "Call/Put options (European/American).",
    "FUTUR": "Standardized forward contracts.",
    "SWPPV": "Fixed vs floating interest rate swaps.",
    "SWAPS": "Cross-currency and multi-leg swaps.",
    "CAPFL": "Interest rate caps and floors.",
    "CEG": "Credit protection.",
    "CEC": "Collateral management.",
}

# ---------------------------------------------------------------------------
# Schema data (mirrored from MCP tools/contracts.py get_contract_schema)
# ---------------------------------------------------------------------------

BASE_REQUIRED: dict[str, str] = {
    "contract_type": "ContractType enum (e.g., 'PAM', 'LAM', 'SWPPV')",
    "contract_id": "str - Unique contract identifier",
    "status_date": "ActusDateTime - Analysis/valuation date (ISO: 'YYYY-MM-DD')",
    "contract_role": "ContractRole enum - RPA, RPL, RFL, PFL, BUY, SEL, LG, ST",
}

SPECIFIC_REQUIRED: dict[str, dict[str, str]] = {
    "PAM": {
        "initial_exchange_date": "ActusDateTime",
        "maturity_date": "ActusDateTime",
        "notional_principal": "float",
        "nominal_interest_rate": "float (recommended, defaults to 0)",
        "day_count_convention": "DayCountConvention (recommended, defaults to A360)",
    },
    "LAM": {
        "initial_exchange_date": "ActusDateTime",
        "notional_principal": "float (recommended)",
        "nominal_interest_rate": "float (recommended)",
        "maturity_date": "ActusDateTime (or principal_redemption_cycle)",
        "principal_redemption_cycle": "str (or maturity_date)",
        "next_principal_redemption_amount": "float (recommended)",
    },
    "LAX": {
        "initial_exchange_date": "ActusDateTime (recommended)",
        "maturity_date": "ActusDateTime (recommended)",
        "notional_principal": "float (recommended)",
        "nominal_interest_rate": "float (recommended)",
        "array_pr_cycle": "list[str]",
        "array_pr_next": "list[float]",
        "array_pr_anchor": "list[ActusDateTime]",
        "array_increase_decrease": "list[str] - 'INC' or 'DEC'",
    },
    "NAM": {
        "initial_exchange_date": "ActusDateTime",
        "notional_principal": "float (recommended)",
        "nominal_interest_rate": "float (recommended)",
        "maturity_date": "ActusDateTime (or principal_redemption_cycle)",
        "principal_redemption_cycle": "str (or maturity_date)",
        "next_principal_redemption_amount": "float (recommended)",
    },
    "ANN": {
        "initial_exchange_date": "ActusDateTime",
        "principal_redemption_cycle": "str - Payment frequency (e.g., '1M', '3M')",
        "notional_principal": "float (recommended)",
        "nominal_interest_rate": "float (recommended)",
        "maturity_date": "ActusDateTime (recommended)",
    },
    "CLM": {
        "initial_exchange_date": "ActusDateTime (recommended)",
        "notional_principal": "float (recommended)",
        "nominal_interest_rate": "float (recommended)",
        "interest_payment_cycle": "str (recommended)",
    },
    "UMP": {
        "initial_exchange_date": "ActusDateTime (recommended)",
        "notional_principal": "float (recommended)",
        "nominal_interest_rate": "float (recommended)",
    },
    "CSH": {"notional_principal": "float"},
    "STK": {
        "initial_exchange_date": "ActusDateTime (recommended)",
        "notional_principal": "float - Number of shares",
    },
    "COM": {
        "initial_exchange_date": "ActusDateTime (recommended)",
        "notional_principal": "float - Commodity value (recommended)",
    },
    "FXOUT": {
        "initial_exchange_date": "ActusDateTime",
        "maturity_date": "ActusDateTime",
        "notional_principal": "float",
        "delivery_settlement": "str ('D' or 'S')",
        "currency_2": "str - Second currency ISO code",
        "notional_principal_2": "float",
    },
    "OPTNS": {
        "initial_exchange_date": "ActusDateTime",
        "maturity_date": "ActusDateTime",
        "notional_principal": "float",
        "contract_structure": "str (JSON)",
        "option_type": "str ('C', 'P', 'CP')",
        "option_strike_1": "float",
        "option_exercise_type": "str ('E', 'A', 'B')",
    },
    "FUTUR": {
        "initial_exchange_date": "ActusDateTime",
        "maturity_date": "ActusDateTime",
        "notional_principal": "float",
        "contract_structure": "str (JSON)",
    },
    "SWPPV": {
        "initial_exchange_date": "ActusDateTime",
        "maturity_date": "ActusDateTime",
        "notional_principal": "float",
        "nominal_interest_rate": "float - Fixed leg rate",
        "nominal_interest_rate_2": "float - Initial floating rate",
        "interest_payment_cycle": "str (e.g., '3M', '6M')",
        "rate_reset_cycle": "str",
    },
    "SWAPS": {
        "initial_exchange_date": "ActusDateTime",
        "maturity_date": "ActusDateTime",
        "notional_principal": "float",
        "contract_structure": "str (JSON) - Swap legs config",
    },
    "CAPFL": {
        "initial_exchange_date": "ActusDateTime",
        "maturity_date": "ActusDateTime",
        "notional_principal": "float",
        "rate_reset_cycle": "str",
        "nominal_interest_rate": "float (recommended)",
        "rate_reset_cap": "float (recommended)",
        "rate_reset_floor": "float (recommended)",
    },
    "CEG": {
        "initial_exchange_date": "ActusDateTime",
        "maturity_date": "ActusDateTime",
        "notional_principal": "float",
        "contract_structure": "str (JSON)",
        "coverage": "float (recommended)",
    },
    "CEC": {
        "initial_exchange_date": "ActusDateTime",
        "maturity_date": "ActusDateTime",
        "notional_principal": "float",
        "contract_structure": "str (JSON)",
        "coverage": "float (recommended)",
    },
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@contract_app.command("list")
def list_contracts() -> None:
    """List all 18 supported ACTUS contract types."""
    from jactus.cli import get_state

    state = get_state()
    available = sorted(ct.name for ct in CONTRACT_REGISTRY)

    if state.output == OutputFormat.JSON:
        data = [
            {
                "type": ct,
                "name": DESCRIPTIONS.get(ct, ct),
                "category": CATEGORIES.get(ct, "unknown"),
                "description": SHORT_DESCRIPTIONS.get(ct, ""),
            }
            for ct in available
        ]
        print_json(data, state.pretty)
    elif state.output == OutputFormat.CSV:
        rows = [
            [ct, DESCRIPTIONS.get(ct, ct), CATEGORIES.get(ct, ""), SHORT_DESCRIPTIONS.get(ct, "")]
            for ct in available
        ]
        print_csv_output(["type", "name", "category", "description"], rows)
    else:
        rows = [
            [CATEGORIES.get(ct, "").replace("_", " ").title(), ct, DESCRIPTIONS.get(ct, ct)]
            for ct in available
        ]
        print_table(
            f"CONTRACT TYPES ({len(available)})", ["Category", "Type", "Name"], rows, state.no_color
        )


@contract_app.command("schema")
def schema(
    contract_type: str = typer.Option(..., "--type", help="ACTUS contract type (e.g., PAM, ANN)"),
    required_only: bool = typer.Option(False, "--required-only", help="Show only required fields"),
) -> None:
    """Print the JSON schema for a contract type's attributes."""
    from jactus.cli import get_state

    state = get_state()
    ct = contract_type.upper()

    if ct not in DESCRIPTIONS:
        print_error(f"Unknown contract type: {ct}")
        raise typer.Exit(code=1)

    required = {**BASE_REQUIRED}
    if ct in SPECIFIC_REQUIRED:
        required.update(SPECIFIC_REQUIRED[ct])

    if state.output == OutputFormat.JSON:
        data: dict[str, Any] = {
            "type": ct,
            "required_fields": [
                {"name": k, "type": v.split(" - ")[0].strip(), "description": v}
                for k, v in required.items()
            ],
        }
        if not required_only:
            data["optional_fields"] = _optional_fields(ct)
        print_json(data, state.pretty)
    else:
        rows = [[k, v] for k, v in required.items()]
        print_table(
            f"{ct} — Required Fields", ["Field", "Type / Description"], rows, state.no_color
        )
        if not required_only:
            opt = _optional_fields(ct)
            if opt:
                opt_rows = [[f["name"], f["description"]] for f in opt]
                print_table(
                    f"{ct} — Optional Fields", ["Field", "Description"], opt_rows, state.no_color
                )


@contract_app.command("validate")
def validate(
    contract_type: str = typer.Option(..., "--type", help="Contract type"),
    attrs: str | None = typer.Option(None, "--attrs", help="Inline JSON or file path"),  # noqa: UP007
    stdin: bool = typer.Option(False, "--stdin", help="Read attrs from stdin"),
) -> None:
    """Validate contract attributes without simulating. Exit 0 if valid, 3 if invalid."""
    from pydantic import ValidationError

    from jactus.cli import get_state, load_attrs, prepare_attributes

    state = get_state()

    try:
        raw = load_attrs(attrs, stdin)
    except Exception as e:
        print_error(str(e))
        raise typer.Exit(code=1) from None

    raw.setdefault("contract_type", contract_type.upper())
    raw.setdefault("contract_role", raw.get("contract_role", "RPA"))

    # Detect unknown fields
    warnings = _detect_unknown_fields(raw)

    try:
        prepared = prepare_attributes(raw)
        valid_fields = set(ContractAttributes.model_fields.keys())
        cleaned = {k: v for k, v in prepared.items() if k in valid_fields}
        contract_attrs = ContractAttributes(**cleaned)
    except ValidationError as e:
        errors = [
            {"field": " -> ".join(str(loc) for loc in err["loc"]), "message": err["msg"]}
            for err in e.errors()
        ]
        result = {
            "valid": False,
            "contract_type": contract_type.upper(),
            "errors": errors,
            "warnings": warnings,
        }
        if state.output == OutputFormat.JSON:
            print_json(result, state.pretty)
        else:
            print_error("Validation failed")
            for err in errors:
                print_error(f"  {err['field']}: {err['message']}")
        raise typer.Exit(code=3) from None
    except Exception as e:
        result = {
            "valid": False,
            "contract_type": contract_type.upper(),
            "errors": [{"field": "", "message": str(e)}],
            "warnings": warnings,
        }
        if state.output == OutputFormat.JSON:
            print_json(result, state.pretty)
        else:
            print_error(f"Validation error: {e}")
        raise typer.Exit(code=3) from None

    # Contract-specific checks
    ct = contract_type.upper()
    specific_warnings = _contract_specific_checks(contract_attrs, ct)
    warnings.extend(specific_warnings)

    result = {"valid": True, "contract_type": ct, "warnings": warnings or []}
    if state.output == OutputFormat.JSON:
        print_json(result, state.pretty)
    else:
        from jactus.cli.output import console

        console.print(f"[green]Valid[/green] {ct} contract")
        for w in warnings:
            console.print(f"  [yellow]Warning:[/yellow] {w}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_unknown_fields(attributes: dict[str, Any]) -> list[str]:
    """Detect unknown fields and suggest corrections."""
    valid_fields = set(ContractAttributes.model_fields.keys())
    warnings = []
    for key in attributes:
        if key in valid_fields:
            continue
        if key in ATTRIBUTE_MAP:
            warnings.append(
                f"Unknown field '{key}'. ACTUS short name — use '{ATTRIBUTE_MAP[key]}' instead."
            )
            continue
        close = difflib.get_close_matches(key, valid_fields, n=1, cutoff=0.6)
        if close:
            warnings.append(f"Unknown field '{key}'. Did you mean '{close[0]}'?")
        else:
            warnings.append(f"Unknown field '{key}' — it will be ignored.")
    return warnings


def _contract_specific_checks(attrs: ContractAttributes, ct: str) -> list[str]:
    """Run contract-specific validation checks, return warnings."""
    warnings: list[str] = []

    if ct in ("PAM", "LAM", "LAX", "NAM", "ANN", "CLM", "UMP"):
        if attrs.initial_exchange_date is None:
            warnings.append(f"{ct} typically requires initial_exchange_date")
        if attrs.notional_principal is None:
            warnings.append(f"{ct} typically requires notional_principal")

    if ct in ("PAM", "LAM", "NAM", "ANN"):
        if attrs.nominal_interest_rate is None:
            warnings.append(f"{ct} typically requires nominal_interest_rate")

    if ct == "PAM" and attrs.maturity_date is None:
        warnings.append("PAM requires maturity_date")

    if ct in ("LAM", "NAM"):
        if attrs.maturity_date is None and attrs.principal_redemption_cycle is None:
            warnings.append(f"{ct} requires either maturity_date or principal_redemption_cycle")

    if ct == "ANN" and attrs.principal_redemption_cycle is None:
        warnings.append("ANN requires principal_redemption_cycle")

    if ct == "SWPPV":
        if not attrs.interest_payment_cycle:
            warnings.append("SWPPV requires interest_payment_cycle")
        if not attrs.rate_reset_cycle:
            warnings.append("SWPPV requires rate_reset_cycle")
        if attrs.nominal_interest_rate_2 is None:
            warnings.append("SWPPV requires nominal_interest_rate_2")

    return warnings


def _optional_fields(ct: str) -> list[dict[str, str]]:
    """Return optional fields for a contract type, excluding those in SPECIFIC_REQUIRED."""
    specific = SPECIFIC_REQUIRED.get(ct, {})
    excluded = set(BASE_REQUIRED) | set(specific)

    all_optional: dict[str, str] = {
        "currency": "str - ISO currency code (default: USD)",
        "currency_2": "str - Second currency for FX/swap contracts",
        "contract_deal_date": "ActusDateTime",
        "purchase_date": "ActusDateTime - Secondary market purchase",
        "termination_date": "ActusDateTime - Early termination",
        "settlement_date": "ActusDateTime - Derivative settlement",
        "amortization_date": "ActusDateTime - ANN amortization end date",
        "nominal_interest_rate": "float - Nominal interest rate",
        "nominal_interest_rate_2": "float - Second rate for swaps",
        "notional_principal_2": "float - Second notional for FX/swaps",
        "next_principal_redemption_amount": "float - Next PR payment",
        "interest_payment_cycle": "str - e.g. '6M', '1Y'",
        "interest_payment_anchor": "ActusDateTime",
        "principal_redemption_cycle": "str",
        "principal_redemption_anchor": "ActusDateTime",
        "fee_payment_cycle": "str",
        "fee_payment_anchor": "ActusDateTime",
        "rate_reset_cycle": "str",
        "rate_reset_anchor": "ActusDateTime",
        "day_count_convention": "DayCountConvention: AA, A360, A365, E30360ISDA, E30360, B30360, BUS252",
        "business_day_convention": "BusinessDayConvention: NULL, SCF, SCMF, CSF, CSMF, SCP, SCMP, CSP, CSMP",
        "end_of_month_convention": "EndOfMonthConvention: EOM, SD",
        "calendar": "Calendar: NO_CALENDAR, MONDAY_TO_FRIDAY, TARGET, US_NYSE, UK_SETTLEMENT",
        "rate_reset_market_object": "str - Market reference for rate reset",
        "rate_reset_spread": "float",
        "rate_reset_floor": "float",
        "rate_reset_cap": "float",
        "fee_rate": "float",
        "fee_basis": "FeeBasis: A (absolute), N (notional percentage)",
        "prepayment_effect": "PrepaymentEffect: N, A, M",
        "scaling_effect": "ScalingEffect: 000, I00, 0N0, IN0, 00M, I0M, 0NM, INM",
        "option_type": "str: 'C' (call), 'P' (put), 'CP' (collar)",
        "option_strike_1": "float",
        "option_strike_2": "float",
        "option_exercise_type": "str: 'E', 'A', 'B'",
        "contract_structure": "str (JSON) - Reference to child contracts",
        "coverage": "float - Coverage ratio",
        "accrued_interest": "float",
        "contract_performance": "ContractPerformance: PF, DL, DQ, DF",
    }

    return [{"name": k, "description": v} for k, v in all_optional.items() if k not in excluded]
