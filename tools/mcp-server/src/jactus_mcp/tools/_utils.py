"""Shared utilities for JACTUS MCP tools."""

from pathlib import Path
from typing import Any

from jactus.core import ActusDateTime, ContractRole, ContractType
from jactus.core.types import (
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


def get_jactus_root() -> Path:
    """Get the root directory of the JACTUS repository."""
    current = Path(__file__).parent
    for _ in range(10):  # Safety limit
        if (current / "src" / "jactus").exists():
            return current
        current = current.parent

    # Fallback: 5 levels up
    return Path(__file__).parent.parent.parent.parent.parent


# All single-value ActusDateTime fields in ContractAttributes.
# Used by simulate.py and validation.py for string→ActusDateTime conversion.
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

# All list[ActusDateTime] fields in ContractAttributes.
ARRAY_DATE_FIELDS = [
    "analysis_dates",
    "array_pr_anchor",
    "array_ip_anchor",
    "array_rr_anchor",
]


# ---- Enum conversion utilities ----


def convert_enum(value: str, enum_class: type, aliases: dict[str, Any] | None = None) -> Any:
    """Convert a string to an enum member, checking aliases then name then value."""
    if aliases and value in aliases:
        return aliases[value]
    # Try by name (e.g. "E30360" for DayCountConvention.E30360)
    try:
        return enum_class[value]
    except KeyError:
        pass
    # Try by value (e.g. "30E360" for DayCountConvention.E30360 whose value is "30E360")
    try:
        return enum_class(value)
    except ValueError:
        pass
    # Return as-is; Pydantic will report the error
    return value


# Aliases map enum NAMES to members where name != value
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

# Enum fields and their classes (+ optional alias dicts)
ENUM_FIELDS: list[tuple[str, type, dict | None]] = [
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


def parse_datetime(value: str) -> ActusDateTime:
    """Parse an ISO date string to ActusDateTime.

    Supports: YYYY-MM-DD, YYYY-MM-DDTHH:MM:SS
    """
    if "T" in value:
        return ActusDateTime.from_iso(value)
    parts = value.split("-")
    return ActusDateTime(int(parts[0]), int(parts[1]), int(parts[2]))


def prepare_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Convert string values to proper JACTUS types.

    Handles all enum fields, date fields, and array date fields.
    Shared by simulate.py and validation.py for consistent conversion.
    """
    attrs = dict(attributes)

    # Convert all enum string values to proper enum members
    for field_name, enum_class, aliases in ENUM_FIELDS:
        if field_name in attrs and isinstance(attrs[field_name], str):
            attrs[field_name] = convert_enum(attrs[field_name], enum_class, aliases)

    # Convert single date strings to ActusDateTime
    for field in DATE_FIELDS:
        if field in attrs and isinstance(attrs[field], str):
            attrs[field] = parse_datetime(attrs[field])

    # Convert array date fields (list of date strings → list of ActusDateTime)
    for field in ARRAY_DATE_FIELDS:
        if field in attrs and isinstance(attrs[field], list):
            attrs[field] = [
                parse_datetime(v) if isinstance(v, str) else v
                for v in attrs[field]
            ]

    return attrs
