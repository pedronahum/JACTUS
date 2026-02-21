"""Shared utilities for JACTUS MCP tools."""

from pathlib import Path


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
