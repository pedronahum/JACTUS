"""Contract discovery and introspection tools."""

import json
from typing import Any

from jactus.contracts import CONTRACT_REGISTRY
from jactus.core import ContractType, EventType, ContractAttributes


def list_contracts() -> dict[str, Any]:
    """List all available ACTUS contract types in JACTUS.

    Returns:
        Dictionary with contract types organized by category.
    """
    # Organize contracts by category
    principal = ["PAM", "LAM", "LAX", "NAM", "ANN", "CLM"]
    non_principal = ["UMP", "CSH", "STK"]
    exotic = ["COM"]
    derivative = ["FXOUT", "OPTNS", "FUTUR", "SWPPV", "SWAPS", "CAPFL", "CEG", "CEC"]

    available = [ct.name for ct in CONTRACT_REGISTRY.keys()]

    return {
        "total_contracts": len(available),
        "categories": {
            "principal": [ct for ct in principal if ct in available],
            "non_principal": [ct for ct in non_principal if ct in available],
            "exotic": [ct for ct in exotic if ct in available],
            "derivative": [ct for ct in derivative if ct in available],
        },
        "all_contracts": sorted(available),
    }


def get_contract_info(contract_type: str) -> dict[str, Any]:
    """Get detailed information about a specific contract type.

    Args:
        contract_type: ACTUS contract type code (e.g., "PAM", "SWPPV")

    Returns:
        Dictionary with contract information including description and category.
    """
    descriptions = {
        "PAM": "Principal at Maturity - Interest-only loans and bonds",
        "LAM": "Linear Amortizer - Fixed principal amortization loans",
        "LAX": "Exotic Linear Amortizer - Variable amortization schedules",
        "NAM": "Negative Amortizer - Loans with increasing principal balance",
        "ANN": "Annuity - Mortgages and equal payment loans",
        "CLM": "Call Money - Variable principal with on-demand repayment",
        "UMP": "Undefined Maturity Profile - Revolving credit lines",
        "CSH": "Cash - Money market accounts and escrow",
        "STK": "Stock - Equity positions",
        "COM": "Commodity - Physical commodities and futures underliers",
        "FXOUT": "Foreign Exchange Outright - FX forwards and swaps",
        "OPTNS": "Options - Call/Put options (European/American)",
        "FUTUR": "Futures - Standardized forward contracts",
        "SWPPV": "Plain Vanilla Swap - Fixed vs floating interest rate swaps",
        "SWAPS": "Generic Swap - Cross-currency and multi-leg swaps",
        "CAPFL": "Cap/Floor - Interest rate caps and floors",
        "CEG": "Credit Enhancement Guarantee - Credit protection",
        "CEC": "Credit Enhancement Collateral - Collateral management",
    }

    categories = {
        "PAM": "principal", "LAM": "principal", "LAX": "principal",
        "NAM": "principal", "ANN": "principal", "CLM": "principal",
        "UMP": "non_principal", "CSH": "non_principal", "STK": "non_principal",
        "COM": "exotic",
        "FXOUT": "derivative", "OPTNS": "derivative", "FUTUR": "derivative",
        "SWPPV": "derivative", "SWAPS": "derivative", "CAPFL": "derivative",
        "CEG": "derivative", "CEC": "derivative",
    }

    try:
        ct = ContractType[contract_type]

        if ct not in CONTRACT_REGISTRY:
            return {
                "error": f"Contract type {contract_type} not implemented in JACTUS",
                "available": list_contracts()["all_contracts"],
            }

        contract_class = CONTRACT_REGISTRY[ct]

        return {
            "contract_type": contract_type,
            "description": descriptions.get(contract_type, "No description available"),
            "category": categories.get(contract_type, "unknown"),
            "implemented": True,
            "class_name": contract_class.__name__,
            "module": contract_class.__module__,
        }

    except KeyError:
        return {
            "error": f"Unknown contract type: {contract_type}",
            "available": list_contracts()["all_contracts"],
        }


def get_contract_schema(contract_type: str) -> dict[str, Any]:
    """Get required and optional parameters for a contract type.

    Args:
        contract_type: ACTUS contract type code

    Returns:
        Dictionary with required fields, optional fields, and their types.
    """
    # Get Pydantic schema
    schema = ContractAttributes.model_json_schema()

    # Base required fields (always needed)
    base_required = {
        "contract_type": "ContractType enum",
        "status_date": "ActusDateTime - Contract status date",
        "contract_role": "ContractRole enum - RPA (lender) or RPL (borrower)",
    }

    # Contract-specific required fields (all 18 types)
    specific_required = {
        # Principal contracts
        "PAM": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "nominal_interest_rate": "float",
            "day_count_convention": "DayCountConvention enum",
        },
        "LAM": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "nominal_interest_rate": "float",
            "next_principal_redemption_amount": "float - Fixed principal payment amount",
            "principal_redemption_cycle": "str (e.g., '1M', '3M')",
        },
        "LAX": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "nominal_interest_rate": "float",
            "array_pr_cycle": "list[str] - Array of PR cycles for exotic amortization",
            "array_pr_next": "list[float] - Array of next PR amounts",
        },
        "NAM": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "nominal_interest_rate": "float",
            "next_principal_redemption_amount": "float",
            "principal_redemption_cycle": "str (e.g., '1M', '3M')",
        },
        "ANN": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "nominal_interest_rate": "float",
            "interest_payment_cycle": "str",
            "principal_redemption_cycle": "str",
        },
        "CLM": {
            "initial_exchange_date": "ActusDateTime",
            "notional_principal": "float",
            "nominal_interest_rate": "float",
            "interest_payment_cycle": "str",
        },
        # Non-principal contracts
        "UMP": {
            "initial_exchange_date": "ActusDateTime",
            "notional_principal": "float",
            "nominal_interest_rate": "float",
        },
        "CSH": {
            "notional_principal": "float",
        },
        "STK": {
            "initial_exchange_date": "ActusDateTime",
            "notional_principal": "float - Number of shares or position value",
        },
        # Exotic contracts
        "COM": {
            "initial_exchange_date": "ActusDateTime",
            "notional_principal": "float - Commodity value",
        },
        # Derivative contracts
        "FXOUT": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "delivery_settlement": "str ('D' for delivery/net or 'S' for settlement/gross)",
            "currency_2": "str - Second currency ISO code",
            "notional_principal_2": "float - Second currency notional",
        },
        "OPTNS": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "option_type": "str ('C' for call, 'P' for put)",
            "option_strike_1": "float",
            "option_exercise_type": "str ('E' European, 'A' American, 'B' Bermudan)",
            "contract_structure": "str (JSON) - Reference to underlier contract",
        },
        "FUTUR": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "contract_structure": "str (JSON) - Reference to underlier contract",
        },
        "SWPPV": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "nominal_interest_rate": "float - Fixed leg rate",
            "interest_payment_cycle": "str - Payment frequency (e.g., '3M', '6M')",
            "rate_reset_cycle": "str - Floating leg reset frequency",
        },
        "SWAPS": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "contract_structure": "str (JSON) - Defines swap legs configuration",
        },
        "CAPFL": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "nominal_interest_rate": "float",
            "rate_reset_cap": "float - Interest rate cap level",
            "rate_reset_floor": "float - Interest rate floor level",
            "rate_reset_cycle": "str - Reset frequency",
        },
        "CEG": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "coverage": "float - Coverage ratio",
            "contract_structure": "str (JSON) - Reference to guaranteed contract",
        },
        "CEC": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "coverage": "float - Coverage ratio",
            "contract_structure": "str (JSON) - Reference to collateral contract",
        },
    }

    required = {**base_required}
    if contract_type in specific_required:
        required.update(specific_required[contract_type])

    # Common optional fields
    optional = {
        "contract_id": "str - Unique identifier",
        "currency": "str - ISO currency code (default: USD)",
        "interest_payment_cycle": "str - Interest payment cycle, IPCL (e.g., '6M', '1Y')",
        "day_count_convention": "DayCountConvention enum",
        "business_day_convention": "BusinessDayConvention enum",
        "premium_discount_at_ied": "float",
        "interest_payment_anchor": "ActusDateTime - Interest payment anchor date (IPANX)",
        "principal_redemption_anchor": "ActusDateTime - Principal redemption anchor date (PRANX)",
    }

    return {
        "contract_type": contract_type,
        "required_fields": required,
        "optional_fields": optional,
        "example_usage": f"""
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver

attrs = ContractAttributes(
    contract_type=ContractType.{contract_type},
    contract_role=ContractRole.RPA,
    # Add required fields listed above
)

rf_observer = ConstantRiskFactorObserver()
contract = create_contract(attrs, rf_observer)
result = contract.simulate()
""",
    }


def get_event_types() -> dict[str, Any]:
    """List all ACTUS event types.

    Returns:
        Dictionary with event types and descriptions.
    """
    event_descriptions = {
        "IED": "Initial Exchange Date - Contract inception",
        "IP": "Interest Payment - Periodic interest payment",
        "IPCI": "Interest Capitalization - Interest added to principal",
        "PR": "Principal Redemption - Partial principal repayment",
        "MD": "Maturity Date - Contract maturity and final payment",
        "PP": "Principal Prepayment - Unscheduled principal payment",
        "PY": "Penalty Payment - Penalty or fee payment",
        "FP": "Fee Payment - Periodic fee payment",
        "PRD": "Purchase/Redemption - Asset purchase or redemption",
        "TD": "Termination Date - Contract termination",
        "DV": "Dividend - Dividend payment (stocks)",
        "RR": "Rate Reset - Interest rate reset",
        "RRF": "Rate Reset with Fixing - Rate reset with fixing period",
        "SC": "Scaling Index Revision - Notional/interest scaling",
        "AD": "Monitoring Date - Account monitoring",
        "XD": "Exercise Date - Option exercise",
    }

    all_events = [e.name for e in EventType]

    return {
        "total_events": len(all_events),
        "event_types": {
            event: event_descriptions.get(event, "No description")
            for event in sorted(all_events)
        },
    }
