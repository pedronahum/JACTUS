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
        "contract_id": "str - Unique contract identifier",
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
            "nominal_interest_rate_2": "float - Initial floating leg rate (IPNR2)",
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
        "currency": "str - ISO currency code (default: USD)",
        "contract_deal_date": "ActusDateTime (CDD)",
        # Schedule cycles and anchors
        "interest_payment_cycle": "str - Interest payment cycle (IPCL), e.g. '6M', '1Y'",
        "interest_payment_anchor": "ActusDateTime - Interest payment anchor (IPANX)",
        "interest_capitalization_end_date": "ActusDateTime (IPCED)",
        "principal_redemption_cycle": "str - Principal redemption cycle (PRCL)",
        "principal_redemption_anchor": "ActusDateTime (PRANX)",
        "fee_payment_cycle": "str - Fee payment cycle (FECL)",
        "fee_payment_anchor": "ActusDateTime (FEANX)",
        "rate_reset_cycle": "str - Rate reset cycle (RRCL)",
        "rate_reset_anchor": "ActusDateTime (RRANX)",
        "scaling_index_cycle": "str - Scaling index cycle (SCCL)",
        "scaling_index_anchor": "ActusDateTime (SCANX)",
        # Conventions
        "day_count_convention": "DayCountConvention enum (DCC)",
        "business_day_convention": "BusinessDayConvention enum (BDC)",
        "end_of_month_convention": "EndOfMonthConvention enum (EOMC)",
        "calendar": "Calendar enum (CLDR)",
        # Rate reset
        "rate_reset_market_object": "str - Market reference for rate reset (RRMO)",
        "rate_reset_multiplier": "float (RRMLT)",
        "rate_reset_spread": "float (RRSP)",
        "rate_reset_floor": "float (RRLF)",
        "rate_reset_cap": "float (RRLC)",
        "rate_reset_next": "float (RRNXT)",
        # Fees
        "fee_rate": "float (FER)",
        "fee_basis": "FeeBasis enum (FEB)",
        "fee_accrued": "float (FEAC)",
        # Other
        "premium_discount_at_ied": "float (PDIED)",
        "accrued_interest": "float (IPAC)",
        "purchase_date": "ActusDateTime (PRD)",
        "price_at_purchase_date": "float (PPRD)",
        "termination_date": "ActusDateTime (TD)",
        "price_at_termination_date": "float (PTD)",
        "settlement_date": "ActusDateTime (STD)",
        "amortization_date": "ActusDateTime (AMD)",
        "penalty_type": "str (PYTP)",
        "penalty_rate": "float (PYRT)",
        "nominal_interest_rate_2": "float - Second nominal interest rate for swaps (IPNR2)",
        "delivery_settlement": "str - Settlement type: 'D' (delivery/net) or 'S' (settlement/gross)",
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

rf_observer = ConstantRiskFactorObserver(constant_value=0.0)
contract = create_contract(attrs, rf_observer)
result = contract.simulate()
""",
    }


def list_risk_factor_observers() -> dict[str, Any]:
    """List all available risk factor observer types with usage guidance.

    Returns:
        Dictionary with observer types, descriptions, and usage guidance.
    """
    return {
        "observers": {
            "ConstantRiskFactorObserver": {
                "description": "Returns the same constant value for all risk factors",
                "use_case": "Fixed-rate contracts, testing, simple scenarios",
                "mcp_param": "constant_value",
                "example": {"constant_value": 0.05},
            },
            "DictRiskFactorObserver": {
                "description": "Maps risk factor identifiers to fixed values",
                "use_case": "Contracts needing different fixed values per risk factor identifier",
                "mcp_param": "risk_factors",
                "example": {"risk_factors": {"LIBOR-3M": 0.05, "USD/EUR": 1.18}},
            },
            "TimeSeriesRiskFactorObserver": {
                "description": "Maps identifiers to time series with step or linear interpolation",
                "use_case": "Floating-rate contracts with rate resets that need time-varying data",
                "mcp_param": "time_series",
                "interpolation_note": (
                    "Step vs linear only differ when a query date falls BETWEEN data points. "
                    "If rate reset dates align exactly with data points, both modes give identical "
                    "results. To see the difference, place data points at different dates than "
                    "the reset dates (e.g., quarterly data with monthly resets)."
                ),
                "example": {
                    "time_series": {
                        "LIBOR-3M": [
                            ["2024-01-01", 0.04],
                            ["2024-07-01", 0.045],
                            ["2025-01-01", 0.05],
                        ]
                    },
                    "interpolation": "step",
                },
            },
            "CurveRiskFactorObserver": {
                "description": "Yield/rate curves keyed by tenor for term structure modeling",
                "use_case": "Term structure modeling, yield curve-dependent pricing",
                "python_only": True,
            },
            "CompositeRiskFactorObserver": {
                "description": "Chains multiple observers with fallback behavior",
                "use_case": "Complex scenarios needing different data sources per risk factor",
                "python_only": True,
            },
            "CallbackRiskFactorObserver": {
                "description": "Delegates to user-provided Python callables",
                "use_case": "Custom pricing models, external data integration",
                "python_only": True,
            },
            "JaxRiskFactorObserver": {
                "description": "Integer-indexed, fully JAX-compatible for jit/grad/vmap",
                "use_case": "Automatic differentiation, sensitivity analysis, batch scenarios",
                "python_only": True,
            },
        },
        "guidance": {
            "simple_fixed_rate": (
                "Use constant_value=0.0 (default) for contracts with fixed rates "
                "and no market-dependent features"
            ),
            "multiple_market_factors": (
                "Use risk_factors dict when the contract references specific market "
                "objects (e.g., rate_reset_market_object)"
            ),
            "time_varying_rates": (
                "Use time_series for floating-rate contracts with rate resets "
                "that need temporal market data"
            ),
            "advanced": (
                "For yield curves, composites, callbacks, or JAX gradients, "
                "use the Python API directly"
            ),
        },
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
