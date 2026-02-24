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

    # Composite contracts that require child_contracts parameter
    _requires_child_contracts = {"CAPFL", "SWAPS", "CEG", "CEC"}

    try:
        ct = ContractType[contract_type]

        if ct not in CONTRACT_REGISTRY:
            return {
                "error": f"Contract type {contract_type} not implemented in JACTUS",
                "available": list_contracts()["all_contracts"],
            }

        contract_class = CONTRACT_REGISTRY[ct]

        result = {
            "contract_type": contract_type,
            "description": descriptions.get(contract_type, "No description available"),
            "category": categories.get(contract_type, "unknown"),
            "implemented": True,
            "mcp_simulatable": True,
            "class_name": contract_class.__name__,
            "module": contract_class.__module__,
        }

        if contract_type in _requires_child_contracts:
            result["requires_child_contracts"] = True
            result["mcp_note"] = (
                f"{contract_type} is a composite contract — pass a child_contracts "
                f"dict to jactus_simulate_contract. Each child is simulated first, "
                f"then its results feed into the {contract_type} parent. "
                f"Use jactus_get_contract_schema('{contract_type}') for the format."
            )

        return result

    except KeyError:
        return {
            "error": f"Unknown contract type: {contract_type}",
            "available": list_contracts()["all_contracts"],
        }


def _child_observer_example(contract_type: str) -> str:
    """Return a concrete Python example for a child-observer contract type."""
    examples = {
        "SWAPS": '''
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver
from jactus.observers.child_contract import SimulatedChildContractObserver

rf_observer = ConstantRiskFactorObserver(constant_value=0.05)

# Step 1: Create and simulate the two swap legs as separate contracts
leg1_attrs = ContractAttributes(
    contract_id="LEG1", contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=1_000_000.0,
    nominal_interest_rate=0.04,
    interest_payment_cycle="6M",
)
leg2_attrs = ContractAttributes(
    contract_id="LEG2", contract_type=ContractType.PAM,
    contract_role=ContractRole.RPL,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=1_000_000.0,
    nominal_interest_rate=0.035,
    interest_payment_cycle="6M",
)
leg1_result = create_contract(leg1_attrs, rf_observer).simulate()
leg2_result = create_contract(leg2_attrs, rf_observer).simulate()

# Step 2: Register child simulation results in a ChildContractObserver
child_obs = SimulatedChildContractObserver()
child_obs.register_simulation("LEG1", leg1_result.events, leg1_attrs, leg1_result.initial_state)
child_obs.register_simulation("LEG2", leg2_result.events, leg2_attrs, leg2_result.initial_state)

# Step 3: Create the SWAPS parent contract referencing both legs
swap_attrs = ContractAttributes(
    contract_id="SWAP-001", contract_type=ContractType.SWAPS,
    contract_role=ContractRole.RFL,
    status_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    contract_structure=\'{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}\',
)

# Step 4: Simulate — pass BOTH the risk factor observer AND the child observer
swap = create_contract(swap_attrs, rf_observer, child_obs)
result = swap.simulate()
for event in result.events:
    print(f"{event.event_type.name:6s}  {event.event_time}  payoff={float(event.payoff):>12.2f}")
''',
        "CAPFL": '''
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver
from jactus.observers.child_contract import SimulatedChildContractObserver

rf_observer = ConstantRiskFactorObserver(constant_value=0.06)

# Step 1: Create and simulate the underlier loan
loan_attrs = ContractAttributes(
    contract_id="LOAN-001", contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=1_000_000.0,
    nominal_interest_rate=0.05,
    rate_reset_cycle="3M",
    rate_reset_market_object="LIBOR-3M",
)
loan_result = create_contract(loan_attrs, rf_observer).simulate()

# Step 2: Register child simulation results
child_obs = SimulatedChildContractObserver()
child_obs.register_simulation("LOAN-001", loan_result.events, loan_attrs, loan_result.initial_state)

# Step 3: Create the CAPFL (cap) contract referencing the underlier
cap_attrs = ContractAttributes(
    contract_id="CAP-001", contract_type=ContractType.CAPFL,
    contract_role=ContractRole.BUY,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=1_000_000.0,
    rate_reset_cycle="3M",
    rate_reset_cap=0.055,  # Cap rate at 5.5%
)

# Step 4: Simulate with both observers
cap = create_contract(cap_attrs, rf_observer, child_obs)
result = cap.simulate()
for event in result.events:
    print(f"{event.event_type.name:6s}  {event.event_time}  payoff={float(event.payoff):>12.2f}")
''',
        "CEG": '''
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver
from jactus.observers.child_contract import SimulatedChildContractObserver

rf_observer = ConstantRiskFactorObserver(constant_value=0.0)

# Step 1: Create and simulate the covered (guaranteed) loan
loan_attrs = ContractAttributes(
    contract_id="LOAN-001", contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=500_000.0,
    nominal_interest_rate=0.04,
)
loan_result = create_contract(loan_attrs, rf_observer).simulate()

# Step 2: Register child simulation results
child_obs = SimulatedChildContractObserver()
child_obs.register_simulation("LOAN-001", loan_result.events, loan_attrs, loan_result.initial_state)

# Step 3: Create the CEG (guarantee) contract
ceg_attrs = ContractAttributes(
    contract_id="CEG-001", contract_type=ContractType.CEG,
    contract_role=ContractRole.BUY,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=500_000.0,
    coverage=1.0,  # 100% coverage
    contract_structure=\'{"CoveredContract": "LOAN-001"}\',
)

# Step 4: Simulate with both observers
ceg = create_contract(ceg_attrs, rf_observer, child_obs)
result = ceg.simulate()
for event in result.events:
    print(f"{event.event_type.name:6s}  {event.event_time}  payoff={float(event.payoff):>12.2f}")
''',
        "CEC": '''
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver
from jactus.observers.child_contract import SimulatedChildContractObserver

rf_observer = ConstantRiskFactorObserver(constant_value=0.0)

# Step 1: Create and simulate the covered loan (the contract being collateralized)
loan_attrs = ContractAttributes(
    contract_id="LOAN-001", contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=500_000.0,
    nominal_interest_rate=0.04,
)
loan_result = create_contract(loan_attrs, rf_observer).simulate()

# Step 2: Register child simulation results
child_obs = SimulatedChildContractObserver()
child_obs.register_simulation("LOAN-001", loan_result.events, loan_attrs, loan_result.initial_state)

# Step 3: Create the CEC (collateral) contract
cec_attrs = ContractAttributes(
    contract_id="CEC-001", contract_type=ContractType.CEC,
    contract_role=ContractRole.BUY,
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 1),
    maturity_date=ActusDateTime(2029, 1, 1),
    notional_principal=500_000.0,
    coverage=1.0,  # 100% collateral coverage
    contract_structure=\'{"CoveredContract": "LOAN-001"}\',
)

# Step 4: Simulate with both observers
cec = create_contract(cec_attrs, rf_observer, child_obs)
result = cec.simulate()
for event in result.events:
    print(f"{event.event_type.name:6s}  {event.event_time}  payoff={float(event.payoff):>12.2f}")
''',
    }
    return examples.get(contract_type, "")


def _child_contract_mcp_example(contract_type: str) -> dict[str, Any]:
    """Return an MCP-friendly JSON example for a composite contract type."""
    examples = {
        "SWAPS": {
            "attributes": {
                "contract_type": "SWAPS",
                "contract_id": "SWAP-001",
                "contract_role": "RFL",
                "status_date": "2024-01-01",
                "maturity_date": "2029-01-01",
                "contract_structure": '{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
            },
            "child_contracts": {
                "LEG1": {
                    "contract_type": "PAM", "contract_id": "LEG1",
                    "contract_role": "RPA", "status_date": "2024-01-01",
                    "initial_exchange_date": "2024-01-01", "maturity_date": "2029-01-01",
                    "notional_principal": 1000000.0, "nominal_interest_rate": 0.04,
                    "interest_payment_cycle": "6M",
                },
                "LEG2": {
                    "contract_type": "PAM", "contract_id": "LEG2",
                    "contract_role": "RPL", "status_date": "2024-01-01",
                    "initial_exchange_date": "2024-01-01", "maturity_date": "2029-01-01",
                    "notional_principal": 1000000.0, "nominal_interest_rate": 0.035,
                    "interest_payment_cycle": "6M",
                },
            },
        },
        "CAPFL": {
            "attributes": {
                "contract_type": "CAPFL",
                "contract_id": "CAP-001",
                "contract_role": "BUY",
                "status_date": "2024-01-01",
                "initial_exchange_date": "2024-01-01",
                "maturity_date": "2029-01-01",
                "notional_principal": 1000000.0,
                "rate_reset_cycle": "3M",
                "rate_reset_cap": 0.055,
            },
            "child_contracts": {
                "LOAN-001": {
                    "contract_type": "PAM", "contract_id": "LOAN-001",
                    "contract_role": "RPA", "status_date": "2024-01-01",
                    "initial_exchange_date": "2024-01-01", "maturity_date": "2029-01-01",
                    "notional_principal": 1000000.0, "nominal_interest_rate": 0.05,
                    "rate_reset_cycle": "3M",
                },
            },
            "constant_value": 0.06,
        },
        "CEG": {
            "attributes": {
                "contract_type": "CEG",
                "contract_id": "CEG-001",
                "contract_role": "BUY",
                "status_date": "2024-01-01",
                "initial_exchange_date": "2024-01-01",
                "maturity_date": "2029-01-01",
                "notional_principal": 500000.0,
                "coverage": 1.0,
                "contract_structure": '{"CoveredContract": "LOAN-001"}',
            },
            "child_contracts": {
                "LOAN-001": {
                    "contract_type": "PAM", "contract_id": "LOAN-001",
                    "contract_role": "RPA", "status_date": "2024-01-01",
                    "initial_exchange_date": "2024-01-01", "maturity_date": "2029-01-01",
                    "notional_principal": 500000.0, "nominal_interest_rate": 0.04,
                },
            },
        },
        "CEC": {
            "attributes": {
                "contract_type": "CEC",
                "contract_id": "CEC-001",
                "contract_role": "BUY",
                "status_date": "2024-01-01",
                "initial_exchange_date": "2024-01-01",
                "maturity_date": "2029-01-01",
                "notional_principal": 500000.0,
                "coverage": 1.0,
                "contract_structure": '{"CoveredContract": "LOAN-001"}',
            },
            "child_contracts": {
                "LOAN-001": {
                    "contract_type": "PAM", "contract_id": "LOAN-001",
                    "contract_role": "RPA", "status_date": "2024-01-01",
                    "initial_exchange_date": "2024-01-01", "maturity_date": "2029-01-01",
                    "notional_principal": 500000.0, "nominal_interest_rate": 0.04,
                },
            },
        },
    }
    return examples.get(contract_type, {})


def get_contract_schema(contract_type: str) -> dict[str, Any]:
    """Get required and optional parameters for a contract type.

    Args:
        contract_type: ACTUS contract type code

    Returns:
        Dictionary with required fields, optional fields, and their types.
    """
    # Get Pydantic schema
    schema = ContractAttributes.model_json_schema()

    # Base required fields (always needed by Pydantic)
    base_required = {
        "contract_type": "ContractType enum (e.g., 'PAM', 'LAM', 'SWPPV')",
        "contract_id": "str - Unique contract identifier",
        "status_date": "ActusDateTime - Analysis/valuation date (ISO string: 'YYYY-MM-DD')",
        "contract_role": (
            "ContractRole enum - RPA (Real Position Asset/lender), "
            "RPL (Real Position Liability/borrower), RFL (Receive First Leg), "
            "PFL (Pay First Leg), BUY (Protection Buyer), SEL (Protection Seller), "
            "LG (Long), ST (Short)"
        ),
    }

    # Contract-specific typical fields.
    # Fields marked "(recommended)" have sensible defaults but are usually
    # needed for meaningful simulations. Fields without the marker are strictly
    # required by the contract logic.
    specific_required = {
        # Principal contracts
        "PAM": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "nominal_interest_rate": "float (recommended, defaults to 0)",
            "day_count_convention": "DayCountConvention enum (recommended, defaults to A360)",
        },
        "LAM": {
            "initial_exchange_date": "ActusDateTime",
            "notional_principal": "float (recommended)",
            "nominal_interest_rate": "float (recommended, defaults to 0)",
            "maturity_date": "ActusDateTime - Required if principal_redemption_cycle not set",
            "principal_redemption_cycle": "str (e.g., '1M', '3M') - Required if maturity_date not set",
            "next_principal_redemption_amount": "float (recommended, auto-calculated if omitted)",
        },
        "LAX": {
            "initial_exchange_date": "ActusDateTime (recommended)",
            "maturity_date": "ActusDateTime (recommended)",
            "notional_principal": "float (recommended)",
            "nominal_interest_rate": "float (recommended, defaults to 0)",
            "array_pr_cycle": "list[str] - Array of PR cycles for exotic amortization",
            "array_pr_next": "list[float] - Array of next PR amounts",
            "array_pr_anchor": "list[ActusDateTime] - Array of PR anchor dates",
            "array_increase_decrease": "list[str] - 'INC' or 'DEC' per segment",
        },
        "NAM": {
            "initial_exchange_date": "ActusDateTime",
            "notional_principal": "float (recommended)",
            "nominal_interest_rate": "float (recommended, defaults to 0)",
            "maturity_date": "ActusDateTime - Required if principal_redemption_cycle not set",
            "principal_redemption_cycle": "str (e.g., '1M', '3M') - Required if maturity_date not set",
            "next_principal_redemption_amount": "float (recommended, auto-calculated if omitted)",
        },
        "ANN": {
            "initial_exchange_date": "ActusDateTime",
            "principal_redemption_cycle": "str - Payment frequency (e.g., '1M', '3M')",
            "notional_principal": "float (recommended)",
            "nominal_interest_rate": "float (recommended, defaults to 0)",
            "maturity_date": "ActusDateTime (recommended, or use amortization_date)",
        },
        "CLM": {
            "initial_exchange_date": "ActusDateTime (recommended)",
            "notional_principal": "float (recommended)",
            "nominal_interest_rate": "float (recommended)",
            "interest_payment_cycle": "str (recommended)",
        },
        # Non-principal contracts
        "UMP": {
            "initial_exchange_date": "ActusDateTime (recommended)",
            "notional_principal": "float (recommended)",
            "nominal_interest_rate": "float (recommended)",
        },
        "CSH": {
            "notional_principal": "float",
        },
        "STK": {
            "initial_exchange_date": "ActusDateTime (recommended)",
            "notional_principal": "float - Number of shares or position value",
        },
        # Exotic contracts
        "COM": {
            "initial_exchange_date": "ActusDateTime (recommended)",
            "notional_principal": "float - Commodity value (recommended)",
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
            "contract_structure": "str (JSON) - Reference to underlier contract",
            "option_type": "str ('C' for call, 'P' for put, 'CP' for collar)",
            "option_strike_1": "float",
            "option_exercise_type": "str ('E' European, 'A' American, 'B' Bermudan)",
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
            "rate_reset_cycle": "str - Reset frequency",
            "nominal_interest_rate": "float (recommended)",
            "rate_reset_cap": "float - Interest rate cap level (recommended)",
            "rate_reset_floor": "float - Interest rate floor level (recommended)",
        },
        "CEG": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "contract_structure": "str (JSON) - Reference to guaranteed contract",
            "coverage": "float - Coverage ratio (recommended)",
        },
        "CEC": {
            "initial_exchange_date": "ActusDateTime",
            "maturity_date": "ActusDateTime",
            "notional_principal": "float",
            "contract_structure": "str (JSON) - Reference to collateral contract",
            "coverage": "float - Coverage ratio (recommended)",
        },
    }

    required = {**base_required}
    if contract_type in specific_required:
        required.update(specific_required[contract_type])

    # Optional fields — comprehensive list of all ContractAttributes fields
    # not already in base_required. Fields that appear in specific_required
    # for the current contract_type are excluded dynamically below.
    optional = {
        # Currency
        "currency": "str - ISO currency code (default: USD)",
        "currency_2": "str - Second currency for FX/swap contracts (CUR2)",
        # Dates
        "contract_deal_date": "ActusDateTime (CDD)",
        "initial_exchange_date": "ActusDateTime - Contract inception (IED)",
        "maturity_date": "ActusDateTime - Contract maturity (MD)",
        "purchase_date": "ActusDateTime - Secondary market purchase (PRD)",
        "termination_date": "ActusDateTime - Early termination (TD)",
        "settlement_date": "ActusDateTime - Derivative settlement (STD)",
        "amortization_date": "ActusDateTime - ANN amortization end date (AMD)",
        "analysis_dates": "list[ActusDateTime] - Array of analysis dates (AD)",
        # Notional and rates
        "notional_principal": "float - Principal amount (NT)",
        "nominal_interest_rate": "float - Nominal interest rate as decimal (IPNR)",
        "nominal_interest_rate_2": "float - Second rate for swaps (IPNR2)",
        "notional_principal_2": "float - Second notional for FX/swaps (NT2)",
        "next_principal_redemption_amount": "float - Next PR payment amount (PRNXT)",
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
        "interest_calculation_base_cycle": "str - Interest calc base reset cycle (IPCBCL)",
        "interest_calculation_base_anchor": "ActusDateTime (IPCBANX)",
        # Array schedule attributes (for LAX and exotic schedules)
        "array_pr_anchor": "list[ActusDateTime] - Array of PR anchors (ARPRANX)",
        "array_pr_cycle": "list[str] - Array of PR cycles (ARPRCL)",
        "array_pr_next": "list[float] - Array of next PR amounts (ARPRNXT)",
        "array_increase_decrease": "list[str] - 'INC' or 'DEC' per segment (ARINCDEC)",
        "array_ip_anchor": "list[ActusDateTime] - Array of IP anchors (ARIPANX)",
        "array_ip_cycle": "list[str] - Array of IP cycles (ARIPCL)",
        "array_rr_anchor": "list[ActusDateTime] - Array of RR anchors (ARRRANX)",
        "array_rr_cycle": "list[str] - Array of RR cycles (ARRRCL)",
        "array_rate": "list[float] - Array of interest rates (ARRATE)",
        "array_fixed_variable": "list[str] - 'F' or 'V' per segment (ARFIXVAR)",
        # Conventions
        "day_count_convention": "DayCountConvention enum: AA, A360, A365, E30360ISDA, E30360, B30360, BUS252 (DCC)",
        "business_day_convention": "BusinessDayConvention enum: NULL, SCF, SCMF, CSF, CSMF, SCP, SCMP, CSP, CSMP (BDC)",
        "end_of_month_convention": "EndOfMonthConvention enum: EOM, SD (EOMC)",
        "calendar": "Calendar enum: NO_CALENDAR, MONDAY_TO_FRIDAY, TARGET, US_NYSE, UK_SETTLEMENT (CLDR)",
        # Rate reset
        "rate_reset_market_object": "str - Market reference for rate reset (RRMO)",
        "rate_reset_multiplier": "float (RRMLT)",
        "rate_reset_spread": "float (RRSP)",
        "rate_reset_floor": "float (RRLF)",
        "rate_reset_cap": "float (RRLC)",
        "rate_reset_next": "float (RRNXT)",
        # Fees
        "fee_rate": "float (FER)",
        "fee_basis": "FeeBasis enum: A (absolute), N (notional percentage) (FEB)",
        "fee_accrued": "float (FEAC)",
        # Interest calculation base
        "interest_calculation_base": "InterestCalculationBase enum: NT, NTIED, NTL (IPCB)",
        "interest_calculation_base_amount": "float (IPCBA)",
        # Prepayment
        "prepayment_effect": "PrepaymentEffect enum: N, A, M (PPEF, default: N)",
        "penalty_type": "str (PYTP)",
        "penalty_rate": "float (PYRT)",
        # Scaling
        "scaling_effect": "ScalingEffect enum: 000, I00, 0N0, IN0, 00M, I0M, 0NM, INM (SCEF, default: 000)",
        "scaling_index_at_status_date": "float (SCIXSD)",
        "scaling_index_at_contract_deal_date": "float (SCIXCDD)",
        "scaling_market_object": "str (SCMO)",
        # Options and derivatives
        "option_type": "str: 'C' (call), 'P' (put), 'CP' (collar) (OPTP)",
        "option_strike_1": "float - Primary strike price (OPS1)",
        "option_strike_2": "float - Secondary strike for collars (OPS2)",
        "option_exercise_type": "str: 'E' (European), 'A' (American), 'B' (Bermudan) (OPXT)",
        "option_exercise_end_date": "ActusDateTime (OPXED)",
        "exercise_date": "ActusDateTime - Option/derivative exercise date (XD)",
        "exercise_amount": "float - Amount determined at exercise (XA)",
        "settlement_period": "str - Period between exercise and settlement (STPD)",
        "delivery_type": "str: 'P' (physical), 'C' (cash) (DVTP)",
        "delivery_settlement": "str: 'D' (delivery/net), 'S' (settlement/gross) (DS)",
        "contract_structure": "str (JSON) - Reference to underlier/child contracts (CTST)",
        "future_price": "float - Agreed futures price (PFUT)",
        "settlement_currency": "str - Settlement currency (CURS)",
        "fixing_period": "str - Period between rate observation and reset (RRFIX)",
        "x_day_notice": "str - Notice period for call/put (XDN)",
        # Commodity and equity
        "quantity": "float - Quantity of commodity/stock (QT)",
        "unit": "str - Unit of measurement (UNIT)",
        "market_object_code": "str - Market object code for price observation (MOC)",
        "market_object_code_of_dividends": "str - Dividend observation code (DVMO)",
        "dividend_cycle": "str - Dividend payment cycle (DVCL)",
        "dividend_anchor": "ActusDateTime - Dividend payment anchor (DVANX)",
        # Credit enhancement
        "coverage": "float - Coverage ratio (CECV)",
        "credit_event_type": "ContractPerformance enum: PF, DL, DQ, DF (CET)",
        "credit_enhancement_guarantee_extent": "str: NO, NI, or MV (CEGE)",
        # Other
        "premium_discount_at_ied": "float (PDIED)",
        "accrued_interest": "float (IPAC)",
        "price_at_purchase_date": "float (PPRD)",
        "price_at_termination_date": "float (PTD)",
        "contract_performance": "ContractPerformance enum: PF, DL, DQ, DF (PRF, default: PF)",
    }

    # Remove fields that are already in specific_required for this contract type
    specific = specific_required.get(contract_type, {})
    filtered_optional = {k: v for k, v in optional.items() if k not in specific and k not in base_required}

    # Composite contracts that require child_contracts parameter
    _requires_child_contracts = {"CAPFL", "SWAPS", "CEG", "CEC"}

    # Build example code — MCP JSON for composite, Python for standard
    if contract_type in _requires_child_contracts:
        mcp_example = _child_contract_mcp_example(contract_type)
        python_example = _child_observer_example(contract_type)
    else:
        mcp_example = None
        python_example = f"""
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
"""

    result = {
        "contract_type": contract_type,
        "required_fields": required,
        "optional_fields": filtered_optional,
        "example_usage": python_example,
    }

    if contract_type in _requires_child_contracts:
        result["requires_child_contracts"] = True
        result["mcp_example"] = mcp_example
        result["mcp_note"] = (
            f"{contract_type} is a composite contract. Pass a child_contracts dict "
            f"to jactus_simulate_contract — each child is simulated automatically, "
            f"then its results feed into the parent. See the mcp_example field."
        )

    return result


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
