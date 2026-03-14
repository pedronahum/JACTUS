#!/usr/bin/env python3
"""Validate and simulate an ACTUS contract from a JSON file.

Usage:
    python validate_and_simulate.py contract.json
    python validate_and_simulate.py contract.json --dry-run

Exit codes:
    0 — Success
    1 — Validation error (missing/invalid fields)
    2 — Simulation error (runtime failure)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


REQUIRED_FIELDS = {
    "contract_id": str,
    "contract_type": str,
    "contract_role": str,
    "status_date": str,
}

TYPE_SPECIFIC_REQUIRED = {
    "PAM": ["initial_exchange_date", "maturity_date", "notional_principal"],
    "LAM": ["initial_exchange_date", "notional_principal"],
    "LAX": ["initial_exchange_date", "notional_principal", "array_pr_cycle", "array_pr_next"],
    "NAM": ["initial_exchange_date", "notional_principal"],
    "ANN": ["initial_exchange_date", "notional_principal", "principal_redemption_cycle"],
    "CLM": ["initial_exchange_date", "notional_principal"],
    "UMP": ["initial_exchange_date", "notional_principal"],
    "CSH": ["notional_principal"],
    "STK": ["notional_principal"],
    "COM": ["notional_principal"],
    "FXOUT": ["initial_exchange_date", "maturity_date", "notional_principal"],
    "OPTNS": ["initial_exchange_date", "maturity_date", "notional_principal"],
    "FUTUR": ["initial_exchange_date", "maturity_date", "notional_principal"],
    "SWPPV": ["initial_exchange_date", "maturity_date", "notional_principal"],
    "SWAPS": ["maturity_date", "contract_structure"],
    "CAPFL": ["maturity_date", "notional_principal"],
    "CEG": ["maturity_date", "notional_principal", "contract_structure"],
    "CEC": ["maturity_date", "notional_principal", "contract_structure"],
}

VALID_CONTRACT_TYPES = list(TYPE_SPECIFIC_REQUIRED.keys())
VALID_ROLES = ["RPA", "RPL", "RFL", "PFL", "BUY", "SEL", "LG", "ST"]

DATE_FIELDS = [
    "status_date", "initial_exchange_date", "maturity_date",
    "contract_deal_date", "purchase_date", "termination_date",
    "settlement_date", "amortization_date", "interest_payment_anchor",
    "principal_redemption_anchor", "rate_reset_anchor", "fee_payment_anchor",
    "dividend_anchor",
]


def validate_attributes(attrs: dict) -> list[str]:
    """Validate contract attributes and return a list of error messages."""
    errors = []

    # Check universal required fields
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in attrs:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(attrs[field], expected_type):
            errors.append(f"Field '{field}' must be a {expected_type.__name__}, got {type(attrs[field]).__name__}")

    if errors:
        return errors

    # Validate contract_type
    ct = attrs["contract_type"].upper()
    if ct not in VALID_CONTRACT_TYPES:
        errors.append(f"Invalid contract_type '{ct}'. Valid types: {', '.join(VALID_CONTRACT_TYPES)}")
        return errors

    # Validate contract_role
    role = attrs["contract_role"].upper()
    if role not in VALID_ROLES:
        errors.append(f"Invalid contract_role '{role}'. Valid roles: {', '.join(VALID_ROLES)}")

    # Validate type-specific required fields
    for field in TYPE_SPECIFIC_REQUIRED.get(ct, []):
        if field not in attrs:
            errors.append(f"Missing required field for {ct}: {field}")

    # Validate date formats
    for field in DATE_FIELDS:
        if field in attrs and isinstance(attrs[field], str):
            try:
                datetime.fromisoformat(attrs[field])
            except ValueError:
                errors.append(f"Invalid date format for '{field}': '{attrs[field]}'. Use ISO format: YYYY-MM-DD")

    # Validate numeric fields
    numeric_fields = [
        "notional_principal", "nominal_interest_rate", "notional_principal_2",
        "nominal_interest_rate_2", "next_principal_redemption_amount",
        "option_strike_1", "option_strike_2", "future_price", "coverage",
        "fee_rate", "rate_reset_spread", "rate_reset_multiplier",
        "rate_reset_floor", "rate_reset_cap",
    ]
    for field in numeric_fields:
        if field in attrs and not isinstance(attrs[field], (int, float)):
            errors.append(f"Field '{field}' must be numeric, got {type(attrs[field]).__name__}")

    return errors


def simulate_contract(attrs: dict) -> dict:
    """Simulate a contract and return the result as a dict."""
    try:
        from jactus.contracts import create_contract
        from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
        from jactus.observers import ConstantRiskFactorObserver
    except ImportError as e:
        raise RuntimeError(f"JACTUS not installed: {e}. Run: pip install git+https://github.com/pedronahum/JACTUS.git")

    # Convert string dates to ActusDateTime
    processed = dict(attrs)
    for field in DATE_FIELDS:
        if field in processed and isinstance(processed[field], str):
            dt = datetime.fromisoformat(processed[field])
            processed[field] = ActusDateTime(dt.year, dt.month, dt.day)

    # Convert array date fields
    for field in ["array_pr_anchor", "array_ip_anchor", "array_rr_anchor"]:
        if field in processed and isinstance(processed[field], list):
            converted = []
            for d in processed[field]:
                if isinstance(d, str):
                    dt = datetime.fromisoformat(d)
                    converted.append(ActusDateTime(dt.year, dt.month, dt.day))
                else:
                    converted.append(d)
            processed[field] = converted

    # Convert enums
    processed["contract_type"] = ContractType[processed["contract_type"].upper()]
    processed["contract_role"] = ContractRole[processed["contract_role"].upper()]

    # Create and simulate
    rf_rate = processed.pop("risk_factor_rate", 0.0)
    contract_attrs = ContractAttributes(**processed)
    rf_observer = ConstantRiskFactorObserver(constant_value=rf_rate)
    contract = create_contract(contract_attrs, rf_observer)
    result = contract.simulate()

    # Format output
    events = []
    for event in result.events:
        events.append({
            "time": str(event.event_time),
            "type": event.event_type.name,
            "payoff": float(event.payoff),
            "currency": getattr(event, "currency", "USD"),
        })

    total_inflows = sum(e["payoff"] for e in events if e["payoff"] > 0)
    total_outflows = sum(e["payoff"] for e in events if e["payoff"] < 0)

    return {
        "contract_id": attrs["contract_id"],
        "contract_type": attrs["contract_type"],
        "events": events,
        "summary": {
            "total_inflows": total_inflows,
            "total_outflows": total_outflows,
            "net_cashflow": total_inflows + total_outflows,
            "event_count": len(events),
            "non_zero_events": sum(1 for e in events if e["payoff"] != 0),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate and simulate an ACTUS contract from a JSON file."
    )
    parser.add_argument("json_file", help="Path to JSON file with contract attributes")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate only, do not simulate",
    )
    args = parser.parse_args()

    # Load JSON
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(json.dumps({"error": f"File not found: {args.json_file}"}), file=sys.stderr)
        sys.exit(1)

    try:
        with open(json_path) as f:
            attrs = json.load(f)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {e}"}), file=sys.stderr)
        sys.exit(1)

    # Validate
    errors = validate_attributes(attrs)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2), file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(json.dumps({"valid": True, "message": "Validation passed"}, indent=2))
        sys.exit(0)

    # Simulate
    try:
        result = simulate_contract(attrs)
        print(json.dumps(result, indent=2))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"error": f"Simulation failed: {e}"}), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
