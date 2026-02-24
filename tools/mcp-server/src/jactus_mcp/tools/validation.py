"""Contract validation tools."""

import difflib
from typing import Any

from jactus.core import ContractAttributes
from jactus.core.attributes import ATTRIBUTE_MAP
from pydantic import ValidationError

from jactus_mcp.tools._utils import prepare_attributes


def _detect_unknown_fields(attributes: dict[str, Any]) -> list[str]:
    """Detect unknown fields in contract attributes and suggest corrections.

    Returns a list of warning strings for any keys not recognized as valid
    ContractAttributes fields or ACTUS short names.
    """
    valid_fields = set(ContractAttributes.model_fields.keys())
    actus_short_names = ATTRIBUTE_MAP  # maps ACTUS short name -> python name

    warnings = []
    for key in attributes:
        if key in valid_fields:
            continue

        # Check if it's an ACTUS short name
        if key in actus_short_names:
            python_name = actus_short_names[key]
            warnings.append(
                f"Unknown field '{key}'. This is an ACTUS short name "
                f"— use the Python name '{python_name}' instead."
            )
            continue

        # Try to find close matches among valid field names
        close_matches = difflib.get_close_matches(
            key, valid_fields, n=1, cutoff=0.6
        )
        if close_matches:
            warnings.append(
                f"Unknown field '{key}'. Did you mean '{close_matches[0]}'?"
            )
        else:
            warnings.append(f"Unknown field '{key}' — it will be ignored.")

    return warnings


def validate_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Validate contract attributes for correctness.

    Uses the same enum/date conversion as simulate_contract to ensure
    consistent behavior between validation and simulation.

    Args:
        attributes: Dictionary of contract attributes

    Returns:
        Validation result with success status and any errors.
    """
    try:
        # Detect unknown fields before conversion
        unknown_field_warnings = _detect_unknown_fields(attributes)

        # Use shared prepare_attributes for consistent enum/date conversion
        try:
            prepared = prepare_attributes(attributes)
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Attribute conversion error: {e!s}"],
                "hint": "Check date formats (YYYY-MM-DD) and enum values.",
            }

        # Strip unknown keys so Pydantic doesn't silently ignore them
        valid_fields = set(ContractAttributes.model_fields.keys())
        cleaned = {k: v for k, v in prepared.items() if k in valid_fields}

        # Validate with Pydantic
        attrs = ContractAttributes(**cleaned)

        # Additional contract-specific validation
        warnings = list(unknown_field_warnings)
        contract_type = attrs.contract_type.name if hasattr(attrs.contract_type, 'name') else str(attrs.contract_type)

        # ---- Contract-specific required field checks ----

        # Principal contracts
        if contract_type in ("PAM", "LAM", "LAX", "NAM", "ANN", "CLM", "UMP"):
            if attrs.initial_exchange_date is None:
                warnings.append(f"{contract_type} typically requires initial_exchange_date")
            if attrs.notional_principal is None:
                warnings.append(f"{contract_type} typically requires notional_principal")

        if contract_type in ("PAM", "LAM", "NAM", "ANN"):
            if attrs.nominal_interest_rate is None:
                warnings.append(f"{contract_type} typically requires nominal_interest_rate")

        if contract_type == "PAM":
            if attrs.maturity_date is None:
                warnings.append("PAM requires maturity_date")

        if contract_type in ("LAM", "NAM"):
            if attrs.maturity_date is None and attrs.principal_redemption_cycle is None:
                return {
                    "valid": False,
                    "errors": [f"{contract_type} requires either maturity_date or principal_redemption_cycle"],
                }

        if contract_type == "ANN":
            if attrs.principal_redemption_cycle is None:
                return {
                    "valid": False,
                    "errors": ["ANN requires principal_redemption_cycle"],
                }
            if attrs.maturity_date is None and attrs.amortization_date is None:
                warnings.append("ANN typically requires maturity_date or amortization_date")

        if contract_type == "CLM":
            if attrs.interest_payment_cycle is None:
                warnings.append("CLM typically requires interest_payment_cycle")

        # Non-principal contracts
        if contract_type == "CSH":
            if attrs.notional_principal is None:
                return {
                    "valid": False,
                    "errors": ["CSH requires notional_principal"],
                }

        if contract_type in ("STK", "COM"):
            if attrs.initial_exchange_date is None:
                warnings.append(f"{contract_type} typically requires initial_exchange_date")
            if attrs.notional_principal is None:
                warnings.append(f"{contract_type} typically requires notional_principal")

        # Derivative contracts
        if contract_type == "FXOUT":
            if not attrs.delivery_settlement:
                return {
                    "valid": False,
                    "errors": ["FXOUT requires delivery_settlement ('D' or 'S')"],
                }
            if not attrs.currency_2:
                warnings.append("FXOUT typically requires currency_2")
            if attrs.notional_principal_2 is None:
                warnings.append("FXOUT typically requires notional_principal_2")

        if contract_type == "OPTNS":
            if not attrs.contract_structure:
                return {
                    "valid": False,
                    "errors": ["OPTNS requires contract_structure (JSON string)"],
                }
            if not attrs.option_type:
                warnings.append("OPTNS typically requires option_type ('C' or 'P')")
            if attrs.option_strike_1 is None:
                warnings.append("OPTNS typically requires option_strike_1")

        if contract_type == "FUTUR":
            if not attrs.contract_structure:
                return {
                    "valid": False,
                    "errors": ["FUTUR requires contract_structure (JSON string)"],
                }

        if contract_type == "SWPPV":
            if not attrs.interest_payment_cycle:
                return {
                    "valid": False,
                    "errors": ["SWPPV requires interest_payment_cycle"],
                }
            if not attrs.rate_reset_cycle:
                return {
                    "valid": False,
                    "errors": ["SWPPV requires rate_reset_cycle"],
                }
            if attrs.nominal_interest_rate_2 is None:
                return {
                    "valid": False,
                    "errors": ["SWPPV requires nominal_interest_rate_2 (initial floating leg rate)"],
                }

        if contract_type == "SWAPS":
            if not attrs.contract_structure:
                return {
                    "valid": False,
                    "errors": ["SWAPS requires contract_structure (JSON string)"],
                }

        if contract_type == "CAPFL":
            if not attrs.rate_reset_cycle:
                return {
                    "valid": False,
                    "errors": ["CAPFL requires rate_reset_cycle"],
                }
            if attrs.rate_reset_cap is None and attrs.rate_reset_floor is None:
                warnings.append("CAPFL typically requires rate_reset_cap and/or rate_reset_floor")

        if contract_type in ("CEG", "CEC"):
            if not attrs.contract_structure:
                return {
                    "valid": False,
                    "errors": [f"{contract_type} requires contract_structure (JSON string)"],
                }
            if attrs.coverage is None:
                warnings.append(f"{contract_type} typically requires coverage")

        if contract_type == "LAX":
            if not attrs.array_pr_cycle and not attrs.array_pr_next:
                warnings.append("LAX typically requires array_pr_cycle and array_pr_next")

        return {
            "valid": True,
            "contract_type": contract_type,
            "attributes": {
                "contract_id": attrs.contract_id,
                "notional_principal": attrs.notional_principal,
                "currency": attrs.currency,
                "initial_exchange_date": str(attrs.initial_exchange_date) if attrs.initial_exchange_date else None,
                "maturity_date": str(attrs.maturity_date) if attrs.maturity_date else None,
            },
            "warnings": warnings if warnings else None,
            "message": "Contract attributes are valid",
        }

    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
            msg = error["msg"]
            errors.append(f"{field}: {msg}")

        return {
            "valid": False,
            "errors": errors,
            "message": "Validation failed",
            "hint": "Check required fields for this contract type using jactus_get_contract_schema",
        }

    except Exception as e:
        return {
            "valid": False,
            "errors": [str(e)],
            "message": "Unexpected validation error",
        }


def get_validation_example() -> str:
    """Get an example of how to use validation.

    Returns:
        Example code showing validation usage.
    """
    return '''"""Validation Example"""

from jactus_mcp.tools import validation

# Example 1: Valid PAM contract
valid_attrs = {
    "contract_type": "PAM",
    "contract_role": "RPA",
    "status_date": "2024-01-01",
    "initial_exchange_date": "2024-01-15",
    "maturity_date": "2025-01-15",
    "notional_principal": 100000.0,
    "nominal_interest_rate": 0.05,
    "day_count_convention": "E30360",
}

result = validation.validate_attributes(valid_attrs)
print(result)
# Output: {"valid": True, "message": "Contract attributes are valid", ...}

# Example 2: Invalid contract (missing required field)
invalid_attrs = {
    "contract_type": "PAM",
    "contract_role": "RPA",
    # Missing status_date!
    "notional_principal": 100000.0,
}

result = validation.validate_attributes(invalid_attrs)
print(result)
# Output: {"valid": False, "errors": ["status_date: Field required"], ...}
'''
