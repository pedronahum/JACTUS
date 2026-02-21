"""Contract validation tools."""

import difflib
import json
from typing import Any

from pydantic import ValidationError
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.core.attributes import ATTRIBUTE_MAP


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

    Args:
        attributes: Dictionary of contract attributes

    Returns:
        Validation result with success status and any errors.
    """
    try:
        # Convert string contract_type to enum if needed
        if "contract_type" in attributes and isinstance(attributes["contract_type"], str):
            try:
                attributes["contract_type"] = ContractType[attributes["contract_type"]]
            except KeyError:
                return {
                    "valid": False,
                    "errors": [
                        f"Invalid contract_type: {attributes['contract_type']}. "
                        f"Must be one of: {[ct.name for ct in ContractType]}"
                    ],
                }

        # Convert string contract_role to enum if needed
        if "contract_role" in attributes and isinstance(attributes["contract_role"], str):
            try:
                attributes["contract_role"] = ContractRole[attributes["contract_role"]]
            except KeyError:
                return {
                    "valid": False,
                    "errors": [
                        f"Invalid contract_role: {attributes['contract_role']}. "
                        f"Must be one of: {[cr.name for cr in ContractRole]}"
                    ],
                }

        # Convert date strings to ActusDateTime if needed
        date_fields = [
            "status_date",
            "initial_exchange_date",
            "maturity_date",
            "cycle_anchor_date_of_interest_payment",
            "cycle_anchor_date_of_principal_redemption",
            "purchase_date",
            "termination_date",
            "capitalization_end_date",
        ]

        for field in date_fields:
            if field in attributes and isinstance(attributes[field], str):
                try:
                    # Parse ISO format: YYYY-MM-DD
                    parts = attributes[field].split("-")
                    if len(parts) == 3:
                        year, month, day = map(int, parts)
                        attributes[field] = ActusDateTime(year, month, day)
                except Exception as e:
                    return {
                        "valid": False,
                        "errors": [f"Invalid date format for {field}: {str(e)}"],
                    }

        # Detect unknown fields before Pydantic validation
        unknown_field_warnings = _detect_unknown_fields(attributes)

        # Strip unknown keys so Pydantic doesn't silently ignore them
        valid_fields = set(ContractAttributes.model_fields.keys())
        cleaned = {k: v for k, v in attributes.items() if k in valid_fields}

        # Validate with Pydantic
        attrs = ContractAttributes(**cleaned)

        # Additional contract-specific validation
        warnings = list(unknown_field_warnings)
        contract_type = attrs.contract_type.name if hasattr(attrs.contract_type, 'name') else str(attrs.contract_type)

        # Check for common required fields by contract type
        if contract_type in ["PAM", "LAM", "ANN", "SWPPV"]:
            if attrs.notional_principal is None or attrs.notional_principal == 0:
                warnings.append("notional_principal should typically be non-zero")

            if attrs.nominal_interest_rate is None:
                warnings.append("nominal_interest_rate is typically required")

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

        if contract_type == "FXOUT":
            if not attrs.delivery_settlement:
                return {
                    "valid": False,
                    "errors": ["FXOUT requires delivery_settlement ('D' or 'S')"],
                }

        if contract_type == "OPTNS":
            if not attrs.contract_structure:
                return {
                    "valid": False,
                    "errors": ["OPTNS requires contract_structure (JSON string)"],
                }
            if not attrs.option_type:
                warnings.append("option_type should be 'C' (call) or 'P' (put)")

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
    "day_count_convention": "30E360",
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
