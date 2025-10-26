"""Tests for validation tools."""

import pytest
from jactus_mcp.tools import validation


def test_validate_valid_pam():
    """Test validation of a valid PAM contract."""
    attrs = {
        "contract_id": "LOAN-001",
        "contract_type": "PAM",
        "contract_role": "RPA",
        "status_date": "2024-01-01",
        "initial_exchange_date": "2024-01-15",
        "maturity_date": "2025-01-15",
        "notional_principal": 100000.0,
        "nominal_interest_rate": 0.05,
        "day_count_convention": "30E360",
    }

    result = validation.validate_attributes(attrs)

    assert result["valid"] is True
    assert result["contract_type"] == "PAM"
    assert "message" in result


def test_validate_missing_required_field():
    """Test validation with missing required field."""
    attrs = {
        "contract_type": "PAM",
        "contract_role": "RPA",
        # Missing status_date
        "notional_principal": 100000.0,
    }

    result = validation.validate_attributes(attrs)

    assert result["valid"] is False
    assert "errors" in result
    assert len(result["errors"]) > 0


def test_validate_invalid_contract_type():
    """Test validation with invalid contract type."""
    attrs = {
        "contract_type": "INVALID_TYPE",
        "contract_role": "RPA",
        "status_date": "2024-01-01",
    }

    result = validation.validate_attributes(attrs)

    assert result["valid"] is False
    assert "errors" in result


def test_validate_swppv_missing_cycles():
    """Test SWPPV validation requires cycles."""
    attrs = {
        "contract_id": "SWAP-001",
        "contract_type": "SWPPV",
        "contract_role": "RPA",
        "status_date": "2024-01-01",
        "initial_exchange_date": "2024-01-15",
        "maturity_date": "2025-01-15",
        "notional_principal": 10000000.0,
        # Missing interest_payment_cycle and rate_reset_cycle
    }

    result = validation.validate_attributes(attrs)

    assert result["valid"] is False
    # Should fail on missing cycles
    assert any("interest_payment_cycle" in str(e).lower() or "rate_reset_cycle" in str(e).lower() for e in result["errors"])


def test_validate_with_warnings():
    """Test validation that JACTUS rejects zero notional."""
    attrs = {
        "contract_id": "LOAN-002",
        "contract_type": "PAM",
        "contract_role": "RPA",
        "status_date": "2024-01-01",
        "initial_exchange_date": "2024-01-15",
        "maturity_date": "2025-01-15",
        "notional_principal": 0.0,  # Zero notional (JACTUS rejects this)
        "day_count_convention": "30E360",
    }

    result = validation.validate_attributes(attrs)

    # JACTUS now rejects zero notional as invalid
    assert result["valid"] is False
    assert any("notional" in str(e).lower() for e in result["errors"])
