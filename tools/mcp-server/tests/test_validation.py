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


def test_validate_unknown_field_warning():
    """Test that unknown fields produce warnings but still validate."""
    attrs = {
        "contract_id": "LOAN-001",
        "contract_type": "PAM",
        "contract_role": "RPA",
        "status_date": "2024-01-01",
        "initial_exchange_date": "2024-01-15",
        "maturity_date": "2025-01-15",
        "notional_principal": 100000.0,
        "nominal_interest_rate": 0.05,
        "totally_fake_field": "hello",
    }

    result = validation.validate_attributes(attrs)

    assert result["valid"] is True
    assert result["warnings"] is not None
    assert any("totally_fake_field" in w for w in result["warnings"])


def test_validate_unknown_field_suggestion():
    """Test that close misspellings get 'did you mean' suggestions."""
    attrs = {
        "contract_id": "LOAN-001",
        "contract_type": "PAM",
        "contract_role": "RPA",
        "status_date": "2024-01-01",
        "initial_exchange_date": "2024-01-15",
        "maturity_date": "2025-01-15",
        "notional_principal": 100000.0,
        "nominal_interest_rate": 0.05,
        "cycle_of_interest_payment": "1M",  # misspelling of interest_payment_cycle
    }

    result = validation.validate_attributes(attrs)

    assert result["valid"] is True
    assert result["warnings"] is not None
    assert any("cycle_of_interest_payment" in w for w in result["warnings"])
    assert any("Did you mean" in w for w in result["warnings"])


def test_validate_swppv_missing_nominal_interest_rate_2():
    """Test SWPPV validation requires nominal_interest_rate_2."""
    attrs = {
        "contract_id": "SWAP-001",
        "contract_type": "SWPPV",
        "contract_role": "RPA",
        "status_date": "2024-01-01",
        "initial_exchange_date": "2024-01-15",
        "maturity_date": "2025-01-15",
        "notional_principal": 10000000.0,
        "nominal_interest_rate": 0.05,
        "interest_payment_cycle": "3M",
        "rate_reset_cycle": "3M",
        # Missing nominal_interest_rate_2
    }

    result = validation.validate_attributes(attrs)

    assert result["valid"] is False
    assert any("nominal_interest_rate_2" in str(e) for e in result["errors"])


def test_validate_actus_short_name_suggestion():
    """Test that ACTUS short names are recognized and suggest Python names."""
    attrs = {
        "contract_id": "LOAN-001",
        "contract_type": "PAM",
        "contract_role": "RPA",
        "status_date": "2024-01-01",
        "initial_exchange_date": "2024-01-15",
        "maturity_date": "2025-01-15",
        "notional_principal": 100000.0,
        "nominal_interest_rate": 0.05,
        "IPCL": "1M",  # ACTUS short name for interest_payment_cycle
    }

    result = validation.validate_attributes(attrs)

    assert result["valid"] is True
    assert result["warnings"] is not None
    assert any("IPCL" in w for w in result["warnings"])
    assert any("interest_payment_cycle" in w for w in result["warnings"])
