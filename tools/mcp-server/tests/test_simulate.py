"""Tests for contract simulation tool."""

import json

import pytest

from jactus_mcp.tools import simulate


@pytest.fixture
def pam_attributes():
    """Basic PAM contract attributes."""
    return {
        "contract_type": "PAM",
        "contract_id": "TEST-PAM-001",
        "contract_role": "RPA",
        "status_date": "2024-01-01",
        "initial_exchange_date": "2024-01-15",
        "maturity_date": "2025-01-15",
        "notional_principal": 100000.0,
        "nominal_interest_rate": 0.05,
        "day_count_convention": "30E360",
    }


def test_simulate_pam_basic(pam_attributes):
    """Test basic PAM simulation returns structured results."""
    result = simulate.simulate_contract(pam_attributes)

    assert result["success"] is True
    assert result["contract_type"] == "PAM"
    assert result["num_events"] > 0
    assert len(result["events"]) == result["num_events"]
    assert result["summary"] is not None
    assert result["initial_state"] is not None
    assert result["final_state"] is not None


def test_simulate_pam_events_are_json_serializable(pam_attributes):
    """Test that simulation results can be JSON-serialized (no JAX arrays)."""
    result = simulate.simulate_contract(pam_attributes)

    # This should not raise - all values must be JSON-compatible
    serialized = json.dumps(result)
    assert isinstance(serialized, str)

    # Verify round-trip
    parsed = json.loads(serialized)
    assert parsed["success"] is True
    assert isinstance(parsed["events"], list)


def test_simulate_pam_event_structure(pam_attributes):
    """Test that each event has the expected fields."""
    result = simulate.simulate_contract(pam_attributes)

    for event in result["events"]:
        assert "event_type" in event
        assert "event_time" in event
        assert "payoff" in event
        assert "currency" in event
        assert isinstance(event["payoff"], (int, float))
        assert isinstance(event["event_time"], str)


def test_simulate_pam_summary(pam_attributes):
    """Test summary statistics are computed correctly."""
    result = simulate.simulate_contract(pam_attributes)
    summary = result["summary"]

    assert "total_cashflows" in summary
    assert "total_inflows" in summary
    assert "total_outflows" in summary
    assert "net_cashflow" in summary
    assert "first_event" in summary
    assert "last_event" in summary
    assert isinstance(summary["total_inflows"], (int, float))
    assert isinstance(summary["total_outflows"], (int, float))


def test_simulate_with_include_states(pam_attributes):
    """Test that include_states=True returns state dicts."""
    result = simulate.simulate_contract(pam_attributes, include_states=True)

    assert result["success"] is True
    # At least some events should have states
    events_with_states = [e for e in result["events"] if e.get("state_post") is not None]
    assert len(events_with_states) > 0


def test_simulate_with_constant_value(pam_attributes):
    """Test simulation with explicit constant risk factor value."""
    result = simulate.simulate_contract(pam_attributes, constant_value=0.05)

    assert result["success"] is True
    assert result["num_events"] > 0


def test_simulate_with_dict_risk_factors(pam_attributes):
    """Test simulation with DictRiskFactorObserver."""
    result = simulate.simulate_contract(
        pam_attributes,
        risk_factors={"LIBOR-3M": 0.05, "USD/EUR": 1.18},
    )

    assert result["success"] is True
    assert result["num_events"] > 0


def test_simulate_invalid_contract_type():
    """Test error handling for invalid contract type."""
    result = simulate.simulate_contract({
        "contract_type": "INVALID",
        "contract_id": "TEST",
        "contract_role": "RPA",
        "status_date": "2024-01-01",
    })

    assert result["success"] is False
    assert "error" in result


def test_simulate_missing_required_fields():
    """Test error handling for missing required fields."""
    result = simulate.simulate_contract({
        "contract_type": "PAM",
        "contract_role": "RPA",
        # Missing status_date and other required fields
    })

    assert result["success"] is False
    assert "error" in result


def test_prepare_attributes_converts_enums():
    """Test that _prepare_attributes correctly converts string enums."""
    from jactus.core import ContractType, ContractRole

    attrs = simulate._prepare_attributes({
        "contract_type": "PAM",
        "contract_role": "RPA",
    })

    assert attrs["contract_type"] == ContractType.PAM
    assert attrs["contract_role"] == ContractRole.RPA


def test_prepare_attributes_converts_dates():
    """Test that _prepare_attributes correctly converts date strings."""
    from jactus.core import ActusDateTime

    attrs = simulate._prepare_attributes({
        "status_date": "2024-01-15",
        "initial_exchange_date": "2024-06-01T00:00:00",
    })

    assert isinstance(attrs["status_date"], ActusDateTime)
    assert isinstance(attrs["initial_exchange_date"], ActusDateTime)
