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
    """Test that prepare_attributes correctly converts string enums."""
    from jactus.core import ContractType, ContractRole
    from jactus_mcp.tools._utils import prepare_attributes

    attrs = prepare_attributes({
        "contract_type": "PAM",
        "contract_role": "RPA",
    })

    assert attrs["contract_type"] == ContractType.PAM
    assert attrs["contract_role"] == ContractRole.RPA


def test_prepare_attributes_converts_dates():
    """Test that prepare_attributes correctly converts date strings."""
    from jactus.core import ActusDateTime
    from jactus_mcp.tools._utils import prepare_attributes

    attrs = prepare_attributes({
        "status_date": "2024-01-15",
        "initial_exchange_date": "2024-06-01T00:00:00",
    })

    assert isinstance(attrs["status_date"], ActusDateTime)
    assert isinstance(attrs["initial_exchange_date"], ActusDateTime)


def test_simulate_with_time_series(pam_attributes):
    """Test simulation with TimeSeriesRiskFactorObserver."""
    result = simulate.simulate_contract(
        pam_attributes,
        time_series={
            "RATE": [
                ["2024-01-01", 0.04],
                ["2024-07-01", 0.045],
                ["2025-01-01", 0.05],
            ]
        },
    )

    assert result["success"] is True
    assert result["num_events"] > 0


def test_simulate_time_series_linear_interpolation(pam_attributes):
    """Test simulation with linear interpolation time series."""
    result = simulate.simulate_contract(
        pam_attributes,
        time_series={"RATE": [["2024-01-01", 0.04], ["2025-01-01", 0.05]]},
        interpolation="linear",
    )
    assert result["success"] is True


def test_simulate_time_series_takes_priority_over_risk_factors(pam_attributes):
    """Test that time_series takes priority when both are provided."""
    result = simulate.simulate_contract(
        pam_attributes,
        risk_factors={"RATE": 999.0},
        time_series={"RATE": [["2024-01-01", 0.04]]},
    )
    assert result["success"] is True


def test_simulate_time_series_invalid_format():
    """Test error handling for invalid time series format."""
    attrs = {
        "contract_type": "PAM",
        "contract_id": "TEST",
        "contract_role": "RPA",
        "status_date": "2024-01-01",
        "initial_exchange_date": "2024-01-15",
        "maturity_date": "2025-01-15",
        "notional_principal": 100000.0,
        "nominal_interest_rate": 0.05,
    }
    result = simulate.simulate_contract(
        attrs,
        time_series={"RATE": [["2024-01-01"]]},  # Missing value
    )
    assert result["success"] is False
    assert "error" in result


def test_simulate_granular_error_invalid_enum():
    """Test that invalid enum values return an error with error_type field."""
    result = simulate.simulate_contract({
        "contract_type": "INVALID",
        "contract_role": "RPA",
        "status_date": "2024-01-01",
    })
    assert result["success"] is False
    assert result["error_type"] in ("invalid_attribute", "validation_error")


def test_simulate_granular_error_missing_fields():
    """Test that missing required fields return appropriate error."""
    result = simulate.simulate_contract({
        "contract_type": "PAM",
        "contract_role": "RPA",
        # Missing status_date
    })
    assert result["success"] is False
    assert "error_type" in result


def test_simulate_composite_contract_without_child_contracts():
    """Test that composite contracts without child_contracts return helpful error."""
    for ct in ("CAPFL", "SWAPS", "CEG", "CEC"):
        result = simulate.simulate_contract({
            "contract_type": ct,
            "contract_id": f"TEST-{ct}",
            "contract_role": "RPA",
            "status_date": "2024-01-01",
        })
        assert result["success"] is False
        assert result["error_type"] == "missing_child_contracts"
        assert "child_contracts" in result["error"]
        assert "jactus_get_contract_schema" in result["hint"]


# ---- Child contract simulation tests ----


def test_simulate_swaps_with_child_contracts():
    """Test simulating a SWAPS contract with two PAM legs via child_contracts."""
    parent_attrs = {
        "contract_type": "SWAPS",
        "contract_id": "SWAP-001",
        "contract_role": "RFL",
        "status_date": "2024-01-01",
        "maturity_date": "2029-01-01",
        "contract_structure": '{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
    }
    child_contracts = {
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
    }
    result = simulate.simulate_contract(parent_attrs, child_contracts=child_contracts)
    assert result["success"] is True
    assert result["contract_type"] == "SWAPS"
    assert result["num_events"] > 0
    assert "child_results" in result
    assert "LEG1" in result["child_results"]
    assert "LEG2" in result["child_results"]


def test_simulate_ceg_with_child_contracts():
    """Test simulating a CEG contract with an underlier loan via child_contracts."""
    parent_attrs = {
        "contract_type": "CEG",
        "contract_id": "CEG-001",
        "contract_role": "BUY",
        "status_date": "2024-01-01",
        "initial_exchange_date": "2024-01-01",
        "maturity_date": "2029-01-01",
        "notional_principal": 500000.0,
        "coverage": 1.0,
        "contract_structure": '{"CoveredContract": "LOAN-001"}',
    }
    child_contracts = {
        "LOAN-001": {
            "contract_type": "PAM", "contract_id": "LOAN-001",
            "contract_role": "RPA", "status_date": "2024-01-01",
            "initial_exchange_date": "2024-01-01", "maturity_date": "2029-01-01",
            "notional_principal": 500000.0, "nominal_interest_rate": 0.04,
        },
    }
    result = simulate.simulate_contract(parent_attrs, child_contracts=child_contracts)
    assert result["success"] is True
    assert result["contract_type"] == "CEG"
    assert "child_results" in result
    assert "LOAN-001" in result["child_results"]


def test_simulate_child_contract_invalid_child():
    """Test that an invalid child contract returns a clear error."""
    parent_attrs = {
        "contract_type": "SWAPS",
        "contract_id": "SWAP-001",
        "contract_role": "RFL",
        "status_date": "2024-01-01",
        "maturity_date": "2029-01-01",
        "contract_structure": '{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
    }
    child_contracts = {
        "LEG1": {
            "contract_type": "PAM", "contract_id": "LEG1",
            "contract_role": "RPA", "status_date": "2024-01-01",
            "initial_exchange_date": "2024-01-01", "maturity_date": "2029-01-01",
            "notional_principal": 1000000.0,
        },
        "LEG2": {
            # Missing contract_role — should fail
            "contract_type": "INVALID_TYPE", "contract_id": "LEG2",
        },
    }
    result = simulate.simulate_contract(parent_attrs, child_contracts=child_contracts)
    assert result["success"] is False
    assert result["error_type"] == "child_simulation_error"
    assert "LEG2" in result["error"]


def test_simulate_child_results_summary():
    """Test that child_results includes summary for each child."""
    parent_attrs = {
        "contract_type": "CEG",
        "contract_id": "CEG-001",
        "contract_role": "BUY",
        "status_date": "2024-01-01",
        "initial_exchange_date": "2024-01-01",
        "maturity_date": "2029-01-01",
        "notional_principal": 500000.0,
        "coverage": 1.0,
        "contract_structure": '{"CoveredContract": "LOAN-001"}',
    }
    child_contracts = {
        "LOAN-001": {
            "contract_type": "PAM", "contract_id": "LOAN-001",
            "contract_role": "RPA", "status_date": "2024-01-01",
            "initial_exchange_date": "2024-01-01", "maturity_date": "2029-01-01",
            "notional_principal": 500000.0, "nominal_interest_rate": 0.04,
        },
    }
    result = simulate.simulate_contract(parent_attrs, child_contracts=child_contracts)
    assert result["success"] is True
    child = result["child_results"]["LOAN-001"]
    assert "contract_type" in child
    assert child["contract_type"] == "PAM"
    assert "num_events" in child
    assert child["num_events"] > 0
    assert "net_cashflow" in child


# ---- Pagination tests ----


@pytest.fixture
def lam_attributes():
    """LAM contract with monthly payments over 4+ years (many events)."""
    return {
        "contract_type": "LAM",
        "contract_id": "TEST-LAM-001",
        "contract_role": "RPA",
        "status_date": "2024-01-01",
        "initial_exchange_date": "2024-01-01",
        "maturity_date": "2028-01-01",
        "notional_principal": 500000.0,
        "nominal_interest_rate": 0.05,
        "interest_payment_cycle": "1M",
        "principal_redemption_cycle": "1M",
        "next_principal_redemption_amount": 10000.0,
        "day_count_convention": "A365",
    }


def test_simulate_event_limit(lam_attributes):
    """Test that event_limit restricts the number of returned events."""
    result = simulate.simulate_contract(lam_attributes, event_limit=3)

    assert result["success"] is True
    assert len(result["events"]) == 3
    assert result["num_events"] > 3  # total is larger
    assert "pagination" in result
    assert result["pagination"]["limit"] == 3
    assert result["pagination"]["returned"] == 3
    assert result["pagination"]["total_events"] == result["num_events"]


def test_simulate_event_offset(lam_attributes):
    """Test that event_offset skips events from the beginning."""
    full = simulate.simulate_contract(lam_attributes)
    offset_result = simulate.simulate_contract(lam_attributes, event_offset=2)

    assert offset_result["success"] is True
    assert "pagination" in offset_result
    assert offset_result["pagination"]["offset"] == 2
    # First event of offset result should be the third event of the full result
    assert offset_result["events"][0] == full["events"][2]


def test_simulate_event_limit_and_offset(lam_attributes):
    """Test combined limit and offset for pagination."""
    full = simulate.simulate_contract(lam_attributes)
    page = simulate.simulate_contract(lam_attributes, event_limit=2, event_offset=1)

    assert page["success"] is True
    assert len(page["events"]) == 2
    assert page["events"][0] == full["events"][1]
    assert page["events"][1] == full["events"][2]
    assert page["pagination"]["total_events"] == full["num_events"]


def test_simulate_summary_covers_all_events(lam_attributes):
    """Test that summary is computed from all events, not just the page."""
    full = simulate.simulate_contract(lam_attributes)
    page = simulate.simulate_contract(lam_attributes, event_limit=2)

    # Summaries should be identical regardless of pagination
    assert page["summary"] == full["summary"]


def test_simulate_auto_truncation_with_states(lam_attributes):
    """Test that include_states auto-truncates when output is too large."""
    result = simulate.simulate_contract(lam_attributes, include_states=True)

    assert result["success"] is True
    # With 4 years of monthly events + states, output should be auto-truncated
    if result["num_events"] > 10:
        assert "pagination" in result
        assert result["pagination"].get("truncated") is True
        assert len(result["events"]) <= 10
        assert result["pagination"]["total_events"] == result["num_events"]


def test_simulate_no_truncation_without_states(lam_attributes):
    """Test that without include_states, output is not auto-truncated."""
    result = simulate.simulate_contract(lam_attributes, include_states=False)

    assert result["success"] is True
    assert len(result["events"]) == result["num_events"]
    assert "pagination" not in result


def test_simulate_explicit_limit_skips_auto_truncation(lam_attributes):
    """Test that explicit event_limit prevents auto-truncation."""
    result = simulate.simulate_contract(
        lam_attributes, include_states=True, event_limit=5
    )

    assert result["success"] is True
    assert len(result["events"]) == 5
    assert result["pagination"]["limit"] == 5
    # Should not have truncated flag since explicit limit was used
    assert result["pagination"].get("truncated") is not True
