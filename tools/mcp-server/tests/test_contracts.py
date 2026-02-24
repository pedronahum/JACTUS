"""Tests for contract discovery tools."""

import pytest
from jactus_mcp.tools import contracts


def test_list_contracts():
    """Test listing all contracts."""
    result = contracts.list_contracts()

    assert "total_contracts" in result
    assert result["total_contracts"] == 18
    assert "categories" in result
    assert "principal" in result["categories"]
    assert "derivative" in result["categories"]
    assert "all_contracts" in result

    # Check specific contracts exist
    assert "PAM" in result["all_contracts"]
    assert "SWPPV" in result["all_contracts"]
    assert "OPTNS" in result["all_contracts"]


def test_get_contract_info_valid():
    """Test getting info for a valid contract."""
    result = contracts.get_contract_info("PAM")

    assert result["contract_type"] == "PAM"
    assert result["implemented"] is True
    assert "description" in result
    assert "Principal at Maturity" in result["description"]
    assert result["category"] == "principal"


def test_get_contract_info_invalid():
    """Test getting info for an invalid contract."""
    result = contracts.get_contract_info("INVALID")

    assert "error" in result
    assert "available" in result


def test_get_contract_schema():
    """Test getting contract schema."""
    result = contracts.get_contract_schema("PAM")

    assert result["contract_type"] == "PAM"
    assert "required_fields" in result
    assert "optional_fields" in result
    assert "contract_type" in result["required_fields"]
    assert "contract_id" in result["required_fields"]
    assert "status_date" in result["required_fields"]
    assert "contract_id" not in result["optional_fields"]
    assert "example_usage" in result


def test_get_contract_schema_swppv():
    """Test schema for swap contract."""
    result = contracts.get_contract_schema("SWPPV")

    assert "interest_payment_cycle" in result["required_fields"]
    assert "rate_reset_cycle" in result["required_fields"]
    assert "nominal_interest_rate_2" in result["required_fields"]


def test_get_event_types():
    """Test getting event types."""
    result = contracts.get_event_types()

    assert "total_events" in result
    assert result["total_events"] > 0
    assert "event_types" in result

    # Check common events exist
    assert "IED" in result["event_types"]
    assert "IP" in result["event_types"]
    assert "MD" in result["event_types"]

    # Check descriptions
    assert "Initial Exchange" in result["event_types"]["IED"]


def test_list_risk_factor_observers():
    """Test listing available risk factor observers."""
    result = contracts.list_risk_factor_observers()

    assert "observers" in result
    assert "guidance" in result
    assert "ConstantRiskFactorObserver" in result["observers"]
    assert "DictRiskFactorObserver" in result["observers"]
    assert "TimeSeriesRiskFactorObserver" in result["observers"]
    assert "CurveRiskFactorObserver" in result["observers"]
    assert "CompositeRiskFactorObserver" in result["observers"]
    assert "CallbackRiskFactorObserver" in result["observers"]
    assert "JaxRiskFactorObserver" in result["observers"]


def test_list_risk_factor_observers_has_mcp_params():
    """Test that MCP-available observers have mcp_param field."""
    result = contracts.list_risk_factor_observers()

    assert "mcp_param" in result["observers"]["ConstantRiskFactorObserver"]
    assert "mcp_param" in result["observers"]["DictRiskFactorObserver"]
    assert "mcp_param" in result["observers"]["TimeSeriesRiskFactorObserver"]


def test_list_risk_factor_observers_python_only_flagged():
    """Test that Python-only observers are flagged."""
    result = contracts.list_risk_factor_observers()

    assert result["observers"]["CurveRiskFactorObserver"].get("python_only") is True
    assert result["observers"]["CompositeRiskFactorObserver"].get("python_only") is True
    assert result["observers"]["CallbackRiskFactorObserver"].get("python_only") is True
    assert result["observers"]["JaxRiskFactorObserver"].get("python_only") is True
