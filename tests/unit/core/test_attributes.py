"""Unit tests for contract attributes.

Test ID: T1.3
"""

import pytest
from pydantic import ValidationError

from jactus.core.attributes import ATTRIBUTE_MAP, ContractAttributes
from jactus.core.time import ActusDateTime
from jactus.core.types import (
    BusinessDayConvention,
    Calendar,
    ContractPerformance,
    ContractRole,
    ContractType,
    DayCountConvention,
    EndOfMonthConvention,
    FeeBasis,
    PrepaymentEffect,
    ScalingEffect,
)


class TestContractAttributesBasic:
    """Test basic Contract Attributes creation and validation."""

    def test_minimal_creation(self):
        """Test creating contract attributes with minimal required fields."""
        attrs = ContractAttributes(
            contract_id="TEST-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        assert attrs.contract_id == "TEST-001"
        assert attrs.contract_type == ContractType.PAM
        assert attrs.contract_role == ContractRole.RPA
        assert attrs.status_date == ActusDateTime(2024, 1, 1, 0, 0, 0)

    def test_full_pam_contract(self):
        """Test creating a complete PAM contract."""
        attrs = ContractAttributes(
            contract_id="LOAN-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime.from_iso("2024-01-01T00:00:00"),
            initial_exchange_date=ActusDateTime.from_iso("2024-01-15T00:00:00"),
            maturity_date=ActusDateTime.from_iso("2029-01-15T00:00:00"),
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            currency="USD",
            day_count_convention=DayCountConvention.A360,
            business_day_convention=BusinessDayConvention.SCF,
            interest_payment_cycle="3M",
            interest_payment_anchor=ActusDateTime.from_iso("2024-04-15T00:00:00"),
        )

        assert attrs.notional_principal == 100000.0
        assert attrs.nominal_interest_rate == 0.05
        assert attrs.currency == "USD"
        assert attrs.day_count_convention == DayCountConvention.A360
        assert attrs.interest_payment_cycle == "3M"

    def test_defaults_applied(self):
        """Test that default values are properly applied."""
        attrs = ContractAttributes(
            contract_id="TEST-002",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )

        assert attrs.currency == "USD"
        assert attrs.business_day_convention == BusinessDayConvention.NULL
        assert attrs.end_of_month_convention == EndOfMonthConvention.SD
        assert attrs.calendar == Calendar.NO_CALENDAR
        assert attrs.prepayment_effect == PrepaymentEffect.N
        assert attrs.scaling_effect == ScalingEffect.S000
        assert attrs.contract_performance == ContractPerformance.PF


class TestValidation:
    """Test Pydantic validation rules."""

    def test_interest_rate_validation_negative(self):
        """Test that interest rate can be negative but not <= -1."""
        # Negative rates are OK (down to -0.99...)
        attrs = ContractAttributes(
            contract_id="TEST-003",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nominal_interest_rate=-0.005,  # -0.5% is valid
        )
        assert attrs.nominal_interest_rate == -0.005

        # But -1 or lower should fail
        with pytest.raises(ValidationError) as exc_info:
            ContractAttributes(
                contract_id="TEST-004",
                contract_type=ContractType.PAM,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                nominal_interest_rate=-1.0,
            )
        assert "must be > -1" in str(exc_info.value)

    def test_notional_validation_non_zero(self):
        """Test that notional must be non-zero."""
        with pytest.raises(ValidationError) as exc_info:
            ContractAttributes(
                contract_id="TEST-005",
                contract_type=ContractType.PAM,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                notional_principal=0.0,
            )
        assert "must be non-zero" in str(exc_info.value)

    def test_currency_validation_format(self):
        """Test currency code validation."""
        # Valid currency
        attrs = ContractAttributes(
            contract_id="TEST-006",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="EUR",
        )
        assert attrs.currency == "EUR"

        # Too short
        with pytest.raises(ValidationError):
            ContractAttributes(
                contract_id="TEST-007",
                contract_type=ContractType.PAM,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="US",
            )

        # Too long
        with pytest.raises(ValidationError):
            ContractAttributes(
                contract_id="TEST-008",
                contract_type=ContractType.PAM,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="USDD",
            )

        # Not uppercase
        with pytest.raises(ValidationError):
            ContractAttributes(
                contract_id="TEST-009",
                contract_type=ContractType.PAM,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="usd",
            )

    def test_date_ordering_ied_after_sd(self):
        """Test that IED must be >= SD."""
        # Valid: IED after SD
        attrs = ContractAttributes(
            contract_id="TEST-010",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
        )
        assert attrs.initial_exchange_date > attrs.status_date

        # IED before SD is allowed per ACTUS spec (contract already existed)
        attrs2 = ContractAttributes(
            contract_id="TEST-011",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )
        assert attrs2.initial_exchange_date < attrs2.status_date

    def test_date_ordering_md_after_ied(self):
        """Test that MD must be > IED."""
        # Valid: MD after IED
        attrs = ContractAttributes(
            contract_id="TEST-012",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
        )
        assert attrs.maturity_date > attrs.initial_exchange_date

        # Invalid: MD same as IED
        with pytest.raises(ValidationError) as exc_info:
            ContractAttributes(
                contract_id="TEST-013",
                contract_type=ContractType.PAM,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
                maturity_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            )
        assert "must be >" in str(exc_info.value)

    def test_array_schedule_length_matching(self):
        """Test that array schedule elements have matching lengths."""
        # Valid: all PR arrays same length
        attrs = ContractAttributes(
            contract_id="TEST-014",
            contract_type=ContractType.ANN,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            array_pr_anchor=[
                ActusDateTime(2024, 1, 15, 0, 0, 0),
                ActusDateTime(2024, 4, 15, 0, 0, 0),
            ],
            array_pr_cycle=["3M", "3M"],
            array_pr_next=[1000.0, 1000.0],
        )
        assert len(attrs.array_pr_anchor) == 2

        # Invalid: mismatched lengths
        with pytest.raises(ValidationError) as exc_info:
            ContractAttributes(
                contract_id="TEST-015",
                contract_type=ContractType.ANN,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                array_pr_anchor=[
                    ActusDateTime(2024, 1, 15, 0, 0, 0),
                    ActusDateTime(2024, 4, 15, 0, 0, 0),
                ],
                array_pr_cycle=["3M"],  # Only 1 element!
            )
        assert "must have same length" in str(exc_info.value)


class TestAttributeMap:
    """Test ACTUS name mapping."""

    def test_attribute_map_completeness(self):
        """Test that ATTRIBUTE_MAP is defined."""
        assert isinstance(ATTRIBUTE_MAP, dict)
        assert len(ATTRIBUTE_MAP) > 0

    def test_attribute_map_keys(self):
        """Test some expected ACTUS names are in map."""
        expected_keys = ["NT", "IPNR", "IED", "MD", "SD", "CT", "CNTRL", "DCC", "CUR"]
        for key in expected_keys:
            assert key in ATTRIBUTE_MAP

    def test_attribute_map_values(self):
        """Test that mapped values are valid Python attribute names."""
        assert ATTRIBUTE_MAP["NT"] == "notional_principal"
        assert ATTRIBUTE_MAP["IPNR"] == "nominal_interest_rate"
        assert ATTRIBUTE_MAP["IED"] == "initial_exchange_date"
        assert ATTRIBUTE_MAP["MD"] == "maturity_date"


class TestGetSetAttribute:
    """Test get_attribute and set_attribute methods."""

    def test_get_attribute_by_actus_name(self):
        """Test getting attributes by ACTUS short name."""
        attrs = ContractAttributes(
            contract_id="TEST-016",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
        )

        assert attrs.get_attribute("NT") == 100000.0
        assert attrs.get_attribute("IPNR") == 0.05
        assert attrs.get_attribute("CT") == ContractType.PAM

    def test_get_attribute_invalid_name(self):
        """Test that invalid ACTUS names raise KeyError."""
        attrs = ContractAttributes(
            contract_id="TEST-017",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )

        with pytest.raises(KeyError):
            attrs.get_attribute("INVALID")

    def test_set_attribute_by_actus_name(self):
        """Test setting attributes by ACTUS short name."""
        attrs = ContractAttributes(
            contract_id="TEST-018",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            notional_principal=100000.0,
        )

        # Change notional
        attrs.set_attribute("NT", 150000.0)
        assert attrs.notional_principal == 150000.0
        assert attrs.get_attribute("NT") == 150000.0

        # Set interest rate
        attrs.set_attribute("IPNR", 0.045)
        assert attrs.nominal_interest_rate == 0.045

    def test_set_attribute_validates(self):
        """Test that set_attribute enforces validation."""
        attrs = ContractAttributes(
            contract_id="TEST-019",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )

        # Should fail validation
        with pytest.raises(ValidationError):
            attrs.set_attribute("NT", 0.0)  # Zero notional

    def test_is_attribute_defined(self):
        """Test checking if attribute is defined."""
        attrs = ContractAttributes(
            contract_id="TEST-020",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            notional_principal=100000.0,
        )

        assert attrs.is_attribute_defined("NT") is True
        assert attrs.is_attribute_defined("IPNR") is False  # Not set
        assert attrs.is_attribute_defined("MD") is False  # Not set


class TestSerialization:
    """Test dictionary serialization."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        attrs = ContractAttributes(
            contract_id="TEST-021",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            notional_principal=100000.0,
            currency="EUR",
        )

        # Pydantic v2 model_dump
        data = attrs.model_dump()

        assert isinstance(data, dict)
        assert data["contract_id"] == "TEST-021"
        assert data["contract_type"] == ContractType.PAM
        assert data["notional_principal"] == 100000.0
        assert data["currency"] == "EUR"

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "contract_id": "TEST-022",
            "contract_type": ContractType.PAM,
            "contract_role": ContractRole.RPA,
            "status_date": ActusDateTime(2024, 1, 1, 0, 0, 0),
            "notional_principal": 100000.0,
        }

        # Pydantic v2 model_validate
        attrs = ContractAttributes.model_validate(data)

        assert attrs.contract_id == "TEST-022"
        assert attrs.notional_principal == 100000.0

    def test_roundtrip_serialization(self):
        """Test that serialization round-trips correctly."""
        original = ContractAttributes(
            contract_id="TEST-023",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            currency="USD",
            day_count_convention=DayCountConvention.A360,
        )

        # Convert to dict and back
        data = original.model_dump()
        restored = ContractAttributes.model_validate(data)

        assert restored.contract_id == original.contract_id
        assert restored.notional_principal == original.notional_principal
        assert restored.nominal_interest_rate == original.nominal_interest_rate
        assert restored.day_count_convention == original.day_count_convention


class TestComplexContracts:
    """Test complex contract configurations."""

    def test_ann_with_array_schedule(self):
        """Test ANN contract with array schedules."""
        attrs = ContractAttributes(
            contract_id="ANN-001",
            contract_type=ContractType.ANN,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            currency="USD",
            # Array schedule for principal redemption
            array_pr_anchor=[
                ActusDateTime(2024, 4, 15, 0, 0, 0),
                ActusDateTime(2024, 7, 15, 0, 0, 0),
                ActusDateTime(2024, 10, 15, 0, 0, 0),
            ],
            array_pr_cycle=["3M", "3M", "3M"],
            array_pr_next=[5000.0, 5000.0, 5000.0],
        )

        assert len(attrs.array_pr_anchor) == 3
        assert attrs.array_pr_cycle[0] == "3M"
        assert attrs.array_pr_next[2] == 5000.0

    def test_variable_rate_contract(self):
        """Test contract with rate reset attributes."""
        attrs = ContractAttributes(
            contract_id="VAR-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            notional_principal=100000.0,
            nominal_interest_rate=0.03,
            currency="USD",
            rate_reset_cycle="6M",
            rate_reset_anchor=ActusDateTime(2024, 7, 15, 0, 0, 0),
            rate_reset_market_object="LIBOR_6M",
            rate_reset_spread=0.02,
            rate_reset_floor=0.01,
            rate_reset_cap=0.10,
        )

        assert attrs.rate_reset_cycle == "6M"
        assert attrs.rate_reset_market_object == "LIBOR_6M"
        assert attrs.rate_reset_spread == 0.02
        assert attrs.rate_reset_floor == 0.01
        assert attrs.rate_reset_cap == 0.10

    def test_contract_with_fees(self):
        """Test contract with fee attributes."""
        attrs = ContractAttributes(
            contract_id="FEE-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            notional_principal=100000.0,
            currency="USD",
            fee_rate=0.001,
            fee_basis=FeeBasis.N,  # Notional percentage
            fee_payment_cycle="1Y",
            fee_payment_anchor=ActusDateTime(2025, 1, 15, 0, 0, 0),
        )

        assert attrs.fee_rate == 0.001
        assert attrs.fee_basis == FeeBasis.N
        assert attrs.fee_payment_cycle == "1Y"
