"""Integration tests for contract attributes validation.

T1.11: Contract Attributes Validation
Tests complex validation scenarios across contract components:
- Complete PAM contract with all schedule components
- ANN contract with rate resets (simplified)
- SWAPS dual legs (simplified)
- Cross-module validation (attributes + schedules + calendars)
"""

from jactus.core import ActusDateTime, ContractAttributes
from jactus.core.types import (
    BusinessDayConvention,
    Calendar,
    ContractRole,
    ContractType,
    DayCountConvention,
    EndOfMonthConvention,
)
from jactus.utilities import MondayToFridayCalendar, generate_schedule


class TestPAMContractComplete:
    """Test complete PAM contract with schedules."""

    def test_pam_with_interest_payment_schedule(self):
        """PAM contract integrated with schedule generation."""
        # Create contract attributes
        attrs = ContractAttributes(
            contract_id="PAM001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A365,
            currency="USD",
            # Schedule parameters
            interest_payment_cycle="3M",
            interest_payment_anchor=ActusDateTime(2024, 1, 15, 0, 0, 0),
            business_day_convention=BusinessDayConvention.SCF,
            calendar=Calendar.MONDAY_TO_FRIDAY,
        )

        # Generate interest payment schedule using contract attributes
        calendar = MondayToFridayCalendar()
        schedule = generate_schedule(
            start=attrs.interest_payment_anchor,
            cycle=attrs.interest_payment_cycle,
            end=attrs.maturity_date,
            business_day_convention=attrs.business_day_convention,
            calendar=calendar,
        )

        # Verify schedule is consistent with contract
        assert schedule[0] == attrs.interest_payment_anchor
        assert schedule[-1] >= attrs.maturity_date or schedule[-2] == ActusDateTime(
            2029, 1, 15, 0, 0, 0
        )
        # Quarterly payments for 5 years = 20 payments
        assert len(schedule) >= 19

    def test_pam_minimal(self):
        """Minimal valid PAM contract."""
        attrs = ContractAttributes(
            contract_id="PAM002",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A365,
            currency="USD",
        )

        assert attrs.notional_principal == 100000.0
        assert attrs.nominal_interest_rate == 0.05
        assert attrs.contract_type == ContractType.PAM


class TestANNContractWithSchedules:
    """Test ANN contract with principal redemption schedules."""

    def test_ann_with_monthly_payments(self):
        """ANN contract with monthly principal + interest payments."""
        attrs = ContractAttributes(
            contract_id="ANN001",
            contract_type=ContractType.ANN,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            notional_principal=100000.0,
            nominal_interest_rate=0.045,
            day_count_convention=DayCountConvention.A360,
            currency="USD",
            principal_redemption_cycle="1M",
        )

        # Generate payment schedule
        if attrs.principal_redemption_cycle:
            schedule = generate_schedule(
                start=attrs.initial_exchange_date,
                cycle=attrs.principal_redemption_cycle,
                end=attrs.maturity_date,
            )

            # Monthly payments for 5 years = 60 payments
            assert len(schedule) == 61  # Including start


class TestSWAPSDualLegs:
    """Test swap contract structure (dual legs)."""

    def test_swaps_complementary_legs(self):
        """SWAPS modeled as two complementary contracts."""
        # Fixed leg (receive fixed)
        fixed_leg = ContractAttributes(
            contract_id="SWAP001_FIXED",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.04,
            day_count_convention=DayCountConvention.A360,
            currency="USD",
            interest_payment_cycle="6M",
        )

        # Floating leg (pay floating)
        floating_leg = ContractAttributes(
            contract_id="SWAP001_FLOATING",
            contract_type=ContractType.SWPPV,
            contract_role=ContractRole.PFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            notional_principal=1000000.0,
            nominal_interest_rate=0.03,
            day_count_convention=DayCountConvention.A360,
            currency="USD",
            interest_payment_cycle="3M",
        )

        # Verify complementary structure
        assert fixed_leg.notional_principal == floating_leg.notional_principal
        assert fixed_leg.maturity_date == floating_leg.maturity_date
        assert fixed_leg.contract_role == ContractRole.RFL
        assert floating_leg.contract_role == ContractRole.PFL


class TestCrossModuleIntegration:
    """Test integration between attributes, schedules, and calendars."""

    def test_contract_with_eom_and_calendar(self):
        """Contract with EOM convention and business day calendar."""
        attrs = ContractAttributes(
            contract_id="INT001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 31, 0, 0, 0),
            maturity_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A365,
            currency="USD",
            interest_payment_cycle="1M",
            interest_payment_anchor=ActusDateTime(2024, 1, 31, 0, 0, 0),
            business_day_convention=BusinessDayConvention.SCF,
            calendar=Calendar.MONDAY_TO_FRIDAY,
            end_of_month_convention=EndOfMonthConvention.EOM,
        )

        # Generate schedule using all conventions
        calendar = MondayToFridayCalendar()
        schedule = generate_schedule(
            start=attrs.interest_payment_anchor,
            cycle=attrs.interest_payment_cycle,
            end=attrs.maturity_date,
            end_of_month_convention=attrs.end_of_month_convention,
            business_day_convention=attrs.business_day_convention,
            calendar=calendar,
        )

        # All dates should be business days after adjustment
        for date in schedule:
            assert calendar.is_business_day(date)

        # Should have monthly payments
        assert len(schedule) == 12

    def test_multiple_contracts_same_calendar(self):
        """Multiple contracts sharing same calendar."""
        calendar = MondayToFridayCalendar()

        # Contract 1: Quarterly payments
        contract1 = ContractAttributes(
            contract_id="C001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A365,
            currency="USD",
            interest_payment_cycle="3M",
        )

        # Contract 2: Monthly payments
        contract2 = ContractAttributes(
            contract_id="C002",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            notional_principal=200000.0,
            nominal_interest_rate=0.04,
            day_count_convention=DayCountConvention.A360,
            currency="USD",
            interest_payment_cycle="1M",
        )

        # Generate schedules
        schedule1 = generate_schedule(
            start=contract1.initial_exchange_date,
            cycle=contract1.interest_payment_cycle,
            end=contract1.maturity_date,
            business_day_convention=BusinessDayConvention.SCF,
            calendar=calendar,
        )

        schedule2 = generate_schedule(
            start=contract2.initial_exchange_date,
            cycle=contract2.interest_payment_cycle,
            end=contract2.maturity_date,
            business_day_convention=BusinessDayConvention.SCF,
            calendar=calendar,
        )

        # Verify different schedules
        assert len(schedule1) == 5  # Quarterly
        assert len(schedule2) == 13  # Monthly
        # First schedule dates are subset
        for date in schedule1:
            if date in schedule2:
                # Some dates may overlap
                pass


class TestAttributeAccessors:
    """Test ACTUS attribute name accessors."""

    def test_get_set_by_actus_names(self):
        """Get and set attributes by ACTUS short names."""
        attrs = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A365,
            currency="USD",
        )

        # Get by ACTUS name
        assert attrs.get_attribute("CNTRL") == ContractRole.RPA
        assert attrs.get_attribute("NT") == 100000.0
        assert attrs.get_attribute("IPNR") == 0.05

        # Set by ACTUS name - modifies in place and returns None
        attrs.set_attribute("IPNR", 0.06)
        assert attrs.get_attribute("IPNR") == 0.06
        assert attrs.nominal_interest_rate == 0.06

    def test_is_attribute_defined(self):
        """Check if attributes are defined."""
        attrs = ContractAttributes(
            contract_id="TEST002",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A365,
            currency="USD",
        )

        # Required attributes
        assert attrs.is_attribute_defined("IED")
        assert attrs.is_attribute_defined("NT")
        # Optional not set
        assert not attrs.is_attribute_defined("IPCL")
