"""Property-based tests for contract invariants (T3.11).

Uses Hypothesis to test fundamental properties that should hold
for all contracts regardless of parameters.
"""

import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from jactus.contracts import create_contract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractType,
    DayCountConvention,
)
from jactus.observers import ConstantRiskFactorObserver


class TestCashContractProperties:
    """Property tests for CSH contract."""

    @given(
        notional=st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_csh_notional_absolute_value_preserved(self, notional):
        """Test CSH contract preserves notional absolute value in state."""
        attrs = ContractAttributes(
            contract_id="CSH-PROP-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=notional,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
        contract = create_contract(attrs, rf_obs)

        # Initialize state
        state = contract.initialize_state()

        # Notional absolute value should match
        assert abs(abs(float(state.nt)) - notional) < notional * 0.001

    @given(
        notional=st.floats(min_value=1.0, max_value=1e9, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_csh_simulation_always_succeeds(self, notional):
        """Test CSH simulation never fails regardless of notional."""
        attrs = ContractAttributes(
            contract_id="CSH-PROP-002",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=notional,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
        contract = create_contract(attrs, rf_obs)

        # Should never raise
        result = contract.simulate()
        assert result is not None
        assert len(result.events) >= 1


class TestPAMContractProperties:
    """Property tests for PAM contract."""

    @given(
        notional=st.floats(min_value=1000.0, max_value=1e7, allow_nan=False, allow_infinity=False),
        rate=st.floats(min_value=0.0, max_value=0.2, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_pam_ied_payoff_equals_notional(self, notional, rate):
        """Test PAM IED event payoff equals notional."""
        attrs = ContractAttributes(
            contract_id="PAM-PROP-001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=notional,
            nominal_interest_rate=rate,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=rate)
        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        # Find IED event
        ied_event = next((e for e in result.events if e.event_type.value == "IED"), None)
        assert ied_event is not None

        # IED payoff should equal notional (negative for borrower)
        assert abs(float(ied_event.payoff) + notional) < notional * 0.001  # 0.1% tolerance

    @given(
        notional=st.floats(min_value=1000.0, max_value=1e7, allow_nan=False, allow_infinity=False),
        rate=st.floats(min_value=0.01, max_value=0.2, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_pam_generates_interest_payments(self, notional, rate):
        """Test PAM contracts generate interest payment events."""
        attrs = ContractAttributes(
            contract_id="PAM-PROP-002",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 7, 15, 0, 0, 0),  # 18 months
            currency="USD",
            notional_principal=notional,
            nominal_interest_rate=rate,
            day_count_convention=DayCountConvention.A360,
            interest_payment_cycle="6M",  # Semi-annual
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=rate)
        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        # Count IP events
        ip_events = [e for e in result.events if e.event_type.value == "IP"]

        # Should have at least one IP event
        assert len(ip_events) >= 1

    @given(
        notional=st.floats(min_value=1000.0, max_value=1e7, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30, deadline=None)
    def test_pam_md_repays_principal(self, notional):
        """Test PAM MD event repays principal."""
        attrs = ContractAttributes(
            contract_id="PAM-PROP-003",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=notional,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        # Find MD event
        md_event = next((e for e in result.events if e.event_type.value == "MD"), None)
        assert md_event is not None

        # MD payoff should be at least notional (positive for borrower paying back)
        assert float(md_event.payoff) >= notional * 0.99  # Allow small rounding


class TestStockContractProperties:
    """Property tests for STK contract."""

    @given(
        purchase_price=st.floats(
            min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        sale_price=st.floats(
            min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_stk_profit_loss_calculation(self, purchase_price, sale_price):
        """Test STK contract correctly calculates profit/loss."""
        attrs = ContractAttributes(
            contract_id="STK-PROP-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            termination_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            price_at_purchase_date=purchase_price,
            price_at_termination_date=sale_price,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=100.0)
        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        # Calculate net cashflow
        cashflows = result.get_cashflows()
        net_cashflow = sum(float(amt) for _, amt, _ in cashflows)

        # Net should equal sale - purchase
        expected_profit = sale_price - purchase_price
        assert abs(net_cashflow - expected_profit) < 0.01


class TestCommodityContractProperties:
    """Property tests for COM contract."""

    @given(
        purchase_price=st.floats(
            min_value=100.0, max_value=10000.0, allow_nan=False, allow_infinity=False
        ),
        sale_price=st.floats(
            min_value=100.0, max_value=10000.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_com_profit_loss_calculation(self, purchase_price, sale_price):
        """Test COM contract correctly calculates profit/loss."""
        attrs = ContractAttributes(
            contract_id="COM-PROP-001",
            contract_type=ContractType.COM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            termination_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            price_at_purchase_date=purchase_price,
            price_at_termination_date=sale_price,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=5000.0)
        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        # Calculate net cashflow
        cashflows = result.get_cashflows()
        net_cashflow = sum(float(amt) for _, amt, _ in cashflows)

        # Net should equal sale - purchase
        expected_profit = sale_price - purchase_price
        assert abs(net_cashflow - expected_profit) < 0.01


class TestGeneralContractProperties:
    """General properties that should hold for all contracts."""

    @given(
        contract_type=st.sampled_from([ContractType.CSH, ContractType.STK, ContractType.COM]),
    )
    @settings(max_examples=20, deadline=None)
    def test_events_monotonically_increasing(self, contract_type):
        """Test contract events are monotonically increasing in time."""
        if contract_type == ContractType.CSH:
            attrs = ContractAttributes(
                contract_id="PROP-001",
                contract_type=contract_type,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="USD",
                notional_principal=100000.0,
            )
        else:  # STK or COM
            attrs = ContractAttributes(
                contract_id="PROP-001",
                contract_type=contract_type,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
                termination_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
                currency="USD",
                price_at_purchase_date=100.0,
                price_at_termination_date=120.0,
            )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        # Check monotonic time
        for i in range(len(result.events) - 1):
            curr_time = result.events[i].event_time
            next_time = result.events[i + 1].event_time
            assert curr_time <= next_time, "Events must be chronological"

    @given(
        contract_type=st.sampled_from([ContractType.CSH, ContractType.STK, ContractType.COM]),
    )
    @settings(max_examples=20, deadline=None)
    def test_all_events_have_valid_payoffs(self, contract_type):
        """Test all events have finite payoffs."""
        if contract_type == ContractType.CSH:
            attrs = ContractAttributes(
                contract_id="PROP-002",
                contract_type=contract_type,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="USD",
                notional_principal=100000.0,
            )
        else:  # STK or COM
            attrs = ContractAttributes(
                contract_id="PROP-002",
                contract_type=contract_type,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
                termination_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
                currency="USD",
                price_at_purchase_date=100.0,
                price_at_termination_date=120.0,
            )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        # All payoffs should be finite
        for event in result.events:
            payoff = float(event.payoff)
            assert jnp.isfinite(payoff), f"Payoff must be finite, got {payoff}"

    @given(
        contract_type=st.sampled_from([ContractType.CSH, ContractType.STK, ContractType.COM]),
    )
    @settings(max_examples=20, deadline=None)
    def test_state_transitions_preserve_currency(self, contract_type):
        """Test all events maintain the same currency."""
        if contract_type == ContractType.CSH:
            attrs = ContractAttributes(
                contract_id="PROP-003",
                contract_type=contract_type,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="EUR",
                notional_principal=100000.0,
            )
        else:  # STK or COM
            attrs = ContractAttributes(
                contract_id="PROP-003",
                contract_type=contract_type,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
                termination_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
                currency="EUR",
                price_at_purchase_date=100.0,
                price_at_termination_date=120.0,
            )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = create_contract(attrs, rf_obs)
        result = contract.simulate()

        # All events should have EUR currency
        for event in result.events:
            assert event.currency == "EUR"
