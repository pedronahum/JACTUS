"""Unit tests for Stock (STK) contract implementation.

Tests the StockContract, StockPayoffFunction, and StockStateTransitionFunction
classes for correctness according to the ACTUS specification.
"""

import jax.numpy as jnp
import pytest

from jactus.contracts.stk import (
    StockContract,
    StockPayoffFunction,
    StockStateTransitionFunction,
)
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractState,
    ContractType,
    EventType,
)
from jactus.observers import ConstantRiskFactorObserver

# ============================================================================
# Test StockPayoffFunction
# ============================================================================


class TestStockPayoffFunction:
    """Test StockPayoffFunction class."""

    def test_initialization(self):
        """Test StockPayoffFunction can be created."""
        pof = StockPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        assert pof.contract_role == ContractRole.RPA
        assert pof.currency == "USD"

    def test_pof_ad_returns_zero(self):
        """Test POF_AD_STK returns zero (no cashflow at analysis)."""
        pof = StockPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        state = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2025, 1, 1, 0, 0, 0),
            nt=jnp.array(0.0),
            ipnr=jnp.array(0.0),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="STK-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)

        payoff = pof.calculate_payoff(
            EventType.AD,
            state,
            attrs,
            ActusDateTime(2024, 1, 1, 0, 0, 0),
            rf_obs,
        )

        assert float(payoff) == 0.0

    def test_pof_prd_pays_purchase_price(self):
        """Test POF_PRD_STK pays purchase price (negative cashflow)."""
        pof = StockPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        state = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2025, 1, 1, 0, 0, 0),
            nt=jnp.array(0.0),
            ipnr=jnp.array(0.0),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="STK-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            market_object_code="AAPL",
            price_at_purchase_date=150.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)

        payoff = pof.calculate_payoff(
            EventType.PRD,
            state,
            attrs,
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # PRD should pay -150 (negative = outflow for buyer)
        assert float(payoff) == -150.0

    def test_pof_td_receives_termination_price(self):
        """Test POF_TD_STK receives termination price (positive cashflow)."""
        pof = StockPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        state = ContractState(
            sd=ActusDateTime(2024, 6, 1, 0, 0, 0),
            tmd=ActusDateTime(2025, 1, 1, 0, 0, 0),
            nt=jnp.array(0.0),
            ipnr=jnp.array(0.0),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="STK-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            termination_date=ActusDateTime(2025, 1, 1, 0, 0, 0),
            currency="USD",
            market_object_code="AAPL",
            price_at_termination_date=175.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=175.0)

        payoff = pof.calculate_payoff(
            EventType.TD,
            state,
            attrs,
            ActusDateTime(2025, 1, 1, 0, 0, 0),
            rf_obs,
        )

        # TD should receive +175 (positive = inflow for seller)
        assert float(payoff) == 175.0

    def test_pof_td_returns_zero_if_price_not_set(self):
        """Test POF_TD_STK returns zero if price not specified."""
        pof = StockPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        state = ContractState(
            sd=ActusDateTime(2024, 6, 1, 0, 0, 0),
            tmd=ActusDateTime(2025, 1, 1, 0, 0, 0),
            nt=jnp.array(0.0),
            ipnr=jnp.array(0.0),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="STK-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            termination_date=ActusDateTime(2025, 1, 1, 0, 0, 0),
            currency="USD",
            price_at_termination_date=None,  # Not specified
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=180.0)

        payoff = pof.calculate_payoff(
            EventType.TD,
            state,
            attrs,
            ActusDateTime(2025, 1, 1, 0, 0, 0),
            rf_obs,
        )

        # Should return 0 when price not set
        assert float(payoff) == 0.0

    def test_pof_dv_returns_zero_if_not_defined(self):
        """Test POF_DV_STK returns zero if no dividend defined."""
        pof = StockPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        state = ContractState(
            sd=ActusDateTime(2024, 3, 1, 0, 0, 0),
            tmd=ActusDateTime(2025, 1, 1, 0, 0, 0),
            nt=jnp.array(0.0),
            ipnr=jnp.array(0.0),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="STK-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)

        payoff = pof.calculate_payoff(
            EventType.DV,
            state,
            attrs,
            ActusDateTime(2024, 4, 1, 0, 0, 0),
            rf_obs,
        )

        # No dividend defined
        assert float(payoff) == 0.0

    def test_pof_ce_returns_zero(self):
        """Test POF_CE_STK returns zero."""
        pof = StockPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        state = ContractState(
            sd=ActusDateTime(2024, 3, 1, 0, 0, 0),
            tmd=ActusDateTime(2025, 1, 1, 0, 0, 0),
            nt=jnp.array(0.0),
            ipnr=jnp.array(0.0),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="STK-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)

        payoff = pof.calculate_payoff(
            EventType.CE,
            state,
            attrs,
            ActusDateTime(2024, 6, 1, 0, 0, 0),
            rf_obs,
        )

        assert float(payoff) == 0.0


# ============================================================================
# Test StockStateTransitionFunction
# ============================================================================


class TestStockStateTransitionFunction:
    """Test StockStateTransitionFunction class."""

    def test_initialization(self):
        """Test StockStateTransitionFunction can be created."""
        stf = StockStateTransitionFunction()
        assert stf is not None

    def test_stf_updates_status_date(self):
        """Test STF updates status date for all events."""
        stf = StockStateTransitionFunction()

        state_pre = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2025, 1, 1, 0, 0, 0),
            nt=jnp.array(0.0),
            ipnr=jnp.array(0.0),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="STK-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)

        # Test with DV event
        state_post = stf.transition_state(
            EventType.DV,
            state_pre,
            attrs,
            ActusDateTime(2024, 4, 1, 0, 0, 0),
            rf_obs,
        )

        # Status date should be updated
        assert state_post.sd == ActusDateTime(2024, 4, 1, 0, 0, 0)

        # All other state should remain unchanged
        assert float(state_post.nt) == 0.0
        assert float(state_post.ipnr) == 0.0


# ============================================================================
# Test StockContract
# ============================================================================


class TestStockContract:
    """Test StockContract class."""

    def test_initialization_success(self):
        """Test STK contract can be created with valid attributes."""
        attrs = ContractAttributes(
            contract_id="STK-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)

        contract = StockContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        assert contract.attributes.contract_id == "STK-001"
        assert contract.attributes.contract_type == ContractType.STK

    def test_initialization_requires_contract_type(self):
        """Test STK contract requires STK contract type."""
        with pytest.raises(ValueError) as exc_info:
            attrs = ContractAttributes(
                contract_id="STK-001",
                contract_type=ContractType.CSH,  # Wrong type
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="USD",
            )

            rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

            StockContract(
                attributes=attrs,
                risk_factor_observer=rf_obs,
            )

        assert "contract type must be stk" in str(exc_info.value).lower()

    def test_initialize_state_minimal(self):
        """Test STK state initialization is minimal."""
        attrs = ContractAttributes(
            contract_id="STK-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)

        contract = StockContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        state = contract.initialize_state()

        # Status date should be set
        assert state.sd == ActusDateTime(2024, 1, 1, 0, 0, 0)

        # All amounts should be zero (STK doesn't use these)
        assert float(state.nt) == 0.0
        assert float(state.ipac) == 0.0

    def test_generate_event_schedule_with_purchase_only(self):
        """Test STK event schedule with only purchase."""
        attrs = ContractAttributes(
            contract_id="STK-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            currency="USD",
            market_object_code="AAPL",
            price_at_purchase_date=150.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)

        contract = StockContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        schedule = contract.generate_event_schedule()

        # Should have only PRD event
        assert len(schedule.events) == 1
        assert schedule.events[0].event_type == EventType.PRD
        assert schedule.events[0].event_time == ActusDateTime(2024, 1, 15, 0, 0, 0)

    def test_generate_event_schedule_with_purchase_and_termination(self):
        """Test STK event schedule with purchase and termination."""
        attrs = ContractAttributes(
            contract_id="STK-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            termination_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            market_object_code="AAPL",
            price_at_purchase_date=150.0,
            price_at_termination_date=175.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)

        contract = StockContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        schedule = contract.generate_event_schedule()

        # Should have PRD + TD = 2 events
        assert len(schedule.events) == 2
        assert schedule.events[0].event_type == EventType.PRD
        assert schedule.events[1].event_type == EventType.TD

    def test_simulate_purchase_and_sale(self):
        """Test STK simulation with purchase and sale."""
        attrs = ContractAttributes(
            contract_id="STK-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            termination_date=ActusDateTime(2024, 6, 15, 0, 0, 0),
            currency="USD",
            market_object_code="AAPL",
            price_at_purchase_date=150.0,
            price_at_termination_date=175.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)

        contract = StockContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        result = contract.simulate()

        # Should have 2 events
        assert len(result.events) == 2

        # First event should be PRD with negative payoff
        prd_event = result.events[0]
        assert prd_event.event_type == EventType.PRD
        assert float(prd_event.payoff) < 0  # Paid for stock

        # Second event should be TD with positive payoff
        td_event = result.events[1]
        assert td_event.event_type == EventType.TD
        assert float(td_event.payoff) > 0  # Received from sale

        # Net should be positive (profit)
        net_cashflow = float(prd_event.payoff) + float(td_event.payoff)
        assert net_cashflow > 0  # Made profit (175 - 150 = 25)

    def test_get_payoff_function(self):
        """Test get_payoff_function returns StockPayoffFunction."""
        attrs = ContractAttributes(
            contract_id="STK-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)

        contract = StockContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        pof = contract.get_payoff_function(EventType.DV)
        assert isinstance(pof, StockPayoffFunction)

    def test_get_state_transition_function(self):
        """Test get_state_transition_function returns StockStateTransitionFunction."""
        attrs = ContractAttributes(
            contract_id="STK-001",
            contract_type=ContractType.STK,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=150.0)

        contract = StockContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        stf = contract.get_state_transition_function(EventType.DV)
        assert isinstance(stf, StockStateTransitionFunction)
