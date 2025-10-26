"""Tests for COM (Commodity) contract implementation.

This module tests the COM contract type implementation including:
- Payoff functions for all COM events
- State transition functions
- Event schedule generation
- Contract simulation

ACTUS Reference:
    ACTUS v1.1 Section 7.10 - COM: Commodity
"""

import jax.numpy as jnp
import pytest

from jactus.contracts.com import (
    CommodityContract,
    CommodityPayoffFunction,
    CommodityStateTransitionFunction,
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


class TestCommodityPayoffFunction:
    """Test COM payoff functions."""

    @pytest.fixture
    def attrs(self):
        """Standard COM contract attributes."""
        return ContractAttributes(
            contract_id="COM-001",
            contract_type=ContractType.COM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            price_at_purchase_date=7500.0,  # Total purchase price
            price_at_termination_date=8200.0,  # Total sale price
        )

    @pytest.fixture
    def state(self):
        """Standard COM contract state."""
        return ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 12, 31, 0, 0, 0),
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
        )

    @pytest.fixture
    def rf_obs(self):
        """Mock risk factor observer."""
        return ConstantRiskFactorObserver(constant_value=80.0)

    def test_pof_ad_returns_zero(self, attrs, state, rf_obs):
        """Test POF_AD_COM returns zero."""
        pof = CommodityPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        payoff = pof.calculate_payoff(
            EventType.AD,
            state,
            attrs,
            ActusDateTime(2024, 6, 1, 0, 0, 0),
            rf_obs,
        )

        assert float(payoff) == 0.0

    def test_pof_prd_pays_purchase_price(self, attrs, state, rf_obs):
        """Test POF_PRD_COM pays purchase price (negative cashflow)."""
        pof = CommodityPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        payoff = pof.calculate_payoff(
            EventType.PRD,
            state,
            attrs,
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # Purchase: -7500 (negative = outflow for buyer)
        assert float(payoff) == -7500.0

    def test_pof_td_receives_termination_price(self, attrs, state, rf_obs):
        """Test POF_TD_COM receives termination price (positive cashflow)."""
        pof = CommodityPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        payoff = pof.calculate_payoff(
            EventType.TD,
            state,
            attrs,
            ActusDateTime(2024, 12, 31, 0, 0, 0),
            rf_obs,
        )

        # Termination: +8200 (positive = inflow for seller)
        assert float(payoff) == 8200.0

    def test_pof_ce_returns_zero(self, attrs, state, rf_obs):
        """Test POF_CE_COM returns zero."""
        pof = CommodityPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        payoff = pof.calculate_payoff(
            EventType.CE,
            state,
            attrs,
            ActusDateTime(2024, 6, 1, 0, 0, 0),
            rf_obs,
        )

        assert float(payoff) == 0.0

    def test_pof_unknown_event_returns_zero(self, attrs, state, rf_obs):
        """Test unknown event type returns zero."""
        pof = CommodityPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        # Use an event type that COM doesn't handle (e.g., IP)
        payoff = pof.calculate_payoff(
            EventType.IP,
            state,
            attrs,
            ActusDateTime(2024, 6, 1, 0, 0, 0),
            rf_obs,
        )

        assert float(payoff) == 0.0

    def test_pof_prd_with_single_unit(self, state, rf_obs):
        """Test POF_PRD_COM with single unit purchase (e.g., one gold ounce)."""
        attrs = ContractAttributes(
            contract_id="COM-002",
            contract_type=ContractType.COM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            price_at_purchase_date=1500.0,  # $1500 total (e.g., one gold ounce)
        )

        pof = CommodityPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        payoff = pof.calculate_payoff(
            EventType.PRD,
            state,
            attrs,
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            rf_obs,
        )

        # Purchase: -1500
        assert float(payoff) == -1500.0


class TestCommodityStateTransitionFunction:
    """Test COM state transition functions."""

    @pytest.fixture
    def attrs(self):
        """Standard COM contract attributes."""
        return ContractAttributes(
            contract_id="COM-001",
            contract_type=ContractType.COM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

    @pytest.fixture
    def state_pre(self):
        """State before event."""
        return ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 12, 31, 0, 0, 0),
            nt=jnp.array(0.0, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
        )

    @pytest.fixture
    def rf_obs(self):
        """Mock risk factor observer."""
        return ConstantRiskFactorObserver(constant_value=80.0)

    def test_stf_updates_status_date(self, attrs, state_pre, rf_obs):
        """Test COM state transition updates status date."""
        stf = CommodityStateTransitionFunction()

        new_time = ActusDateTime(2024, 6, 1, 0, 0, 0)
        state_post = stf.transition_state(
            EventType.PRD,
            state_pre,
            attrs,
            new_time,
            rf_obs,
        )

        # Status date should be updated
        assert state_post.sd == new_time

        # All other state variables should remain unchanged
        assert state_post.tmd == state_pre.tmd
        assert float(state_post.nt) == float(state_pre.nt)
        assert float(state_post.ipnr) == float(state_pre.ipnr)


class TestCommodityContract:
    """Test COM contract class."""

    @pytest.fixture
    def attrs(self):
        """Standard COM contract attributes."""
        return ContractAttributes(
            contract_id="COM-001",
            contract_type=ContractType.COM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            purchase_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            termination_date=ActusDateTime(2024, 12, 31, 0, 0, 0),
            currency="USD",
            price_at_purchase_date=7500.0,
            price_at_termination_date=8200.0,
        )

    @pytest.fixture
    def rf_obs(self):
        """Mock risk factor observer."""
        return ConstantRiskFactorObserver(constant_value=80.0)

    def test_contract_initialization(self, attrs, rf_obs):
        """Test COM contract initialization."""
        contract = CommodityContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        assert contract.attributes.contract_id == "COM-001"
        assert contract.attributes.contract_type == ContractType.COM

    def test_contract_initialization_wrong_type_raises_error(self, attrs, rf_obs):
        """Test COM contract rejects wrong contract type."""
        attrs_wrong = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,  # Wrong type!
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        with pytest.raises(ValueError, match="Contract type must be COM"):
            CommodityContract(
                attributes=attrs_wrong,
                risk_factor_observer=rf_obs,
            )

    def test_generate_event_schedule_with_purchase_and_termination(self, attrs, rf_obs):
        """Test COM event schedule with purchase and termination."""
        contract = CommodityContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        schedule = contract.generate_event_schedule()

        # Should have 2 events: PRD and TD
        assert len(schedule.events) == 2

        # First event should be PRD
        assert schedule.events[0].event_type == EventType.PRD
        assert schedule.events[0].event_time == ActusDateTime(2024, 1, 15, 0, 0, 0)

        # Second event should be TD
        assert schedule.events[1].event_type == EventType.TD
        assert schedule.events[1].event_time == ActusDateTime(2024, 12, 31, 0, 0, 0)

    def test_generate_event_schedule_without_dates(self, rf_obs):
        """Test COM event schedule without purchase/termination dates."""
        attrs_no_dates = ContractAttributes(
            contract_id="COM-002",
            contract_type=ContractType.COM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
        )

        contract = CommodityContract(
            attributes=attrs_no_dates,
            risk_factor_observer=rf_obs,
        )

        schedule = contract.generate_event_schedule()

        # Should have no events
        assert len(schedule.events) == 0

    def test_initialize_state(self, attrs, rf_obs):
        """Test COM state initialization."""
        contract = CommodityContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        state = contract.initialize_state()

        # Check status date
        assert state.sd == attrs.status_date

        # Check maturity date (should use termination date)
        assert state.tmd == attrs.termination_date

        # Check all state variables are initialized
        assert float(state.nt) == 0.0
        assert float(state.ipnr) == 0.0
        assert float(state.nsc) == 1.0
        assert float(state.isc) == 1.0

    def test_simulate_purchase_and_sale(self, attrs, rf_obs):
        """Test COM simulation with purchase and sale."""
        contract = CommodityContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        result = contract.simulate()

        # Should have 2 events: PRD and TD
        assert len(result.events) == 2

        # First event should be PRD with negative payoff
        prd_event = result.events[0]
        assert prd_event.event_type == EventType.PRD
        assert float(prd_event.payoff) < 0  # Paying for commodity
        assert float(prd_event.payoff) == -7500.0

        # Second event should be TD with positive payoff
        td_event = result.events[1]
        assert td_event.event_type == EventType.TD
        assert float(td_event.payoff) > 0  # Receiving sale proceeds
        assert float(td_event.payoff) == 8200.0

        # Net should be positive (profit: 8200 - 7500 = 700)
        net_cashflow = float(prd_event.payoff) + float(td_event.payoff)
        assert net_cashflow == 700.0

    def test_simulate_tracks_state_changes(self, attrs, rf_obs):
        """Test COM simulation tracks state changes correctly."""
        contract = CommodityContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        result = contract.simulate()

        # Each event should have state_pre and state_post
        for event in result.events:
            assert event.state_pre is not None
            assert event.state_post is not None

            # Status date should be updated in state_post
            assert event.state_post.sd == event.event_time

    def test_get_payoff_function_returns_correct_type(self, attrs, rf_obs):
        """Test get_payoff_function returns CommodityPayoffFunction."""
        contract = CommodityContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        pof = contract.get_payoff_function(EventType.PRD)

        assert isinstance(pof, CommodityPayoffFunction)
        assert pof.contract_role == attrs.contract_role
        assert pof.currency == attrs.currency

    def test_get_state_transition_function_returns_correct_type(self, attrs, rf_obs):
        """Test get_state_transition_function returns CommodityStateTransitionFunction."""
        contract = CommodityContract(
            attributes=attrs,
            risk_factor_observer=rf_obs,
        )

        stf = contract.get_state_transition_function(EventType.PRD)

        assert isinstance(stf, CommodityStateTransitionFunction)
