"""Unit tests for Cash (CSH) contract implementation.

Tests the CashContract, CashPayoffFunction, and CashStateTransitionFunction
classes for correctness according to the ACTUS specification.
"""

import jax.numpy as jnp

from jactus.contracts.csh import (
    CashContract,
    CashPayoffFunction,
    CashStateTransitionFunction,
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
# Test CashPayoffFunction
# ============================================================================


class TestCashPayoffFunction:
    """Test CashPayoffFunction class."""

    def test_initialization(self):
        """Test CashPayoffFunction can be created."""
        pof = CashPayoffFunction(
            contract_role=ContractRole.RPA,
            currency="USD",
            settlement_currency=None,
        )

        assert pof.contract_role == ContractRole.RPA
        assert pof.currency == "USD"

    def test_calculate_payoff_returns_zero(self):
        """Test that CSH payoffs are always zero."""
        pof = CashPayoffFunction(
            contract_role=ContractRole.RPA, currency="USD", settlement_currency=None
        )

        state = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.0),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=10000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        payoff = pof.calculate_payoff(
            EventType.AD,
            state,
            attrs,
            ActusDateTime(2024, 1, 1, 0, 0, 0),
            rf_obs,
        )

        assert float(payoff) == 0.0

    def test_payoff_is_jax_array(self):
        """Test that payoff returns JAX array."""
        pof = CashPayoffFunction(
            contract_role=ContractRole.RPA, currency="USD", settlement_currency=None
        )

        state = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.0),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=10000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        payoff = pof.calculate_payoff(
            EventType.AD, state, attrs, ActusDateTime(2024, 1, 1, 0, 0, 0), rf_obs
        )

        assert isinstance(payoff, jnp.ndarray)


# ============================================================================
# Test CashStateTransitionFunction
# ============================================================================


class TestCashStateTransitionFunction:
    """Test CashStateTransitionFunction class."""

    def test_initialization(self):
        """Test CashStateTransitionFunction can be created."""
        stf = CashStateTransitionFunction()
        assert stf is not None

    def test_transition_updates_status_date(self):
        """Test that state transition updates status date."""
        stf = CashStateTransitionFunction()

        state_pre = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.0),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=10000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
        new_time = ActusDateTime(2024, 2, 1, 0, 0, 0)

        state_post = stf.transition_state(EventType.AD, state_pre, attrs, new_time, rf_obs)

        assert state_post.sd == new_time

    def test_transition_preserves_notional(self):
        """Test that state transition preserves notional."""
        stf = CashStateTransitionFunction()

        state_pre = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.0),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=10000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        state_post = stf.transition_state(
            EventType.AD,
            state_pre,
            attrs,
            ActusDateTime(2024, 2, 1, 0, 0, 0),
            rf_obs,
        )

        assert float(state_post.nt) == float(state_pre.nt)

    def test_transition_preserves_all_other_states(self):
        """Test that only status date changes, all other states unchanged."""
        stf = CashStateTransitionFunction()

        state_pre = ContractState(
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            tmd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(10000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(100.0),
            feac=jnp.array(50.0),
            nsc=jnp.array(1.5),
            isc=jnp.array(2.0),
        )

        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=10000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        state_post = stf.transition_state(
            EventType.AD,
            state_pre,
            attrs,
            ActusDateTime(2024, 2, 1, 0, 0, 0),
            rf_obs,
        )

        # Only sd should change
        assert state_post.sd != state_pre.sd
        # All others should be unchanged
        assert float(state_post.nt) == float(state_pre.nt)
        assert float(state_post.ipnr) == float(state_pre.ipnr)
        assert float(state_post.ipac) == float(state_pre.ipac)
        assert float(state_post.feac) == float(state_pre.feac)
        assert float(state_post.nsc) == float(state_pre.nsc)
        assert float(state_post.isc) == float(state_pre.isc)


# ============================================================================
# Test CashContract
# ============================================================================


class TestCashContract:
    """Test CashContract class."""

    def test_initialization_valid(self):
        """Test CashContract initializes with valid attributes."""
        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=10000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
        contract = CashContract(attributes=attrs, risk_factor_observer=rf_obs)

        assert contract.attributes.contract_id == "CSH-001"
        assert contract.attributes.contract_type == ContractType.CSH

    def test_initialization_rejects_wrong_contract_type(self):
        """Test CashContract rejects wrong contract type."""
        attrs = ContractAttributes(
            contract_id="PAM-001",
            contract_type=ContractType.PAM,  # Wrong type!
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=10000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        try:
            CashContract(attributes=attrs, risk_factor_observer=rf_obs)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Contract type must be CSH" in str(e)

    def test_initialization_requires_notional(self):
        """Test CashContract requires notional_principal."""
        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=None,  # Missing!
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)

        try:
            CashContract(attributes=attrs, risk_factor_observer=rf_obs)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "notional_principal" in str(e).lower()

    def test_initialization_requires_role(self):
        """Test CashContract requires contract_role."""
        # Pydantic will reject None for contract_role at validation time
        import pytest
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            _attrs = ContractAttributes(
                contract_id="CSH-001",
                contract_type=ContractType.CSH,
                contract_role=None,  # Missing!
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency="USD",
                notional_principal=10000.0,
            )

        assert "contract_role" in str(exc_info.value).lower()

    def test_initialization_requires_currency(self):
        """Test CashContract requires currency."""
        # Pydantic will reject None for currency at validation time
        import pytest
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            _attrs = ContractAttributes(
                contract_id="CSH-001",
                contract_type=ContractType.CSH,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                currency=None,  # Missing!
                notional_principal=10000.0,
            )

        assert "currency" in str(exc_info.value).lower()

    def test_generate_event_schedule_single_ad(self):
        """Test CSH generates single AD event."""
        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=10000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
        contract = CashContract(attributes=attrs, risk_factor_observer=rf_obs)

        schedule = contract.generate_event_schedule()

        assert len(schedule.events) == 1
        assert schedule.events[0].event_type == EventType.AD
        assert schedule.events[0].event_time == attrs.status_date

    def test_initialize_state_applies_role_sign(self):
        """Test state initialization applies role sign to notional."""
        # Test RPA (positive sign)
        attrs_rpa = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=10000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
        contract_rpa = CashContract(attributes=attrs_rpa, risk_factor_observer=rf_obs)

        state_rpa = contract_rpa.initialize_state()
        assert float(state_rpa.nt) == 10000.0  # Positive

        # Test RPL (negative sign)
        attrs_rpl = ContractAttributes(
            contract_id="CSH-002",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=10000.0,
        )

        contract_rpl = CashContract(attributes=attrs_rpl, risk_factor_observer=rf_obs)

        state_rpl = contract_rpl.initialize_state()
        assert float(state_rpl.nt) == -10000.0  # Negative

    def test_initialize_state_sets_status_date(self):
        """Test state initialization sets status date correctly."""
        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 15, 12, 30, 0),
            currency="USD",
            notional_principal=10000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
        contract = CashContract(attributes=attrs, risk_factor_observer=rf_obs)

        state = contract.initialize_state()
        assert state.sd == attrs.status_date

    def test_simulate_completes_successfully(self):
        """Test CSH simulation completes without errors."""
        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=10000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
        contract = CashContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        assert result is not None
        assert len(result.events) == 1
        assert len(result.states) == 1

    def test_simulate_produces_zero_cashflow(self):
        """Test CSH simulation produces zero total cashflow."""
        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=10000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
        contract = CashContract(attributes=attrs, risk_factor_observer=rf_obs)

        result = contract.simulate()

        # All CSH payoffs should be zero
        assert all(float(event.payoff) == 0.0 for event in result.events)

    def test_get_payoff_function_returns_correct_type(self):
        """Test get_payoff_function returns CashPayoffFunction."""
        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=10000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
        contract = CashContract(attributes=attrs, risk_factor_observer=rf_obs)

        pof = contract.get_payoff_function(EventType.AD)
        assert isinstance(pof, CashPayoffFunction)

    def test_get_state_transition_function_returns_correct_type(self):
        """Test get_state_transition_function returns CashStateTransitionFunction."""
        attrs = ContractAttributes(
            contract_id="CSH-001",
            contract_type=ContractType.CSH,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            currency="USD",
            notional_principal=10000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
        contract = CashContract(attributes=attrs, risk_factor_observer=rf_obs)

        stf = contract.get_state_transition_function(EventType.AD)
        assert isinstance(stf, CashStateTransitionFunction)
