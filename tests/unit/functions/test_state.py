"""Unit tests for state transition function framework.

T2.2: State Transition Function Framework Tests

Tests for:
- StateTransitionFunction Protocol
- BaseStateTransitionFunction ABC
- create_state_pre helper
- validate_state_transition helper
- Interest accrual
- Fee accrual (both bases)
- JAX JIT compatibility
"""

from unittest.mock import Mock

import jax
import jax.numpy as jnp

from jactus.core import ActusDateTime, ContractAttributes, ContractState
from jactus.core.types import (
    ContractRole,
    ContractType,
    DayCountConvention,
    EventType,
    FeeBasis,
)
from jactus.functions import (
    BaseStateTransitionFunction,
    StateTransitionFunction,
    create_state_pre,
    validate_state_transition,
)


class TestStateTransitionFunctionProtocol:
    """Test StateTransitionFunction protocol enforcement."""

    def test_protocol_requires_call_method(self):
        """StateTransitionFunction protocol requires __call__ method."""

        class ValidSTF:
            def __call__(self, event_type, state_pre, attributes, time, risk_factor_observer):
                return state_pre

        assert isinstance(ValidSTF(), StateTransitionFunction)

    def test_protocol_rejects_without_call_method(self):
        """Objects without __call__ are not StateTransitionFunctions."""

        class InvalidSTF:
            pass

        assert not isinstance(InvalidSTF(), StateTransitionFunction)


class ConcreteStateTransitionFunction(BaseStateTransitionFunction):
    """Concrete implementation for testing BaseStateTransitionFunction."""

    def transition_state(self, event_type, state_pre, attributes, time, risk_factor_observer):
        """Simple state transition: just return a copy of the state."""
        return state_pre


class TestBaseStateTransitionFunctionInit:
    """Test BaseStateTransitionFunction initialization."""

    def test_init_with_dcc(self):
        """Initialize with explicit day count convention."""
        stf = ConcreteStateTransitionFunction(day_count_convention=DayCountConvention.A360)

        assert stf.day_count_convention == DayCountConvention.A360

    def test_init_without_dcc(self):
        """Initialize without day count convention (uses contract's DCC)."""
        stf = ConcreteStateTransitionFunction()

        assert stf.day_count_convention is None


class TestUpdateStatusDate:
    """Test update_status_date method."""

    def test_update_status_date(self):
        """Status date is updated correctly."""
        stf = ConcreteStateTransitionFunction()

        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        old_sd = ActusDateTime(2024, 1, 15, 0, 0, 0)
        new_sd = ActusDateTime(2024, 6, 15, 0, 0, 0)

        state = ContractState(
            tmd=tmd,
            sd=old_sd,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        updated = stf.update_status_date(state, new_sd)

        assert updated.sd == new_sd
        assert updated.nt == state.nt  # Other fields unchanged
        assert updated.ipnr == state.ipnr


class TestAccrueInterest:
    """Test accrue_interest method."""

    def test_accrue_interest_one_year_a360(self):
        """Accrue interest for one year using A360."""
        stf = ConcreteStateTransitionFunction(day_count_convention=DayCountConvention.A360)

        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)
        state = ContractState(
            tmd=tmd,
            sd=sd,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),  # 5% annual rate
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attributes = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=sd,
            currency="USD",
            day_count_convention=DayCountConvention.A360,
        )

        from_date = ActusDateTime(2024, 1, 15, 0, 0, 0)
        to_date = ActusDateTime(2025, 1, 15, 0, 0, 0)

        accrued = stf.accrue_interest(state, attributes, from_date, to_date)

        # A360: 366/360 = 1.0167 years (2024 is leap year)
        # Interest = 100000 * 0.05 * 1.0167 = 5083.33
        expected = jnp.array(5083.33, dtype=jnp.float32)
        assert jnp.allclose(accrued, expected, atol=0.01)

    def test_accrue_interest_six_months_a365(self):
        """Accrue interest for six months using A365."""
        stf = ConcreteStateTransitionFunction(day_count_convention=DayCountConvention.A365)

        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)
        state = ContractState(
            tmd=tmd,
            sd=sd,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.06),  # 6% annual rate
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attributes = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=sd,
            currency="USD",
            day_count_convention=DayCountConvention.A365,
        )

        from_date = ActusDateTime(2024, 1, 15, 0, 0, 0)
        to_date = ActusDateTime(2024, 7, 15, 0, 0, 0)

        accrued = stf.accrue_interest(state, attributes, from_date, to_date)

        # A365: 182/365 ≈ 0.4986 years
        # Interest = 100000 * 0.06 * 0.4986 ≈ 2991.78
        expected = jnp.array(2991.78, dtype=jnp.float32)
        assert jnp.allclose(accrued, expected, atol=1.0)

    def test_accrue_interest_uses_contract_dcc_if_none(self):
        """Uses contract's DCC when STF's DCC is None."""
        stf = ConcreteStateTransitionFunction(day_count_convention=None)

        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)
        state = ContractState(
            tmd=tmd,
            sd=sd,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attributes = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=sd,
            currency="USD",
            day_count_convention=DayCountConvention.A360,
        )

        from_date = ActusDateTime(2024, 1, 15, 0, 0, 0)
        to_date = ActusDateTime(2025, 1, 15, 0, 0, 0)

        accrued = stf.accrue_interest(state, attributes, from_date, to_date)

        # Should use A360 from attributes
        # A360: 366/360 = 1.0167 years (2024 is leap year)
        # Interest = 100000 * 0.05 * 1.0167 = 5083.33
        expected = jnp.array(5083.33, dtype=jnp.float32)
        assert jnp.allclose(accrued, expected, atol=0.01)

    def test_accrue_interest_zero_notional(self):
        """Zero notional results in zero interest."""
        stf = ConcreteStateTransitionFunction(day_count_convention=DayCountConvention.A360)

        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)
        state = ContractState(
            tmd=tmd,
            sd=sd,
            nt=jnp.array(0.0),  # Zero notional
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attributes = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=sd,
            currency="USD",
            day_count_convention=DayCountConvention.A360,
        )

        from_date = ActusDateTime(2024, 1, 15, 0, 0, 0)
        to_date = ActusDateTime(2025, 1, 15, 0, 0, 0)

        accrued = stf.accrue_interest(state, attributes, from_date, to_date)

        assert jnp.allclose(accrued, jnp.array(0.0, dtype=jnp.float32))

    def test_accrue_interest_zero_rate(self):
        """Zero interest rate results in zero interest."""
        stf = ConcreteStateTransitionFunction(day_count_convention=DayCountConvention.A360)

        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)
        state = ContractState(
            tmd=tmd,
            sd=sd,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.0),  # Zero rate
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attributes = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=sd,
            currency="USD",
            day_count_convention=DayCountConvention.A360,
        )

        from_date = ActusDateTime(2024, 1, 15, 0, 0, 0)
        to_date = ActusDateTime(2025, 1, 15, 0, 0, 0)

        accrued = stf.accrue_interest(state, attributes, from_date, to_date)

        assert jnp.allclose(accrued, jnp.array(0.0, dtype=jnp.float32))


class TestAccrueFees:
    """Test accrue_fees method."""

    def test_accrue_fees_absolute_basis(self):
        """Accrue fees with absolute basis."""
        stf = ConcreteStateTransitionFunction(day_count_convention=DayCountConvention.A360)

        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)
        state = ContractState(
            tmd=tmd,
            sd=sd,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attributes = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=sd,
            currency="USD",
            day_count_convention=DayCountConvention.A360,
        )

        from_date = ActusDateTime(2024, 1, 15, 0, 0, 0)
        to_date = ActusDateTime(2025, 1, 15, 0, 0, 0)
        fee_rate = jnp.array(1000.0, dtype=jnp.float32)  # 1000 per year

        accrued = stf.accrue_fees(state, attributes, from_date, to_date, fee_rate, FeeBasis.A)

        # Absolute: FER * YF = 1000 * 1.0167 = 1016.67 (2024 is leap year)
        expected = jnp.array(1016.67, dtype=jnp.float32)
        assert jnp.allclose(accrued, expected, atol=0.01)

    def test_accrue_fees_notional_basis(self):
        """Accrue fees with notional basis."""
        stf = ConcreteStateTransitionFunction(day_count_convention=DayCountConvention.A360)

        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)
        state = ContractState(
            tmd=tmd,
            sd=sd,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attributes = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=sd,
            currency="USD",
            day_count_convention=DayCountConvention.A360,
        )

        from_date = ActusDateTime(2024, 1, 15, 0, 0, 0)
        to_date = ActusDateTime(2025, 1, 15, 0, 0, 0)
        fee_rate = jnp.array(0.01, dtype=jnp.float32)  # 1% of notional per year

        accrued = stf.accrue_fees(state, attributes, from_date, to_date, fee_rate, FeeBasis.N)

        # Notional: NT * FER * YF = 100000 * 0.01 * 1.0167 = 1016.67 (2024 is leap year)
        expected = jnp.array(1016.67, dtype=jnp.float32)
        assert jnp.allclose(accrued, expected, atol=0.01)

    def test_accrue_fees_six_months(self):
        """Accrue fees for partial year."""
        stf = ConcreteStateTransitionFunction(day_count_convention=DayCountConvention.A360)

        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)
        state = ContractState(
            tmd=tmd,
            sd=sd,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attributes = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=sd,
            currency="USD",
            day_count_convention=DayCountConvention.A360,
        )

        from_date = ActusDateTime(2024, 1, 15, 0, 0, 0)
        to_date = ActusDateTime(2024, 7, 15, 0, 0, 0)
        fee_rate = jnp.array(1200.0, dtype=jnp.float32)  # 1200 per year

        accrued = stf.accrue_fees(state, attributes, from_date, to_date, fee_rate, FeeBasis.A)

        # A360: 182/360 = 0.5056 years
        # Absolute: 1200 * 0.5056 = 606.67
        expected = jnp.array(606.67, dtype=jnp.float32)
        assert jnp.allclose(accrued, expected, atol=0.01)


class TestStateTransitionCall:
    """Test complete __call__ pipeline."""

    def test_call_updates_status_date(self):
        """__call__ updates status date to event time."""
        stf = ConcreteStateTransitionFunction()

        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        old_sd = ActusDateTime(2024, 1, 15, 0, 0, 0)
        event_time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        state_pre = ContractState(
            tmd=tmd,
            sd=old_sd,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attributes = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=old_sd,
            currency="USD",
        )

        observer = Mock()

        state_post = stf(EventType.IP, state_pre, attributes, event_time, observer)

        # Status date should be updated to event time
        assert state_post.sd == event_time

    def test_call_executes_transition_state(self):
        """__call__ executes transition_state method."""

        class CustomSTF(BaseStateTransitionFunction):
            def transition_state(
                self, event_type, state_pre, attributes, time, risk_factor_observer
            ):
                # Reset accrued interest on IP event
                if event_type == EventType.IP:
                    return state_pre.replace(ipac=jnp.array(0.0))
                return state_pre

        stf = CustomSTF()

        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)
        event_time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        state_pre = ContractState(
            tmd=tmd,
            sd=sd,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(2500.0),  # Has accrued interest
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attributes = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=sd,
            currency="USD",
        )

        observer = Mock()

        state_post = stf(EventType.IP, state_pre, attributes, event_time, observer)

        # IPAC should be reset to zero
        assert jnp.allclose(state_post.ipac, jnp.array(0.0))
        # Status date should be updated
        assert state_post.sd == event_time


class TestCreateStatePre:
    """Test create_state_pre helper function."""

    def test_create_state_pre_minimal(self):
        """Create state with minimal required parameters."""
        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)

        state = create_state_pre(tmd=tmd, sd=sd, nt=100000.0, ipnr=0.05)

        assert state.tmd == tmd
        assert state.sd == sd
        assert jnp.allclose(state.nt, jnp.array(100000.0))
        assert jnp.allclose(state.ipnr, jnp.array(0.05))
        # Check defaults
        assert jnp.allclose(state.ipac, jnp.array(0.0))
        assert jnp.allclose(state.feac, jnp.array(0.0))
        assert jnp.allclose(state.nsc, jnp.array(1.0))
        assert jnp.allclose(state.isc, jnp.array(1.0))

    def test_create_state_pre_with_all_params(self):
        """Create state with all parameters specified."""
        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)

        state = create_state_pre(
            tmd=tmd,
            sd=sd,
            nt=100000.0,
            ipnr=0.05,
            ipac=1250.0,
            feac=100.0,
            nsc=0.9,
            isc=1.1,
        )

        assert jnp.allclose(state.ipac, jnp.array(1250.0))
        assert jnp.allclose(state.feac, jnp.array(100.0))
        assert jnp.allclose(state.nsc, jnp.array(0.9))
        assert jnp.allclose(state.isc, jnp.array(1.1))

    def test_create_state_pre_returns_contract_state(self):
        """Returns proper ContractState instance."""
        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)

        state = create_state_pre(tmd=tmd, sd=sd, nt=100000.0, ipnr=0.05)

        assert isinstance(state, ContractState)


class TestValidateStateTransition:
    """Test validate_state_transition helper function."""

    def test_valid_transition(self):
        """Valid state transition passes validation."""
        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd_pre = ActusDateTime(2024, 1, 15, 0, 0, 0)
        sd_post = ActusDateTime(2024, 6, 15, 0, 0, 0)

        state_pre = ContractState(
            tmd=tmd,
            sd=sd_pre,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        state_post = ContractState(
            tmd=tmd,
            sd=sd_post,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(2500.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        is_valid = validate_state_transition(state_pre, state_post, EventType.IP)

        assert is_valid is True

    def test_invalid_status_date_backwards(self):
        """Status date going backwards fails validation."""
        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd_pre = ActusDateTime(2024, 6, 15, 0, 0, 0)
        sd_post = ActusDateTime(2024, 1, 15, 0, 0, 0)  # Earlier than pre

        state_pre = ContractState(
            tmd=tmd,
            sd=sd_pre,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        state_post = ContractState(
            tmd=tmd,
            sd=sd_post,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        is_valid = validate_state_transition(state_pre, state_post, EventType.IP)

        assert is_valid is False

    def test_invalid_negative_notional(self):
        """Negative notional fails validation."""
        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd_pre = ActusDateTime(2024, 1, 15, 0, 0, 0)
        sd_post = ActusDateTime(2024, 6, 15, 0, 0, 0)

        state_pre = ContractState(
            tmd=tmd,
            sd=sd_pre,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        state_post = ContractState(
            tmd=tmd,
            sd=sd_post,
            nt=jnp.array(-1000.0),  # Negative
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        is_valid = validate_state_transition(state_pre, state_post, EventType.IP)

        assert is_valid is False

    def test_invalid_negative_scaling_factor(self):
        """Negative or zero scaling factor fails validation."""
        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd_pre = ActusDateTime(2024, 1, 15, 0, 0, 0)
        sd_post = ActusDateTime(2024, 6, 15, 0, 0, 0)

        state_pre = ContractState(
            tmd=tmd,
            sd=sd_pre,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        state_post = ContractState(
            tmd=tmd,
            sd=sd_post,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(0.0),  # Zero scaling factor
            isc=jnp.array(1.0),
        )

        is_valid = validate_state_transition(state_pre, state_post, EventType.IP)

        assert is_valid is False


class TestJAXJITCompatibility:
    """Test JAX JIT compatibility for state transition functions."""

    def test_accrue_interest_jit_compatible(self):
        """accrue_interest is JIT-compatible."""

        @jax.jit
        def jitted_accrue_interest(nt, ipnr, yf):
            return nt * ipnr * yf

        nt = jnp.array(100000.0)
        ipnr = jnp.array(0.05)
        yf = jnp.array(1.0)

        result = jitted_accrue_interest(nt, ipnr, yf)

        expected = jnp.array(5000.0)
        assert jnp.allclose(result, expected)

    def test_accrue_fees_jit_compatible(self):
        """Fee accrual logic is JIT-compatible."""

        @jax.jit
        def jitted_accrue_fees_absolute(fer, yf):
            return fer * yf

        @jax.jit
        def jitted_accrue_fees_notional(nt, fer, yf):
            return nt * fer * yf

        # Absolute basis
        fer = jnp.array(1000.0)
        yf = jnp.array(1.0)
        result_abs = jitted_accrue_fees_absolute(fer, yf)
        assert jnp.allclose(result_abs, jnp.array(1000.0))

        # Notional basis
        nt = jnp.array(100000.0)
        fer = jnp.array(0.01)
        result_not = jitted_accrue_fees_notional(nt, fer, yf)
        assert jnp.allclose(result_not, jnp.array(1000.0))

    def test_state_transition_returns_jax_compatible_state(self):
        """State transitions return JAX-compatible states."""
        stf = ConcreteStateTransitionFunction()

        tmd = ActusDateTime(2029, 1, 15, 0, 0, 0)
        sd = ActusDateTime(2024, 1, 15, 0, 0, 0)

        state = ContractState(
            tmd=tmd,
            sd=sd,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        attributes = ContractAttributes(
            contract_id="TEST001",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=sd,
            currency="USD",
        )

        observer = Mock()
        event_time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        result = stf(EventType.IP, state, attributes, event_time, observer)

        # Verify all numeric fields are JAX arrays
        assert isinstance(result.nt, jnp.ndarray)
        assert isinstance(result.ipnr, jnp.ndarray)
        assert isinstance(result.ipac, jnp.ndarray)
        assert isinstance(result.feac, jnp.ndarray)
        assert isinstance(result.nsc, jnp.ndarray)
        assert isinstance(result.isc, jnp.ndarray)
