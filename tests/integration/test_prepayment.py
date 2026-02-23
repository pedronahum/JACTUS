"""Integration tests for prepayment (PP) event handling.

Tests that PP events properly reduce notional via observe_event()
when a DictRiskFactorObserver provides prepayment data.
"""

import jax.numpy as jnp
import pytest

from jactus.contracts import create_contract
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractState,
    ContractType,
    DayCountConvention,
    EventType,
)
from jactus.core.types import ContractRole
from jactus.observers import ConstantRiskFactorObserver, DictRiskFactorObserver

_MD = ActusDateTime(2025, 1, 15, 0, 0, 0)


def _state(sd_args, nt=100_000.0, ipnr=0.05, ipac=0.0, ipcb=None):
    """Create a ContractState with all required fields."""
    return ContractState(
        tmd=_MD,
        sd=ActusDateTime(*sd_args),
        nt=jnp.array(nt, dtype=jnp.float32),
        ipnr=jnp.array(ipnr, dtype=jnp.float32),
        ipac=jnp.array(ipac, dtype=jnp.float32),
        feac=jnp.array(0.0, dtype=jnp.float32),
        nsc=jnp.array(1.0, dtype=jnp.float32),
        isc=jnp.array(1.0, dtype=jnp.float32),
        ipcb=jnp.array(ipcb, dtype=jnp.float32) if ipcb is not None else None,
    )


def _make_pam_attrs(contract_id="PP-TEST-001"):
    """Create basic PAM contract attributes for PP testing."""
    return ContractAttributes(
        contract_id=contract_id,
        contract_type=ContractType.PAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
        maturity_date=_MD,
        notional_principal=100_000.0,
        nominal_interest_rate=0.05,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="3M",
    )


def _make_lam_attrs(contract_id="PP-TEST-002"):
    """Create basic LAM contract attributes for PP testing."""
    return ContractAttributes(
        contract_id=contract_id,
        contract_type=ContractType.LAM,
        contract_role=ContractRole.RPA,
        status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
        maturity_date=_MD,
        notional_principal=100_000.0,
        nominal_interest_rate=0.05,
        day_count_convention=DayCountConvention.A360,
        interest_payment_cycle="3M",
        principal_redemption_cycle="3M",
        next_principal_redemption_amount=25_000.0,
    )


class TestPAMPrepaymentPayoff:
    """Test PAM _pof_pp with observe_event."""

    def test_pp_payoff_with_observer(self):
        """PP payoff should return observed amount from DictRiskFactorObserver."""
        attrs = _make_pam_attrs()
        pp_amount = 10_000.0
        observer = DictRiskFactorObserver(
            risk_factors={},
            event_data={attrs.contract_id: pp_amount},
        )
        contract = create_contract(attrs, observer)
        pof = contract.get_payoff_function(EventType.PP)
        state = _state((2024, 4, 15, 0, 0, 0))
        time = ActusDateTime(2024, 7, 15, 0, 0, 0)

        payoff = pof._pof_pp(state, attrs, time, observer)
        assert float(payoff) == pytest.approx(pp_amount)

    def test_pp_payoff_no_observer_data(self):
        """PP payoff returns 0 when observer has no PP data."""
        attrs = _make_pam_attrs()
        observer = DictRiskFactorObserver(risk_factors={})
        contract = create_contract(attrs, observer)
        pof = contract.get_payoff_function(EventType.PP)
        state = _state((2024, 4, 15, 0, 0, 0))
        time = ActusDateTime(2024, 7, 15, 0, 0, 0)

        payoff = pof._pof_pp(state, attrs, time, observer)
        assert float(payoff) == pytest.approx(0.0)


class TestPAMPrepaymentSTF:
    """Test PAM _stf_pp reduces notional."""

    def test_stf_pp_reduces_notional(self):
        """STF_PP should reduce notional by PP amount from observer."""
        attrs = _make_pam_attrs()
        pp_amount = 20_000.0
        observer = DictRiskFactorObserver(
            risk_factors={},
            event_data={attrs.contract_id: pp_amount},
        )
        contract = create_contract(attrs, observer)
        stf = contract.get_state_transition_function(EventType.PP)
        state = _state((2024, 4, 15, 0, 0, 0))
        time = ActusDateTime(2024, 7, 15, 0, 0, 0)

        new_state = stf._stf_pp(state, attrs, time, observer)

        assert float(new_state.nt) == pytest.approx(100_000.0 - pp_amount)
        assert new_state.sd == time

    def test_stf_pp_accrues_interest(self):
        """STF_PP should also accrue interest from Sd to t."""
        attrs = _make_pam_attrs()
        observer = DictRiskFactorObserver(
            risk_factors={},
            event_data={attrs.contract_id: 10_000.0},
        )
        contract = create_contract(attrs, observer)
        stf = contract.get_state_transition_function(EventType.PP)
        state = _state((2024, 4, 15, 0, 0, 0))
        time = ActusDateTime(2024, 7, 15, 0, 0, 0)

        new_state = stf._stf_pp(state, attrs, time, observer)

        # 91 days between Apr 15 and Jul 15, A/360
        expected_ipac = 91 / 360.0 * 0.05 * 100_000.0
        assert float(new_state.ipac) == pytest.approx(expected_ipac, rel=1e-4)

    def test_stf_pp_no_observer_data_backward_compatible(self):
        """When observer has no PP data, notional is unchanged."""
        attrs = _make_pam_attrs()
        observer = DictRiskFactorObserver(risk_factors={})
        contract = create_contract(attrs, observer)
        stf = contract.get_state_transition_function(EventType.PP)
        state = _state((2024, 4, 15, 0, 0, 0))
        time = ActusDateTime(2024, 7, 15, 0, 0, 0)

        new_state = stf._stf_pp(state, attrs, time, observer)

        assert float(new_state.nt) == pytest.approx(100_000.0)

    def test_stf_pp_with_constant_observer(self):
        """ConstantRiskFactorObserver returns constant for observe_event too."""
        attrs = _make_pam_attrs()
        observer = ConstantRiskFactorObserver(constant_value=0.0)
        contract = create_contract(attrs, observer)
        stf = contract.get_state_transition_function(EventType.PP)
        state = _state((2024, 4, 15, 0, 0, 0))
        time = ActusDateTime(2024, 7, 15, 0, 0, 0)

        new_state = stf._stf_pp(state, attrs, time, observer)

        # ConstantRiskFactorObserver returns 0.0, so notional unchanged
        assert float(new_state.nt) == pytest.approx(100_000.0)


class TestLAMPrepayment:
    """Test LAM PP payoff and STF with observe_event."""

    def test_lam_pp_payoff_with_observer(self):
        """LAM PP payoff should return observed amount."""
        attrs = _make_lam_attrs()
        pp_amount = 15_000.0
        observer = DictRiskFactorObserver(
            risk_factors={},
            event_data={attrs.contract_id: pp_amount},
        )
        contract = create_contract(attrs, observer)
        pof = contract.get_payoff_function(EventType.PP)
        state = _state((2024, 4, 15, 0, 0, 0), ipcb=100_000.0)
        time = ActusDateTime(2024, 7, 15, 0, 0, 0)

        payoff = pof._pof_pp(state, attrs, time, observer)
        assert float(payoff) == pytest.approx(pp_amount)

    def test_lam_stf_pp_reduces_notional_and_ipcb(self):
        """LAM STF_PP should reduce notional and update IPCB (NT mode)."""
        attrs = _make_lam_attrs()
        pp_amount = 20_000.0
        observer = DictRiskFactorObserver(
            risk_factors={},
            event_data={attrs.contract_id: pp_amount},
        )
        contract = create_contract(attrs, observer)
        stf = contract.get_state_transition_function(EventType.PP)
        state = _state((2024, 4, 15, 0, 0, 0), ipcb=100_000.0)
        time = ActusDateTime(2024, 7, 15, 0, 0, 0)

        new_state = stf._stf_pp(state, attrs, time, observer)

        expected_nt = 100_000.0 - pp_amount
        assert float(new_state.nt) == pytest.approx(expected_nt)
        # IPCB should track notional in NT mode (default)
        assert float(new_state.ipcb) == pytest.approx(expected_nt)

    def test_lam_stf_pp_no_observer_data(self):
        """When no PP data, LAM notional is unchanged (backward compatible)."""
        attrs = _make_lam_attrs()
        observer = DictRiskFactorObserver(risk_factors={})
        contract = create_contract(attrs, observer)
        stf = contract.get_state_transition_function(EventType.PP)
        state = _state((2024, 4, 15, 0, 0, 0), ipcb=100_000.0)
        time = ActusDateTime(2024, 7, 15, 0, 0, 0)

        new_state = stf._stf_pp(state, attrs, time, observer)

        assert float(new_state.nt) == pytest.approx(100_000.0)
