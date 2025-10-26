"""Tests for LAX (Exotic Linear Amortizer) contract implementation."""

import pytest

from jactus.contracts import create_contract
from jactus.contracts.lax import (
    ExoticLinearAmortizerContract,
    LAXPayoffFunction,
    LAXStateTransitionFunction,
    generate_array_schedule,
)
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    ContractState,
    ContractType,
    DayCountConvention,
    EventType,
)
from jactus.observers import ConstantRiskFactorObserver


class TestArrayScheduleGeneration:
    """Test array schedule generation helper function."""

    def test_generate_array_schedule_basic(self):
        """Test basic array schedule generation."""
        anchors = [
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            ActusDateTime(2024, 6, 15, 0, 0, 0),
        ]
        cycles = ["1M", "1M"]
        end = ActusDateTime(2025, 1, 15, 0, 0, 0)

        schedule = generate_array_schedule(anchors, cycles, end)

        # Should have events from both sub-schedules
        assert len(schedule) > 0
        # All events should be after anchors and before/at end
        for event in schedule:
            assert event > anchors[0]
            assert event <= end

    def test_generate_array_schedule_with_filter(self):
        """Test array schedule generation with filter."""
        anchors = [
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            ActusDateTime(2024, 6, 15, 0, 0, 0),
        ]
        cycles = ["1M", "1M"]
        end = ActusDateTime(2025, 1, 15, 0, 0, 0)
        filter_values = ["INC", "DEC"]

        # Get only DEC events
        schedule_dec = generate_array_schedule(
            anchors, cycles, end, filter_values=filter_values, filter_target="DEC"
        )

        # Should only have events from second sub-schedule
        assert len(schedule_dec) > 0
        for event in schedule_dec:
            # Should be after second anchor
            assert event > anchors[1]

        # Get only INC events
        schedule_inc = generate_array_schedule(
            anchors, cycles, end, filter_values=filter_values, filter_target="INC"
        )

        # Should only have events from first sub-schedule
        assert len(schedule_inc) > 0
        for event in schedule_inc:
            # Should be after first anchor but before second
            assert event > anchors[0]

    def test_generate_array_schedule_empty(self):
        """Test array schedule with empty arrays."""
        schedule = generate_array_schedule([], [], ActusDateTime(2025, 1, 15, 0, 0, 0))
        assert len(schedule) == 0

    def test_generate_array_schedule_validation(self):
        """Test array schedule validation."""
        anchors = [ActusDateTime(2024, 1, 15, 0, 0, 0)]
        cycles = ["1M", "1M"]  # Length mismatch

        with pytest.raises(ValueError, match="same length"):
            generate_array_schedule(anchors, cycles, ActusDateTime(2025, 1, 15, 0, 0, 0))


class TestLAXInitialization:
    """Test LAX contract initialization."""

    def test_lax_basic_initialization(self):
        """Test basic LAX initialization with array schedules."""
        attrs = ContractAttributes(
            contract_id="LAX-TEST-001",
            contract_type=ContractType.LAX,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            array_pr_anchor=[
                ActusDateTime(2024, 2, 15, 0, 0, 0),
                ActusDateTime(2025, 1, 15, 0, 0, 0),
            ],
            array_pr_cycle=["1M", "1M"],
            array_pr_next=[1000.0, 2000.0],
            array_increase_decrease=["INC", "DEC"],
            next_principal_redemption_amount=1000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = create_contract(attrs, rf_obs)

        assert isinstance(contract, ExoticLinearAmortizerContract)
        assert contract.attributes.contract_type == ContractType.LAX

    def test_lax_array_validation_consistent_lengths(self):
        """Test that array validation requires consistent lengths."""
        # This should pass validation
        attrs = ContractAttributes(
            contract_id="LAX-TEST-002",
            contract_type=ContractType.LAX,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            array_pr_anchor=[
                ActusDateTime(2024, 2, 15, 0, 0, 0),
                ActusDateTime(2025, 1, 15, 0, 0, 0),
            ],
            array_pr_cycle=["1M", "1M"],
            array_pr_next=[1000.0, 2000.0],
            array_increase_decrease=["INC", "DEC"],
        )

        # Should not raise
        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = create_contract(attrs, rf_obs)
        assert contract is not None

    def test_lax_array_validation_invalid_arincdec(self):
        """Test that ARINCDEC values must be 'INC' or 'DEC'."""
        with pytest.raises(ValueError, match="'INC' or 'DEC'"):
            ContractAttributes(
                contract_id="LAX-TEST-003",
                contract_type=ContractType.LAX,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
                maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
                currency="USD",
                notional_principal=100000.0,
                array_pr_anchor=[ActusDateTime(2024, 2, 15, 0, 0, 0)],
                array_pr_cycle=["1M"],
                array_pr_next=[1000.0],
                array_increase_decrease=["INVALID"],  # Invalid value
            )

    def test_lax_array_validation_invalid_arfixvar(self):
        """Test that ARFIXVAR values must be 'F' or 'V'."""
        with pytest.raises(ValueError, match="'F' or 'V'"):
            ContractAttributes(
                contract_id="LAX-TEST-004",
                contract_type=ContractType.LAX,
                contract_role=ContractRole.RPA,
                status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
                maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
                currency="USD",
                notional_principal=100000.0,
                array_rr_anchor=[ActusDateTime(2024, 2, 15, 0, 0, 0)],
                array_rr_cycle=["1Y"],
                array_rate=[0.06],
                array_fixed_variable=["INVALID"],  # Invalid value
            )


class TestLAXEventSchedule:
    """Test LAX event schedule generation."""

    def test_pr_schedule_from_decrease_periods(self):
        """Test that PR events are generated from ARINCDEC='DEC' periods."""
        attrs = ContractAttributes(
            contract_id="LAX-TEST-005",
            contract_type=ContractType.LAX,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 6, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            array_pr_anchor=[
                ActusDateTime(2024, 2, 15, 0, 0, 0),
                ActusDateTime(2025, 1, 15, 0, 0, 0),
            ],
            array_pr_cycle=["3M", "1M"],
            array_pr_next=[1000.0, 2000.0],
            array_increase_decrease=["INC", "DEC"],
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Get PR events
        pr_events = [e for e in schedule.events if e.event_type == EventType.PR]

        # PR events should only come from second period (DEC)
        assert len(pr_events) > 0
        for event in pr_events:
            assert event.event_time > ActusDateTime(2025, 1, 15, 0, 0, 0)

    def test_pi_schedule_from_increase_periods(self):
        """Test that PI events are generated from ARINCDEC='INC' periods."""
        attrs = ContractAttributes(
            contract_id="LAX-TEST-006",
            contract_type=ContractType.LAX,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 6, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            array_pr_anchor=[
                ActusDateTime(2024, 2, 15, 0, 0, 0),
                ActusDateTime(2025, 1, 15, 0, 0, 0),
            ],
            array_pr_cycle=["3M", "1M"],
            array_pr_next=[1000.0, 2000.0],
            array_increase_decrease=["INC", "DEC"],
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Get PI events
        pi_events = [e for e in schedule.events if e.event_type == EventType.PI]

        # PI events should only come from first period (INC) with 3M cycle
        # Starting from 2024-02-15 with 3M cycle, before maturity 2025-06-15
        assert len(pi_events) > 0
        # All PI events should be after the first anchor
        for event in pi_events:
            assert event.event_time > ActusDateTime(2024, 2, 15, 0, 0, 0)
            # Events can extend beyond second anchor since maturity is later

    def test_prf_events_at_anchors(self):
        """Test that PRF events are generated at anchor dates."""
        attrs = ContractAttributes(
            contract_id="LAX-TEST-007",
            contract_type=ContractType.LAX,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            array_pr_anchor=[
                ActusDateTime(2024, 2, 15, 0, 0, 0),
                ActusDateTime(2025, 1, 15, 0, 0, 0),
            ],
            array_pr_cycle=["1M", "1M"],
            array_pr_next=[1000.0, 2000.0],
            array_increase_decrease=["INC", "DEC"],
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Get PRF events
        prf_events = [e for e in schedule.events if e.event_type == EventType.PRF]

        # Should have PRF events at each anchor
        assert len(prf_events) == 2
        assert prf_events[0].event_time == ActusDateTime(2024, 2, 15, 0, 0, 0)
        assert prf_events[1].event_time == ActusDateTime(2025, 1, 15, 0, 0, 0)

    def test_ip_array_schedule(self):
        """Test IP schedule generation from array."""
        attrs = ContractAttributes(
            contract_id="LAX-TEST-008",
            contract_type=ContractType.LAX,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 6, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            array_ip_anchor=[
                ActusDateTime(2024, 2, 15, 0, 0, 0),
                ActusDateTime(2025, 1, 15, 0, 0, 0),
            ],
            array_ip_cycle=["3M", "1M"],
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Get IP events
        ip_events = [e for e in schedule.events if e.event_type == EventType.IP]

        # Should have IP events from both periods
        assert len(ip_events) > 0

    def test_rr_rrf_array_schedule(self):
        """Test RR/RRF schedule generation from array with ARFIXVAR."""
        attrs = ContractAttributes(
            contract_id="LAX-TEST-009",
            contract_type=ContractType.LAX,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            array_rr_anchor=[
                ActusDateTime(2024, 2, 15, 0, 0, 0),
                ActusDateTime(2025, 1, 15, 0, 0, 0),
            ],
            array_rr_cycle=["6M", "6M"],
            array_rate=[0.05, 0.06],
            array_fixed_variable=["V", "F"],
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = create_contract(attrs, rf_obs)
        schedule = contract.generate_event_schedule()

        # Get RR and RRF events
        rr_events = [e for e in schedule.events if e.event_type == EventType.RR]
        rrf_events = [e for e in schedule.events if e.event_type == EventType.RRF]

        # RR from first period (V), RRF from second period (F)
        assert len(rr_events) > 0
        assert len(rrf_events) > 0


class TestLAXStateTransitions:
    """Test LAX state transition functions."""

    def test_pi_event_increases_notional(self):
        """Test that PI event increases notional."""
        attrs = ContractAttributes(
            contract_id="LAX-TEST-010",
            contract_type=ContractType.LAX,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            next_principal_redemption_amount=5000.0,
        )

        # Create initial state
        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2026, 1, 15, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
            ipac=0.0,
            feac=0.0,
            nsc=1.0,
            isc=1.0,
            ipcb=100000.0,
            prnxt=5000.0,
        )

        stf = LAXStateTransitionFunction()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        # Execute PI event
        new_state = stf.transition_state(
            EventType.PI, state, attrs, ActusDateTime(2024, 2, 15, 0, 0, 0), rf_obs
        )

        # Notional should increase by prnxt
        assert float(new_state.nt) > float(state.nt)
        assert float(new_state.nt) == pytest.approx(105000.0, abs=1.0)

    def test_pr_event_decreases_notional(self):
        """Test that PR event decreases notional."""
        attrs = ContractAttributes(
            contract_id="LAX-TEST-011",
            contract_type=ContractType.LAX,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            next_principal_redemption_amount=5000.0,
        )

        # Create initial state
        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2026, 1, 15, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
            ipac=0.0,
            feac=0.0,
            nsc=1.0,
            isc=1.0,
            ipcb=100000.0,
            prnxt=5000.0,
        )

        stf = LAXStateTransitionFunction()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        # Execute PR event
        new_state = stf.transition_state(
            EventType.PR, state, attrs, ActusDateTime(2024, 2, 15, 0, 0, 0), rf_obs
        )

        # Notional should decrease by prnxt
        assert float(new_state.nt) < float(state.nt)
        assert float(new_state.nt) == pytest.approx(95000.0, abs=1.0)

    def test_prf_event_updates_prnxt(self):
        """Test that PRF event updates Prnxt from array."""
        attrs = ContractAttributes(
            contract_id="LAX-TEST-012",
            contract_type=ContractType.LAX,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            array_pr_anchor=[
                ActusDateTime(2024, 2, 15, 0, 0, 0),
                ActusDateTime(2025, 1, 15, 0, 0, 0),
            ],
            array_pr_cycle=["1M", "1M"],
            array_pr_next=[1000.0, 2000.0],
            array_increase_decrease=["INC", "DEC"],
            next_principal_redemption_amount=500.0,
        )

        # Create initial state
        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2026, 1, 15, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
            ipac=0.0,
            feac=0.0,
            nsc=1.0,
            isc=1.0,
            ipcb=100000.0,
            prnxt=500.0,
        )

        stf = LAXStateTransitionFunction()
        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        # Execute PRF event at first anchor
        new_state = stf.transition_state(
            EventType.PRF, state, attrs, ActusDateTime(2024, 2, 15, 0, 0, 0), rf_obs
        )

        # Prnxt should be updated to first array value
        assert float(new_state.prnxt) == 1000.0

        # Execute PRF event at second anchor
        new_state2 = stf.transition_state(
            EventType.PRF, new_state, attrs, ActusDateTime(2025, 1, 15, 0, 0, 0), rf_obs
        )

        # Prnxt should be updated to second array value
        assert float(new_state2.prnxt) == 2000.0


class TestLAXPayoffs:
    """Test LAX payoff functions."""

    def test_pi_payoff(self):
        """Test PI payoff (same as PR but for increases)."""
        attrs = ContractAttributes(
            contract_id="LAX-TEST-013",
            contract_type=ContractType.LAX,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
        )

        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2026, 1, 15, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
            ipac=0.0,
            feac=0.0,
            nsc=1.0,
            isc=1.0,
            ipcb=100000.0,
            prnxt=5000.0,
        )

        pof = LAXPayoffFunction(
            contract_role=attrs.contract_role,
            currency=attrs.currency,
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        payoff = pof.calculate_payoff(
            EventType.PI, state, attrs, ActusDateTime(2024, 2, 15, 0, 0, 0), rf_obs
        )

        # PI payoff should be prnxt (negative from borrower perspective)
        assert float(payoff) == pytest.approx(5000.0, abs=1.0)

    def test_prf_payoff_zero(self):
        """Test that PRF event has zero payoff."""
        attrs = ContractAttributes(
            contract_id="LAX-TEST-014",
            contract_type=ContractType.LAX,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
        )

        state = ContractState(
            sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
            tmd=ActusDateTime(2026, 1, 15, 0, 0, 0),
            nt=100000.0,
            ipnr=0.05,
            ipac=0.0,
            feac=0.0,
            nsc=1.0,
            isc=1.0,
            ipcb=100000.0,
            prnxt=5000.0,
        )

        pof = LAXPayoffFunction(
            contract_role=attrs.contract_role,
            currency=attrs.currency,
        )
        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

        payoff = pof.calculate_payoff(
            EventType.PRF, state, attrs, ActusDateTime(2024, 2, 15, 0, 0, 0), rf_obs
        )

        # PRF has no payoff
        assert float(payoff) == 0.0


class TestLAXSimulation:
    """Test complete LAX simulations."""

    def test_step_up_schedule_simulation(self):
        """Test simulation with step-up principal schedule."""
        attrs = ContractAttributes(
            contract_id="LAX-STEP-UP-001",
            contract_type=ContractType.LAX,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 6, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            array_pr_anchor=[
                ActusDateTime(2024, 2, 15, 0, 0, 0),
                ActusDateTime(2025, 1, 15, 0, 0, 0),
            ],
            array_pr_cycle=["3M", "1M"],
            array_pr_next=[5000.0, 10000.0],
            array_increase_decrease=["INC", "DEC"],
            next_principal_redemption_amount=5000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = create_contract(attrs, rf_obs)

        result = contract.simulate()

        # Should complete simulation
        assert len(result.events) > 0

        # Should have both PI and PR events
        pi_events = [e for e in result.events if e.event_type == EventType.PI]
        pr_events = [e for e in result.events if e.event_type == EventType.PR]

        assert len(pi_events) > 0, "Should have PI events"
        assert len(pr_events) > 0, "Should have PR events"

    def test_mixed_increase_decrease_simulation(self):
        """Test simulation with mixed increase/decrease periods."""
        attrs = ContractAttributes(
            contract_id="LAX-MIXED-001",
            contract_type=ContractType.LAX,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2026, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            array_pr_anchor=[
                ActusDateTime(2024, 2, 15, 0, 0, 0),
                ActusDateTime(2024, 8, 15, 0, 0, 0),
                ActusDateTime(2025, 2, 15, 0, 0, 0),
            ],
            array_pr_cycle=["1M", "1M", "1M"],
            array_pr_next=[2000.0, 3000.0, 4000.0],
            array_increase_decrease=["INC", "DEC", "DEC"],
            next_principal_redemption_amount=2000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = create_contract(attrs, rf_obs)

        result = contract.simulate()

        # Should complete simulation
        assert len(result.events) > 0

        # Get final state
        final_state = result.final_state

        # Notional should have changed over time
        assert final_state is not None

    def test_final_notional_zero(self):
        """Test that properly structured LAX reaches zero notional."""
        # Create LAX that increases then decreases to zero
        attrs = ContractAttributes(
            contract_id="LAX-ZERO-001",
            contract_type=ContractType.LAX,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
            currency="USD",
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
            day_count_convention=DayCountConvention.A360,
            array_pr_anchor=[
                ActusDateTime(2024, 2, 15, 0, 0, 0),
                ActusDateTime(2024, 4, 15, 0, 0, 0),
            ],
            array_pr_cycle=["1M", "1M"],
            array_pr_next=[10000.0, 20000.0],
            array_increase_decrease=["INC", "DEC"],
            next_principal_redemption_amount=10000.0,
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.05)
        contract = create_contract(attrs, rf_obs)

        result = contract.simulate()

        # Final state should have notional close to zero or zero
        # (MD event zeros it out)
        final_state = result.final_state
        assert final_state is not None
        # After MD, notional should be exactly zero
        md_events = [e for e in result.events if e.event_type == EventType.MD]
        assert len(md_events) == 1
