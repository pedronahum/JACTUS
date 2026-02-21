"""Unit tests for SWAPS (Generic Swap) contract."""

import jax.numpy as jnp
import pytest

from jactus.contracts import GenericSwapContract, create_contract
from jactus.contracts.swaps import determine_leg_roles, merge_congruent_events
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractEvent,
    ContractRole,
    ContractState,
    ContractType,
    EventType,
)
from jactus.observers import ConstantRiskFactorObserver, MockChildContractObserver


class TestSWAPSInitialization:
    """Test SWAPS contract initialization and validation."""

    def test_swaps_contract_creation(self):
        """Test successful SWAPS contract creation."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
            delivery_settlement="D",
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        swap = GenericSwapContract(attrs, rf_obs, child_obs)
        assert swap is not None
        assert swap.attributes.contract_type == ContractType.SWAPS

    def test_swaps_requires_child_observer(self):
        """Test that SWAPS requires child contract observer."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)

        with pytest.raises(ValueError, match="child_contract_observer is required"):
            GenericSwapContract(attrs, rf_obs, None)

    def test_swaps_requires_contract_structure(self):
        """Test that SWAPS requires contract structure."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        with pytest.raises(ValueError, match="contract_structure.*is required"):
            GenericSwapContract(attrs, rf_obs, child_obs)

    def test_swaps_requires_first_and_second_leg(self):
        """Test that contract structure must have FirstLeg and SecondLeg."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_structure='{"FirstLeg": "LEG1"}',  # Missing SecondLeg
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        with pytest.raises(ValueError, match="FirstLeg.*SecondLeg"):
            GenericSwapContract(attrs, rf_obs, child_obs)

    def test_swaps_wrong_contract_type(self):
        """Test that wrong contract type raises error."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.PAM,  # Wrong type
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        with pytest.raises(ValueError, match="Expected contract_type=SWAPS"):
            GenericSwapContract(attrs, rf_obs, child_obs)


class TestLegRoleAssignment:
    """Test leg role assignment logic."""

    def test_rfl_assigns_rpa_rpl(self):
        """Test RFL (Receive First Leg) assigns RPA to first, RPL to second."""
        first_role, second_role = determine_leg_roles(ContractRole.RFL)
        assert first_role == ContractRole.RPA
        assert second_role == ContractRole.RPL

    def test_pfl_assigns_rpl_rpa(self):
        """Test PFL (Pay First Leg) assigns RPL to first, RPA to second."""
        first_role, second_role = determine_leg_roles(ContractRole.PFL)
        assert first_role == ContractRole.RPL
        assert second_role == ContractRole.RPA

    def test_rpa_defaults_to_pfl(self):
        """Test RPA (not RFL) defaults to PFL logic."""
        first_role, second_role = determine_leg_roles(ContractRole.RPA)
        assert first_role == ContractRole.RPL
        assert second_role == ContractRole.RPA


class TestEventMerging:
    """Test event merging for congruent events."""

    def test_merge_congruent_events(self):
        """Test merging two events with same time and type."""
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        event1 = ContractEvent(
            event_type=EventType.IP,
            event_time=time,
            payoff=1000.0,
            currency="USD",
        )

        event2 = ContractEvent(
            event_type=EventType.IP,
            event_time=time,
            payoff=-800.0,
            currency="USD",
        )

        merged = merge_congruent_events(event1, event2)

        assert merged.event_type == EventType.IP
        assert merged.event_time == time
        assert float(merged.payoff) == pytest.approx(200.0, abs=0.01)

    def test_merge_preserves_event_type(self):
        """Test that merged event preserves event type."""
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        event1 = ContractEvent(
            event_type=EventType.PR,
            event_time=time,
            payoff=10000.0,
            currency="USD",
        )

        event2 = ContractEvent(
            event_type=EventType.PR,
            event_time=time,
            payoff=-10000.0,
            currency="USD",
        )

        merged = merge_congruent_events(event1, event2)

        assert merged.event_type == EventType.PR
        assert float(merged.payoff) == pytest.approx(0.0, abs=0.01)


class TestSWAPSEventSchedule:
    """Test SWAPS event schedule generation."""

    def test_swaps_generates_events_from_legs(self):
        """Test that SWAPS generates events from child legs."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2025, 1, 1, 0, 0, 0),
            delivery_settlement="D",  # Net settlement
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        # Register child events
        leg1_events = [
            ContractEvent(
                event_type=EventType.IED,
                event_time=ActusDateTime(2024, 1, 15, 0, 0, 0),
                payoff=0.0,
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 6, 15, 0, 0, 0),
                payoff=1000.0,
                currency="USD",
            ),
        ]

        leg2_events = [
            ContractEvent(
                event_type=EventType.IED,
                event_time=ActusDateTime(2024, 1, 15, 0, 0, 0),
                payoff=0.0,
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 6, 15, 0, 0, 0),
                payoff=-800.0,
                currency="USD",
            ),
        ]

        child_obs.register_child("LEG1", events=leg1_events)
        child_obs.register_child("LEG2", events=leg2_events)

        swap = GenericSwapContract(attrs, rf_obs, child_obs)
        schedule = swap.generate_event_schedule()

        # Should have merged IED and IP events (net settlement)
        assert len(schedule.events) >= 2

    def test_swaps_net_settlement_merges_congruent(self):
        """Test that DS='S' (cash/net settlement) merges congruent events."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            delivery_settlement="S",  # Cash/net settlement
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        # Same time IP events
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        leg1_events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=time,
                payoff=1000.0,
                currency="USD",
            ),
        ]

        leg2_events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=time,
                payoff=-800.0,
                currency="USD",
            ),
        ]

        child_obs.register_child("LEG1", events=leg1_events)
        child_obs.register_child("LEG2", events=leg2_events)

        swap = GenericSwapContract(attrs, rf_obs, child_obs)
        schedule = swap.generate_event_schedule()

        # Should have 1 merged IP event with net payoff
        ip_events = [e for e in schedule.events if e.event_type == EventType.IP]
        assert len(ip_events) == 1
        assert float(ip_events[0].payoff) == pytest.approx(200.0, abs=0.01)

    def test_swaps_gross_settlement_keeps_separate(self):
        """Test that DS='D' (delivery/gross) keeps events separate."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            delivery_settlement="D",  # Delivery/gross settlement
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        # Same time IP events
        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        leg1_events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=time,
                payoff=1000.0,
                currency="USD",
            ),
        ]

        leg2_events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=time,
                payoff=-800.0,
                currency="USD",
            ),
        ]

        child_obs.register_child("LEG1", events=leg1_events)
        child_obs.register_child("LEG2", events=leg2_events)

        swap = GenericSwapContract(attrs, rf_obs, child_obs)
        schedule = swap.generate_event_schedule()

        # Should have 2 separate IP events
        ip_events = [e for e in schedule.events if e.event_type == EventType.IP]
        assert len(ip_events) == 2

    def test_swaps_non_congruent_events_not_merged(self):
        """Test that non-congruent event types are not merged."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            delivery_settlement="D",  # Net settlement
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        time = ActusDateTime(2024, 6, 15, 0, 0, 0)

        # Different event types (not congruent)
        leg1_events = [
            ContractEvent(
                event_type=EventType.RR,  # Rate reset
                event_time=time,
                payoff=0.0,
                currency="USD",
            ),
        ]

        leg2_events = [
            ContractEvent(
                event_type=EventType.FP,  # Fee payment
                event_time=time,
                payoff=100.0,
                currency="USD",
            ),
        ]

        child_obs.register_child("LEG1", events=leg1_events)
        child_obs.register_child("LEG2", events=leg2_events)

        swap = GenericSwapContract(attrs, rf_obs, child_obs)
        schedule = swap.generate_event_schedule()

        # Should have both events (not merged - different types)
        assert len([e for e in schedule.events if e.event_type == EventType.RR]) == 1
        assert len([e for e in schedule.events if e.event_type == EventType.FP]) == 1


class TestSWAPSStateInitialization:
    """Test SWAPS state initialization."""

    def test_swaps_initialize_state_aggregates_from_legs(self):
        """Test that initial state is aggregated from child legs."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        # Register child states
        leg1_state = ContractState(
            tmd=ActusDateTime(2029, 1, 1, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(100000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.05, dtype=jnp.float32),
            ipac=jnp.array(500.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
        )

        leg2_state = ContractState(
            tmd=ActusDateTime(2029, 1, 1, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(100000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.03, dtype=jnp.float32),
            ipac=jnp.array(-300.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
        )

        child_obs.register_child("LEG1", state=leg1_state)
        child_obs.register_child("LEG2", state=leg2_state)

        swap = GenericSwapContract(attrs, rf_obs, child_obs)
        state = swap.initialize_state()

        # ipac should be sum of both legs
        assert float(state.ipac) == pytest.approx(200.0, abs=0.01)


class TestSWAPSPayoffAndStateTransition:
    """Test SWAPS payoff and state transition functions."""

    def test_swaps_payoff_returns_zero(self):
        """Test that SWAPS payoff function returns zero (payoffs from children)."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        # Register child states
        leg1_state = ContractState(
            tmd=ActusDateTime(2029, 1, 1, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(100000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.05, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
        )
        child_obs.register_child("LEG1", state=leg1_state)
        child_obs.register_child("LEG2", state=leg1_state)

        swap = GenericSwapContract(attrs, rf_obs, child_obs)
        state = swap.initialize_state()
        pof = swap.get_payoff_function(EventType.IP)

        payoff = pof.calculate_payoff(
            EventType.IP,
            state,
            attrs,
            ActusDateTime(2024, 6, 15, 0, 0, 0),
            rf_obs,
        )

        # SWAPS payoffs come from child events, parent returns zero
        assert float(payoff) == pytest.approx(0.0, abs=0.01)

    def test_swaps_state_transition_returns_unchanged(self):
        """Test that SWAPS state transition returns state unchanged."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        # Register child states
        leg1_state = ContractState(
            tmd=ActusDateTime(2029, 1, 1, 0, 0, 0),
            sd=ActusDateTime(2024, 1, 1, 0, 0, 0),
            nt=jnp.array(100000.0, dtype=jnp.float32),
            ipnr=jnp.array(0.05, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
        )
        child_obs.register_child("LEG1", state=leg1_state)
        child_obs.register_child("LEG2", state=leg1_state)

        swap = GenericSwapContract(attrs, rf_obs, child_obs)
        state = swap.initialize_state()
        stf = swap.get_state_transition_function(EventType.IP)

        state_post = stf.transition_state(
            EventType.IP,
            state,
            attrs,
            ActusDateTime(2024, 6, 15, 0, 0, 0),
            rf_obs,
        )

        # SWAPS state changes come from children, parent returns unchanged
        assert state_post == state


class TestSWAPSEdgeCases:
    """Test SWAPS edge cases and error handling."""

    def test_swaps_invalid_json_structure(self):
        """Test that invalid JSON in contract_structure raises error."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_structure="{invalid json",  # Invalid JSON
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        with pytest.raises(ValueError, match="valid JSON"):
            GenericSwapContract(attrs, rf_obs, child_obs)

    def test_swaps_non_dict_structure(self):
        """Test that non-dict contract_structure raises error."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_structure='"just a string"',  # Valid JSON but not a dict
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        with pytest.raises(ValueError, match="JSON object"):
            GenericSwapContract(attrs, rf_obs, child_obs)

    def test_swaps_with_analysis_dates(self):
        """Test that SWAPS can include analysis dates."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
            analysis_dates=[
                ActusDateTime(2024, 3, 1, 0, 0, 0),
                ActusDateTime(2024, 6, 1, 0, 0, 0),
            ],
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        child_obs.register_child("LEG1", events=[])
        child_obs.register_child("LEG2", events=[])

        swap = GenericSwapContract(attrs, rf_obs, child_obs)
        schedule = swap.generate_event_schedule()

        ad_events = [e for e in schedule.events if e.event_type == EventType.AD]
        assert len(ad_events) == 2

    def test_swaps_with_termination_date(self):
        """Test that SWAPS can include termination date."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
            termination_date=ActusDateTime(2025, 12, 31, 0, 0, 0),
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        child_obs.register_child("LEG1", events=[])
        child_obs.register_child("LEG2", events=[])

        swap = GenericSwapContract(attrs, rf_obs, child_obs)
        schedule = swap.generate_event_schedule()

        td_events = [e for e in schedule.events if e.event_type == EventType.TD]
        assert len(td_events) == 1
        assert td_events[0].event_time == attrs.termination_date


class TestSWAPSFactory:
    """Test SWAPS factory creation."""

    def test_create_swaps_via_factory(self):
        """Test creating SWAPS contract using create_contract factory."""
        attrs = ContractAttributes(
            contract_id="SWAP001",
            contract_type=ContractType.SWAPS,
            contract_role=ContractRole.RFL,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            maturity_date=ActusDateTime(2029, 1, 1, 0, 0, 0),
            contract_structure='{"FirstLeg": "LEG1", "SecondLeg": "LEG2"}',
        )

        rf_obs = ConstantRiskFactorObserver(constant_value=0.03)
        child_obs = MockChildContractObserver()

        swap = create_contract(attrs, rf_obs, child_obs)
        assert isinstance(swap, GenericSwapContract)
        assert swap.attributes.contract_type == ContractType.SWAPS
