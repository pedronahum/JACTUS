"""Unit tests for child contract observer.

T2.4: Child Contract Observer Tests

Tests for:
- ChildContractObserver Protocol
- BaseChildContractObserver ABC
- MockChildContractObserver
- Condition application
- Multiple child contracts
"""

import jax.numpy as jnp
import pytest

from jactus.core import ActusDateTime, ContractAttributes, ContractEvent, ContractState
from jactus.core.types import ContractRole, ContractType, EventType
from jactus.observers.child_contract import (
    BaseChildContractObserver,
    ChildContractObserver,
    MockChildContractObserver,
)


class TestChildContractObserverProtocol:
    """Test ChildContractObserver protocol enforcement."""

    def test_protocol_is_runtime_checkable(self):
        """Test that protocol can be checked with isinstance()."""
        observer = MockChildContractObserver()
        assert isinstance(observer, ChildContractObserver)

    def test_protocol_requires_observe_events(self):
        """Test that protocol requires observe_events method."""

        class IncompleteObserver:
            def observe_state(self, identifier, time, state=None, attributes=None):
                pass

            def observe_attribute(self, identifier, attribute_name):
                pass

        observer = IncompleteObserver()
        assert not isinstance(observer, ChildContractObserver)

    def test_protocol_requires_observe_state(self):
        """Test that protocol requires observe_state method."""

        class IncompleteObserver:
            def observe_events(self, identifier, time, attributes=None):
                pass

            def observe_attribute(self, identifier, attribute_name):
                pass

        observer = IncompleteObserver()
        assert not isinstance(observer, ChildContractObserver)

    def test_protocol_requires_observe_attribute(self):
        """Test that protocol requires observe_attribute method."""

        class IncompleteObserver:
            def observe_events(self, identifier, time, attributes=None):
                pass

            def observe_state(self, identifier, time, state=None, attributes=None):
                pass

        observer = IncompleteObserver()
        assert not isinstance(observer, ChildContractObserver)


class TestBaseChildContractObserver:
    """Test BaseChildContractObserver abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseChildContractObserver()  # type: ignore

    def test_subclass_must_implement_abstract_methods(self):
        """Test that subclass must implement all abstract methods."""

        class IncompleteObserver(BaseChildContractObserver):
            def _get_events(self, identifier, time, attributes):
                return []

            # Missing _get_state and _get_attribute

        with pytest.raises(TypeError):
            IncompleteObserver()  # type: ignore

    def test_observe_events_wraps_get_events(self):
        """Test that observe_events calls _get_events."""

        class TestObserver(BaseChildContractObserver):
            def _get_events(self, identifier, time, attributes):
                return [
                    ContractEvent(
                        event_type=EventType.IED,
                        event_time=time,
                        payoff=jnp.array(0.0),
                        currency="USD",
                        state_pre=None,
                        state_post=None,
                    )
                ]

            def _get_state(self, identifier, time, state, attributes):
                return ContractState(
                    sd=time,
                    tmd=time,
                    nt=jnp.array(100000.0),
                    ipnr=jnp.array(0.05),
                    ipac=jnp.array(0.0),
                    feac=jnp.array(0.0),
                    nsc=jnp.array(1.0),
                    isc=jnp.array(1.0),
                )

            def _get_attribute(self, identifier, attribute_name):
                return jnp.array(100000.0)

        observer = TestObserver()
        time = ActusDateTime(2024, 1, 1, 0, 0, 0)
        events = observer.observe_events("child1", time)
        assert len(events) == 1
        assert events[0].event_type == EventType.IED

    def test_observe_events_validates_return_type(self):
        """Test that observe_events validates return type is list."""

        class BadObserver(BaseChildContractObserver):
            def _get_events(self, identifier, time, attributes):
                return "not a list"  # type: ignore

            def _get_state(self, identifier, time, state, attributes):
                return ContractState(
                    sd=time,
                    tmd=time,
                    nt=jnp.array(100000.0),
                    ipnr=jnp.array(0.05),
                    ipac=jnp.array(0.0),
                    feac=jnp.array(0.0),
                    nsc=jnp.array(1.0),
                    isc=jnp.array(1.0),
                )

            def _get_attribute(self, identifier, attribute_name):
                return jnp.array(100000.0)

        observer = BadObserver()
        time = ActusDateTime(2024, 1, 1, 0, 0, 0)
        with pytest.raises(ValueError, match="Expected list of events"):
            observer.observe_events("child1", time)

    def test_observe_state_wraps_get_state(self):
        """Test that observe_state calls _get_state."""

        class TestObserver(BaseChildContractObserver):
            def _get_events(self, identifier, time, attributes):
                return []

            def _get_state(self, identifier, time, state, attributes):
                return ContractState(
                    sd=time,
                    tmd=time,
                    nt=jnp.array(100000.0),
                    ipnr=jnp.array(0.05),
                    ipac=jnp.array(0.0),
                    feac=jnp.array(0.0),
                    nsc=jnp.array(1.0),
                    isc=jnp.array(1.0),
                )

            def _get_attribute(self, identifier, attribute_name):
                return jnp.array(100000.0)

        observer = TestObserver()
        time = ActusDateTime(2024, 1, 1, 0, 0, 0)
        state = observer.observe_state("child1", time)
        assert isinstance(state, ContractState)
        assert state.nt == jnp.array(100000.0)

    def test_observe_state_validates_return_type(self):
        """Test that observe_state validates return type is ContractState."""

        class BadObserver(BaseChildContractObserver):
            def _get_events(self, identifier, time, attributes):
                return []

            def _get_state(self, identifier, time, state, attributes):
                return "not a state"  # type: ignore

            def _get_attribute(self, identifier, attribute_name):
                return jnp.array(100000.0)

        observer = BadObserver()
        time = ActusDateTime(2024, 1, 1, 0, 0, 0)
        with pytest.raises(ValueError, match="Expected ContractState"):
            observer.observe_state("child1", time)

    def test_observe_attribute_wraps_get_attribute(self):
        """Test that observe_attribute calls _get_attribute."""

        class TestObserver(BaseChildContractObserver):
            def _get_events(self, identifier, time, attributes):
                return []

            def _get_state(self, identifier, time, state, attributes):
                return ContractState(
                    sd=time,
                    tmd=time,
                    nt=jnp.array(100000.0),
                    ipnr=jnp.array(0.05),
                    ipac=jnp.array(0.0),
                    feac=jnp.array(0.0),
                    nsc=jnp.array(1.0),
                    isc=jnp.array(1.0),
                )

            def _get_attribute(self, identifier, attribute_name):
                return jnp.array(100000.0)

        observer = TestObserver()
        value = observer.observe_attribute("child1", "notional_principal")
        assert isinstance(value, jnp.ndarray)
        assert value == jnp.array(100000.0)

    def test_observe_attribute_converts_to_jax_array(self):
        """Test that observe_attribute converts return value to JAX array."""

        class TestObserver(BaseChildContractObserver):
            def _get_events(self, identifier, time, attributes):
                return []

            def _get_state(self, identifier, time, state, attributes):
                return ContractState(
                    status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
                    nt=jnp.array(100000.0),
                    ipnr=jnp.array(0.05),
                )

            def _get_attribute(self, identifier, attribute_name):
                return 100000.0  # Python float

        observer = TestObserver()
        value = observer.observe_attribute("child1", "notional_principal")
        assert isinstance(value, jnp.ndarray)
        assert value.dtype == jnp.float32


class TestMockChildContractObserver:
    """Test MockChildContractObserver implementation."""

    def test_init_creates_empty_observer(self):
        """Test that __init__ creates empty dictionaries."""
        observer = MockChildContractObserver()
        assert observer.child_events == {}
        assert observer.child_states == {}
        assert observer.child_attributes == {}

    def test_register_child_with_events(self):
        """Test registering child with events."""
        observer = MockChildContractObserver()
        time = ActusDateTime(2024, 1, 1, 0, 0, 0)
        events = [
            ContractEvent(
                event_type=EventType.IED,
                event_time=time,
                payoff=jnp.array(0.0),
                currency="USD",
                state_pre=None,
                state_post=None,
            )
        ]
        observer.register_child("child1", events=events)
        assert "child1" in observer.child_events
        assert observer.child_events["child1"] == events

    def test_register_child_with_state(self):
        """Test registering child with state."""
        observer = MockChildContractObserver()
        time = ActusDateTime(2024, 1, 1, 0, 0, 0)
        state = ContractState(
            sd=time,
            tmd=time,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )
        observer.register_child("child1", state=state)
        assert "child1" in observer.child_states
        assert observer.child_states["child1"] == state

    def test_register_child_with_attributes(self):
        """Test registering child with attributes."""
        observer = MockChildContractObserver()
        attributes = {"notional_principal": 100000.0, "nominal_interest_rate": 0.05}
        observer.register_child("child1", attributes=attributes)
        assert "child1" in observer.child_attributes
        assert observer.child_attributes["child1"] == attributes

    def test_register_child_with_all_data(self):
        """Test registering child with events, state, and attributes."""
        observer = MockChildContractObserver()
        time = ActusDateTime(2024, 1, 1, 0, 0, 0)

        events = [
            ContractEvent(
                event_type=EventType.IED,
                event_time=time,
                payoff=jnp.array(0.0),
                currency="USD",
                state_pre=None,
                state_post=None,
            )
        ]
        state = ContractState(
            sd=time,
            tmd=time,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )
        attributes = {"notional_principal": 100000.0}

        observer.register_child("child1", events=events, state=state, attributes=attributes)

        assert "child1" in observer.child_events
        assert "child1" in observer.child_states
        assert "child1" in observer.child_attributes

    def test_observe_events_returns_all_events_after_time(self):
        """Test that observe_events filters by time."""
        observer = MockChildContractObserver()
        t1 = ActusDateTime(2024, 1, 1, 0, 0, 0)
        t2 = ActusDateTime(2024, 2, 1, 0, 0, 0)
        t3 = ActusDateTime(2024, 3, 1, 0, 0, 0)

        events = [
            ContractEvent(
                event_type=EventType.IED,
                event_time=t1,
                payoff=jnp.array(0.0),
                currency="USD",
                state_pre=None,
                state_post=None,
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=t2,
                payoff=jnp.array(500.0),
                currency="USD",
                state_pre=None,
                state_post=None,
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=t3,
                payoff=jnp.array(500.0),
                currency="USD",
                state_pre=None,
                state_post=None,
            ),
        ]
        observer.register_child("child1", events=events)

        # Observe from t2 - should get t2 and t3
        observed = observer.observe_events("child1", t2)
        assert len(observed) == 2
        assert observed[0].event_time == t2
        assert observed[1].event_time == t3

    def test_observe_events_raises_on_missing_child(self):
        """Test that observe_events raises KeyError for missing child."""
        observer = MockChildContractObserver()
        time = ActusDateTime(2024, 1, 1, 0, 0, 0)
        with pytest.raises(KeyError, match="Child contract not found: child1"):
            observer.observe_events("child1", time)

    def test_observe_state_returns_child_state(self):
        """Test that observe_state returns registered state."""
        observer = MockChildContractObserver()
        time = ActusDateTime(2024, 1, 1, 0, 0, 0)
        state = ContractState(
            sd=time,
            tmd=time,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )
        observer.register_child("child1", state=state)

        observed_state = observer.observe_state("child1", time)
        assert observed_state == state

    def test_observe_state_raises_on_missing_child(self):
        """Test that observe_state raises KeyError for missing child."""
        observer = MockChildContractObserver()
        time = ActusDateTime(2024, 1, 1, 0, 0, 0)
        with pytest.raises(KeyError, match="Child contract not found: child1"):
            observer.observe_state("child1", time)

    def test_observe_attribute_returns_attribute_value(self):
        """Test that observe_attribute returns registered attribute."""
        observer = MockChildContractObserver()
        attributes = {"notional_principal": 100000.0}
        observer.register_child("child1", attributes=attributes)

        value = observer.observe_attribute("child1", "notional_principal")
        assert isinstance(value, jnp.ndarray)
        assert value == jnp.array(100000.0, dtype=jnp.float32)

    def test_observe_attribute_raises_on_missing_child(self):
        """Test that observe_attribute raises KeyError for missing child."""
        observer = MockChildContractObserver()
        with pytest.raises(KeyError, match="Child contract not found: child1"):
            observer.observe_attribute("child1", "notional_principal")

    def test_observe_attribute_raises_on_missing_attribute(self):
        """Test that observe_attribute raises AttributeError for missing attribute."""
        observer = MockChildContractObserver()
        observer.register_child("child1", attributes={"notional_principal": 100000.0})
        with pytest.raises(AttributeError, match="Attribute 'rate' not found"):
            observer.observe_attribute("child1", "rate")


class TestConditionApplication:
    """Test condition application for temporary attribute overrides."""

    def test_apply_conditions_overrides_single_attribute(self):
        """Test that apply_conditions overrides a single attribute."""
        observer = MockChildContractObserver()
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            notional_principal=100000.0,
        )

        new_attrs = observer.apply_conditions(attrs, {"notional_principal": 200000.0})
        assert new_attrs.notional_principal == 200000.0
        # Original should be unchanged
        assert attrs.notional_principal == 100000.0

    def test_apply_conditions_overrides_multiple_attributes(self):
        """Test that apply_conditions overrides multiple attributes."""
        observer = MockChildContractObserver()
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
        )

        new_attrs = observer.apply_conditions(
            attrs, {"notional_principal": 200000.0, "nominal_interest_rate": 0.06}
        )
        assert new_attrs.notional_principal == 200000.0
        assert new_attrs.nominal_interest_rate == 0.06
        # Original unchanged
        assert attrs.notional_principal == 100000.0
        assert attrs.nominal_interest_rate == 0.05

    def test_apply_conditions_preserves_other_attributes(self):
        """Test that apply_conditions preserves attributes not in overrides."""
        observer = MockChildContractObserver()
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            notional_principal=100000.0,
            nominal_interest_rate=0.05,
        )

        new_attrs = observer.apply_conditions(attrs, {"notional_principal": 200000.0})
        assert new_attrs.nominal_interest_rate == 0.05  # Preserved
        assert new_attrs.contract_type == ContractType.PAM  # Preserved
        assert new_attrs.contract_role == ContractRole.RPA  # Preserved

    def test_apply_conditions_is_immutable(self):
        """Test that apply_conditions does not modify original attributes."""
        observer = MockChildContractObserver()
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            notional_principal=100000.0,
        )

        original_notional = attrs.notional_principal
        observer.apply_conditions(attrs, {"notional_principal": 200000.0})
        assert attrs.notional_principal == original_notional  # Unchanged


class TestMultipleChildContracts:
    """Test handling multiple child contracts."""

    def test_register_multiple_children(self):
        """Test registering multiple child contracts."""
        observer = MockChildContractObserver()
        time = ActusDateTime(2024, 1, 1, 0, 0, 0)

        state1 = ContractState(
            sd=time,
            tmd=time,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )
        state2 = ContractState(
            sd=time,
            tmd=time,
            nt=jnp.array(200000.0),
            ipnr=jnp.array(0.06),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        observer.register_child("child1", state=state1)
        observer.register_child("child2", state=state2)

        assert len(observer.child_states) == 2
        assert observer.observe_state("child1", time).nt == jnp.array(100000.0)
        assert observer.observe_state("child2", time).nt == jnp.array(200000.0)

    def test_observe_different_children(self):
        """Test observing different children returns different data."""
        observer = MockChildContractObserver()

        attrs1 = {"notional_principal": 100000.0}
        attrs2 = {"notional_principal": 200000.0}

        observer.register_child("loan1", attributes=attrs1)
        observer.register_child("loan2", attributes=attrs2)

        value1 = observer.observe_attribute("loan1", "notional_principal")
        value2 = observer.observe_attribute("loan2", "notional_principal")

        assert value1 == jnp.array(100000.0)
        assert value2 == jnp.array(200000.0)

    def test_children_are_independent(self):
        """Test that child contracts are independent."""
        observer = MockChildContractObserver()
        time = ActusDateTime(2024, 1, 1, 0, 0, 0)

        event1 = ContractEvent(
            event_type=EventType.IED,
            event_time=time,
            payoff=jnp.array(0.0),
            currency="USD",
            state_pre=None,
            state_post=None,
        )
        event2 = ContractEvent(
            event_type=EventType.IP,
            event_time=time,
            payoff=jnp.array(500.0),
            currency="EUR",
            state_pre=None,
            state_post=None,
        )

        observer.register_child("child1", events=[event1])
        observer.register_child("child2", events=[event2])

        events1 = observer.observe_events("child1", time)
        events2 = observer.observe_events("child2", time)

        assert len(events1) == 1
        assert len(events2) == 1
        assert events1[0].event_type == EventType.IED
        assert events2[0].event_type == EventType.IP


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_observe_events_with_empty_list(self):
        """Test observing events when child has empty event list."""
        observer = MockChildContractObserver()
        time = ActusDateTime(2024, 1, 1, 0, 0, 0)
        observer.register_child("child1", events=[])

        events = observer.observe_events("child1", time)
        assert events == []

    def test_observe_events_with_no_matching_events(self):
        """Test observing events when no events match time filter."""
        observer = MockChildContractObserver()
        t1 = ActusDateTime(2024, 1, 1, 0, 0, 0)
        t2 = ActusDateTime(2024, 12, 31, 0, 0, 0)

        event = ContractEvent(
            event_type=EventType.IED,
            event_time=t1,
            payoff=jnp.array(0.0),
            currency="USD",
            state_pre=None,
            state_post=None,
        )
        observer.register_child("child1", events=[event])

        # Observe from future time - no events should match
        events = observer.observe_events("child1", t2)
        assert events == []

    def test_register_child_updates_existing_data(self):
        """Test that re-registering a child updates its data."""
        observer = MockChildContractObserver()
        time = ActusDateTime(2024, 1, 1, 0, 0, 0)

        state1 = ContractState(
            sd=time,
            tmd=time,
            nt=jnp.array(100000.0),
            ipnr=jnp.array(0.05),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )
        state2 = ContractState(
            sd=time,
            tmd=time,
            nt=jnp.array(200000.0),
            ipnr=jnp.array(0.06),
            ipac=jnp.array(0.0),
            feac=jnp.array(0.0),
            nsc=jnp.array(1.0),
            isc=jnp.array(1.0),
        )

        observer.register_child("child1", state=state1)
        assert observer.observe_state("child1", time).nt == jnp.array(100000.0)

        # Re-register with new state
        observer.register_child("child1", state=state2)
        assert observer.observe_state("child1", time).nt == jnp.array(200000.0)

    def test_apply_conditions_with_empty_overrides(self):
        """Test that apply_conditions with empty dict returns equivalent attributes."""
        observer = MockChildContractObserver()
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            notional_principal=100000.0,
        )

        new_attrs = observer.apply_conditions(attrs, {})
        assert new_attrs.notional_principal == attrs.notional_principal
        assert new_attrs.contract_type == attrs.contract_type

    def test_apply_conditions_ignores_unknown_attributes(self):
        """Test that apply_conditions ignores attributes not in ContractAttributes."""
        observer = MockChildContractObserver()
        attrs = ContractAttributes(
            contract_id="TEST",
            contract_type=ContractType.PAM,
            contract_role=ContractRole.RPA,
            status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
            notional_principal=100000.0,
        )

        # This should not raise an error, just ignore the unknown attribute
        new_attrs = observer.apply_conditions(
            attrs, {"notional_principal": 200000.0, "unknown_attr": 999.0}
        )
        assert new_attrs.notional_principal == 200000.0
