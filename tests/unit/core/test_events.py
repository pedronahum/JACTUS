"""Unit tests for contract events.

Test ID: T1.5
"""

import jax.numpy as jnp
import pytest

from jactus.core.events import (
    EVENT_SEQUENCE_ORDER,
    ContractEvent,
    EventSchedule,
    merge_congruent_events,
    phi,
    sort_events,
    tau,
)
from jactus.core.time import ActusDateTime
from jactus.core.types import EventType


class TestContractEvent:
    """Test ContractEvent class."""

    def test_create_event(self):
        """Test creating a basic event."""
        event = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
            sequence=EVENT_SEQUENCE_ORDER[EventType.IP],
        )

        assert event.event_type == EventType.IP
        assert event.event_time == ActusDateTime(2024, 4, 15, 0, 0, 0)
        assert float(event.payoff) == pytest.approx(1250.0)
        assert event.currency == "USD"
        assert event.sequence == EVENT_SEQUENCE_ORDER[EventType.IP]

    def test_event_comparison_by_time(self):
        """Test that events are compared by time."""
        event1 = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
            payoff=jnp.array(1000.0),
            currency="USD",
        )

        event2 = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 7, 15, 0, 0, 0),
            payoff=jnp.array(1000.0),
            currency="USD",
        )

        assert event1 < event2
        assert event2 > event1
        assert event1 <= event2
        assert event2 >= event1

    def test_event_comparison_by_sequence(self):
        """Test that events at same time are compared by sequence."""
        # IED has sequence 1, IP has sequence 8
        event_ied = ContractEvent(
            event_type=EventType.IED,
            event_time=ActusDateTime(2024, 1, 15, 0, 0, 0),
            payoff=jnp.array(-100000.0),
            currency="USD",
            sequence=EVENT_SEQUENCE_ORDER[EventType.IED],
        )

        event_ip = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 1, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
            sequence=EVENT_SEQUENCE_ORDER[EventType.IP],
        )

        # IED should come before IP at same time
        assert event_ied < event_ip
        assert event_ip > event_ied

    def test_event_equality(self):
        """Test event equality."""
        event1 = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
            sequence=8,
        )

        event2 = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
            sequence=8,
        )

        assert event1 == event2

    def test_event_hash(self):
        """Test that events can be hashed."""
        event = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
        )

        # Should be hashable
        d = {event: "value"}
        assert d[event] == "value"


class TestEventSerialization:
    """Test event serialization."""

    def test_to_dict(self):
        """Test converting event to dictionary."""
        event = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
            sequence=8,
        )

        data = event.to_dict()

        assert data["event_type"] == "IP"
        assert data["event_time"] == "2024-04-15T00:00:00"
        assert data["payoff"] == pytest.approx(1250.0)
        assert data["currency"] == "USD"
        assert data["sequence"] == 8

    def test_from_dict(self):
        """Test creating event from dictionary."""
        data = {
            "event_type": "IP",
            "event_time": "2024-04-15T00:00:00",
            "payoff": 1250.0,
            "currency": "USD",
            "sequence": 8,
        }

        event = ContractEvent.from_dict(data)

        assert event.event_type == EventType.IP
        assert event.event_time == ActusDateTime(2024, 4, 15, 0, 0, 0)
        assert float(event.payoff) == pytest.approx(1250.0)
        assert event.currency == "USD"

    def test_roundtrip_serialization(self):
        """Test that serialization round-trips correctly."""
        original = ContractEvent(
            event_type=EventType.PR,
            event_time=ActusDateTime(2024, 1, 15, 0, 0, 0),
            payoff=jnp.array(5000.0),
            currency="EUR",
            sequence=3,
        )

        data = original.to_dict()
        restored = ContractEvent.from_dict(data)

        assert restored.event_type == original.event_type
        assert restored.event_time == original.event_time
        assert restored.currency == original.currency


class TestEventSchedule:
    """Test EventSchedule class."""

    def test_create_schedule(self):
        """Test creating an event schedule."""
        events = [
            ContractEvent(
                event_type=EventType.IED,
                event_time=ActusDateTime(2024, 1, 15, 0, 0, 0),
                payoff=jnp.array(-100000.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
        ]

        schedule = EventSchedule(tuple(events), "LOAN-001")

        assert len(schedule) == 2
        assert schedule.contract_id == "LOAN-001"

    def test_schedule_iteration(self):
        """Test iterating over schedule."""
        events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 7, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
        ]

        schedule = EventSchedule(tuple(events), "LOAN-001")

        count = 0
        for event in schedule:
            assert isinstance(event, ContractEvent)
            count += 1

        assert count == 2

    def test_schedule_indexing(self):
        """Test accessing events by index."""
        events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
        ]

        schedule = EventSchedule(tuple(events), "LOAN-001")

        assert schedule[0].event_type == EventType.IP

    def test_add_event(self):
        """Test adding an event to schedule."""
        event1 = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
        )

        schedule = EventSchedule(tuple([event1]), "LOAN-001")

        event2 = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 7, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
        )

        new_schedule = schedule.add_event(event2)

        assert len(schedule) == 1  # Original unchanged
        assert len(new_schedule) == 2  # New schedule has both

    def test_filter_by_type(self):
        """Test filtering events by type."""
        events = [
            ContractEvent(
                event_type=EventType.IED,
                event_time=ActusDateTime(2024, 1, 15, 0, 0, 0),
                payoff=jnp.array(-100000.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 7, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.MD,
                event_time=ActusDateTime(2029, 1, 15, 0, 0, 0),
                payoff=jnp.array(100000.0),
                currency="USD",
            ),
        ]

        schedule = EventSchedule(tuple(events), "LOAN-001")
        ip_events = schedule.filter_by_type(EventType.IP)

        assert len(ip_events) == 2
        assert all(e.event_type == EventType.IP for e in ip_events)

    def test_filter_by_time_range(self):
        """Test filtering events by time range."""
        events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 1, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 7, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
        ]

        schedule = EventSchedule(tuple(events), "LOAN-001")
        q2_events = schedule.filter_by_time_range(
            ActusDateTime(2024, 4, 1, 0, 0, 0),
            ActusDateTime(2024, 6, 30, 0, 0, 0),
        )

        assert len(q2_events) == 1
        assert q2_events[0].event_time == ActusDateTime(2024, 4, 15, 0, 0, 0)

    def test_merge_schedules(self):
        """Test merging two schedules."""
        events1 = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
        ]

        events2 = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 7, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
        ]

        schedule1 = EventSchedule(tuple(events1), "LOAN-001")
        schedule2 = EventSchedule(tuple(events2), "LOAN-001")

        merged = schedule1.merge(schedule2)

        assert len(merged) == 2
        assert merged[0].event_time < merged[1].event_time  # Sorted

    def test_get_payoffs(self):
        """Test extracting all payoffs."""
        events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 7, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
        ]

        schedule = EventSchedule(tuple(events), "LOAN-001")
        payoffs = schedule.get_payoffs()

        assert len(payoffs) == 2
        assert float(payoffs[0]) == pytest.approx(1250.0)
        assert float(payoffs[1]) == pytest.approx(1250.0)

    def test_get_times(self):
        """Test extracting all event times."""
        events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 7, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
        ]

        schedule = EventSchedule(tuple(events), "LOAN-001")
        times = schedule.get_times()

        assert len(times) == 2
        assert times[0] == ActusDateTime(2024, 4, 15, 0, 0, 0)
        assert times[1] == ActusDateTime(2024, 7, 15, 0, 0, 0)


class TestOperators:
    """Test τ (tau) and φ (phi) operators."""

    def test_tau_single_event(self):
        """Test τ operator on single event."""
        event = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
        )

        t = tau(event)
        assert t == ActusDateTime(2024, 4, 15, 0, 0, 0)

    def test_tau_schedule(self):
        """Test τ operator on schedule."""
        events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 7, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
        ]

        schedule = EventSchedule(tuple(events), "LOAN-001")
        times = tau(schedule)

        assert len(times) == 2
        assert times[0] == ActusDateTime(2024, 4, 15, 0, 0, 0)

    def test_phi_single_event(self):
        """Test φ operator on single event."""
        event = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
        )

        p = phi(event)
        assert float(p) == pytest.approx(1250.0)

    def test_phi_schedule(self):
        """Test φ operator on schedule."""
        events = [
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
            ContractEvent(
                event_type=EventType.IP,
                event_time=ActusDateTime(2024, 7, 15, 0, 0, 0),
                payoff=jnp.array(1250.0),
                currency="USD",
            ),
        ]

        schedule = EventSchedule(tuple(events), "LOAN-001")
        payoffs = phi(schedule)

        assert len(payoffs) == 2
        assert float(payoffs[0]) == pytest.approx(1250.0)


class TestEventFunctions:
    """Test event utility functions."""

    def test_sort_events(self):
        """Test sorting events."""
        event1 = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 7, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
        )

        event2 = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
        )

        event3 = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 10, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
        )

        unsorted = [event1, event2, event3]
        sorted_events = sort_events(unsorted)

        assert sorted_events[0] == event2
        assert sorted_events[1] == event1
        assert sorted_events[2] == event3

    def test_merge_congruent_events(self):
        """Test merging events at same time."""
        event1 = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
            sequence=8,
        )

        event2 = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
            payoff=jnp.array(750.0),
            currency="USD",
            sequence=8,
        )

        merged = merge_congruent_events(event1, event2)

        assert merged.event_time == ActusDateTime(2024, 4, 15, 0, 0, 0)
        assert float(merged.payoff) == pytest.approx(2000.0)  # Sum of payoffs
        assert merged.currency == "USD"

    def test_merge_different_times_fails(self):
        """Test that merging events at different times fails."""
        event1 = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
        )

        event2 = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 7, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
        )

        with pytest.raises(ValueError) as exc_info:
            merge_congruent_events(event1, event2)

        assert "different times" in str(exc_info.value)

    def test_merge_different_currencies_fails(self):
        """Test that merging events with different currencies fails."""
        event1 = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
            payoff=jnp.array(1250.0),
            currency="USD",
        )

        event2 = ContractEvent(
            event_type=EventType.IP,
            event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
            payoff=jnp.array(1000.0),
            currency="EUR",
        )

        with pytest.raises(ValueError) as exc_info:
            merge_congruent_events(event1, event2)

        assert "different currencies" in str(exc_info.value)


class TestEventSequenceOrder:
    """Test event sequence ordering."""

    def test_sequence_order_defined(self):
        """Test that EVENT_SEQUENCE_ORDER is defined."""
        assert isinstance(EVENT_SEQUENCE_ORDER, dict)
        assert len(EVENT_SEQUENCE_ORDER) > 0

    def test_ied_comes_first(self):
        """Test that IED has lowest sequence (comes first)."""
        assert EVENT_SEQUENCE_ORDER[EventType.IED] == 1

    def test_unique_sequences(self):
        """Test that all sequences are unique."""
        sequences = list(EVENT_SEQUENCE_ORDER.values())
        assert len(sequences) == len(set(sequences))

    def test_all_event_types_covered(self):
        """Test that all event types have sequence defined."""
        # Check major event types
        major_types = [
            EventType.IED,
            EventType.IP,
            EventType.PR,
            EventType.MD,
            EventType.RR,
            EventType.FP,
        ]

        for event_type in major_types:
            assert event_type in EVENT_SEQUENCE_ORDER
