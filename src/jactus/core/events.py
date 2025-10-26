"""Contract event structures for ACTUS contracts.

This module provides ContractEvent and EventSchedule classes for representing
and managing contract events (cash flows and state transitions).

References:
    ACTUS Technical Specification v1.1, Section 2.5, 2.9 (Events)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from jactus.core.states import ContractState
from jactus.core.time import ActusDateTime
from jactus.core.types import EventType


@dataclass
class ContractEvent:
    """Represents a single contract event.

    An event is a discrete occurrence on the timeline that may generate
    a cash flow (payoff) and/or change the contract state.

    Attributes:
        event_type: Type of event (IED, IP, PR, MD, etc.)
        event_time: When the event occurs (τ operator)
        payoff: Cash flow amount (φ operator)
        currency: Currency of payoff
        state_pre: Contract state before event
        state_post: Contract state after event
        sequence: Sequence number for ordering same-time events

    Example:
        >>> event = ContractEvent(
        ...     event_type=EventType.IP,
        ...     event_time=ActusDateTime(2024, 4, 15, 0, 0, 0),
        ...     payoff=jnp.array(1250.0),
        ...     currency="USD",
        ...     sequence=EVENT_SEQUENCE_ORDER[EventType.IP],
        ... )

    References:
        ACTUS Technical Specification v1.1, Section 2.5
    """

    event_type: EventType
    event_time: ActusDateTime
    payoff: jnp.ndarray
    currency: str
    state_pre: ContractState | None = None
    state_post: ContractState | None = None
    sequence: int = 0

    def __lt__(self, other: ContractEvent) -> bool:
        """Compare by time, then sequence."""
        if self.event_time != other.event_time:
            return self.event_time < other.event_time
        return self.sequence < other.sequence

    def __le__(self, other: ContractEvent) -> bool:
        """Compare by time, then sequence."""
        if self.event_time != other.event_time:
            return self.event_time <= other.event_time
        return self.sequence <= other.sequence

    def __gt__(self, other: ContractEvent) -> bool:
        """Compare by time, then sequence."""
        if self.event_time != other.event_time:
            return self.event_time > other.event_time
        return self.sequence > other.sequence

    def __ge__(self, other: ContractEvent) -> bool:
        """Compare by time, then sequence."""
        if self.event_time != other.event_time:
            return self.event_time >= other.event_time
        return self.sequence >= other.sequence

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, ContractEvent):
            return NotImplemented
        return (
            self.event_type == other.event_type
            and self.event_time == other.event_time
            and self.currency == other.currency
            and bool(jnp.allclose(self.payoff, other.payoff))
            and self.sequence == other.sequence
        )

    def __hash__(self) -> int:
        """Hash for use in dicts/sets."""
        return hash((self.event_type, self.event_time, self.currency, self.sequence))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "event_type": self.event_type.value,
            "event_time": self.event_time.to_iso(),
            "payoff": float(self.payoff),
            "currency": self.currency,
            "sequence": self.sequence,
            # Note: state_pre/state_post not serialized by default (can be large)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContractEvent:
        """Create ContractEvent from dictionary.

        Args:
            data: Dictionary with event data

        Returns:
            New ContractEvent instance
        """
        return cls(
            event_type=EventType(data["event_type"]),
            event_time=ActusDateTime.from_iso(data["event_time"]),
            payoff=jnp.array(data["payoff"]),
            currency=data["currency"],
            sequence=data.get("sequence", 0),
        )


@dataclass(frozen=True)
class EventSchedule:
    """Immutable container for a sequence of events.

    Represents the complete event schedule for a contract, maintaining
    events in chronological order.

    Attributes:
        events: Immutable tuple of events
        contract_id: Associated contract identifier

    Example:
        >>> events = [event1, event2, event3]
        >>> schedule = EventSchedule(
        ...     events=tuple(sorted(events)),
        ...     contract_id="LOAN-001",
        ... )

    References:
        ACTUS Technical Specification v1.1, Section 2.9
    """

    events: tuple[ContractEvent, ...]
    contract_id: str

    def __len__(self) -> int:
        """Return number of events."""
        return len(self.events)

    def __iter__(self) -> Any:
        """Iterate over events."""
        return iter(self.events)

    def __getitem__(self, index: int) -> ContractEvent:
        """Get event by index."""
        return self.events[index]

    def add_event(self, event: ContractEvent) -> EventSchedule:
        """Add an event and return new schedule.

        Since schedules are immutable, this creates a new schedule
        with the event added in sorted order.

        Args:
            event: Event to add

        Returns:
            New EventSchedule with event added

        Example:
            >>> new_schedule = schedule.add_event(new_event)
        """
        new_events = list(self.events)
        new_events.append(event)
        new_events.sort()
        return EventSchedule(tuple(new_events), self.contract_id)

    def filter_by_type(self, event_type: EventType) -> EventSchedule:
        """Filter events by type.

        Args:
            event_type: Type to filter for

        Returns:
            New EventSchedule with only matching events

        Example:
            >>> ip_events = schedule.filter_by_type(EventType.IP)
        """
        filtered = [e for e in self.events if e.event_type == event_type]
        return EventSchedule(tuple(filtered), self.contract_id)

    def filter_by_time_range(self, start: ActusDateTime, end: ActusDateTime) -> EventSchedule:
        """Filter events by time range.

        Args:
            start: Start time (inclusive)
            end: End time (inclusive)

        Returns:
            New EventSchedule with events in range

        Example:
            >>> range_events = schedule.filter_by_time_range(
            ...     ActusDateTime(2024, 1, 1, 0, 0, 0),
            ...     ActusDateTime(2024, 12, 31, 0, 0, 0),
            ... )
        """
        filtered = [e for e in self.events if start <= e.event_time <= end]
        return EventSchedule(tuple(filtered), self.contract_id)

    def merge(self, other: EventSchedule) -> EventSchedule:
        """Merge with another schedule.

        Combines events from both schedules and sorts by time/sequence.

        Args:
            other: Other schedule to merge

        Returns:
            New EventSchedule with merged events

        Example:
            >>> combined = schedule1.merge(schedule2)
        """
        merged = list(self.events) + list(other.events)
        merged.sort()
        return EventSchedule(tuple(merged), self.contract_id)

    def get_payoffs(self) -> jnp.ndarray:
        """Extract all payoffs as a JAX array.

        Returns:
            Array of payoff values

        Example:
            >>> payoffs = schedule.get_payoffs()
        """
        if not self.events:
            return jnp.array([])
        return jnp.array([float(e.payoff) for e in self.events])

    def get_times(self) -> list[ActusDateTime]:
        """Extract all event times.

        Returns:
            List of event times

        Example:
            >>> times = schedule.get_times()
        """
        return [e.event_time for e in self.events]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "contract_id": self.contract_id,
            "events": [e.to_dict() for e in self.events],
        }


# Event sequence ordering per ACTUS specification Section 2.9
# Events occurring at the same time are processed in this order
EVENT_SEQUENCE_ORDER: dict[EventType, int] = {
    EventType.IED: 1,  # Initial Exchange
    EventType.FP: 2,  # Fee Payment
    EventType.PR: 3,  # Principal Redemption
    EventType.PI: 4,  # Principal Increase
    EventType.PRD: 5,  # Principal Redemption Drawing
    EventType.PY: 6,  # Penalty Payment
    EventType.PP: 7,  # Principal Prepayment
    EventType.IP: 8,  # Interest Payment
    EventType.IPCI: 9,  # Interest Capitalization
    EventType.RR: 10,  # Rate Reset
    EventType.RRF: 11,  # Rate Reset Fixing
    EventType.DV: 12,  # Dividend Payment
    EventType.SC: 13,  # Scaling Index Fixing
    EventType.IPCB: 14,  # Interest Calculation Base Fixing
    EventType.XD: 15,  # Exercise
    EventType.STD: 16,  # Settlement
    EventType.MD: 17,  # Maturity
    EventType.TD: 18,  # Termination
    EventType.CE: 19,  # Credit Event
    EventType.AD: 20,  # Monitoring
    EventType.PRF: 21,  # Performance
}


def tau(event: ContractEvent | EventSchedule) -> ActusDateTime | list[ActusDateTime]:
    """τ (tau) operator - Extract event time(s).

    The τ operator extracts the time component of events.

    Args:
        event: Single event or event schedule

    Returns:
        Event time(s)

    Example:
        >>> t = tau(event)
        >>> times = tau(schedule)

    References:
        ACTUS Technical Specification v1.1, Section 2.5
    """
    if isinstance(event, ContractEvent):
        return event.event_time
    return event.get_times()


def phi(event: ContractEvent | EventSchedule) -> jnp.ndarray:
    """φ (phi) operator - Extract payoff(s).

    The φ operator extracts the payoff (cash flow) component of events.

    Args:
        event: Single event or event schedule

    Returns:
        Payoff(s) as JAX array

    Example:
        >>> p = phi(event)
        >>> payoffs = phi(schedule)

    References:
        ACTUS Technical Specification v1.1, Section 2.5
    """
    if isinstance(event, ContractEvent):
        return event.payoff
    return event.get_payoffs()


def sort_events(events: list[ContractEvent]) -> list[ContractEvent]:
    """Sort events by time, then by sequence.

    Args:
        events: List of events to sort

    Returns:
        Sorted list of events

    Example:
        >>> sorted_events = sort_events([event3, event1, event2])

    References:
        ACTUS Technical Specification v1.1, Section 2.9
    """
    return sorted(events)


def merge_congruent_events(event1: ContractEvent, event2: ContractEvent) -> ContractEvent:
    """Merge two events at the same time.

    Used for composite contracts where multiple events occur simultaneously.
    Payoffs are summed, and the earliest event type takes precedence.

    Args:
        event1: First event
        event2: Second event

    Returns:
        Merged event

    Raises:
        ValueError: If events have different times or currencies

    Example:
        >>> merged = merge_congruent_events(event1, event2)

    References:
        ACTUS Technical Specification v1.1, Section 2.9
    """
    if event1.event_time != event2.event_time:
        raise ValueError("Cannot merge events at different times")

    if event1.currency != event2.currency:
        raise ValueError("Cannot merge events with different currencies")

    # Use the earlier sequence (higher priority event type)
    primary = event1 if event1.sequence < event2.sequence else event2

    # Sum payoffs
    merged_payoff = event1.payoff + event2.payoff

    return ContractEvent(
        event_type=primary.event_type,
        event_time=primary.event_time,
        payoff=merged_payoff,
        currency=primary.currency,
        sequence=primary.sequence,
        state_pre=primary.state_pre,
        state_post=event2.state_post,  # Use final state
    )
