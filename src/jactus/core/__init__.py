"""Core type definitions and fundamental structures for ACTUS contracts.

This module provides the foundational types, enumerations, and data structures
used throughout the JACTUS package.
"""

from jactus.core.attributes import ATTRIBUTE_MAP, ContractAttributes
from jactus.core.events import (
    EVENT_SEQUENCE_ORDER,
    ContractEvent,
    EventSchedule,
    merge_congruent_events,
    phi,
    sort_events,
    tau,
)
from jactus.core.states import ContractState, initialize_state
from jactus.core.time import (
    ActusDateTime,
    add_period,
    adjust_to_business_day,
    is_business_day,
    parse_cycle,
    parse_iso_datetime,
)
from jactus.core.types import (
    # Type aliases
    Amount,
    # Enumerations
    BusinessDayConvention,
    Calendar,
    ContractPerformance,
    ContractRole,
    ContractType,
    Cycle,
    CyclePointOfInterestPayment,
    DayCountConvention,
    EndOfMonthConvention,
    EventType,
    FeeBasis,
    InterestCalculationBase,
    Percentage,
    PrepaymentEffect,
    Rate,
    ScalingEffect,
    Timestamp,
)

__all__ = [
    # Type aliases
    "Timestamp",
    "Amount",
    "Rate",
    "Percentage",
    "Cycle",
    # Enumerations
    "EventType",
    "ContractType",
    "ContractRole",
    "DayCountConvention",
    "BusinessDayConvention",
    "EndOfMonthConvention",
    "Calendar",
    "ContractPerformance",
    "FeeBasis",
    "InterestCalculationBase",
    "CyclePointOfInterestPayment",
    "PrepaymentEffect",
    "ScalingEffect",
    # Date/Time
    "ActusDateTime",
    "parse_iso_datetime",
    "parse_cycle",
    "add_period",
    "is_business_day",
    "adjust_to_business_day",
    # Contract Attributes
    "ContractAttributes",
    "ATTRIBUTE_MAP",
    # Contract States
    "ContractState",
    "initialize_state",
    # Events
    "ContractEvent",
    "EventSchedule",
    "EVENT_SEQUENCE_ORDER",
    "tau",
    "phi",
    "sort_events",
    "merge_congruent_events",
]
