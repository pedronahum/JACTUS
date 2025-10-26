"""Contract lifetime and lifecycle management.

This module provides functions for managing contract lifecycles, including
calculating contract end dates, determining contract phases, and filtering
events by lifecycle status.

References:
    ACTUS v1.1 Section 4 - Contract Lifecycle
"""

from enum import Enum

from jactus.core import ActusDateTime, ContractAttributes, ContractEvent

# Constants
DEFAULT_T_MAX_YEARS = 100  # Default maximum contract duration in years


class ContractPhase(str, Enum):
    """Contract lifecycle phases.

    Attributes:
        PRE_INCEPTION: Before initial exchange date (IED)
        ACTIVE: Between IED and maturity/termination
        MATURED: After maturity date (MD) or amortization end (AMD)
        TERMINATED: After termination date (TD)
    """

    PRE_INCEPTION = "pre_inception"
    ACTIVE = "active"
    MATURED = "matured"
    TERMINATED = "terminated"


def calculate_contract_end(
    attributes: ContractAttributes,
    t_max_years: int = DEFAULT_T_MAX_YEARS,
) -> ActusDateTime:
    """Calculate the end date of a contract.

    Determines when a contract ends based on ACTUS termination rules:
    1. Termination Date (TD) if set (highest priority)
    2. Maturity Date (MD) if set
    3. Amortization End Date (AMD) if calculable
    4. Purchase Date (STD/PRD) + t_max if perpetual
    5. Status Date + t_max as fallback

    Args:
        attributes: Contract attributes
        t_max_years: Maximum contract duration in years (default: 100)

    Returns:
        Contract end date

    Example:
        >>> attrs = ContractAttributes(
        ...     contract_id="TEST",
        ...     contract_type=ContractType.PAM,
        ...     contract_role=ContractRole.RPA,
        ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        ...     maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
        ... )
        >>> end_date = calculate_contract_end(attrs)
        >>> # Returns maturity_date: 2034-01-01

    Note:
        For perpetual contracts (no MD), uses status_date + t_max_years
        to provide a practical simulation boundary.

    References:
        ACTUS v1.1 Section 4.1 - Contract Termination
    """
    # Priority 1: Termination date (explicit termination)
    if attributes.termination_date is not None:
        return attributes.termination_date

    # Priority 2: Maturity date (natural end)
    if attributes.maturity_date is not None:
        return attributes.maturity_date

    # Priority 3: Amortization end date
    # Note: AMD calculation depends on contract type and schedule
    # For now, we check if it's explicitly set
    # TODO: Calculate AMD from amortization schedule when available

    # Priority 4: Purchase/Status date + t_max (perpetual contracts)
    # Use purchase date if available, otherwise status date
    start_date = attributes.purchase_date or attributes.status_date

    # Calculate end date by adding t_max_years
    return start_date.add_period(f"{t_max_years}Y")


def is_contract_active(
    attributes: ContractAttributes,
    current_time: ActusDateTime,
    t_max_years: int = DEFAULT_T_MAX_YEARS,
) -> bool:
    """Check if a contract is active at a given time.

    A contract is active if:
    - Current time >= Initial Exchange Date (IED) or Status Date
    - Current time <= Contract end date

    Args:
        attributes: Contract attributes
        current_time: Time to check
        t_max_years: Maximum contract duration in years

    Returns:
        True if contract is active, False otherwise

    Example:
        >>> attrs = ContractAttributes(
        ...     contract_id="TEST",
        ...     contract_type=ContractType.PAM,
        ...     contract_role=ContractRole.RPA,
        ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        ...     initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
        ...     maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
        ... )
        >>> # Before IED
        >>> is_contract_active(attrs, ActusDateTime(2024, 1, 10, 0, 0, 0))
        False
        >>> # During active period
        >>> is_contract_active(attrs, ActusDateTime(2025, 1, 1, 0, 0, 0))
        True
        >>> # After maturity
        >>> is_contract_active(attrs, ActusDateTime(2035, 1, 1, 0, 0, 0))
        False

    References:
        ACTUS v1.1 Section 4.2 - Contract Activity
    """
    # Determine start date (IED or status date)
    start_date = attributes.initial_exchange_date or attributes.status_date

    # Get end date
    end_date = calculate_contract_end(attributes, t_max_years)

    # Contract is active if current time is between start and end
    return start_date <= current_time <= end_date


def get_contract_phase(
    attributes: ContractAttributes,
    current_time: ActusDateTime,
    t_max_years: int = DEFAULT_T_MAX_YEARS,
) -> ContractPhase:
    """Get the current phase of a contract's lifecycle.

    Determines which phase the contract is in:
    - PRE_INCEPTION: Before initial exchange date
    - ACTIVE: Between inception and maturity/termination
    - MATURED: After maturity date (natural end)
    - TERMINATED: After termination date (forced end)

    Args:
        attributes: Contract attributes
        current_time: Time to evaluate
        t_max_years: Maximum contract duration in years

    Returns:
        Current contract phase

    Example:
        >>> attrs = ContractAttributes(
        ...     contract_id="TEST",
        ...     contract_type=ContractType.PAM,
        ...     contract_role=ContractRole.RPA,
        ...     status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
        ...     initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
        ...     maturity_date=ActusDateTime(2034, 1, 1, 0, 0, 0),
        ... )
        >>> phase = get_contract_phase(attrs, ActusDateTime(2024, 1, 10, 0, 0, 0))
        >>> assert phase == ContractPhase.PRE_INCEPTION
        >>> phase = get_contract_phase(attrs, ActusDateTime(2025, 1, 1, 0, 0, 0))
        >>> assert phase == ContractPhase.ACTIVE
        >>> phase = get_contract_phase(attrs, ActusDateTime(2035, 1, 1, 0, 0, 0))
        >>> assert phase == ContractPhase.MATURED

    References:
        ACTUS v1.1 Section 4.3 - Contract Phases
    """
    # Determine start date (IED or status date)
    start_date = attributes.initial_exchange_date or attributes.status_date

    # Check if before inception
    if current_time < start_date:
        return ContractPhase.PRE_INCEPTION

    # Check if terminated (explicit termination takes precedence)
    if attributes.termination_date is not None and current_time >= attributes.termination_date:
        return ContractPhase.TERMINATED

    # Check if matured (natural end)
    if attributes.maturity_date is not None and current_time >= attributes.maturity_date:
        return ContractPhase.MATURED

    # Calculate end date to check against t_max
    end_date = calculate_contract_end(attributes, t_max_years)
    if current_time > end_date:
        # Past the calculated end (for perpetual contracts)
        return ContractPhase.MATURED

    # Otherwise, contract is active
    return ContractPhase.ACTIVE


def filter_events_by_lifecycle(
    events: list[ContractEvent],
    attributes: ContractAttributes,
    include_pre_inception: bool = False,
    include_post_maturity: bool = False,
    t_max_years: int = DEFAULT_T_MAX_YEARS,
) -> list[ContractEvent]:
    """Filter events based on contract lifecycle status.

    Filters events to only include those that occur during the active
    contract period. Optionally includes pre-inception and post-maturity events.

    Args:
        events: List of contract events
        attributes: Contract attributes
        include_pre_inception: If True, include events before IED
        include_post_maturity: If True, include events after maturity/termination
        t_max_years: Maximum contract duration in years

    Returns:
        Filtered list of events

    Example:
        >>> events = [...]  # List of ContractEvent objects
        >>> attrs = ContractAttributes(...)
        >>> # Get only active period events
        >>> active_events = filter_events_by_lifecycle(events, attrs)
        >>> # Include all events including pre-inception
        >>> all_events = filter_events_by_lifecycle(
        ...     events, attrs,
        ...     include_pre_inception=True,
        ...     include_post_maturity=True
        ... )

    References:
        ACTUS v1.1 Section 4.4 - Event Filtering
    """
    filtered = []
    for event in events:
        phase = get_contract_phase(attributes, event.event_time, t_max_years)

        # Always include active phase events
        if phase == ContractPhase.ACTIVE:
            filtered.append(event)
            continue

        # Include pre-inception if requested
        if include_pre_inception and phase == ContractPhase.PRE_INCEPTION:
            filtered.append(event)
            continue

        # Include post-maturity if requested
        if include_post_maturity and phase in (
            ContractPhase.MATURED,
            ContractPhase.TERMINATED,
        ):
            filtered.append(event)
            continue

    return filtered
