"""Schedule generation utilities for ACTUS contracts.

This module provides functions for generating event schedules according to
ACTUS specifications, including handling of cycle notation, end-of-month
conventions, and business day conventions.

References:
    ACTUS Technical Specification v1.1, Section 3 (Schedule Generation)
"""

from __future__ import annotations

from jactus.core.time import ActusDateTime, adjust_to_business_day, parse_cycle
from jactus.core.types import BusinessDayConvention, Calendar, EndOfMonthConvention


def generate_schedule(
    start: ActusDateTime | None,
    cycle: str | None,
    end: ActusDateTime | None,
    end_of_month_convention: EndOfMonthConvention = EndOfMonthConvention.SD,
    business_day_convention: BusinessDayConvention = BusinessDayConvention.NULL,
    calendar: Calendar = Calendar.NO_CALENDAR,
) -> list[ActusDateTime]:
    """Generate a regular event schedule S(s, c, T).

    Generates dates starting from 'start', adding 'cycle' repeatedly until
    reaching or exceeding 'end'. Applies end-of-month and business day
    conventions.

    Args:
        start: Schedule start date (anchor)
        cycle: Cycle string in NPS format (e.g., '3M', '1Y', '1Q+')
        end: Schedule end date
        end_of_month_convention: How to handle month-end dates
        business_day_convention: How to adjust non-business days
        calendar: Business day calendar to use

    Returns:
        List of dates in chronological order

    Example:
        >>> schedule = generate_schedule(
        ...     start=ActusDateTime(2024, 1, 15, 0, 0, 0),
        ...     cycle="3M",
        ...     end=ActusDateTime(2025, 1, 15, 0, 0, 0),
        ... )

    References:
        ACTUS Technical Specification v1.1, Section 3.1
    """
    # Handle empty schedules
    if start is None or end is None:
        return []

    if cycle is None or cycle == "":
        return [start]

    # Parse cycle
    multiplier, period, stub = parse_cycle(cycle)

    # Generate base schedule
    dates = []
    current = start

    while current <= end:
        dates.append(current)
        # Add cycle to get next date
        current = current.add_period(f"{multiplier}{period}", end_of_month_convention)

    # Handle stub indicator
    if stub == "-" and len(dates) > 1:
        # Short stub: remove last date if it's beyond end
        if dates[-1] > end:
            dates = dates[:-1]
    elif stub == "+":
        # Long stub: keep dates beyond end up to one cycle
        pass  # Already done

    # Apply end-of-month convention
    if end_of_month_convention == EndOfMonthConvention.EOM:
        dates = apply_end_of_month_convention(dates, start, cycle, end_of_month_convention)

    # Apply business day convention
    if business_day_convention != BusinessDayConvention.NULL:
        dates = apply_business_day_convention(dates, business_day_convention, calendar)

    # Remove duplicates and sort
    return sorted(set(dates))


def generate_array_schedule(
    anchors: list[ActusDateTime],
    cycles: list[str],
    end: ActusDateTime,
    end_of_month_convention: EndOfMonthConvention = EndOfMonthConvention.SD,
    business_day_convention: BusinessDayConvention = BusinessDayConvention.NULL,
    calendar: Calendar = Calendar.NO_CALENDAR,
) -> list[ActusDateTime]:
    """Generate array schedule S~(~s, ~c, T).

    Generates a schedule from multiple (anchor, cycle) pairs. Each pair
    generates a sub-schedule that ends at the next anchor or final end.

    Args:
        anchors: List of anchor dates
        cycles: List of cycle strings (same length as anchors)
        end: Final end date
        end_of_month_convention: How to handle month-end dates
        business_day_convention: How to adjust non-business days
        calendar: Business day calendar

    Returns:
        Sorted list of all dates from all sub-schedules

    Example:
        >>> schedule = generate_array_schedule(
        ...     anchors=[
        ...         ActusDateTime(2024, 1, 15, 0, 0, 0),
        ...         ActusDateTime(2024, 7, 15, 0, 0, 0),
        ...     ],
        ...     cycles=["3M", "6M"],
        ...     end=ActusDateTime(2025, 1, 15, 0, 0, 0),
        ... )

    References:
        ACTUS Technical Specification v1.1, Section 3.2
    """
    if len(anchors) != len(cycles):
        raise ValueError("anchors and cycles must have same length")

    if not anchors:
        return []

    all_dates = []

    for i, (anchor, cycle) in enumerate(zip(anchors, cycles, strict=True)):
        # Determine end for this sub-schedule
        sub_end = anchors[i + 1] if i < len(anchors) - 1 else end

        # Generate sub-schedule
        sub_schedule = generate_schedule(
            start=anchor,
            cycle=cycle,
            end=sub_end,
            end_of_month_convention=end_of_month_convention,
            business_day_convention=business_day_convention,
            calendar=calendar,
        )

        # Don't include the boundary to avoid duplicates (except for last)
        if i < len(anchors) - 1:
            sub_schedule = [d for d in sub_schedule if d < sub_end]

        all_dates.extend(sub_schedule)

    # Add final end date
    all_dates.append(end)

    # Remove duplicates and sort
    return sorted(set(all_dates))


def apply_end_of_month_convention(
    dates: list[ActusDateTime],
    start: ActusDateTime,
    cycle: str,
    convention: EndOfMonthConvention,
) -> list[ActusDateTime]:
    """Apply end-of-month convention to schedule.

    The EOM convention only applies if:
    1. Start date is the last day of a month with <31 days
    2. Cycle is a multiple of 1 month

    Args:
        dates: List of dates to adjust
        start: Original start date
        cycle: Cycle string
        convention: EOM convention to apply

    Returns:
        List of adjusted dates

    References:
        ACTUS Technical Specification v1.1, Section 3.3
    """
    if convention == EndOfMonthConvention.SD:
        return dates  # No adjustment for Same Day

    # Check if EOM applies
    if not start.is_end_of_month():
        return dates

    # Check if cycle is monthly (M, Q, H, Y)
    multiplier, period, _ = parse_cycle(cycle)
    if period not in ("M", "Q", "H", "Y"):
        return dates

    # Apply EOM: move each date to end of its month
    import datetime

    adjusted = []
    for date in dates:
        if not date.is_end_of_month():
            # Find end of current month
            py_dt = date.to_datetime()

            # Find last day of current month
            # Note: ACTUS uses naive datetimes (no timezone)
            if py_dt.month == 12:
                next_month = datetime.datetime(py_dt.year + 1, 1, 1)  # noqa: DTZ001
            else:
                next_month = datetime.datetime(py_dt.year, py_dt.month + 1, 1)  # noqa: DTZ001

            last_day = next_month - datetime.timedelta(days=1)
            adjusted.append(
                ActusDateTime(
                    last_day.year,
                    last_day.month,
                    last_day.day,
                    date.hour,
                    date.minute,
                    date.second,
                )
            )
        else:
            adjusted.append(date)

    return adjusted


def apply_business_day_convention(
    dates: list[ActusDateTime],
    convention: BusinessDayConvention,
    calendar: Calendar,
) -> list[ActusDateTime]:
    """Apply business day convention to schedule.

    Adjusts each date according to the specified convention if it falls on
    a non-business day.

    Args:
        dates: List of dates to adjust
        convention: Business day convention
        calendar: Business day calendar

    Returns:
        List of adjusted dates

    References:
        ACTUS Technical Specification v1.1, Section 3.4
    """
    if convention == BusinessDayConvention.NULL:
        return dates

    adjusted = []
    for date in dates:
        adjusted_date = adjust_to_business_day(date, convention, calendar)
        adjusted.append(adjusted_date)

    return adjusted


def expand_period_to_months(period: str, multiplier: int) -> int | None:
    """Convert period to number of months.

    Args:
        period: Period type (D, W, M, Q, H, Y)
        multiplier: Number of periods

    Returns:
        Number of months, or None for D/W (day/week periods)

    Example:
        >>> expand_period_to_months('Q', 2)
        6
        >>> expand_period_to_months('Y', 1)
        12
    """
    period_to_months = {
        "M": 1,
        "Q": 3,
        "H": 6,
        "Y": 12,
    }

    if period in ("D", "W"):
        return None  # These are handled with timedelta

    return period_to_months.get(period, 0) * multiplier
