"""Day count convention implementations for ACTUS contracts.

This module provides year fraction calculations according to various day count
conventions used in financial contracts.

References:
    ACTUS Technical Specification v1.1, Section 4 (Day Count Conventions)
    ISDA 2006 Definitions
"""

from __future__ import annotations

import calendar

from jactus.core.time import ActusDateTime
from jactus.core.types import DayCountConvention
from jactus.utilities.calendars import HolidayCalendar, MondayToFridayCalendar


def year_fraction(
    start: ActusDateTime,
    end: ActusDateTime,
    convention: DayCountConvention,
    maturity: ActusDateTime | None = None,
    calendar: HolidayCalendar | None = None,
) -> float:
    """Calculate year fraction between two dates using specified convention.

    Args:
        start: Start date
        end: End date
        convention: Day count convention to use
        maturity: Maturity date (required for some conventions)
        calendar: Holiday calendar for BUS/252 convention. If None, defaults
            to MondayToFridayCalendar (Mon-Fri only, no public holidays).

    Returns:
        Year fraction as a float

    Example:
        >>> start = ActusDateTime(2024, 1, 15, 0, 0, 0)
        >>> end = ActusDateTime(2024, 7, 15, 0, 0, 0)
        >>> year_fraction(start, end, DayCountConvention.AA)
        0.5

    References:
        ACTUS Technical Specification v1.1, Section 4.1
    """
    if convention == DayCountConvention.AA:
        return _year_fraction_aa(start, end)
    if convention == DayCountConvention.A360:
        return _year_fraction_a360(start, end)
    if convention == DayCountConvention.A365:
        return _year_fraction_a365(start, end)
    if convention == DayCountConvention.E30360:
        return _year_fraction_30e360(start, end)
    if convention == DayCountConvention.E30360ISDA:
        if maturity is None:
            raise ValueError("Maturity date required for 30E/360 ISDA convention")
        return _year_fraction_30e360_isda(start, end, maturity)
    if convention == DayCountConvention.B30360:
        return _year_fraction_30360(start, end)
    if convention == DayCountConvention.BUS252:
        return _year_fraction_bus252(start, end, calendar)
    raise ValueError(f"Unsupported day count convention: {convention}")


def _year_fraction_aa(start: ActusDateTime, end: ActusDateTime) -> float:
    """Actual/Actual ISDA day count convention.

    Year fraction = Sum of (days in each year / days in that year)

    References:
        ACTUS A/A, ISDA 2006 Section 4.16(b)
    """
    if start >= end:
        return 0.0

    total_fraction = 0.0
    current = start

    while current.year < end.year:
        # Find end of current year
        year_end = ActusDateTime(current.year, 12, 31, 0, 0, 0)

        # Days from current to end of year
        days_in_period = current.days_between(year_end) + 1  # Include last day
        days_in_year = 366 if calendar.isleap(current.year) else 365

        total_fraction += days_in_period / days_in_year

        # Move to next year
        current = ActusDateTime(current.year + 1, 1, 1, 0, 0, 0)

    # Handle remaining days in final year
    if current < end:
        days_in_period = current.days_between(end)
        days_in_year = 366 if calendar.isleap(end.year) else 365
        total_fraction += days_in_period / days_in_year

    return total_fraction


def _year_fraction_a360(start: ActusDateTime, end: ActusDateTime) -> float:
    """Actual/360 day count convention.

    Year fraction = actual days / 360

    References:
        ACTUS A/360
    """
    actual_days = start.days_between(end)
    return actual_days / 360.0


def _year_fraction_a365(start: ActusDateTime, end: ActusDateTime) -> float:
    """Actual/365 Fixed day count convention.

    Year fraction = actual days / 365

    References:
        ACTUS A/365
    """
    actual_days = start.days_between(end)
    return actual_days / 365.0


def _year_fraction_30e360(start: ActusDateTime, end: ActusDateTime) -> float:
    """30E/360 (Eurobond basis) day count convention.

    Days = (Y2-Y1)*360 + (M2-M1)*30 + (D2-D1)
    Year fraction = Days / 360

    Adjustments:
    - If D1 = 31, then D1 = 30
    - If D2 = 31, then D2 = 30

    References:
        ACTUS 30E/360, ISDA 2006 Section 4.16(g)
    """
    y1, m1, d1 = start.year, start.month, start.day
    y2, m2, d2 = end.year, end.month, end.day

    # Apply adjustments
    if d1 == 31:
        d1 = 30
    if d2 == 31:
        d2 = 30

    days = (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
    return days / 360.0


def _year_fraction_30e360_isda(
    start: ActusDateTime, end: ActusDateTime, maturity: ActusDateTime
) -> float:
    """30E/360 ISDA day count convention.

    Similar to 30E/360 but with different end-of-month handling.

    Adjustments:
    - If D1 is last day of February, then D1 = 30
    - If D1 = 31, then D1 = 30
    - If D2 is last day of February and not the maturity date, then D2 = 30
    - If D2 = 31, then D2 = 30

    References:
        ACTUS 30E/360 ISDA, ISDA 2006 Section 4.16(h)
    """
    y1, m1, d1 = start.year, start.month, start.day
    y2, m2, d2 = end.year, end.month, end.day

    # Check if dates are last day of February
    def is_last_day_of_feb(dt: ActusDateTime) -> bool:
        if dt.month != 2:
            return False
        last_day = 29 if calendar.isleap(dt.year) else 28
        return dt.day == last_day

    # Apply adjustments
    if is_last_day_of_feb(start) or d1 == 31:
        d1 = 30

    if (is_last_day_of_feb(end) and end != maturity) or d2 == 31:
        d2 = 30

    days = (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
    return days / 360.0


def _year_fraction_30360(start: ActusDateTime, end: ActusDateTime) -> float:
    """30/360 (Bond Basis, US) day count convention.

    Days = (Y2-Y1)*360 + (M2-M1)*30 + (D2-D1)
    Year fraction = Days / 360

    Adjustments:
    - If D1 = 31, then D1 = 30
    - If D1 = 30 or 31, and D2 = 31, then D2 = 30

    References:
        ACTUS 30/360, ISDA 2006 Section 4.16(f)
    """
    y1, m1, d1 = start.year, start.month, start.day
    y2, m2, d2 = end.year, end.month, end.day

    # Apply adjustments
    if d1 == 31:
        d1 = 30
    if d1 >= 30 and d2 == 31:
        d2 = 30

    days = (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
    return days / 360.0


def _year_fraction_bus252(
    start: ActusDateTime,
    end: ActusDateTime,
    calendar: HolidayCalendar | None = None,
) -> float:
    """BUS/252 (Brazilian business days) day count convention.

    Year fraction = business days / 252

    Uses the provided holiday calendar to determine business days. If no
    calendar is provided, defaults to MondayToFridayCalendar (Mon-Fri only,
    no public holidays).

    Args:
        start: Start date
        end: End date
        calendar: Holiday calendar for business day determination.

    References:
        ACTUS BUS/252
    """
    if calendar is None:
        calendar = MondayToFridayCalendar()
    return calendar.business_days_between(start, end) / 252.0


def days_between_30_360_methods(
    start: ActusDateTime,
    end: ActusDateTime,
    method: str = "30E/360",
) -> int:
    """Calculate days between two dates using 30/360 methods.

    Args:
        start: Start date
        end: End date
        method: One of "30E/360", "30/360", "30E/360 ISDA"

    Returns:
        Number of days (can be negative if end < start)

    Example:
        >>> start = ActusDateTime(2024, 2, 15, 0, 0, 0)
        >>> end = ActusDateTime(2024, 8, 15, 0, 0, 0)
        >>> days_between_30_360_methods(start, end, "30E/360")
        180
    """
    y1, m1, d1 = start.year, start.month, start.day
    y2, m2, d2 = end.year, end.month, end.day

    if method == "30E/360":
        if d1 == 31:
            d1 = 30
        if d2 == 31:
            d2 = 30
    elif method == "30/360":
        if d1 == 31:
            d1 = 30
        if d1 >= 30 and d2 == 31:
            d2 = 30
    elif method == "30E/360 ISDA":
        # Simplified - full implementation needs maturity
        def is_last_day_of_feb(dt: ActusDateTime) -> bool:
            if dt.month != 2:
                return False
            last_day = 29 if calendar.isleap(dt.year) else 28
            return dt.day == last_day

        if is_last_day_of_feb(start) or d1 == 31:
            d1 = 30

        if is_last_day_of_feb(end) or d2 == 31:
            d2 = 30
    else:
        raise ValueError(f"Unknown method: {method}")

    return (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
