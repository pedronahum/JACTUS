"""Date and time handling for ACTUS contracts.

This module provides the ActusDateTime class and utilities for parsing,
manipulating, and comparing dates according to ACTUS specifications.

Key features:
- ISO 8601 datetime parsing with ACTUS-specific extensions
- Support for 24:00:00 (end of day) and 00:00:00 (start of day)
- Period/cycle arithmetic (e.g., adding '3M' to a date)
- Month-end handling and leap year support
- JAX pytree registration for functional programming
- Business day awareness

References:
    ACTUS Technical Specification v1.1, Section 3 (Time)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import jax

from jactus.core.types import BusinessDayConvention, Calendar, Cycle, EndOfMonthConvention


@dataclass(frozen=True)
class ActusDateTime:
    """Immutable datetime representation for ACTUS contracts.

    ACTUS uses ISO 8601 datetime strings with special handling:
    - 24:00:00 represents end of day (midnight of next day)
    - 00:00:00 represents start of day (midnight)
    - Dates can be added/subtracted using cycle notation (e.g., '3M', '1Y')

    Attributes:
        year: Year (1-9999)
        month: Month (1-12)
        day: Day of month (1-31)
        hour: Hour (0-24, where 24 = end of day)
        minute: Minute (0-59)
        second: Second (0-59)

    Example:
        >>> dt = ActusDateTime.from_iso("2024-01-15T00:00:00")
        >>> dt.add_period("3M")
        ActusDateTime(2024, 4, 15, 0, 0, 0)

    References:
        ACTUS Technical Specification v1.1, Section 3.1
    """

    year: int
    month: int
    day: int
    hour: int = 0
    minute: int = 0
    second: int = 0

    def __post_init__(self) -> None:
        """Validate datetime components."""
        if not 1 <= self.year <= 9999:
            raise ValueError(f"Year must be 1-9999, got {self.year}")
        if not 1 <= self.month <= 12:
            raise ValueError(f"Month must be 1-12, got {self.month}")
        if not 1 <= self.day <= 31:
            raise ValueError(f"Day must be 1-31, got {self.day}")
        if not 0 <= self.hour <= 24:
            raise ValueError(f"Hour must be 0-24, got {self.hour}")
        if self.hour == 24 and (self.minute != 0 or self.second != 0):
            raise ValueError("24:00:00 is the only valid time with hour=24")
        if not 0 <= self.minute <= 59:
            raise ValueError(f"Minute must be 0-59, got {self.minute}")
        if not 0 <= self.second <= 59:
            raise ValueError(f"Second must be 0-59, got {self.second}")

    @classmethod
    def from_iso(cls, iso_string: str) -> ActusDateTime:
        """Parse ISO 8601 datetime string.

        Supports formats:
        - YYYY-MM-DD
        - YYYY-MM-DDTHH:MM:SS
        - YYYY-MM-DD HH:MM:SS (space separator)

        Special handling for 24:00:00 (end of day).

        Args:
            iso_string: ISO 8601 formatted datetime string

        Returns:
            ActusDateTime instance

        Raises:
            ValueError: If string format is invalid

        Example:
            >>> ActusDateTime.from_iso("2024-01-15T24:00:00")
            ActusDateTime(2024, 1, 15, 24, 0, 0)
        """
        return parse_iso_datetime(iso_string)

    def to_iso(self) -> str:
        """Convert to ISO 8601 string.

        Returns:
            ISO 8601 formatted string (YYYY-MM-DDTHH:MM:SS)

        Example:
            >>> dt = ActusDateTime(2024, 1, 15, 12, 30, 0)
            >>> dt.to_iso()
            '2024-01-15T12:30:00'
        """
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}T{self.hour:02d}:{self.minute:02d}:{self.second:02d}"

    def to_datetime(self) -> datetime:
        """Convert to Python datetime object.

        Note: 24:00:00 is converted to 00:00:00 of the next day.

        Returns:
            Python datetime object

        Example:
            >>> dt = ActusDateTime(2024, 1, 15, 24, 0, 0)
            >>> dt.to_datetime()
            datetime(2024, 1, 16, 0, 0, 0)
        """
        if self.hour == 24:
            # 24:00:00 means midnight of next day
            dt = datetime(self.year, self.month, self.day, 0, 0, 0)
            return dt + timedelta(days=1)
        return datetime(self.year, self.month, self.day, self.hour, self.minute, self.second)

    def add_period(
        self,
        cycle: Cycle,
        end_of_month_convention: EndOfMonthConvention = EndOfMonthConvention.SD,
    ) -> ActusDateTime:
        """Add a period to this datetime.

        Periods are specified in ACTUS cycle notation:
        - NPS where N=number, P=period type, S=stub indicator
        - Period types: D=days, W=weeks, M=months, Q=quarters, H=half-years, Y=years
        - Stub indicator: '-' for short stub, '+' for long stub (optional)

        Args:
            cycle: Period to add (e.g., '3M', '1Y', '2W')
            end_of_month_convention: How to handle month-end dates

        Returns:
            New ActusDateTime after adding period

        Example:
            >>> dt = ActusDateTime(2024, 1, 31, 0, 0, 0)
            >>> dt.add_period("1M", EndOfMonthConvention.EOM)
            ActusDateTime(2024, 2, 29, 0, 0, 0)  # Leap year

        References:
            ACTUS Technical Specification v1.1, Section 3.2
        """
        return add_period(self, cycle, end_of_month_convention)

    def is_end_of_month(self) -> bool:
        """Check if this date is the last day of the month.

        Returns:
            True if this is the last day of the month

        Example:
            >>> ActusDateTime(2024, 2, 29, 0, 0, 0).is_end_of_month()
            True
            >>> ActusDateTime(2024, 2, 28, 0, 0, 0).is_end_of_month()
            False
        """
        # Check if adding one day changes the month
        dt = self.to_datetime()
        next_day = dt + timedelta(days=1)
        return next_day.month != dt.month

    def days_between(self, other: ActusDateTime) -> int:
        """Calculate actual days between this date and another.

        Args:
            other: Other datetime

        Returns:
            Number of days (can be negative if other is earlier)

        Example:
            >>> dt1 = ActusDateTime(2024, 1, 15, 0, 0, 0)
            >>> dt2 = ActusDateTime(2024, 1, 18, 0, 0, 0)
            >>> dt1.days_between(dt2)
            3
        """
        dt1 = self.to_datetime()
        dt2 = other.to_datetime()
        return (dt2 - dt1).days

    def years_between(self, other: ActusDateTime) -> float:
        """Calculate approximate years between dates (actual days / 365.25).

        Args:
            other: Other datetime

        Returns:
            Approximate years (can be negative)

        Example:
            >>> dt1 = ActusDateTime(2024, 1, 15, 0, 0, 0)
            >>> dt2 = ActusDateTime(2025, 1, 15, 0, 0, 0)
            >>> abs(dt1.years_between(dt2) - 1.0) < 0.01
            True
        """
        days = self.days_between(other)
        return days / 365.25

    def __eq__(self, other: object) -> bool:
        """Check equality with another ActusDateTime."""
        if not isinstance(other, ActusDateTime):
            return NotImplemented
        return (
            self.year == other.year
            and self.month == other.month
            and self.day == other.day
            and self.hour == other.hour
            and self.minute == other.minute
            and self.second == other.second
        )

    def __lt__(self, other: ActusDateTime) -> bool:
        """Check if this datetime is before another."""
        return self.to_datetime() < other.to_datetime()

    def __le__(self, other: ActusDateTime) -> bool:
        """Check if this datetime is before or equal to another."""
        return self.to_datetime() <= other.to_datetime()

    def __gt__(self, other: ActusDateTime) -> bool:
        """Check if this datetime is after another."""
        return self.to_datetime() > other.to_datetime()

    def __ge__(self, other: ActusDateTime) -> bool:
        """Check if this datetime is after or equal to another."""
        return self.to_datetime() >= other.to_datetime()

    def __hash__(self) -> int:
        """Hash for use in dicts/sets."""
        return hash((self.year, self.month, self.day, self.hour, self.minute, self.second))


# Register ActusDateTime as a JAX pytree for functional programming
def _actus_datetime_flatten(dt: ActusDateTime) -> tuple[tuple[int, ...], None]:
    """Flatten ActusDateTime for JAX pytree registration."""
    return ((dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second), None)


def _actus_datetime_unflatten(aux_data: None, children: tuple[int, ...]) -> ActusDateTime:
    """Unflatten ActusDateTime for JAX pytree registration."""
    return ActusDateTime(*children)


jax.tree_util.register_pytree_node(
    ActusDateTime,
    _actus_datetime_flatten,
    _actus_datetime_unflatten,
)


def parse_iso_datetime(iso_string: str) -> ActusDateTime:
    """Parse ISO 8601 datetime string into ActusDateTime.

    Supports formats:
    - YYYY-MM-DD
    - YYYY-MM-DDTHH:MM:SS
    - YYYY-MM-DD HH:MM:SS

    Args:
        iso_string: ISO 8601 formatted string

    Returns:
        ActusDateTime instance

    Raises:
        ValueError: If format is invalid

    Example:
        >>> parse_iso_datetime("2024-01-15T12:30:00")
        ActusDateTime(2024, 1, 15, 12, 30, 0)
    """
    # Try full datetime with T separator
    match = re.match(
        r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})$",
        iso_string,
    )
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return ActusDateTime(year, month, day, hour, minute, second)

    # Try full datetime with space separator
    match = re.match(
        r"^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})$",
        iso_string,
    )
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return ActusDateTime(year, month, day, hour, minute, second)

    # Try date only
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", iso_string)
    if match:
        year, month, day = map(int, match.groups())
        return ActusDateTime(year, month, day, 0, 0, 0)

    raise ValueError(
        f"Invalid ISO 8601 datetime format: {iso_string}. "
        f"Expected YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"
    )


def parse_cycle(cycle: Cycle) -> tuple[int, str, str]:
    """Parse ACTUS cycle notation.

    Cycle format: NPS
    - N: number (integer)
    - P: period type (D/W/M/Q/H/Y)
    - S: stub indicator ('-' short, '+' long) - optional

    Args:
        cycle: Cycle string (e.g., '3M', '1Y-', '6M+')

    Returns:
        Tuple of (number, period_type, stub_indicator)

    Raises:
        ValueError: If cycle format is invalid

    Example:
        >>> parse_cycle("3M")
        (3, 'M', '')
        >>> parse_cycle("1Y-")
        (1, 'Y', '-')

    References:
        ACTUS Technical Specification v1.1, Section 3.2
    """
    match = re.match(r"^(\d+)([DWMQHY])([-+]?)$", cycle.upper())
    if not match:
        raise ValueError(
            f"Invalid cycle format: {cycle}. "
            f"Expected format: NPS where N=number, P=D/W/M/Q/H/Y, S='-'/'+' (optional)"
        )

    number_str, period_type, stub = match.groups()
    return int(number_str), period_type, stub


def add_period(
    dt: ActusDateTime,
    cycle: Cycle,
    end_of_month_convention: EndOfMonthConvention = EndOfMonthConvention.SD,
) -> ActusDateTime:
    """Add a period to a datetime according to ACTUS conventions.

    Args:
        dt: Starting datetime
        cycle: Period to add (e.g., '3M', '1Y')
        end_of_month_convention: How to handle month-end dates

    Returns:
        New datetime after adding period

    Example:
        >>> dt = ActusDateTime(2024, 1, 31, 0, 0, 0)
        >>> add_period(dt, "1M", EndOfMonthConvention.EOM)
        ActusDateTime(2024, 2, 29, 0, 0, 0)

    References:
        ACTUS Technical Specification v1.1, Section 3.2, 3.3
    """
    number, period_type, _ = parse_cycle(cycle)

    # Remember if we started at end of month
    started_at_eom = dt.is_end_of_month()

    # Convert to Python datetime for arithmetic
    py_dt = dt.to_datetime()

    # Add the period
    if period_type == "D":
        new_dt = py_dt + timedelta(days=number)
    elif period_type == "W":
        new_dt = py_dt + timedelta(weeks=number)
    elif period_type in ("M", "Q", "H", "Y"):
        # Convert to months
        months_map = {"M": 1, "Q": 3, "H": 6, "Y": 12}
        months_to_add = number * months_map[period_type]

        # Calculate new year and month
        total_months = (py_dt.year * 12 + py_dt.month - 1) + months_to_add
        new_year = total_months // 12
        new_month = (total_months % 12) + 1

        # Handle day overflow (e.g., Jan 31 + 1M = Feb 28/29)
        # First try same day
        try:
            new_dt = datetime(
                new_year, new_month, py_dt.day, py_dt.hour, py_dt.minute, py_dt.second
            )
        except ValueError:
            # Day doesn't exist in target month, use last day of month
            # Find last day by trying from 31 down
            for day in range(31, 27, -1):
                try:
                    new_dt = datetime(
                        new_year, new_month, day, py_dt.hour, py_dt.minute, py_dt.second
                    )
                    break
                except ValueError:
                    continue
    else:
        raise ValueError(f"Unsupported period type: {period_type}")

    # Create new ActusDateTime
    result = ActusDateTime(
        new_dt.year,
        new_dt.month,
        new_dt.day,
        new_dt.hour,
        new_dt.minute,
        new_dt.second,
    )

    # Apply end-of-month convention (only for month-based periods)
    if (
        end_of_month_convention == EndOfMonthConvention.EOM
        and started_at_eom
        and period_type in ("M", "Q", "H", "Y")
    ):
        # Move to end of target month
        result_dt = result.to_datetime()
        # Add days until we're at the last day of the month
        while True:
            next_day = result_dt + timedelta(days=1)
            if next_day.month != result_dt.month:
                # We're at the last day
                break
            result_dt = next_day

        result = ActusDateTime(
            result_dt.year,
            result_dt.month,
            result_dt.day,
            dt.hour,
            dt.minute,
            dt.second,
        )

    return result


def is_business_day(
    dt: ActusDateTime,
    calendar: Calendar = Calendar.MONDAY_TO_FRIDAY,
) -> bool:
    """Check if a date is a business day according to the given calendar.

    Args:
        dt: Date to check
        calendar: Business day calendar to use

    Returns:
        True if date is a business day

    Example:
        >>> dt = ActusDateTime(2024, 1, 15, 0, 0, 0)  # Monday
        >>> is_business_day(dt)
        True

    References:
        ACTUS Technical Specification v1.1, Section 3.4
    """
    py_dt = dt.to_datetime()

    if calendar == Calendar.NO_CALENDAR:
        return True

    # Check if weekend
    if py_dt.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    # For now, we only implement NO_CALENDAR and MONDAY_TO_FRIDAY
    # Other calendars (TARGET, NYSE, etc.) would need holiday data
    if calendar in (Calendar.MONDAY_TO_FRIDAY,):
        return True

    # For calendars we don't support yet, be conservative
    return True


def adjust_to_business_day(
    dt: ActusDateTime,
    convention: BusinessDayConvention,
    calendar: Calendar = Calendar.MONDAY_TO_FRIDAY,
) -> ActusDateTime:
    """Adjust a date to a business day according to the given convention.

    Args:
        dt: Date to adjust
        convention: Business day convention to use
        calendar: Business day calendar

    Returns:
        Adjusted date (may be same as input if already a business day)

    Example:
        >>> dt = ActusDateTime(2024, 1, 13, 0, 0, 0)  # Saturday
        >>> adjust_to_business_day(dt, BusinessDayConvention.SCF)
        ActusDateTime(2024, 1, 15, 0, 0, 0)  # Monday

    References:
        ACTUS Technical Specification v1.1, Section 3.4
    """
    if convention == BusinessDayConvention.NULL:
        return dt

    if is_business_day(dt, calendar):
        return dt

    # Save original date for Modified conventions
    original_dt = dt
    py_dt = dt.to_datetime()
    original_month = py_dt.month

    if "F" in convention.value:  # Following
        # Move forward to next business day
        while not is_business_day(dt, calendar):
            py_dt = dt.to_datetime() + timedelta(days=1)
            dt = ActusDateTime(py_dt.year, py_dt.month, py_dt.day, dt.hour, dt.minute, dt.second)

        # Check if Modified: if we crossed a month boundary, go backward from original instead
        if "M" in convention.value and dt.month != original_month:
            dt = original_dt
            py_dt = original_dt.to_datetime()
            while not is_business_day(dt, calendar):
                py_dt = py_dt - timedelta(days=1)
                dt = ActusDateTime(
                    py_dt.year, py_dt.month, py_dt.day, dt.hour, dt.minute, dt.second
                )

    elif "P" in convention.value:  # Preceding
        # Move backward to previous business day
        while not is_business_day(dt, calendar):
            py_dt = dt.to_datetime() - timedelta(days=1)
            dt = ActusDateTime(py_dt.year, py_dt.month, py_dt.day, dt.hour, dt.minute, dt.second)

        # Check if Modified: if we crossed a month boundary, go forward from original instead
        if "M" in convention.value and dt.month != original_month:
            dt = original_dt
            py_dt = original_dt.to_datetime()
            while not is_business_day(dt, calendar):
                py_dt = py_dt + timedelta(days=1)
                dt = ActusDateTime(
                    py_dt.year, py_dt.month, py_dt.day, dt.hour, dt.minute, dt.second
                )

    return dt
