"""Business day calendar implementations for ACTUS contracts.

This module provides holiday calendar functionality for determining business days
and adjusting dates according to business day conventions.

References:
    ACTUS Technical Specification v1.1, Section 3.4 (Business Day Conventions)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import timedelta

from jactus.core.time import ActusDateTime


class HolidayCalendar(ABC):
    """Abstract base class for holiday calendars.

    A holiday calendar determines which dates are business days and provides
    navigation functions for working with business days.
    """

    @abstractmethod
    def is_business_day(self, date: ActusDateTime) -> bool:
        """Check if a date is a business day.

        Args:
            date: Date to check

        Returns:
            True if the date is a business day

        Example:
            >>> cal = MondayToFridayCalendar()
            >>> cal.is_business_day(ActusDateTime(2024, 1, 15, 0, 0, 0))  # Monday
            True
        """

    def is_holiday(self, date: ActusDateTime) -> bool:
        """Check if a date is a holiday (not a business day).

        Args:
            date: Date to check

        Returns:
            True if the date is a holiday
        """
        return not self.is_business_day(date)

    def next_business_day(self, date: ActusDateTime) -> ActusDateTime:
        """Get the next business day on or after the given date.

        Args:
            date: Starting date

        Returns:
            Next business day (may be the same date if already a business day)

        Example:
            >>> cal = MondayToFridayCalendar()
            >>> saturday = ActusDateTime(2024, 1, 6, 0, 0, 0)
            >>> cal.next_business_day(saturday)
            ActusDateTime(2024, 1, 8, 0, 0, 0)  # Monday
        """
        current = date
        while not self.is_business_day(current):
            py_dt = current.to_datetime() + timedelta(days=1)
            current = ActusDateTime(
                py_dt.year, py_dt.month, py_dt.day, current.hour, current.minute, current.second
            )
        return current

    def previous_business_day(self, date: ActusDateTime) -> ActusDateTime:
        """Get the previous business day on or before the given date.

        Args:
            date: Starting date

        Returns:
            Previous business day (may be the same date if already a business day)

        Example:
            >>> cal = MondayToFridayCalendar()
            >>> sunday = ActusDateTime(2024, 1, 7, 0, 0, 0)
            >>> cal.previous_business_day(sunday)
            ActusDateTime(2024, 1, 5, 0, 0, 0)  # Friday
        """
        current = date
        while not self.is_business_day(current):
            py_dt = current.to_datetime() - timedelta(days=1)
            current = ActusDateTime(
                py_dt.year, py_dt.month, py_dt.day, current.hour, current.minute, current.second
            )
        return current

    def add_business_days(self, date: ActusDateTime, days: int) -> ActusDateTime:
        """Add a number of business days to a date.

        Args:
            date: Starting date
            days: Number of business days to add (can be negative)

        Returns:
            Date after adding the specified business days

        Example:
            >>> cal = MondayToFridayCalendar()
            >>> friday = ActusDateTime(2024, 1, 5, 0, 0, 0)
            >>> cal.add_business_days(friday, 1)  # Next business day
            ActusDateTime(2024, 1, 8, 0, 0, 0)  # Monday
        """
        if days == 0:
            return date

        current = date
        direction = 1 if days > 0 else -1
        remaining = abs(days)

        while remaining > 0:
            # Move to next/previous day
            py_dt = current.to_datetime() + timedelta(days=direction)
            current = ActusDateTime(
                py_dt.year, py_dt.month, py_dt.day, current.hour, current.minute, current.second
            )

            # Count if it's a business day
            if self.is_business_day(current):
                remaining -= 1

        return current

    def business_days_between(
        self, start: ActusDateTime, end: ActusDateTime, include_end: bool = False
    ) -> int:
        """Count business days between two dates.

        Args:
            start: Start date
            end: End date
            include_end: Whether to include the end date in the count

        Returns:
            Number of business days

        Example:
            >>> cal = MondayToFridayCalendar()
            >>> mon = ActusDateTime(2024, 1, 1, 0, 0, 0)
            >>> fri = ActusDateTime(2024, 1, 5, 0, 0, 0)
            >>> cal.business_days_between(mon, fri)
            4
        """
        if start > end:
            return -self.business_days_between(end, start, include_end)

        count = 0
        current = start

        while current < end:
            if self.is_business_day(current):
                count += 1
            py_dt = current.to_datetime() + timedelta(days=1)
            current = ActusDateTime(
                py_dt.year, py_dt.month, py_dt.day, current.hour, current.minute, current.second
            )

        # Handle end date
        if include_end and self.is_business_day(end):
            count += 1

        return count


class NoHolidayCalendar(HolidayCalendar):
    """Calendar with no holidays - every day is a business day.

    Useful for theoretical calculations or contracts that don't respect weekends.
    """

    def is_business_day(self, date: ActusDateTime) -> bool:  # noqa: ARG002
        """Every day is a business day.

        Args:
            date: Date to check

        Returns:
            Always True
        """
        return True


class MondayToFridayCalendar(HolidayCalendar):
    """Calendar with Monday-Friday as business days (no holidays).

    Treats weekends (Saturday/Sunday) as non-business days but doesn't
    account for public holidays.
    """

    def is_business_day(self, date: ActusDateTime) -> bool:
        """Check if date is Monday-Friday.

        Args:
            date: Date to check

        Returns:
            True if Monday-Friday, False if Saturday-Sunday
        """
        py_dt = date.to_datetime()
        # Monday=0, Friday=4, Saturday=5, Sunday=6
        return py_dt.weekday() < 5


class CustomCalendar(HolidayCalendar):
    """Calendar with custom holiday dates.

    Allows specifying specific dates as holidays in addition to weekends.
    """

    def __init__(self, holidays: list[ActusDateTime] | None = None, include_weekends: bool = True):
        """Initialize custom calendar.

        Args:
            holidays: List of holiday dates (defaults to empty)
            include_weekends: Whether weekends are also holidays (default True)
        """
        self.holidays: set[tuple[int, int, int]] = set()
        self.include_weekends = include_weekends

        if holidays:
            for holiday in holidays:
                self.holidays.add((holiday.year, holiday.month, holiday.day))

    def add_holiday(self, date: ActusDateTime) -> None:
        """Add a holiday to the calendar.

        Args:
            date: Date to mark as holiday
        """
        self.holidays.add((date.year, date.month, date.day))

    def remove_holiday(self, date: ActusDateTime) -> None:
        """Remove a holiday from the calendar.

        Args:
            date: Date to remove from holidays
        """
        self.holidays.discard((date.year, date.month, date.day))

    def load_from_list(self, holidays: list[ActusDateTime]) -> None:
        """Load holidays from a list of dates.

        Args:
            holidays: List of holiday dates
        """
        self.holidays.clear()
        for holiday in holidays:
            self.add_holiday(holiday)

    def is_business_day(self, date: ActusDateTime) -> bool:
        """Check if date is a business day.

        A business day is not a weekend (if include_weekends=True) and not
        in the custom holidays list.

        Args:
            date: Date to check

        Returns:
            True if business day, False if weekend or holiday
        """
        # Check if it's a custom holiday
        if (date.year, date.month, date.day) in self.holidays:
            return False

        # Check weekend if enabled
        if self.include_weekends:
            py_dt = date.to_datetime()
            if py_dt.weekday() >= 5:  # Saturday or Sunday
                return False

        return True


def get_calendar(calendar_name: str) -> HolidayCalendar:
    """Factory function to get a calendar by name.

    Args:
        calendar_name: Name of calendar ("NO_CALENDAR", "MONDAY_TO_FRIDAY", etc.)

    Returns:
        HolidayCalendar instance

    Raises:
        ValueError: If calendar name is unknown

    Example:
        >>> cal = get_calendar("MONDAY_TO_FRIDAY")
        >>> cal.is_business_day(ActusDateTime(2024, 1, 6, 0, 0, 0))  # Saturday
        False
    """
    calendar_name_upper = calendar_name.upper()

    if calendar_name_upper in ("NO_CALENDAR", "NONE"):
        return NoHolidayCalendar()
    if calendar_name_upper in ("MONDAY_TO_FRIDAY", "MTF"):
        return MondayToFridayCalendar()
    raise ValueError(
        f"Unknown calendar: {calendar_name}. " f"Supported: NO_CALENDAR, MONDAY_TO_FRIDAY"
    )


def is_weekend(date: ActusDateTime) -> bool:
    """Check if a date falls on a weekend (Saturday or Sunday).

    Args:
        date: Date to check

    Returns:
        True if Saturday or Sunday

    Example:
        >>> is_weekend(ActusDateTime(2024, 1, 6, 0, 0, 0))  # Saturday
        True
        >>> is_weekend(ActusDateTime(2024, 1, 8, 0, 0, 0))  # Monday
        False
    """
    py_dt = date.to_datetime()
    return py_dt.weekday() >= 5
