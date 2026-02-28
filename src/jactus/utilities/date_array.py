"""JAX-native integer date representation for vectorized date arithmetic.

This module provides :class:`DateArray`, a date type that stores dates as
parallel int32 arrays (ordinals, years, months, days).  All arithmetic
uses pure integer operations and is fully vectorised with JAX, enabling
schedule generation and year-fraction computation without Python loops.

``DateArray`` is an **internal** optimisation used by the array-mode
simulation path (``pam_array.py``).  It does **not** replace
:class:`~jactus.core.time.ActusDateTime` in the public API.

Ordinal convention
------------------
``ordinal = 1`` corresponds to January 1, year 1 (proleptic Gregorian),
matching Python's ``datetime.date.toordinal()``.

The conversion algorithm is based on Howard Hinnant's ``date.h`` which
shifts the epoch to March 1, year 0 so that February (the variable-length
month) falls at the *end* of the "year".
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jactus.core.time import ActusDateTime

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

_DAYS_IN_MONTH_TABLE = jnp.array(
    [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=jnp.int32
)

# ---------------------------------------------------------------------------
# Low-level vectorised helpers
# ---------------------------------------------------------------------------


def _is_leap_year(y: jnp.ndarray) -> jnp.ndarray:
    """Vectorised leap-year test.  Returns a bool array."""
    return ((y % 4 == 0) & (y % 100 != 0)) | (y % 400 == 0)


def _days_in_month(y: jnp.ndarray, m: jnp.ndarray) -> jnp.ndarray:
    """Vectorised days-in-month.  Returns an int32 array."""
    base = _DAYS_IN_MONTH_TABLE[m]
    feb_adj = jnp.where((m == 2) & _is_leap_year(y), 1, 0).astype(jnp.int32)
    return base + feb_adj


# ---------------------------------------------------------------------------
# Ordinal <-> Y/M/D  (Hinnant / civil_from_days algorithm)
# ---------------------------------------------------------------------------


def _ymd_to_ordinal(y: jnp.ndarray, m: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """Convert ``(year, month, day)`` int32 arrays to ordinals.

    Ordinal 1 = January 1, year 1 (matches ``datetime.date.toordinal``).
    """
    # Shift to March-based year so Feb is month 11 (end of "year")
    a = jnp.where(m <= 2, 1, 0).astype(jnp.int32)
    y_adj = y - a
    m_adj = m + 12 * a - 3  # Mar=0 … Feb=11

    # Day-of-year from March 1 (exact integer formula for irregular months)
    doy = (153 * m_adj + 2) // 5 + d - 1

    # Decompose into 400-year eras
    era = jnp.where(y_adj >= 0, y_adj // 400, (y_adj - 399) // 400)
    yoe = y_adj - era * 400  # year-of-era [0, 399]

    # Day-of-era
    doe = 365 * yoe + yoe // 4 - yoe // 100 + doy

    # Total ordinal  (era * 146097 days in 400 years)
    # Epoch adjustment: Mar 1, year 0 is doe=0 in era 0.
    # Jan 1, year 1 must map to ordinal 1 (matching Python datetime).
    # Mar 1 yr 0 → doy=0, doe=0 → ordinal = 0 - 305 = -305
    # Jan 1 yr 1 → doy=306, doe=306 → ordinal = 306 - 305 = 1  ✓
    ordinal = era * 146097 + doe - 305

    return ordinal.astype(jnp.int32)


def _ordinal_to_ymd(
    ordinal: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert ordinals to ``(year, month, day)`` int32 arrays."""
    # Shift to March-based epoch (inverse of -305 in _ymd_to_ordinal)
    z = ordinal + 305

    # 400-year eras
    era = jnp.where(z >= 0, z // 146097, (z - 146096) // 146097)
    doe = z - era * 146097  # day-of-era [0, 146096]

    # Year-of-era from day-of-era
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365

    # Day-of-year (March-based)
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)

    # March-based month [0, 11]
    mp = (5 * doy + 2) // 153

    # Day [1, 31]
    d = doy - (153 * mp + 2) // 5 + 1

    # Convert back to January-based month [1, 12]
    m = mp + jnp.where(mp < 10, 3, -9)

    # Adjust year for Jan/Feb
    y = era * 400 + yoe + jnp.where(m <= 2, 1, 0)

    return y.astype(jnp.int32), m.astype(jnp.int32), d.astype(jnp.int32)


# ---------------------------------------------------------------------------
# DateArray
# ---------------------------------------------------------------------------


class DateArray:
    """JAX-native date representation using parallel int32 arrays.

    Attributes
    ----------
    ordinals : jnp.ndarray (int32)
        Days since January 1, year 1 (proleptic Gregorian).
    years, months, days : jnp.ndarray (int32)
        Calendar components.
    """

    __slots__ = ("ordinals", "years", "months", "days")

    def __init__(
        self,
        ordinals: jnp.ndarray,
        years: jnp.ndarray,
        months: jnp.ndarray,
        days: jnp.ndarray,
    ) -> None:
        self.ordinals = ordinals
        self.years = years
        self.months = months
        self.days = days

    # -- Construction -------------------------------------------------------

    @staticmethod
    def from_ymd(
        years: jnp.ndarray,
        months: jnp.ndarray,
        days: jnp.ndarray,
    ) -> DateArray:
        """Create from year/month/day arrays, computing ordinals."""
        ordinals = _ymd_to_ordinal(years, months, days)
        return DateArray(ordinals, years, months, days)

    @staticmethod
    def from_ordinals(ordinals: jnp.ndarray) -> DateArray:
        """Create from ordinal array, computing year/month/day."""
        y, m, d = _ordinal_to_ymd(ordinals)
        return DateArray(ordinals, y, m, d)

    @staticmethod
    def from_single(year: int, month: int, day: int) -> DateArray:
        """Create a scalar DateArray from a single date."""
        y = jnp.array(year, dtype=jnp.int32)
        m = jnp.array(month, dtype=jnp.int32)
        d = jnp.array(day, dtype=jnp.int32)
        return DateArray.from_ymd(y, m, d)

    @staticmethod
    def from_actus_datetime(dt: ActusDateTime) -> DateArray:
        """Convert an ``ActusDateTime`` to a scalar ``DateArray``.

        Handles ``hour=24`` (end-of-day) by advancing to the next day.
        """
        y, m, d = dt.year, dt.month, dt.day
        if dt.hour == 24:
            # Advance by 1 day via ordinal arithmetic
            o = _ymd_to_ordinal(
                jnp.array(y, jnp.int32),
                jnp.array(m, jnp.int32),
                jnp.array(d, jnp.int32),
            )
            return DateArray.from_ordinals(o + 1)
        return DateArray.from_single(y, m, d)

    @staticmethod
    def from_actus_datetimes(dts: list[ActusDateTime]) -> DateArray:
        """Convert a list of ``ActusDateTime`` to a batched ``DateArray``."""
        n = len(dts)
        yy = np.empty(n, dtype=np.int32)
        mm = np.empty(n, dtype=np.int32)
        dd = np.empty(n, dtype=np.int32)
        has_h24 = np.zeros(n, dtype=np.bool_)
        for i, dt in enumerate(dts):
            yy[i] = dt.year
            mm[i] = dt.month
            dd[i] = dt.day
            if dt.hour == 24:
                has_h24[i] = True

        years = jnp.asarray(yy)
        months = jnp.asarray(mm)
        days = jnp.asarray(dd)
        ordinals = _ymd_to_ordinal(years, months, days)

        if has_h24.any():
            adj = jnp.asarray(has_h24.astype(np.int32))
            ordinals = ordinals + adj
            ny, nm, nd = _ordinal_to_ymd(ordinals)
            # Merge: only update the h24 entries
            mask = jnp.asarray(has_h24)
            years = jnp.where(mask, ny, years)
            months = jnp.where(mask, nm, months)
            days = jnp.where(mask, nd, days)

        return DateArray(ordinals, years, months, days)

    # -- Properties ---------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.ordinals.shape

    def __len__(self) -> int:
        return self.ordinals.shape[0] if self.ordinals.ndim > 0 else 1

    # -- Arithmetic ---------------------------------------------------------

    def add_days(self, n: jnp.ndarray | int) -> DateArray:
        """Add *n* days.  Pure ordinal arithmetic."""
        new_ord = self.ordinals + jnp.asarray(n, dtype=jnp.int32)
        return DateArray.from_ordinals(new_ord)

    def add_weeks(self, n: jnp.ndarray | int) -> DateArray:
        """Add *n* weeks."""
        return self.add_days(jnp.asarray(n, dtype=jnp.int32) * 7)

    def add_months(self, n: jnp.ndarray | int) -> DateArray:
        """Add *n* months with day-clamping for short months.

        Always computes from the original year/month (not chained),
        preventing day-clamping drift.
        """
        n = jnp.asarray(n, dtype=jnp.int32)
        total = self.years * 12 + self.months - 1 + n
        new_y = total // 12
        new_m = (total % 12) + 1
        max_d = _days_in_month(new_y, new_m)
        new_d = jnp.minimum(self.days, max_d)
        return DateArray.from_ymd(new_y, new_m, new_d)

    def add_months_eom(self, n: jnp.ndarray | int) -> DateArray:
        """Add *n* months with end-of-month convention.

        If the source date is the last day of its month, the result
        is the last day of the target month.
        """
        n = jnp.asarray(n, dtype=jnp.int32)
        total = self.years * 12 + self.months - 1 + n
        new_y = total // 12
        new_m = (total % 12) + 1
        max_d = _days_in_month(new_y, new_m)
        is_eom = self.is_end_of_month()
        new_d = jnp.where(is_eom, max_d, jnp.minimum(self.days, max_d))
        return DateArray.from_ymd(new_y, new_m, new_d)

    # -- Comparison (ordinal-based) -----------------------------------------

    def __lt__(self, other: DateArray) -> jnp.ndarray:
        return self.ordinals < other.ordinals

    def __le__(self, other: DateArray) -> jnp.ndarray:
        return self.ordinals <= other.ordinals

    def __gt__(self, other: DateArray) -> jnp.ndarray:
        return self.ordinals > other.ordinals

    def __ge__(self, other: DateArray) -> jnp.ndarray:
        return self.ordinals >= other.ordinals

    def __eq__(self, other: object) -> jnp.ndarray:
        if not isinstance(other, DateArray):
            return NotImplemented
        return self.ordinals == other.ordinals

    # -- Queries ------------------------------------------------------------

    def is_end_of_month(self) -> jnp.ndarray:
        """Bool array: ``True`` where day equals days-in-month."""
        return self.days == _days_in_month(self.years, self.months)

    def is_leap_year(self) -> jnp.ndarray:
        """Bool array: ``True`` for leap years."""
        return _is_leap_year(self.years)

    def get_days_in_month(self) -> jnp.ndarray:
        """Int32 array of days in each date's month."""
        return _days_in_month(self.years, self.months)

    # -- Slicing ------------------------------------------------------------

    def __getitem__(self, idx) -> DateArray:
        return DateArray(
            self.ordinals[idx],
            self.years[idx],
            self.months[idx],
            self.days[idx],
        )

    # -- Representation -----------------------------------------------------

    def __repr__(self) -> str:
        if self.ordinals.ndim == 0:
            return f"DateArray({int(self.years)}-{int(self.months):02d}-{int(self.days):02d})"
        n = self.ordinals.shape[0]
        if n <= 4:
            dates = ", ".join(
                f"{int(self.years[i])}-{int(self.months[i]):02d}-{int(self.days[i]):02d}"
                for i in range(n)
            )
        else:
            first = f"{int(self.years[0])}-{int(self.months[0]):02d}-{int(self.days[0]):02d}"
            last = f"{int(self.years[-1])}-{int(self.months[-1]):02d}-{int(self.days[-1]):02d}"
            dates = f"{first}, ..., {last}"
        return f"DateArray([{dates}], n={n})"


# ---------------------------------------------------------------------------
# Vectorised year-fraction functions
# ---------------------------------------------------------------------------


def year_fraction_a360(start: DateArray, end: DateArray) -> jnp.ndarray:
    """Actual/360 year fraction (vectorised)."""
    delta = (end.ordinals - start.ordinals).astype(jnp.float32)
    return delta / 360.0


def year_fraction_a365(start: DateArray, end: DateArray) -> jnp.ndarray:
    """Actual/365 year fraction (vectorised)."""
    delta = (end.ordinals - start.ordinals).astype(jnp.float32)
    return delta / 365.0


def year_fraction_30e360(start: DateArray, end: DateArray) -> jnp.ndarray:
    """30E/360 (Eurobond basis) year fraction (vectorised).

    Adjustments: D1=31→30, D2=31→30.
    """
    d1 = jnp.where(start.days == 31, 30, start.days)
    d2 = jnp.where(end.days == 31, 30, end.days)
    days_360 = (end.years - start.years) * 360 + (end.months - start.months) * 30 + (d2 - d1)
    return days_360.astype(jnp.float32) / 360.0


def year_fraction_b30360(start: DateArray, end: DateArray) -> jnp.ndarray:
    """30/360 US (Bond Basis) year fraction (vectorised).

    Adjustments: D1=31→30; if D1>=30 and D2=31 then D2=30.
    """
    d1 = jnp.where(start.days == 31, 30, start.days)
    d2 = jnp.where((d1 >= 30) & (end.days == 31), 30, end.days)
    days_360 = (end.years - start.years) * 360 + (end.months - start.months) * 30 + (d2 - d1)
    return days_360.astype(jnp.float32) / 360.0


def year_fraction_30e360_isda(start: DateArray, end: DateArray, maturity: DateArray) -> jnp.ndarray:
    """30E/360 ISDA year fraction (vectorised).

    Adjustments: last-day-of-Feb or D=31 → 30.
    For end dates, the Feb adjustment only applies when end != maturity.
    """
    start_dim = _days_in_month(start.years, start.months)
    end_dim = _days_in_month(end.years, end.months)

    start_is_feb_end = (start.months == 2) & (start.days == start_dim)
    end_is_feb_end = (end.months == 2) & (end.days == end_dim)
    is_maturity = end.ordinals == maturity.ordinals

    d1 = jnp.where(start_is_feb_end | (start.days == 31), 30, start.days)
    d2 = jnp.where((end_is_feb_end & ~is_maturity) | (end.days == 31), 30, end.days)

    days_360 = (end.years - start.years) * 360 + (end.months - start.months) * 30 + (d2 - d1)
    return days_360.astype(jnp.float32) / 360.0


# ---------------------------------------------------------------------------
# Vectorised schedule generation
# ---------------------------------------------------------------------------


def generate_month_schedule(
    start: DateArray,
    cycle_months: int,
    end: DateArray,
) -> DateArray:
    """Generate a month-based schedule (vectorised, no Python loop).

    Parameters
    ----------
    start : DateArray (scalar)
        Schedule anchor date.
    cycle_months : int
        Number of months per period (e.g. 1, 3, 6, 12).
    end : DateArray (scalar)
        Schedule end boundary (inclusive).

    Returns
    -------
    DateArray
        Dates from *start* up to and including *end*.
    """
    days_span = int(end.ordinals - start.ordinals)
    # Conservative upper bound: min 28 days/month
    n_max = days_span // (cycle_months * 28) + 2

    offsets = jnp.arange(0, n_max, dtype=jnp.int32) * cycle_months
    base = start.years * 12 + start.months - 1
    target = base + offsets

    new_y = target // 12
    new_m = (target % 12) + 1
    max_d = _days_in_month(new_y, new_m)
    new_d = jnp.minimum(start.days, max_d)

    result = DateArray.from_ymd(new_y, new_m, new_d)

    valid = result.ordinals <= end.ordinals
    n_valid = int(jnp.sum(valid))
    return result[:n_valid]


def generate_month_schedule_eom(
    start: DateArray,
    cycle_months: int,
    end: DateArray,
) -> DateArray:
    """Month schedule with end-of-month convention.

    If *start* is the last day of its month, generated dates are
    also the last day of their respective months.
    """
    days_span = int(end.ordinals - start.ordinals)
    n_max = days_span // (cycle_months * 28) + 2

    offsets = jnp.arange(0, n_max, dtype=jnp.int32) * cycle_months
    base = start.years * 12 + start.months - 1
    target = base + offsets

    new_y = target // 12
    new_m = (target % 12) + 1
    max_d = _days_in_month(new_y, new_m)

    is_eom = start.is_end_of_month()
    new_d = jnp.where(is_eom, max_d, jnp.minimum(start.days, max_d))

    result = DateArray.from_ymd(new_y, new_m, new_d)

    valid = result.ordinals <= end.ordinals
    n_valid = int(jnp.sum(valid))
    return result[:n_valid]


def generate_day_schedule(
    start: DateArray,
    day_step: int,
    end: DateArray,
) -> DateArray:
    """Generate a day/week schedule (vectorised, no Python loop).

    Parameters
    ----------
    start : DateArray (scalar)
        Schedule anchor.
    day_step : int
        Number of days per step (1 for daily, 7 for weekly).
    end : DateArray (scalar)
        End boundary (inclusive).
    """
    days_span = int(end.ordinals - start.ordinals)
    n_max = days_span // day_step + 2

    offsets = jnp.arange(0, n_max, dtype=jnp.int32) * day_step
    new_ord = start.ordinals + offsets

    valid = new_ord <= end.ordinals
    n_valid = int(jnp.sum(valid))
    return DateArray.from_ordinals(new_ord[:n_valid])
