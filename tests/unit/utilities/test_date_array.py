"""Tests for the JAX-native DateArray date representation."""

from __future__ import annotations

import datetime

import jax.numpy as jnp
import pytest

from jactus.core.time import ActusDateTime
from jactus.core.types import DayCountConvention
from jactus.utilities.conventions import year_fraction
from jactus.utilities.date_array import (
    DateArray,
    _days_in_month,
    _is_leap_year,
    _ordinal_to_ymd,
    _ymd_to_ordinal,
    generate_day_schedule,
    generate_month_schedule,
    generate_month_schedule_eom,
    year_fraction_30e360,
    year_fraction_30e360_isda,
    year_fraction_a360,
    year_fraction_a365,
    year_fraction_b30360,
)


# ===================================================================
# Ordinal conversion
# ===================================================================


class TestOrdinalConversion:
    """Round-trip and cross-validation against datetime.date.toordinal."""

    SAMPLE_DATES = [
        (1, 1, 1),
        (1, 1, 2),
        (1, 12, 31),
        (100, 1, 1),
        (400, 1, 1),
        (1582, 10, 15),  # Gregorian reform
        (1900, 1, 1),
        (1900, 2, 28),  # 1900 not a leap year
        (1900, 3, 1),
        (2000, 1, 1),
        (2000, 2, 28),
        (2000, 2, 29),  # 2000 IS a leap year
        (2000, 3, 1),
        (2024, 1, 1),
        (2024, 2, 29),  # Leap year
        (2024, 6, 15),
        (2024, 12, 31),
        (2025, 1, 1),
        (2025, 2, 28),
        (2025, 3, 31),
        (2025, 6, 30),
        (2025, 12, 31),
        (2100, 2, 28),  # 2100 not a leap year
        (2100, 3, 1),
        (9999, 12, 31),  # Max Python date
    ]

    def test_scalar_against_python(self):
        """Each scalar ordinal matches datetime.date.toordinal."""
        for y, m, d in self.SAMPLE_DATES:
            expected = datetime.date(y, m, d).toordinal()
            got = int(
                _ymd_to_ordinal(
                    jnp.array(y, jnp.int32),
                    jnp.array(m, jnp.int32),
                    jnp.array(d, jnp.int32),
                )
            )
            assert got == expected, f"Ordinal mismatch for {y}-{m:02d}-{d:02d}: {got} != {expected}"

    def test_batch_against_python(self):
        """Batch conversion matches per-element Python ordinals."""
        years = jnp.array([y for y, _, _ in self.SAMPLE_DATES], dtype=jnp.int32)
        months = jnp.array([m for _, m, _ in self.SAMPLE_DATES], dtype=jnp.int32)
        days = jnp.array([d for _, _, d in self.SAMPLE_DATES], dtype=jnp.int32)

        ordinals = _ymd_to_ordinal(years, months, days)
        for i, (y, m, d) in enumerate(self.SAMPLE_DATES):
            expected = datetime.date(y, m, d).toordinal()
            assert int(ordinals[i]) == expected, f"Index {i}: {y}-{m:02d}-{d:02d}"

    def test_round_trip_ymd(self):
        """from_ymd -> ordinals -> from_ordinals reproduces Y/M/D."""
        for y, m, d in self.SAMPLE_DATES:
            da = DateArray.from_single(y, m, d)
            rt = DateArray.from_ordinals(da.ordinals)
            assert int(rt.years) == y, f"Year mismatch for {y}-{m:02d}-{d:02d}"
            assert int(rt.months) == m, f"Month mismatch for {y}-{m:02d}-{d:02d}"
            assert int(rt.days) == d, f"Day mismatch for {y}-{m:02d}-{d:02d}"

    def test_round_trip_ordinal(self):
        """from_ordinals -> ymd -> from_ymd reproduces ordinal."""
        test_ordinals = [1, 100, 365, 366, 730, 719163, 738886, 3652059]
        for o in test_ordinals:
            da = DateArray.from_ordinals(jnp.array(o, dtype=jnp.int32))
            rt = DateArray.from_ymd(da.years, da.months, da.days)
            assert int(rt.ordinals) == o, f"Ordinal {o} round-trip failed"

    def test_wide_range(self):
        """Spot-check ordinals across several centuries."""
        import random

        rng = random.Random(42)
        for _ in range(1000):
            y = rng.randint(1, 9999)
            m = rng.randint(1, 12)
            max_d = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m]
            if m == 2 and (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)):
                max_d = 29
            d = rng.randint(1, max_d)

            expected = datetime.date(y, m, d).toordinal()
            got = int(
                _ymd_to_ordinal(
                    jnp.array(y, jnp.int32),
                    jnp.array(m, jnp.int32),
                    jnp.array(d, jnp.int32),
                )
            )
            assert got == expected, f"{y}-{m:02d}-{d:02d}: {got} != {expected}"

            # Round-trip
            ry, rm, rd = _ordinal_to_ymd(jnp.array(expected, dtype=jnp.int32))
            assert (int(ry), int(rm), int(rd)) == (y, m, d)


# ===================================================================
# Leap year / days in month
# ===================================================================


class TestLeapYearDaysInMonth:
    def test_leap_years(self):
        years = jnp.array([1900, 2000, 2024, 2100, 2400], dtype=jnp.int32)
        result = _is_leap_year(years)
        expected = [False, True, True, False, True]
        for i, (got, exp) in enumerate(zip(result, expected)):
            assert bool(got) == exp, f"Year {int(years[i])}: {got} != {exp}"

    def test_days_in_february(self):
        years = jnp.array([1900, 2000, 2024, 2025, 2100], dtype=jnp.int32)
        months = jnp.full_like(years, 2)
        result = _days_in_month(years, months)
        expected = [28, 29, 29, 28, 28]
        for i, (got, exp) in enumerate(zip(result, expected)):
            assert int(got) == exp, f"Year {int(years[i])}: {got} != {exp}"

    def test_days_in_all_months(self):
        y = jnp.full(12, 2025, dtype=jnp.int32)
        m = jnp.arange(1, 13, dtype=jnp.int32)
        result = _days_in_month(y, m)
        expected = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        for i in range(12):
            assert int(result[i]) == expected[i], f"Month {i + 1}"


# ===================================================================
# DateArray construction
# ===================================================================


class TestDateArrayConstruction:
    def test_from_single(self):
        da = DateArray.from_single(2024, 6, 15)
        assert int(da.years) == 2024
        assert int(da.months) == 6
        assert int(da.days) == 15
        expected_ord = datetime.date(2024, 6, 15).toordinal()
        assert int(da.ordinals) == expected_ord

    def test_from_actus_datetime(self):
        adt = ActusDateTime(2024, 3, 15, 0, 0, 0)
        da = DateArray.from_actus_datetime(adt)
        assert int(da.years) == 2024
        assert int(da.months) == 3
        assert int(da.days) == 15

    def test_from_actus_datetime_hour_24(self):
        """hour=24 should advance to the next day."""
        adt = ActusDateTime(2024, 1, 15, 24, 0, 0)
        da = DateArray.from_actus_datetime(adt)
        assert int(da.years) == 2024
        assert int(da.months) == 1
        assert int(da.days) == 16

    def test_from_actus_datetime_hour_24_month_end(self):
        """hour=24 on last day of month crosses into next month."""
        adt = ActusDateTime(2024, 1, 31, 24, 0, 0)
        da = DateArray.from_actus_datetime(adt)
        assert int(da.years) == 2024
        assert int(da.months) == 2
        assert int(da.days) == 1

    def test_from_actus_datetime_hour_24_year_end(self):
        """hour=24 on Dec 31 crosses into next year."""
        adt = ActusDateTime(2024, 12, 31, 24, 0, 0)
        da = DateArray.from_actus_datetime(adt)
        assert int(da.years) == 2025
        assert int(da.months) == 1
        assert int(da.days) == 1

    def test_from_actus_datetimes_batch(self):
        dts = [
            ActusDateTime(2024, 1, 1, 0, 0, 0),
            ActusDateTime(2024, 6, 15, 0, 0, 0),
            ActusDateTime(2024, 12, 31, 0, 0, 0),
        ]
        da = DateArray.from_actus_datetimes(dts)
        assert da.shape == (3,)
        assert int(da.years[0]) == 2024
        assert int(da.months[1]) == 6
        assert int(da.days[2]) == 31

    def test_from_actus_datetimes_with_hour_24(self):
        dts = [
            ActusDateTime(2024, 1, 15, 0, 0, 0),
            ActusDateTime(2024, 1, 31, 24, 0, 0),  # -> Feb 1
            ActusDateTime(2024, 6, 15, 0, 0, 0),
        ]
        da = DateArray.from_actus_datetimes(dts)
        assert int(da.years[1]) == 2024
        assert int(da.months[1]) == 2
        assert int(da.days[1]) == 1
        # Others unchanged
        assert int(da.days[0]) == 15
        assert int(da.days[2]) == 15


# ===================================================================
# DateArray arithmetic
# ===================================================================


class TestDateArrayArithmetic:
    def test_add_days(self):
        da = DateArray.from_single(2024, 1, 1)
        result = da.add_days(31)
        assert int(result.years) == 2024
        assert int(result.months) == 2
        assert int(result.days) == 1

    def test_add_days_negative(self):
        da = DateArray.from_single(2024, 3, 1)
        result = da.add_days(-1)
        assert int(result.years) == 2024
        assert int(result.months) == 2
        assert int(result.days) == 29  # Leap year

    def test_add_weeks(self):
        da = DateArray.from_single(2024, 1, 1)
        result = da.add_weeks(2)
        assert int(result.years) == 2024
        assert int(result.months) == 1
        assert int(result.days) == 15

    def test_add_months_simple(self):
        da = DateArray.from_single(2024, 1, 15)
        result = da.add_months(3)
        assert int(result.years) == 2024
        assert int(result.months) == 4
        assert int(result.days) == 15

    def test_add_months_day_clamping(self):
        """Jan 31 + 1M = Feb 29 (leap year 2024)."""
        da = DateArray.from_single(2024, 1, 31)
        result = da.add_months(1)
        assert int(result.years) == 2024
        assert int(result.months) == 2
        assert int(result.days) == 29

    def test_add_months_day_clamping_non_leap(self):
        """Jan 31 + 1M = Feb 28 (non-leap year 2025)."""
        da = DateArray.from_single(2025, 1, 31)
        result = da.add_months(1)
        assert int(result.years) == 2025
        assert int(result.months) == 2
        assert int(result.days) == 28

    def test_add_months_year_crossing(self):
        """Nov 15 + 3M = Feb 15 next year."""
        da = DateArray.from_single(2024, 11, 15)
        result = da.add_months(3)
        assert int(result.years) == 2025
        assert int(result.months) == 2
        assert int(result.days) == 15

    def test_add_months_12(self):
        """Adding 12 months = 1 year (with day clamping)."""
        da = DateArray.from_single(2024, 2, 29)
        result = da.add_months(12)
        assert int(result.years) == 2025
        assert int(result.months) == 2
        assert int(result.days) == 28  # Non-leap

    def test_add_months_no_drift(self):
        """Verify no day-clamping drift: Jan 31 + n*1M computed from anchor."""
        da = DateArray.from_single(2024, 1, 31)
        # +1M = Feb 29, +2M should be Mar 31 (not Mar 29)
        r2 = da.add_months(2)
        assert int(r2.days) == 31  # Mar 31
        r3 = da.add_months(3)
        assert int(r3.days) == 30  # Apr 30
        r4 = da.add_months(4)
        assert int(r4.days) == 31  # May 31

    def test_add_months_vectorised(self):
        """Batch addition with different month offsets."""
        da = DateArray.from_ymd(
            jnp.array([2024, 2024, 2024], dtype=jnp.int32),
            jnp.array([1, 3, 5], dtype=jnp.int32),
            jnp.array([15, 31, 30], dtype=jnp.int32),
        )
        result = da.add_months(1)
        assert int(result.months[0]) == 2
        assert int(result.days[0]) == 15
        assert int(result.months[1]) == 4
        assert int(result.days[1]) == 30  # Mar 31 + 1M = Apr 30
        assert int(result.months[2]) == 6
        assert int(result.days[2]) == 30

    def test_add_months_eom(self):
        """EOM convention: Feb 29 + 1M = Mar 31."""
        da = DateArray.from_single(2024, 2, 29)
        result = da.add_months_eom(1)
        assert int(result.years) == 2024
        assert int(result.months) == 3
        assert int(result.days) == 31

    def test_add_months_eom_not_at_eom(self):
        """EOM convention does not apply when start is not EOM."""
        da = DateArray.from_single(2024, 1, 15)
        result = da.add_months_eom(1)
        assert int(result.days) == 15  # unchanged


# ===================================================================
# DateArray comparison
# ===================================================================


class TestDateArrayComparison:
    def test_lt(self):
        a = DateArray.from_single(2024, 1, 1)
        b = DateArray.from_single(2024, 1, 2)
        assert bool(a < b)
        assert not bool(b < a)

    def test_le_equal(self):
        a = DateArray.from_single(2024, 6, 15)
        b = DateArray.from_single(2024, 6, 15)
        assert bool(a <= b)

    def test_gt(self):
        a = DateArray.from_single(2025, 1, 1)
        b = DateArray.from_single(2024, 12, 31)
        assert bool(a > b)

    def test_eq(self):
        a = DateArray.from_single(2024, 3, 15)
        b = DateArray.from_single(2024, 3, 15)
        assert bool(a == b)


# ===================================================================
# is_end_of_month
# ===================================================================


class TestIsEndOfMonth:
    def test_eom_dates(self):
        da = DateArray.from_ymd(
            jnp.array([2024, 2024, 2024, 2025, 2024], dtype=jnp.int32),
            jnp.array([1, 2, 2, 2, 6], dtype=jnp.int32),
            jnp.array([31, 29, 28, 28, 30], dtype=jnp.int32),
        )
        result = da.is_end_of_month()
        expected = [True, True, False, True, True]
        for i, (got, exp) in enumerate(zip(result, expected)):
            assert bool(got) == exp, f"Index {i}: {got} != {exp}"


# ===================================================================
# Year fractions — cross-validated against conventions.py
# ===================================================================


class TestYearFractions:
    """Cross-validate vectorised year fractions against scalar year_fraction()."""

    DATE_PAIRS = [
        # (start, end)
        ((2024, 1, 1), (2024, 7, 1)),
        ((2024, 1, 15), (2024, 4, 15)),
        ((2024, 1, 31), (2024, 2, 29)),
        ((2024, 2, 29), (2025, 2, 28)),
        ((2020, 1, 1), (2025, 1, 1)),
        ((2024, 3, 31), (2024, 6, 30)),
        ((2024, 1, 1), (2024, 1, 1)),  # Same day
        ((2024, 12, 31), (2025, 1, 1)),  # 1 day
        ((2024, 1, 30), (2024, 2, 28)),
        ((2024, 3, 31), (2024, 4, 30)),
        ((2023, 1, 31), (2023, 3, 31)),  # Non-leap
    ]

    def _make_pairs(self):
        starts, ends = [], []
        for (y1, m1, d1), (y2, m2, d2) in self.DATE_PAIRS:
            starts.append(DateArray.from_single(y1, m1, d1))
            ends.append(DateArray.from_single(y2, m2, d2))

        # Build batched DateArrays
        start_da = DateArray.from_ymd(
            jnp.array([int(s.years) for s in starts], dtype=jnp.int32),
            jnp.array([int(s.months) for s in starts], dtype=jnp.int32),
            jnp.array([int(s.days) for s in starts], dtype=jnp.int32),
        )
        end_da = DateArray.from_ymd(
            jnp.array([int(e.years) for e in ends], dtype=jnp.int32),
            jnp.array([int(e.months) for e in ends], dtype=jnp.int32),
            jnp.array([int(e.days) for e in ends], dtype=jnp.int32),
        )
        return start_da, end_da

    def _scalar_yf(self, convention, start_ymd, end_ymd, maturity_ymd=None):
        s = ActusDateTime(start_ymd[0], start_ymd[1], start_ymd[2], 0, 0, 0)
        e = ActusDateTime(end_ymd[0], end_ymd[1], end_ymd[2], 0, 0, 0)
        mat = (
            ActusDateTime(maturity_ymd[0], maturity_ymd[1], maturity_ymd[2], 0, 0, 0)
            if maturity_ymd
            else None
        )
        return year_fraction(s, e, convention, maturity=mat)

    def test_a360(self):
        start_da, end_da = self._make_pairs()
        result = year_fraction_a360(start_da, end_da)
        for i, (s, e) in enumerate(self.DATE_PAIRS):
            expected = self._scalar_yf(DayCountConvention.A360, s, e)
            assert abs(float(result[i]) - expected) < 1e-6, (
                f"A360 pair {i}: {float(result[i])} != {expected}"
            )

    def test_a365(self):
        start_da, end_da = self._make_pairs()
        result = year_fraction_a365(start_da, end_da)
        for i, (s, e) in enumerate(self.DATE_PAIRS):
            expected = self._scalar_yf(DayCountConvention.A365, s, e)
            assert abs(float(result[i]) - expected) < 1e-6, (
                f"A365 pair {i}: {float(result[i])} != {expected}"
            )

    def test_30e360(self):
        start_da, end_da = self._make_pairs()
        result = year_fraction_30e360(start_da, end_da)
        for i, (s, e) in enumerate(self.DATE_PAIRS):
            expected = self._scalar_yf(DayCountConvention.E30360, s, e)
            assert abs(float(result[i]) - expected) < 1e-6, (
                f"30E/360 pair {i}: {float(result[i])} != {expected}"
            )

    def test_b30360(self):
        start_da, end_da = self._make_pairs()
        result = year_fraction_b30360(start_da, end_da)
        for i, (s, e) in enumerate(self.DATE_PAIRS):
            expected = self._scalar_yf(DayCountConvention.B30360, s, e)
            assert abs(float(result[i]) - expected) < 1e-6, (
                f"30/360 pair {i}: {float(result[i])} != {expected}"
            )

    def test_30e360_isda(self):
        maturity_ymd = (2025, 1, 1)
        start_da, end_da = self._make_pairs()
        mat_da = DateArray.from_single(*maturity_ymd)
        # Broadcast maturity to match batch size
        n = len(self.DATE_PAIRS)
        mat_broadcast = DateArray.from_ymd(
            jnp.full(n, mat_da.years, dtype=jnp.int32),
            jnp.full(n, mat_da.months, dtype=jnp.int32),
            jnp.full(n, mat_da.days, dtype=jnp.int32),
        )
        result = year_fraction_30e360_isda(start_da, end_da, mat_broadcast)
        for i, (s, e) in enumerate(self.DATE_PAIRS):
            expected = self._scalar_yf(
                DayCountConvention.E30360ISDA, s, e, maturity_ymd
            )
            assert abs(float(result[i]) - expected) < 1e-6, (
                f"30E/360 ISDA pair {i}: {float(result[i])} != {expected}"
            )


# ===================================================================
# Schedule generation
# ===================================================================


class TestScheduleGeneration:
    """Cross-validate against _fast_month_schedule from pam_array.py."""

    def _reference_month_schedule(self, start_y, start_m, start_d, cycle_months, end_dt):
        """Python reference using the same algorithm as pam_array._fast_month_schedule."""
        _dim = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

        def dim(y, m):
            if m == 2 and (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)):
                return 29
            return _dim[m]

        base = start_y * 12 + start_m - 1
        dates = []
        n = 0
        while True:
            total = base + n * cycle_months
            y = total // 12
            m = (total % 12) + 1
            d = min(start_d, dim(y, m))
            current = datetime.date(y, m, d)
            if current > end_dt:
                break
            dates.append(current)
            n += 1
        return dates

    @pytest.mark.parametrize(
        "start,cycle_months,end",
        [
            ((2024, 1, 15), 1, (2025, 1, 15)),
            ((2024, 1, 15), 3, (2025, 1, 15)),
            ((2024, 1, 15), 6, (2030, 1, 15)),
            ((2024, 1, 15), 12, (2034, 1, 15)),
            ((2024, 1, 31), 1, (2025, 1, 31)),  # Day clamping chain
            ((2024, 3, 31), 1, (2025, 3, 31)),
            ((2020, 6, 15), 3, (2030, 6, 15)),  # Long range
            ((2024, 2, 29), 12, (2028, 2, 29)),  # Leap to non-leap
        ],
    )
    def test_month_schedule_matches_reference(self, start, cycle_months, end):
        start_da = DateArray.from_single(*start)
        end_da = DateArray.from_single(*end)
        result = generate_month_schedule(start_da, cycle_months, end_da)

        ref = self._reference_month_schedule(
            start[0], start[1], start[2], cycle_months, datetime.date(*end)
        )

        assert len(result) == len(ref), (
            f"Length mismatch: {len(result)} != {len(ref)} "
            f"for start={start}, cycle={cycle_months}M, end={end}"
        )

        for i, ref_date in enumerate(ref):
            assert int(result.years[i]) == ref_date.year, f"Year at {i}"
            assert int(result.months[i]) == ref_date.month, f"Month at {i}"
            assert int(result.days[i]) == ref_date.day, f"Day at {i}"

    def test_day_schedule(self):
        start = DateArray.from_single(2024, 1, 1)
        end = DateArray.from_single(2024, 1, 10)
        result = generate_day_schedule(start, 1, end)
        assert len(result) == 10  # Jan 1..10

        # Check first and last
        assert int(result.years[0]) == 2024
        assert int(result.months[0]) == 1
        assert int(result.days[0]) == 1
        assert int(result.days[-1]) == 10

    def test_week_schedule(self):
        start = DateArray.from_single(2024, 1, 1)
        end = DateArray.from_single(2024, 3, 1)
        result = generate_day_schedule(start, 7, end)  # weekly

        # Each date should be 7 days apart
        for i in range(1, len(result)):
            delta = int(result.ordinals[i]) - int(result.ordinals[i - 1])
            assert delta == 7, f"Delta at {i}: {delta}"

        # Last date should be <= end
        assert int(result.ordinals[-1]) <= datetime.date(2024, 3, 1).toordinal()

    def test_empty_schedule(self):
        """Start == end should produce a single-element schedule."""
        d = DateArray.from_single(2024, 1, 15)
        result = generate_month_schedule(d, 1, d)
        assert len(result) == 1
        assert int(result.days[0]) == 15

    def test_eom_schedule(self):
        """EOM schedule: start on Jan 31 with 1M cycle."""
        start = DateArray.from_single(2024, 1, 31)
        end = DateArray.from_single(2024, 6, 30)
        result = generate_month_schedule_eom(start, 1, end)

        # All dates should be end-of-month
        for i in range(len(result)):
            da_i = result[i]
            assert bool(da_i.is_end_of_month()), (
                f"Index {i}: {int(da_i.years)}-{int(da_i.months):02d}-{int(da_i.days):02d} "
                f"is not EOM"
            )


# ===================================================================
# Slicing and repr
# ===================================================================


class TestDateArrayMisc:
    def test_getitem(self):
        da = DateArray.from_ymd(
            jnp.array([2024, 2025, 2026], dtype=jnp.int32),
            jnp.array([1, 6, 12], dtype=jnp.int32),
            jnp.array([1, 15, 31], dtype=jnp.int32),
        )
        sliced = da[1:3]
        assert sliced.shape == (2,)
        assert int(sliced.years[0]) == 2025

    def test_repr_scalar(self):
        da = DateArray.from_single(2024, 3, 15)
        assert "2024-03-15" in repr(da)

    def test_repr_batch(self):
        da = DateArray.from_ymd(
            jnp.array([2024, 2025], dtype=jnp.int32),
            jnp.array([1, 6], dtype=jnp.int32),
            jnp.array([1, 15], dtype=jnp.int32),
        )
        r = repr(da)
        assert "n=2" in r
