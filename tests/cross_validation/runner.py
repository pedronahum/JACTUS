"""Generic cross-validation test runner for ACTUS test cases.

Provides reusable comparison logic that aligns events by (date, type)
pairs for robust comparison even when schedule generation differs.
"""

from __future__ import annotations

import math
from typing import Any

from jactus.contracts import create_contract
from jactus.core import ActusDateTime, ContractAttributes, EventType
from jactus.observers import ConstantRiskFactorObserver

from .actus_mapper import TimeSeriesRiskFactorObserver, parse_test_terms

# Tolerance for numeric comparisons
ABS_TOL = 1.0  # Within $1 of expected
REL_TOL = 1e-4  # 0.01% relative tolerance

# Event type string -> JACTUS EventType
EVENT_TYPE_MAP: dict[str, EventType] = {
    "AD": EventType.AD,
    "IED": EventType.IED,
    "IP": EventType.IP,
    "IPCI": EventType.IPCI,
    "MD": EventType.MD,
    "PP": EventType.PP,
    "PR": EventType.PR,
    "PRF": EventType.PRF,
    "IPCB": EventType.IPCB,
    "PRD": EventType.PRD,
    "PY": EventType.PY,
    "FP": EventType.FP,
    "TD": EventType.TD,
    "RR": EventType.RR,
    "RRF": EventType.RRF,
    "SC": EventType.SC,
    "CE": EventType.CE,
    "XD": EventType.XD,
    "STD": EventType.STD,
    "DV": EventType.DV,
}


def _values_close(a: float, b: float) -> bool:
    """Check if two values are close enough."""
    if abs(b) < 1.0:
        return abs(a - b) <= ABS_TOL
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


def run_single_test(test_id: str, test_case: dict[str, Any]) -> list[str]:
    """Run a single ACTUS test case and return list of mismatches.

    Uses alignment by (date, event_type) pairs for robust comparison.

    Args:
        test_id: Test case identifier (e.g., "pam01")
        test_case: Test case data from JSON

    Returns:
        List of error messages (empty = pass)
    """
    errors: list[str] = []

    # Parse terms to JACTUS attributes
    terms = test_case["terms"]
    try:
        kwargs = parse_test_terms(terms)
    except Exception as e:
        return [f"Failed to parse terms: {e}"]

    # For open-ended NAM contracts (no maturityDate), derive effective MD from 'to'
    # LAM computes MD internally from amortization; NAM does not.
    ct = kwargs.get("contract_type", "")
    if ct == "NAM" and ("maturity_date" not in kwargs or kwargs["maturity_date"] is None):
        to_str = test_case.get("to") or terms.get("to")
        if to_str:
            # Normalize format: "2013-12-01T00:00" → "2013-12-01T00:00:00"
            if len(to_str) == 16:  # YYYY-MM-DDTHH:MM
                to_str += ":00"
            to_dt = ActusDateTime.from_iso(to_str)
            # For short stub cycles, MD is one IP period before 'to'
            ip_cycle = kwargs.get("interest_payment_cycle", "")
            if ip_cycle and ip_cycle.endswith("-"):
                # Short stub: subtract one cycle period from 'to'
                from jactus.core.time import parse_cycle
                mult, period, _ = parse_cycle(ip_cycle)
                from dateutil.relativedelta import relativedelta
                period_map = {
                    "D": relativedelta(days=mult),
                    "W": relativedelta(weeks=mult),
                    "M": relativedelta(months=mult),
                    "Q": relativedelta(months=3 * mult),
                    "H": relativedelta(months=6 * mult),
                    "Y": relativedelta(years=mult),
                }
                py_dt = to_dt.to_datetime() - period_map[period]
                kwargs["maturity_date"] = ActusDateTime(
                    py_dt.year, py_dt.month, py_dt.day,
                    py_dt.hour, py_dt.minute, py_dt.second,
                )
            else:
                kwargs["maturity_date"] = to_dt

    try:
        attrs = ContractAttributes(**kwargs)
    except Exception as e:
        return [f"Failed to create attributes: {e}"]

    # Create risk factor observer from dataObserved
    data_observed = test_case.get("dataObserved", {})
    if data_observed:
        rf_observer = TimeSeriesRiskFactorObserver(data_observed)
    else:
        rf_observer = ConstantRiskFactorObserver(constant_value=0.0)

    # Create and simulate contract
    try:
        contract = create_contract(attrs, rf_observer)
        result = contract.simulate()
    except Exception as e:
        return [f"Simulation failed: {e}"]

    # Get expected results
    expected_results = test_case.get("results", [])
    if not expected_results:
        return ["No expected results in test case"]

    # Apply 'to' cutoff if specified (simulation end date)
    to_str = test_case.get("to") or terms.get("to")
    to_date: str | None = to_str[:10] if to_str else None

    # Filter out AD events from both sides
    actual_events = [e for e in result.events if e.event_type != EventType.AD]
    expected_events = [e for e in expected_results if e.get("eventType") != "AD"]

    # Filter by 'to' cutoff date
    if to_date:
        actual_events = [e for e in actual_events if e.event_time.to_iso()[:10] <= to_date]
        expected_events = [e for e in expected_events if (e.get("eventDate", "")[:10]) <= to_date]

    # Build lookup: (date_str, event_type) -> actual event
    actual_lookup: dict[tuple[str, str], Any] = {}
    for e in actual_events:
        key = (e.event_time.to_iso()[:10], e.event_type.value)
        actual_lookup[key] = e

    # Build expected lookup
    expected_lookup: dict[tuple[str, str], dict] = {}
    for e in expected_events:
        date_str = e.get("eventDate", "")[:10]
        etype = e.get("eventType", "")
        key = (date_str, etype)
        expected_lookup[key] = e

    # Find matched, missing, and extra events
    matched_keys = set(actual_lookup.keys()) & set(expected_lookup.keys())
    missing_keys = set(expected_lookup.keys()) - set(actual_lookup.keys())
    extra_keys = set(actual_lookup.keys()) - set(expected_lookup.keys())

    # Report missing events (expected but not generated)
    for date_str, etype in sorted(missing_keys):
        errors.append(f"Missing event: {etype} @ {date_str}")

    # Report extra events (generated but not expected)
    for date_str, etype in sorted(extra_keys):
        errors.append(f"Extra event: {etype} @ {date_str}")

    # Compare matched events
    for key in sorted(matched_keys):
        date_str, etype = key
        actual = actual_lookup[key]
        expected = expected_lookup[key]

        # Payoff
        expected_payoff = expected.get("payoff")
        if expected_payoff is not None:
            actual_payoff = float(actual.payoff)
            exp_payoff = float(expected_payoff)
            if not _values_close(actual_payoff, exp_payoff):
                errors.append(
                    f"{etype} @ {date_str}: payoff mismatch: "
                    f"got {actual_payoff:.4f}, expected {exp_payoff:.4f}"
                )

        # Notional principal (post-event state)
        expected_nt = expected.get("notionalPrincipal")
        if expected_nt is not None and actual.state_post is not None:
            actual_nt = abs(float(actual.state_post.nt))
            exp_nt = abs(float(expected_nt))
            if not _values_close(actual_nt, exp_nt):
                errors.append(
                    f"{etype} @ {date_str}: notional mismatch: "
                    f"got {actual_nt:.4f}, expected {exp_nt:.4f}"
                )

        # Nominal interest rate (post-event state)
        expected_ipnr = expected.get("nominalInterestRate")
        if expected_ipnr is not None and actual.state_post is not None:
            actual_ipnr = float(actual.state_post.ipnr)
            exp_ipnr = float(expected_ipnr)
            if not math.isclose(actual_ipnr, exp_ipnr, abs_tol=1e-6):
                errors.append(
                    f"{etype} @ {date_str}: rate mismatch: "
                    f"got {actual_ipnr:.8f}, expected {exp_ipnr:.8f}"
                )

        # Accrued interest (post-event state)
        expected_ipac = expected.get("accruedInterest")
        if expected_ipac is not None and actual.state_post is not None:
            actual_ipac = float(actual.state_post.ipac)
            exp_ipac = float(expected_ipac)
            if not _values_close(actual_ipac, exp_ipac):
                errors.append(
                    f"{etype} @ {date_str}: accrued interest mismatch: "
                    f"got {actual_ipac:.4f}, expected {exp_ipac:.4f}"
                )

    return errors
