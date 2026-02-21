"""Generic cross-validation test runner for ACTUS test cases.

Provides reusable comparison logic that aligns events by (date, type)
pairs for robust comparison even when schedule generation differs.
"""

from __future__ import annotations

import math
from typing import Any

from jactus.contracts import create_contract
from jactus.core import ActusDateTime, ContractAttributes, ContractState, EventType
from jactus.observers import ConstantRiskFactorObserver
from jactus.observers.child_contract import SimulatedChildContractObserver

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
    "IPFX": EventType.IPFX,
    "IPFL": EventType.IPFL,
}


COMPOSITE_TYPES = {"SWAPS", "CAPFL", "CEG", "CEC"}


def _build_child_observer(
    test_case: dict[str, Any],
    rf_observer: Any,
) -> SimulatedChildContractObserver | None:
    """Build a child contract observer from contractStructure in the test case.

    Parses child contract references, simulates each child contract, and
    registers the results in a SimulatedChildContractObserver.

    Returns None if no composite structure is found.
    """
    import jax.numpy as jnp

    terms = test_case["terms"]
    ct = terms.get("contractType", "")
    cs_raw = terms.get("contractStructure")
    if not cs_raw or ct not in COMPOSITE_TYPES:
        return None

    observer = SimulatedChildContractObserver()
    refs = cs_raw if isinstance(cs_raw, list) else [cs_raw]

    # For SWAPS, determine child leg roles from parent role
    parent_role_str = terms.get("contractRole", "RFL")

    for ref in refs:
        ref_type = ref.get("referenceType", "")
        ref_role = ref.get("referenceRole", "")
        obj = ref.get("object", {})

        if ref_type == "CNT" and isinstance(obj, dict) and "contractType" in obj:
            # Full embedded child contract — simulate it
            child_id = obj.get("contractID", f"child-{ref_role}")
            try:
                child_kwargs = parse_test_terms(obj)
                # Inherit contractRole from parent if child doesn't have one
                if "contract_role" not in child_kwargs:
                    from jactus.core.types import ContractRole
                    if ct == "SWAPS" and ref_role in ("FIL", "SEL"):
                        # Use determine_leg_roles to get correct child roles
                        from jactus.contracts.swaps import determine_leg_roles
                        parent_role = ContractRole(parent_role_str)
                        first_role, second_role = determine_leg_roles(parent_role)
                        child_kwargs["contract_role"] = (
                            first_role if ref_role == "FIL" else second_role
                        )
                    else:
                        role_map = {"UDL": "RPA", "COVE": "RPA", "COVI": "RPA"}
                        role_str = role_map.get(ref_role, "RPA")
                        child_kwargs["contract_role"] = ContractRole(role_str)

                child_attrs = ContractAttributes(**child_kwargs)
                child_contract = create_contract(child_attrs, rf_observer)
                child_result = child_contract.simulate()
                observer.register_simulation(
                    child_id, child_result.events, child_attrs,
                    initial_state=child_result.initial_state,
                )
            except Exception:
                # If child simulation fails, skip this reference
                continue

        elif ref_type == "CID":
            # Contract ID reference — need to build state from eventsObserved
            child_id = obj.get("contractIdentifier", "")
            if child_id:
                _register_cid_child(observer, child_id, test_case, terms)

        elif ref_type == "MOC":
            # Market object code — used by OPTNS/FUTUR, not composite
            pass

    # Inject CE (credit events) from eventsObserved into child histories
    # This enables CEG/CEC contracts to detect credit events on covered contracts
    if ct in ("CEG", "CEC"):
        _inject_credit_events(observer, test_case)

    return observer


def _register_cid_child(
    observer: SimulatedChildContractObserver,
    child_id: str,
    test_case: dict[str, Any],
    terms: dict[str, Any],
) -> None:
    """Register a CID-referenced child with synthetic state from parent terms.

    For CEG contracts with CID references, we construct a synthetic child
    contract state from the parent's notionalPrincipal and eventsObserved.
    """
    import jax.numpy as jnp
    from jactus.core import ContractEvent
    from jactus.core.types import ContractPerformance

    nt_val = float(terms.get("notionalPrincipal", 0))
    status_date_str = terms.get("statusDate", "")
    if len(status_date_str) == 16:
        status_date_str += ":00"
    sd = ActusDateTime.from_iso(status_date_str) if status_date_str else ActusDateTime(2020, 1, 1)

    # Build events from eventsObserved
    events_observed = test_case.get("eventsObserved", [])
    synthetic_events: list[ContractEvent] = []

    # Derive maturity from parent terms
    md_str = terms.get("maturityDate", "")
    if md_str:
        if len(md_str) == 16:
            md_str += ":00"
        tmd = ActusDateTime.from_iso(md_str)
    else:
        tmd = sd

    # Initial state: performant with full notional
    initial_state = ContractState(
        tmd=tmd,
        sd=sd,
        nt=jnp.array(nt_val, dtype=jnp.float32),
        ipnr=jnp.array(0.0, dtype=jnp.float32),
        ipac=jnp.array(0.0, dtype=jnp.float32),
        feac=jnp.array(0.0, dtype=jnp.float32),
        nsc=jnp.array(1.0, dtype=jnp.float32),
        isc=jnp.array(1.0, dtype=jnp.float32),
        prf=ContractPerformance.PF,
    )

    # Create a synthetic IED event at status_date
    synthetic_events.append(ContractEvent(
        event_type=EventType.IED,
        event_time=sd,
        payoff=jnp.array(0.0, dtype=jnp.float32),
        currency="XXX",
        state_pre=initial_state,
        state_post=initial_state,
    ))

    # Process observed credit events
    for eo in events_observed:
        if eo.get("contractId") != child_id:
            continue
        eo_time_str = eo.get("time", "")
        if len(eo_time_str) == 16:
            eo_time_str += ":00"
        eo_time = ActusDateTime.from_iso(eo_time_str) if eo_time_str else sd

        eo_type = eo.get("type", "")
        states_data = eo.get("states", {})
        perf_str = states_data.get("contractPerformance", "PF")
        perf = ContractPerformance(perf_str) if perf_str in ("PF", "DL", "DQ", "DF") else ContractPerformance.PF

        # Build state at credit event time
        ce_state = ContractState(
            tmd=tmd,
            sd=eo_time,
            nt=jnp.array(nt_val, dtype=jnp.float32),
            ipnr=jnp.array(0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
            prf=perf,
        )

        evt_type = EVENT_TYPE_MAP.get(eo_type, EventType.CE)
        synthetic_events.append(ContractEvent(
            event_type=evt_type,
            event_time=eo_time,
            payoff=jnp.array(0.0, dtype=jnp.float32),
            currency="XXX",
            state_pre=initial_state,
            state_post=ce_state,
        ))

    observer.register_simulation(child_id, synthetic_events)


def _inject_credit_events(
    observer: SimulatedChildContractObserver,
    test_case: dict[str, Any],
) -> None:
    """Inject credit events from eventsObserved into child observer histories.

    Enables CEG/CEC contracts to detect credit events on covered contracts
    by querying child state at the credit event time.
    """
    import jax.numpy as jnp
    from jactus.core import ContractEvent
    from jactus.core.types import ContractPerformance

    events_observed = test_case.get("eventsObserved", [])
    for eo in events_observed:
        eo_type = eo.get("type", "")
        contract_id = eo.get("contractId", "")
        if eo_type != "CE" or not contract_id:
            continue

        eo_time_str = eo.get("time", "")
        if len(eo_time_str) == 16:
            eo_time_str += ":00"
        eo_time = ActusDateTime.from_iso(eo_time_str)

        states_data = eo.get("states", {})
        perf_str = states_data.get("contractPerformance", "PF")
        perf = (
            ContractPerformance(perf_str)
            if perf_str in ("PF", "DL", "DQ", "DF")
            else ContractPerformance.PF
        )

        # Get the child's state at CE time (if available)
        try:
            child_state = observer.observe_state(contract_id, eo_time)
        except (KeyError, ValueError):
            # Child not registered — skip
            continue

        # Compute accrued interest from child_state.sd to CE time
        # so the CE state has correct ipac (not reset to 0 after last IP)
        from jactus.utilities.conventions import year_fraction
        child_attrs = observer._attributes.get(contract_id)
        if child_attrs and child_attrs.day_count_convention:
            dcc = child_attrs.day_count_convention
        else:
            from jactus.core.types import DayCountConvention
            dcc = DayCountConvention.A365
        yf = year_fraction(child_state.sd, eo_time, dcc)
        ipac_at_ce = float(child_state.ipac) + yf * float(child_state.ipnr) * float(child_state.nt)

        # Create updated state with credit event performance and proper accrual
        ce_state = ContractState(
            tmd=child_state.tmd,
            sd=eo_time,
            nt=child_state.nt,
            ipnr=child_state.ipnr,
            ipac=jnp.array(ipac_at_ce, dtype=jnp.float32),
            feac=child_state.feac,
            nsc=child_state.nsc,
            isc=child_state.isc,
            prf=perf,
        )

        # Create CE event
        ce_event = ContractEvent(
            event_type=EventType.CE,
            event_time=eo_time,
            payoff=jnp.array(0.0, dtype=jnp.float32),
            currency="XXX",
            state_pre=child_state,
            state_post=ce_state,
        )

        # Inject into child's event and history lists
        observer._events.setdefault(contract_id, []).append(ce_event)
        observer._histories.setdefault(contract_id, []).append((eo_time, ce_state))
        # Re-sort history by time
        observer._histories[contract_id].sort(key=lambda x: (
            x[0].year, x[0].month, x[0].day, x[0].hour, x[0].minute, x[0].second,
        ))


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

    # Apply defaults for fields required by JACTUS but not always in ACTUS tests
    if ct == "FXOUT" and "delivery_settlement" not in kwargs:
        kwargs["delivery_settlement"] = "S"  # Default to gross settlement for FXOUT

    # For UMP/CLM without maturity/termination, use 'to' as end date for schedule generation
    # For CLM, prefer XD from eventsObserved as the effective maturity (call date)
    # STD date = XD + xDayNotice (the actual settlement/termination date)
    clm_xd_date = None
    clm_std_date = None
    if ct in ("UMP", "CLM") and "maturity_date" not in kwargs and "termination_date" not in kwargs:
        # Check eventsObserved for XD (call date)
        events_observed = test_case.get("eventsObserved", [])
        for eo in events_observed:
            if eo.get("type") == "XD":
                xd_str = eo.get("time", "")
                if xd_str:
                    if len(xd_str) == 16:
                        xd_str += ":00"
                    clm_xd_date = ActusDateTime.from_iso(xd_str)
                    # Compute STD date = XD + xDayNotice
                    xdn = terms.get("xDayNotice", "P0D")
                    from jactus.core.time import parse_cycle
                    from dateutil.relativedelta import relativedelta
                    xdn_str = xdn[1:] if xdn.startswith("P") else xdn
                    # Remove any L-suffix
                    if "L" in xdn_str:
                        xdn_str = xdn_str[:xdn_str.index("L")]
                    # Parse: "31D", "1M", "0D", "3W", "9M"
                    import re
                    m = re.match(r"(\d+)([DWMY])", xdn_str)
                    if m:
                        n, unit = int(m.group(1)), m.group(2)
                        xd_py = clm_xd_date.to_datetime()
                        delta_map = {
                            "D": relativedelta(days=n),
                            "W": relativedelta(weeks=n),
                            "M": relativedelta(months=n),
                            "Y": relativedelta(years=n),
                        }
                        std_py = xd_py + delta_map.get(unit, relativedelta())
                        clm_std_date = ActusDateTime(
                            std_py.year, std_py.month, std_py.day,
                            std_py.hour, std_py.minute, std_py.second,
                        )
                    else:
                        clm_std_date = clm_xd_date
                    # Use XD date as maturity for schedule generation (IPCI/RR stop at XD)
                    # IP+MD will be moved to STD date after simulation
                    kwargs["maturity_date"] = clm_xd_date
                    break
        if "maturity_date" not in kwargs:
            to_str = test_case.get("to") or terms.get("to")
            if to_str:
                if len(to_str) == 16:
                    to_str += ":00"
                kwargs["maturity_date"] = ActusDateTime.from_iso(to_str)

    # For LAX without maturityDate, derive from 'to' field
    if ct == "LAX" and "maturity_date" not in kwargs:
        to_str = test_case.get("to") or terms.get("to")
        if to_str:
            if len(to_str) == 16:
                to_str += ":00"
            kwargs["maturity_date"] = ActusDateTime.from_iso(to_str)

    # For STK without maturity/termination, use 'to' as end date for DV schedule
    if ct == "STK" and "maturity_date" not in kwargs and "termination_date" not in kwargs:
        to_str = test_case.get("to") or terms.get("to")
        if to_str:
            if len(to_str) == 16:
                to_str += ":00"
            kwargs["maturity_date"] = ActusDateTime.from_iso(to_str)

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

    # Build child contract observer for composite contracts
    child_observer = None
    if ct in COMPOSITE_TYPES:
        try:
            child_observer = _build_child_observer(test_case, rf_observer)
        except Exception as e:
            return [f"Failed to build child observer: {e}"]

    # Create and simulate contract
    try:
        contract = create_contract(attrs, rf_observer, child_observer)
        result = contract.simulate()
    except Exception as e:
        return [f"Simulation failed: {e}"]

    # Get expected results
    expected_results = test_case.get("results", [])
    if not expected_results:
        return ["No expected results in test case"]

    # Apply 'to' cutoff if specified (simulation end date)
    # For CLM with XD+notice, extend cutoff to include STD date
    to_str = test_case.get("to") or terms.get("to")
    to_date: str | None = to_str[:10] if to_str else None
    if clm_std_date is not None and to_date:
        std_date_str = clm_std_date.to_iso()[:10]
        if std_date_str > to_date:
            to_date = std_date_str

    # For CLM with observed XD: inject XD event, move IP+MD to STD date, rename MD→STD
    if clm_xd_date is not None:
        import jax.numpy as jnp
        from jactus.core import ContractEvent, ContractState
        from jactus.utilities import year_fraction
        std_date = clm_std_date or clm_xd_date
        new_actual = []

        for e in result.events:
            if e.event_type == EventType.IP and e.event_time == clm_xd_date:
                # Compute accrued interest at XD time for the XD event state
                pre = e.state_pre
                dcc = attrs.day_count_convention
                yf = year_fraction(pre.sd, clm_xd_date, dcc)
                xd_ipac = pre.ipac + yf * pre.ipnr * pre.nt
                xd_state = pre.replace(ipac=xd_ipac)

                # Inject XD event before IP (with accrued interest in state)
                xd_event = ContractEvent(
                    event_type=EventType.XD,
                    event_time=clm_xd_date,
                    payoff=jnp.array(0.0, dtype=jnp.float32),
                    currency=e.currency,
                    state_pre=pre,
                    state_post=xd_state,
                )
                new_actual.append(xd_event)
                # Move IP to STD date (may be same as XD if notice=0)
                moved_ip = ContractEvent(
                    event_type=EventType.IP,
                    event_time=std_date,
                    payoff=e.payoff,
                    currency=e.currency,
                    state_pre=e.state_pre,
                    state_post=e.state_post,
                )
                new_actual.append(moved_ip)
            elif e.event_type == EventType.MD and e.event_time == clm_xd_date:
                # Rename MD→STD and move to STD date
                std_event = ContractEvent(
                    event_type=EventType.STD,
                    event_time=std_date,
                    payoff=e.payoff,
                    currency=e.currency,
                    state_pre=e.state_pre,
                    state_post=e.state_post,
                )
                new_actual.append(std_event)
            else:
                new_actual.append(e)
        result_events = new_actual
    else:
        result_events = list(result.events)

    # Filter out AD events from both sides
    actual_events = [e for e in result_events if e.event_type != EventType.AD]
    expected_events = [e for e in expected_results if e.get("eventType") != "AD"]

    # Filter by 'to' cutoff date
    if to_date:
        actual_events = [e for e in actual_events if e.event_time.to_iso()[:10] <= to_date]
        expected_events = [e for e in expected_events if (e.get("eventDate", "")[:10]) <= to_date]

    # Build lookup: (date_str, event_type) -> list of actual events
    # Use list to handle duplicates (e.g., FXOUT with two MD events on same date)
    actual_lookup: dict[tuple[str, str], list[Any]] = {}
    for e in actual_events:
        key = (e.event_time.to_iso()[:10], e.event_type.value)
        actual_lookup.setdefault(key, []).append(e)

    # Build expected lookup
    expected_lookup: dict[tuple[str, str], list[dict]] = {}
    for e in expected_events:
        date_str = e.get("eventDate", "")[:10]
        etype = e.get("eventType", "")
        key = (date_str, etype)
        expected_lookup.setdefault(key, []).append(e)

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
        actual_list = actual_lookup[key]
        expected_list = expected_lookup[key]

        # Match events by position (sorted by payoff, then notional for tie-breaking)
        if len(actual_list) > 1 or len(expected_list) > 1:
            actual_list = sorted(
                actual_list,
                key=lambda e: (float(e.payoff), float(e.state_post.nt) if e.state_post else 0),
            )
            expected_list = sorted(
                expected_list,
                key=lambda e: (float(e.get("payoff", 0)), float(e.get("notionalPrincipal", 0))),
            )

        for i, (actual, expected) in enumerate(zip(actual_list, expected_list)):
            suffix = f" [{i}]" if len(actual_list) > 1 else ""

            # Payoff
            expected_payoff = expected.get("payoff")
            if expected_payoff is not None:
                actual_payoff = float(actual.payoff)
                exp_payoff = float(expected_payoff)
                if not _values_close(actual_payoff, exp_payoff):
                    errors.append(
                        f"{etype} @ {date_str}{suffix}: payoff mismatch: "
                        f"got {actual_payoff:.4f}, expected {exp_payoff:.4f}"
                    )

            # Notional principal (post-event state)
            expected_nt = expected.get("notionalPrincipal")
            if expected_nt is not None and actual.state_post is not None:
                actual_nt = abs(float(actual.state_post.nt))
                exp_nt = abs(float(expected_nt))
                if not _values_close(actual_nt, exp_nt):
                    errors.append(
                        f"{etype} @ {date_str}{suffix}: notional mismatch: "
                        f"got {actual_nt:.4f}, expected {exp_nt:.4f}"
                    )

            # Nominal interest rate (post-event state)
            expected_ipnr = expected.get("nominalInterestRate")
            if expected_ipnr is not None and actual.state_post is not None:
                actual_ipnr = float(actual.state_post.ipnr)
                exp_ipnr = float(expected_ipnr)
                if not math.isclose(actual_ipnr, exp_ipnr, abs_tol=1e-6):
                    errors.append(
                        f"{etype} @ {date_str}{suffix}: rate mismatch: "
                        f"got {actual_ipnr:.8f}, expected {exp_ipnr:.8f}"
                    )

            # Accrued interest (post-event state)
            expected_ipac = expected.get("accruedInterest")
            if expected_ipac is not None and actual.state_post is not None:
                actual_ipac = float(actual.state_post.ipac)
                exp_ipac = float(expected_ipac)
                if not _values_close(actual_ipac, exp_ipac):
                    errors.append(
                        f"{etype} @ {date_str}{suffix}: accrued interest mismatch: "
                        f"got {actual_ipac:.4f}, expected {exp_ipac:.4f}"
                    )

        # Report count mismatch
        if len(actual_list) != len(expected_list):
            errors.append(
                f"{etype} @ {date_str}: count mismatch: "
                f"got {len(actual_list)}, expected {len(expected_list)}"
            )

    return errors
