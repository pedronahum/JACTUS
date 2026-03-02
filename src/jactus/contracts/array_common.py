"""Shared infrastructure for array-mode contract simulations.

This module provides common constants, date/schedule helpers, year-fraction
functions, and batch pre-computation primitives reused by all ``*_array.py``
contract modules (e.g., ``pam_array.py``, ``lam_array.py``).

Extracted from ``pam_array.py`` to avoid code duplication across contract types.
"""

from __future__ import annotations

import re as _re
from collections.abc import Callable, Sequence
from datetime import datetime as _datetime
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from jactus.core.types import DayCountConvention
    from jactus.observers import RiskFactorObserver

import jax.numpy as jnp
import numpy as np

from jactus.core import ActusDateTime, ContractAttributes, ContractRole, EventType
from jactus.core.types import NUM_EVENT_TYPES
from jactus.utilities.conventions import year_fraction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: NOP event index — used to pad shorter contracts in batched simulation.
NOP_EVENT_IDX: int = NUM_EVENT_TYPES  # 24 (one past the last valid EventType.index)

#: Float32 dtype shorthand
F32 = jnp.float32

#: DateArray feature flag — enables vectorised year-fraction pre-computation.
USE_DATE_ARRAY: bool = True

#: Batch schedule feature flag — enables JAX-native batch schedule generation.
USE_BATCH_SCHEDULE: bool = True

# ---------------------------------------------------------------------------
# Cached EventType index values for fast comparison
# ---------------------------------------------------------------------------
AD_IDX = EventType.AD.index
IED_IDX = EventType.IED.index
MD_IDX = EventType.MD.index
PR_IDX = EventType.PR.index
PI_IDX = EventType.PI.index
PP_IDX = EventType.PP.index
PY_IDX = EventType.PY.index
PRF_IDX = EventType.PRF.index
FP_IDX = EventType.FP.index
PRD_IDX = EventType.PRD.index
TD_IDX = EventType.TD.index
IP_IDX = EventType.IP.index
IPCI_IDX = EventType.IPCI.index
IPCB_IDX = EventType.IPCB.index
RR_IDX = EventType.RR.index
RRF_IDX = EventType.RRF.index
DV_IDX = EventType.DV.index
DVF_IDX = EventType.DVF.index
SC_IDX = EventType.SC.index
STD_IDX = EventType.STD.index
XD_IDX = EventType.XD.index
CE_IDX = EventType.CE.index

# DCC encoding for batch path
DCC_A360 = 0
DCC_A365 = 1
DCC_E30360 = 2
DCC_B30360 = 3

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def encode_fee_basis(attrs: ContractAttributes) -> int:
    """Encode fee basis as int: 0=A, 1=N, 2=other."""
    from jactus.core.types import FeeBasis

    if attrs.fee_basis == FeeBasis.A:
        return 0
    if attrs.fee_basis == FeeBasis.N:
        return 1
    return 2


def encode_penalty_type(attrs: ContractAttributes) -> int:
    """Encode penalty type as int: 0=A, 1=N, 2=I."""
    pt = attrs.penalty_type
    if pt == "A":
        return 0
    if pt == "N":
        return 1
    if pt == "I":
        return 2
    return 2  # default


def get_role_sign(role: ContractRole | None) -> float:
    """Get +1.0 or -1.0 for the contract role."""
    if role in (ContractRole.RPA, ContractRole.RFL, ContractRole.LG, ContractRole.BUY):
        return 1.0
    if role in (ContractRole.RPL, ContractRole.PFL, ContractRole.ST, ContractRole.SEL):
        return -1.0
    return 1.0


# ---------------------------------------------------------------------------
# Date conversion helpers
# ---------------------------------------------------------------------------


def adt_to_dt(adt: ActusDateTime) -> _datetime:
    """Convert ActusDateTime to Python datetime (fast path)."""
    if adt.hour == 24:
        from datetime import timedelta

        return _datetime(adt.year, adt.month, adt.day) + timedelta(days=1)  # noqa: DTZ001
    return _datetime(adt.year, adt.month, adt.day, adt.hour, adt.minute, adt.second)  # noqa: DTZ001


def dt_to_adt(dt: _datetime) -> ActusDateTime:
    """Convert Python datetime to ActusDateTime."""
    return ActusDateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


# ---------------------------------------------------------------------------
# Schedule generation helpers
# ---------------------------------------------------------------------------

# Days-in-month lookup (index 0 unused). Avoids calendar.monthrange() overhead.
_DAYS_IN_MONTH = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

CYCLE_MONTHS_MAP = {"M": 1, "Q": 3, "H": 6, "Y": 12}

# Pre-compiled regex for cycle parsing (avoid re-compiling per call)
_CYCLE_RE = _re.compile(r"^(\d+)([DWMQHY])([-+]?)$")


def days_in_month(y: int, m: int) -> int:
    """Return number of days in month ``m`` of year ``y``."""
    if m == 2 and (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)):
        return 29
    return _DAYS_IN_MONTH[m]


def parse_cycle_fast(cycle: str) -> tuple[int, str, str]:
    """Parse cycle string without repeated re.match overhead."""
    m = _CYCLE_RE.match(cycle.upper())
    if not m:
        from jactus.core.time import parse_cycle

        return parse_cycle(cycle)
    return int(m.group(1)), m.group(2), m.group(3)


def fast_month_schedule(
    start_y: int,
    start_m: int,
    start_d: int,
    cycle_months: int,
    end_dt: _datetime,
) -> list[_datetime]:
    """Generate monthly-based schedule using direct arithmetic.

    Computes dates as ``start + n * cycle_months`` for n=0,1,2,...
    until the result exceeds ``end_dt``.  Day is clamped to the target
    month's maximum day.
    """
    base = start_y * 12 + start_m - 1
    dates: list[_datetime] = []
    n = 0
    while True:
        total = base + n * cycle_months
        y = total // 12
        m = (total % 12) + 1
        d = min(start_d, days_in_month(y, m))
        current = _datetime(y, m, d)  # noqa: DTZ001
        if current > end_dt:
            break
        dates.append(current)
        n += 1
    return dates


def fast_schedule(
    start: ActusDateTime | None,
    cycle: str | None,
    end: ActusDateTime | None,
) -> list[_datetime]:
    """Fast schedule generation returning Python datetimes.

    Handles the common case (month-based cycles, EOMC=SD, BDC=NULL).
    """
    if start is None or end is None:
        return []
    if cycle is None or cycle == "":
        return [adt_to_dt(start)]

    multiplier, period, _stub = parse_cycle_fast(cycle)
    end_dt = adt_to_dt(end)

    if period in CYCLE_MONTHS_MAP:
        cycle_months = multiplier * CYCLE_MONTHS_MAP[period]
        return fast_month_schedule(start.year, start.month, start.day, cycle_months, end_dt)

    # Day/week-based — use timedelta
    from datetime import timedelta

    start_dt = adt_to_dt(start)
    if period == "D":
        delta = timedelta(days=multiplier)
    else:
        delta = timedelta(weeks=multiplier)

    dates: list[_datetime] = []
    n = 0
    while True:
        current = start_dt + delta * n
        if current > end_dt:
            break
        dates.append(current)
        n += 1
    return dates


# Cache for EVENT_SCHEDULE_PRIORITY lookups
_EVT_PRIORITY: dict[int, int] = {}


def get_evt_priority(evt_idx: int) -> int:
    """Get sort priority for an event type index (cached)."""
    if not _EVT_PRIORITY:
        from jactus.core.types import EVENT_SCHEDULE_PRIORITY

        for et, pri in EVENT_SCHEDULE_PRIORITY.items():
            _EVT_PRIORITY[et.index] = pri
    return _EVT_PRIORITY.get(evt_idx, 99)


# ---------------------------------------------------------------------------
# Year fraction functions (Python scalar)
# ---------------------------------------------------------------------------


def yf_a360(d1: _datetime, d2: _datetime) -> float:
    return (d2 - d1).days / 360.0


def yf_a365(d1: _datetime, d2: _datetime) -> float:
    return (d2 - d1).days / 365.0


def yf_30e360(d1: _datetime, d2: _datetime) -> float:
    y1, m1, dd1 = d1.year, d1.month, d1.day
    y2, m2, dd2 = d2.year, d2.month, d2.day
    if dd1 == 31:
        dd1 = 30
    if dd2 == 31:
        dd2 = 30
    return ((y2 - y1) * 360 + (m2 - m1) * 30 + (dd2 - dd1)) / 360.0


def yf_b30360(d1: _datetime, d2: _datetime) -> float:
    y1, m1, dd1 = d1.year, d1.month, d1.day
    y2, m2, dd2 = d2.year, d2.month, d2.day
    if dd1 == 31:
        dd1 = 30
    if dd1 >= 30 and dd2 == 31:
        dd2 = 30
    return ((y2 - y1) * 360 + (m2 - m1) * 30 + (dd2 - dd1)) / 360.0


def get_yf_fn(dcc_enum: DayCountConvention) -> Callable[[_datetime, _datetime], float] | None:
    """Return the appropriate scalar year-fraction function for a DCC enum.

    Returns ``None`` for uncommon DCCs that need the full ``year_fraction`` path.
    """
    from jactus.core.types import DayCountConvention

    if dcc_enum == DayCountConvention.A360:
        return yf_a360
    if dcc_enum == DayCountConvention.A365:
        return yf_a365
    if dcc_enum == DayCountConvention.E30360:
        return yf_30e360
    if dcc_enum == DayCountConvention.B30360:
        return yf_b30360
    return None


# ---------------------------------------------------------------------------
# NumPy-backed ordinal & year-fraction helpers (zero JAX overhead)
# ---------------------------------------------------------------------------


def np_ymd_to_ordinal(y: np.ndarray, m: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Hinnant Y/M/D → ordinal, NumPy version (for pre-computation path)."""
    a = np.where(m <= 2, 1, 0).astype(np.int64)
    y_adj = y.astype(np.int64) - a
    m_adj = m.astype(np.int64) + 12 * a - 3

    doy = (153 * m_adj + 2) // 5 + d.astype(np.int64) - 1

    era = np.where(y_adj >= 0, y_adj // 400, (y_adj - 399) // 400)
    yoe = y_adj - era * 400

    doe = 365 * yoe + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 305  # type: ignore[no-any-return]


def np_yf_30e360(
    y1: np.ndarray,
    m1: np.ndarray,
    d1: np.ndarray,
    y2: np.ndarray,
    m2: np.ndarray,
    d2: np.ndarray,
) -> np.ndarray:
    """30E/360 year fraction, vectorised NumPy."""
    dd1 = np.where(d1 == 31, 30, d1)
    dd2 = np.where(d2 == 31, 30, d2)
    days = (y2 - y1) * 360 + (m2 - m1) * 30 + (dd2 - dd1)
    return days.astype(np.float64) / 360.0  # type: ignore[no-any-return]


def np_yf_b30360(
    y1: np.ndarray,
    m1: np.ndarray,
    d1: np.ndarray,
    y2: np.ndarray,
    m2: np.ndarray,
    d2: np.ndarray,
) -> np.ndarray:
    """30/360 US (Bond Basis) year fraction, vectorised NumPy."""
    dd1 = np.where(d1 == 31, 30, d1)
    dd2 = np.where((dd1 >= 30) & (d2 == 31), 30, d2)
    days = (y2 - y1) * 360 + (m2 - m1) * 30 + (dd2 - dd1)
    return days.astype(np.float64) / 360.0  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Batch contract parameters (schedule generation)
# ---------------------------------------------------------------------------


class BatchContractParams(NamedTuple):
    """Contract parameters for JAX-native batch schedule generation.

    All fields are ``jnp.ndarray`` with shape ``(N,)`` where *N* is the number
    of batch-eligible contracts.  This structure is passed to JIT-compiled
    functions so all fields must be JAX arrays.
    """

    ied_y: jnp.ndarray  # int32 — IED year
    ied_m: jnp.ndarray  # int32 — IED month
    ied_d: jnp.ndarray  # int32 — IED day
    ied_ord: jnp.ndarray  # int32 — IED ordinal
    md_ord: jnp.ndarray  # int32 — MD ordinal
    sd_ord: jnp.ndarray  # int32 — SD ordinal
    ip_anchor_y: jnp.ndarray  # int32 — IP anchor year
    ip_anchor_m: jnp.ndarray  # int32 — IP anchor month
    ip_anchor_d: jnp.ndarray  # int32 — IP anchor day
    cycle_months: jnp.ndarray  # int32 — IP cycle in months
    has_ip_cycle: jnp.ndarray  # int32 — 1 if contract has IP cycle, 0 otherwise
    dcc_code: jnp.ndarray  # int32 — 0=A360, 1=A365, 2=E30360, 3=B30360


# ---------------------------------------------------------------------------
# JAX-native batch schedule helpers (GPU/TPU-ready)
# ---------------------------------------------------------------------------


def jax_batch_ip_schedule(
    params: BatchContractParams,
    max_ip: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate IP schedule dates for all contracts simultaneously (JAX-native).

    Args:
        params: Batch contract parameters with shape ``(N,)`` per field.
        max_ip: Maximum IP events to generate (static, determines array shape).

    Returns:
        ``(ip_ordinals, ip_valid)`` — shapes ``(N, max_ip)``.
    """
    from jactus.utilities.date_array import _days_in_month as _jax_days_in_month
    from jactus.utilities.date_array import _ymd_to_ordinal as _jax_ymd_to_ordinal

    # step: (1, max_ip)
    step = jnp.arange(max_ip, dtype=jnp.int32).reshape(1, -1)

    # base month ordinal: (N, 1)
    base = (
        params.ip_anchor_y.astype(jnp.int32) * 12 + params.ip_anchor_m.astype(jnp.int32) - 1
    ).reshape(-1, 1)

    # cycle_months: (N, 1)
    cm = params.cycle_months.astype(jnp.int32).reshape(-1, 1)

    # Broadcast: total months for all contracts × all steps — (N, max_ip)
    total = base + step * cm

    # Decompose into Y/M/D
    gen_y = (total // 12).astype(jnp.int32)
    gen_m = ((total % 12) + 1).astype(jnp.int32)

    # Day clamping: min(anchor_day, days_in_month)
    anchor_d = params.ip_anchor_d.reshape(-1, 1)  # (N, 1)
    dim = _jax_days_in_month(gen_y, gen_m)  # (N, max_ip)
    gen_d = jnp.minimum(anchor_d, dim)  # (N, max_ip)

    # Compute ordinals
    ip_ordinals = _jax_ymd_to_ordinal(gen_y, gen_m, gen_d)  # (N, max_ip)

    # Validity: date >= IED and date <= MD and contract has IP cycle
    md_ord = params.md_ord.reshape(-1, 1)
    ied_ord = params.ied_ord.reshape(-1, 1)
    has_ip = params.has_ip_cycle.reshape(-1, 1).astype(jnp.bool_)
    ip_valid = (ip_ordinals >= ied_ord) & (ip_ordinals <= md_ord) & has_ip

    return ip_ordinals, ip_valid


def jax_batch_year_fractions(
    event_ordinals: jnp.ndarray,
    event_valid: jnp.ndarray,
    params: BatchContractParams,
) -> jnp.ndarray:
    """Compute year fractions for all events in the batch (JAX-native).

    Builds the status-date chain ``[sd, evt_0, evt_1, ...]`` and computes
    year fractions between consecutive entries using the per-contract DCC.

    Returns:
        ``(N, max_events)`` float32 year fractions.
    """
    from jactus.utilities.date_array import _ordinal_to_ymd as _jax_ordinal_to_ymd

    n, max_events = event_ordinals.shape

    # SD chain: sd_chain[i, 0] = sd_ord; sd_chain[i, j>0] = event_ordinals[i, j-1]
    sd_chain = jnp.concatenate(
        [params.sd_ord.reshape(-1, 1), event_ordinals[:, :-1]], axis=1
    )  # (N, max_events)

    # Delta days (for A360/A365)
    delta_days = (event_ordinals - sd_chain).astype(jnp.float32)
    yf_a360 = delta_days / 360.0
    yf_a365 = delta_days / 365.0

    # For 30/360 variants, need Y/M/D components
    sd_y, sd_m, sd_d = _jax_ordinal_to_ymd(sd_chain)
    evt_y, evt_m, evt_d = _jax_ordinal_to_ymd(event_ordinals)

    # 30E/360
    dd1_e = jnp.where(sd_d == 31, 30, sd_d)
    dd2_e = jnp.where(evt_d == 31, 30, evt_d)
    days_30e = (evt_y - sd_y) * 360 + (evt_m - sd_m) * 30 + (dd2_e - dd1_e)
    yf_30e360 = days_30e.astype(jnp.float32) / 360.0

    # 30/360 US (Bond Basis)
    dd1_b = jnp.where(sd_d == 31, 30, sd_d)
    dd2_b = jnp.where((dd1_b >= 30) & (evt_d == 31), 30, evt_d)
    days_30b = (evt_y - sd_y) * 360 + (evt_m - sd_m) * 30 + (dd2_b - dd1_b)
    yf_b30360 = days_30b.astype(jnp.float32) / 360.0

    # Select per-contract DCC
    dcc = params.dcc_code.reshape(-1, 1)  # (N, 1)
    yf = jnp.where(
        dcc == DCC_A360,
        yf_a360,
        jnp.where(
            dcc == DCC_A365,
            yf_a365,
            jnp.where(
                dcc == DCC_E30360,
                yf_30e360,
                yf_b30360,  # dcc == DCC_B30360
            ),
        ),
    )

    # Zero out invalid events
    yf = jnp.where(event_valid, yf, 0.0)

    return yf


def compute_max_ip(params: BatchContractParams) -> int:
    """Compute max possible IP events across all contracts (Python)."""
    md_np = np.asarray(params.md_ord)
    ied_np = np.asarray(params.ied_ord)
    cm_np = np.asarray(params.cycle_months).clip(min=1)
    days_span = md_np.astype(np.int64) - ied_np.astype(np.int64)
    # Conservative: assume ~28 days/month minimum
    max_per = days_span / (cm_np.astype(np.int64) * 28)
    return int(np.max(max_per)) + 3


# ---------------------------------------------------------------------------
# Batch padding helper
# ---------------------------------------------------------------------------


def pad_arrays(
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    max_events: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pad arrays to ``max_events`` length with NOP events.

    Returns ``(event_types, year_fractions, rf_values, mask)`` where mask is
    1.0 for real events and 0.0 for padding.
    """
    n = event_types.shape[0]
    pad_n = max_events - n
    mask = jnp.concatenate([jnp.ones(n, dtype=F32), jnp.zeros(pad_n, dtype=F32)])
    event_types = jnp.concatenate([event_types, jnp.full(pad_n, NOP_EVENT_IDX, dtype=jnp.int32)])
    year_fractions = jnp.concatenate([year_fractions, jnp.zeros(pad_n, dtype=F32)])
    rf_values = jnp.concatenate([rf_values, jnp.zeros(pad_n, dtype=F32)])
    return event_types, year_fractions, rf_values, mask


# ---------------------------------------------------------------------------
# Common pre-computed data container
# ---------------------------------------------------------------------------


class RawPrecomputed(NamedTuple):
    """Pre-computed data as Python types (no JAX overhead).

    ``state`` is a tuple of floats; its length depends on the contract type
    (6 for PAM, 8 for LAM/NAM, etc.).
    """

    state: tuple[float, ...]
    event_types: list[int]
    year_fractions: list[float]
    rf_values: list[float]
    params: dict[str, float | int]


# ---------------------------------------------------------------------------
# DCC-aware vectorised year-fraction computation (for precompute_raw_da)
# ---------------------------------------------------------------------------


def compute_vectorised_year_fractions(
    schedule: list[tuple[int, _datetime, _datetime]],
    init_sd_dt: _datetime,
    dcc_enum: DayCountConvention,
) -> list[float]:
    """Compute year fractions for a schedule using NumPy-vectorised helpers.

    Args:
        schedule: List of ``(evt_idx, evt_dt, calc_dt)`` tuples.
        init_sd_dt: Initial status date (Python datetime).
        dcc_enum: ``DayCountConvention`` enum value.

    Returns:
        List of year fractions, one per event.
    """
    from jactus.core.types import DayCountConvention

    if not schedule:
        return []

    n_events = len(schedule)

    sd_y = np.empty(n_events, dtype=np.int32)
    sd_m = np.empty(n_events, dtype=np.int32)
    sd_d = np.empty(n_events, dtype=np.int32)
    calc_y = np.empty(n_events, dtype=np.int32)
    calc_m = np.empty(n_events, dtype=np.int32)
    calc_d = np.empty(n_events, dtype=np.int32)

    sd_y[0] = init_sd_dt.year
    sd_m[0] = init_sd_dt.month
    sd_d[0] = init_sd_dt.day

    for i in range(n_events):
        _ei, evt_dt, calc_dt = schedule[i]
        calc_y[i] = calc_dt.year
        calc_m[i] = calc_dt.month
        calc_d[i] = calc_dt.day
        if i < n_events - 1:
            sd_y[i + 1] = evt_dt.year
            sd_m[i + 1] = evt_dt.month
            sd_d[i + 1] = evt_dt.day

    if dcc_enum in (DayCountConvention.A360, DayCountConvention.A365):
        sd_ord = np_ymd_to_ordinal(sd_y, sd_m, sd_d)
        calc_ord = np_ymd_to_ordinal(calc_y, calc_m, calc_d)
        delta = (calc_ord - sd_ord).astype(np.float64)
        divisor = 360.0 if dcc_enum == DayCountConvention.A360 else 365.0
        return list((delta / divisor).tolist())
    if dcc_enum == DayCountConvention.E30360:
        return list(np_yf_30e360(sd_y, sd_m, sd_d, calc_y, calc_m, calc_d).tolist())
    if dcc_enum == DayCountConvention.B30360:
        return list(np_yf_b30360(sd_y, sd_m, sd_d, calc_y, calc_m, calc_d).tolist())
    # Fallback to scalar for AA, E30360ISDA, BUS252
    yf_list: list[float] = []
    current_sd_dt = init_sd_dt
    for _ei, evt_dt, calc_dt in schedule:
        yf_list.append(year_fraction(dt_to_adt(current_sd_dt), dt_to_adt(calc_dt), dcc_enum))
        current_sd_dt = evt_dt
    return yf_list


# ---------------------------------------------------------------------------
# Risk factor pre-query helpers
# ---------------------------------------------------------------------------


def prequery_risk_factors(
    schedule: list[tuple[int, _datetime, _datetime]],
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
    additional_rf_events: set[int] | None = None,
) -> list[float]:
    """Pre-query risk factors for all events in a schedule.

    By default handles RR (rate reset) and PP (prepayment) events.
    ``additional_rf_events`` can specify extra event types that need
    risk factor observation.

    Args:
        schedule: List of ``(evt_idx, evt_dt, calc_dt)`` tuples.
        attrs: Contract attributes.
        rf_observer: Risk factor observer.
        additional_rf_events: Optional set of extra event type indices
            that require risk factor observation.

    Returns:
        List of risk factor values (0.0 for events that don't need RF).
    """
    market_object = attrs.rate_reset_market_object or ""
    contract_id = attrs.contract_id or ""

    rf_list: list[float] = []
    for evt_idx, evt_dt, _calc_dt in schedule:
        rf_val = 0.0
        if evt_idx == RR_IDX:
            try:
                rf_val = float(rf_observer.observe_risk_factor(market_object, dt_to_adt(evt_dt)))
            except (KeyError, NotImplementedError, TypeError):
                rf_val = 0.0
        elif evt_idx == PP_IDX:
            try:
                rf_val = float(
                    rf_observer.observe_event(contract_id, EventType.PP, dt_to_adt(evt_dt))
                )
            except (KeyError, NotImplementedError, TypeError):
                rf_val = 0.0
        elif additional_rf_events and evt_idx in additional_rf_events:
            try:
                rf_val = float(rf_observer.observe_risk_factor(market_object, dt_to_adt(evt_dt)))
            except (KeyError, NotImplementedError, TypeError):
                rf_val = 0.0
        rf_list.append(rf_val)
    return rf_list


# ---------------------------------------------------------------------------
# Batch params extraction helper
# ---------------------------------------------------------------------------


def extract_batch_params(
    contracts: Sequence[tuple[ContractAttributes, object]],
    indices: list[int],
) -> BatchContractParams:
    """Extract schedule parameters into JAX arrays for batch processing."""
    from jactus.core.types import DayCountConvention

    dcc_map = {
        DayCountConvention.A360: DCC_A360,
        DayCountConvention.A365: DCC_A365,
        DayCountConvention.E30360: DCC_E30360,
        DayCountConvention.B30360: DCC_B30360,
    }

    n = len(indices)
    ied_y = np.empty(n, dtype=np.int32)
    ied_m = np.empty(n, dtype=np.int32)
    ied_d = np.empty(n, dtype=np.int32)
    md_y = np.empty(n, dtype=np.int32)
    md_m = np.empty(n, dtype=np.int32)
    md_d = np.empty(n, dtype=np.int32)
    sd_y = np.empty(n, dtype=np.int32)
    sd_m = np.empty(n, dtype=np.int32)
    sd_d = np.empty(n, dtype=np.int32)
    ip_anchor_y = np.empty(n, dtype=np.int32)
    ip_anchor_m = np.empty(n, dtype=np.int32)
    ip_anchor_d = np.empty(n, dtype=np.int32)
    cycle_months_arr = np.empty(n, dtype=np.int32)
    has_ip_cycle_arr = np.empty(n, dtype=np.int32)
    dcc_code_arr = np.empty(n, dtype=np.int32)

    for j, idx in enumerate(indices):
        attrs = contracts[idx][0]
        ied = attrs.initial_exchange_date
        md = attrs.maturity_date
        sd = attrs.status_date
        assert ied is not None
        assert md is not None

        ied_dt = adt_to_dt(ied)
        md_dt = adt_to_dt(md)
        sd_dt = adt_to_dt(sd)

        ied_y[j], ied_m[j], ied_d[j] = ied_dt.year, ied_dt.month, ied_dt.day
        md_y[j], md_m[j], md_d[j] = md_dt.year, md_dt.month, md_dt.day
        sd_y[j], sd_m[j], sd_d[j] = sd_dt.year, sd_dt.month, sd_dt.day

        ip_cycle = attrs.interest_payment_cycle
        if ip_cycle:
            has_ip_cycle_arr[j] = 1
            anchor = attrs.interest_payment_anchor or ied
            anchor_dt = adt_to_dt(anchor)
            ip_anchor_y[j] = anchor_dt.year
            ip_anchor_m[j] = anchor_dt.month
            ip_anchor_d[j] = anchor_dt.day
            mult, period, _stub = parse_cycle_fast(ip_cycle)
            cycle_months_arr[j] = mult * CYCLE_MONTHS_MAP[period]
        else:
            has_ip_cycle_arr[j] = 0
            ip_anchor_y[j] = ied_dt.year
            ip_anchor_m[j] = ied_dt.month
            ip_anchor_d[j] = ied_dt.day
            cycle_months_arr[j] = 12  # placeholder

        dcc = attrs.day_count_convention or DayCountConvention.A360
        dcc_code_arr[j] = dcc_map.get(dcc, DCC_A360)

    # Compute ordinals via NumPy, then transfer to JAX
    ied_ord = np_ymd_to_ordinal(ied_y, ied_m, ied_d).astype(np.int32)
    md_ord = np_ymd_to_ordinal(md_y, md_m, md_d).astype(np.int32)
    sd_ord = np_ymd_to_ordinal(sd_y, sd_m, sd_d).astype(np.int32)

    return BatchContractParams(
        ied_y=jnp.asarray(ied_y),
        ied_m=jnp.asarray(ied_m),
        ied_d=jnp.asarray(ied_d),
        ied_ord=jnp.asarray(ied_ord),
        md_ord=jnp.asarray(md_ord),
        sd_ord=jnp.asarray(sd_ord),
        ip_anchor_y=jnp.asarray(ip_anchor_y),
        ip_anchor_m=jnp.asarray(ip_anchor_m),
        ip_anchor_d=jnp.asarray(ip_anchor_d),
        cycle_months=jnp.asarray(cycle_months_arr),
        has_ip_cycle=jnp.asarray(has_ip_cycle_arr),
        dcc_code=jnp.asarray(dcc_code_arr),
    )
