"""Array-mode PAM simulation — JIT-compiled, vmap-able pure JAX.

This module provides a high-performance simulation path for PAM (Principal at
Maturity) contracts using ``jax.lax.scan`` for the event loop and
``jax.lax.switch`` for payoff/state-transition dispatch. The entire simulation
kernel is JIT-compilable and can be vectorized across a portfolio with
``jax.vmap``.

Architecture:
    Pre-computation (Python) → Pure JAX kernel (jit + vmap)

    The existing ``PrincipalAtMaturityContract`` generates event schedules and
    initializes state (Python-level, runs once per contract). This module
    converts the results to JAX arrays and runs the numerical simulation as a
    pure function.

Example::

    from jactus.contracts.pam_array import precompute_pam_arrays, simulate_pam_array

    arrays = precompute_pam_arrays(attrs, rf_observer)
    final_state, payoffs = simulate_pam_array(*arrays)

    # Portfolio:
    from jactus.contracts.pam_array import simulate_pam_portfolio
    result = simulate_pam_portfolio(contracts, discount_rate=0.05)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    EventType,
)
from jactus.core.types import NUM_EVENT_TYPES
from jactus.observers import RiskFactorObserver
from jactus.utilities.conventions import year_fraction

# ---------------------------------------------------------------------------
# NOP event index — used to pad shorter contracts in batched simulation.
# ---------------------------------------------------------------------------
NOP_EVENT_IDX: int = NUM_EVENT_TYPES  # 24 (one past the last valid EventType.index)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class PAMArrayState(NamedTuple):
    """Minimal scan-loop state for PAM simulation.

    All fields are scalar ``jnp.ndarray`` (float32). ``sd`` (status date) is
    omitted because year fractions are pre-computed before the JIT boundary.
    """

    nt: jnp.ndarray  # Notional principal (signed)
    ipnr: jnp.ndarray  # Nominal interest rate
    ipac: jnp.ndarray  # Accrued interest
    feac: jnp.ndarray  # Accrued fees
    nsc: jnp.ndarray  # Notional scaling multiplier
    isc: jnp.ndarray  # Interest scaling multiplier


class PAMArrayParams(NamedTuple):
    """Static contract parameters extracted from ``ContractAttributes``.

    These do not change during the scan loop.  Enum-based branches
    (penalty type, fee basis) are encoded as integers for ``jnp.where``.
    """

    role_sign: jnp.ndarray  # +1.0 or -1.0
    notional_principal: jnp.ndarray
    nominal_interest_rate: jnp.ndarray
    premium_discount_at_ied: jnp.ndarray
    accrued_interest: jnp.ndarray  # IPAC attribute (pre-computed for IED)
    fee_rate: jnp.ndarray
    fee_basis: jnp.ndarray  # 0=A, 1=N, 2=other
    penalty_rate: jnp.ndarray
    penalty_type: jnp.ndarray  # 0=A, 1=N, 2=I
    price_at_purchase_date: jnp.ndarray
    price_at_termination_date: jnp.ndarray
    rate_reset_spread: jnp.ndarray
    rate_reset_multiplier: jnp.ndarray
    rate_reset_floor: jnp.ndarray
    rate_reset_cap: jnp.ndarray
    rate_reset_next: jnp.ndarray
    has_rate_floor: jnp.ndarray  # 1.0 if floor is active, else 0.0
    has_rate_cap: jnp.ndarray  # 1.0 if cap is active, else 0.0
    ied_ipac: jnp.ndarray  # Pre-computed accrued interest at IED


# ============================================================================
# Pure JAX payoff functions  (state, params, yf, rf) → scalar payoff
# ============================================================================

_F32 = jnp.float32


def _pof_ad(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_ied(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return params.role_sign * (-1.0) * (params.notional_principal + params.premium_discount_at_ied)


def _pof_md(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return state.nsc * state.nt + state.isc * state.ipac + state.feac


def _pof_pp(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    # rf = pre-computed prepayment amount from observer
    return rf


def _pof_py(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    # penalty_type: 0=A, 1=N, 2=I
    pof_a = params.penalty_rate
    pof_ni = yf * state.nt * params.penalty_rate
    return jnp.where(params.penalty_type == 0, pof_a, pof_ni)


def _pof_fp(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    # fee_basis: 0=A, 1=N, 2=other
    pof_a = params.fee_rate
    pof_n = yf * state.nt * params.fee_rate + state.feac
    return jnp.where(
        params.fee_basis == 0,
        pof_a,
        jnp.where(params.fee_basis == 1, pof_n, state.feac),
    )


def _pof_prd(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return (-1.0) * (params.price_at_purchase_date + state.ipac + yf * state.ipnr * state.nt)


def _pof_td(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return params.price_at_termination_date + state.ipac + yf * state.ipnr * state.nt


def _pof_ip(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return state.isc * (state.ipac + yf * state.ipnr * state.nt)


def _pof_ipci(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_rr(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_rrf(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_sc(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_ce(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_noop(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


# ============================================================================
# Pure JAX state transition functions  (state, params, yf, rf) → new state
# ============================================================================


def _accrue_interest(state: PAMArrayState, yf: jnp.ndarray) -> jnp.ndarray:
    """Common sub-expression: ipac + yf * ipnr * nt."""
    return state.ipac + yf * state.ipnr * state.nt


def _stf_ad(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> PAMArrayState:
    return state._replace(ipac=_accrue_interest(state, yf))


def _stf_ied(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> PAMArrayState:
    return PAMArrayState(
        nt=params.role_sign * params.notional_principal,
        ipnr=params.nominal_interest_rate,
        ipac=params.ied_ipac,
        feac=jnp.array(0.0, dtype=_F32),
        nsc=jnp.array(1.0, dtype=_F32),
        isc=jnp.array(1.0, dtype=_F32),
    )


def _stf_md(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> PAMArrayState:
    return state._replace(
        nt=jnp.array(0.0, dtype=_F32),
        ipac=jnp.array(0.0, dtype=_F32),
        feac=jnp.array(0.0, dtype=_F32),
    )


def _stf_pp(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> PAMArrayState:
    new_ipac = _accrue_interest(state, yf)
    new_nt = state.nt - rf  # rf = prepayment amount
    return state._replace(nt=new_nt, ipac=new_ipac)


def _stf_py(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> PAMArrayState:
    return _stf_ad(state, params, yf, rf)


def _stf_fp(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> PAMArrayState:
    new_ipac = _accrue_interest(state, yf)
    return state._replace(ipac=new_ipac, feac=jnp.array(0.0, dtype=_F32))


def _stf_prd(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> PAMArrayState:
    return _stf_ad(state, params, yf, rf)


def _stf_td(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> PAMArrayState:
    return state._replace(
        nt=jnp.array(0.0, dtype=_F32),
        ipac=jnp.array(0.0, dtype=_F32),
        feac=jnp.array(0.0, dtype=_F32),
    )


def _stf_ip(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> PAMArrayState:
    return state._replace(ipac=jnp.array(0.0, dtype=_F32))


def _stf_ipci(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> PAMArrayState:
    total_ipac = _accrue_interest(state, yf)
    return state._replace(nt=state.nt + total_ipac, ipac=jnp.array(0.0, dtype=_F32))


def _stf_rr(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> PAMArrayState:
    new_ipac = _accrue_interest(state, yf)
    # rf = observed market rate
    raw_rate = params.rate_reset_multiplier * rf + params.rate_reset_spread
    # Branchless floor/cap using jnp.where + jnp.maximum/minimum
    clamped = jnp.where(
        params.has_rate_floor > 0.5,
        jnp.maximum(raw_rate, params.rate_reset_floor),
        raw_rate,
    )
    clamped = jnp.where(
        params.has_rate_cap > 0.5,
        jnp.minimum(clamped, params.rate_reset_cap),
        clamped,
    )
    return state._replace(ipnr=clamped, ipac=new_ipac)


def _stf_rrf(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> PAMArrayState:
    new_ipac = _accrue_interest(state, yf)
    return state._replace(ipnr=params.rate_reset_next, ipac=new_ipac)


def _stf_sc(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> PAMArrayState:
    return _stf_ad(state, params, yf, rf)


def _stf_ce(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> PAMArrayState:
    return _stf_ad(state, params, yf, rf)


def _stf_noop(
    state: PAMArrayState, params: PAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> PAMArrayState:
    return state


# ============================================================================
# Dispatch tables — indexed by EventType.index (0..23) + NOP (24)
# ============================================================================

# fmt: off
_POF_TABLE: list[Any] = [
    _pof_ad,    # 0  AD
    _pof_ied,   # 1  IED
    _pof_md,    # 2  MD
    _pof_noop,  # 3  PR   (not used in PAM)
    _pof_noop,  # 4  PI   (not used in PAM)
    _pof_pp,    # 5  PP
    _pof_py,    # 6  PY
    _pof_noop,  # 7  PRF  (not used in PAM)
    _pof_fp,    # 8  FP
    _pof_prd,   # 9  PRD
    _pof_td,    # 10 TD
    _pof_ip,    # 11 IP
    _pof_ipci,  # 12 IPCI
    _pof_noop,  # 13 IPCB (not used in PAM)
    _pof_rr,    # 14 RR
    _pof_rrf,   # 15 RRF
    _pof_noop,  # 16 DV   (not used in PAM)
    _pof_noop,  # 17 DVF  (not used in PAM)
    _pof_sc,    # 18 SC
    _pof_noop,  # 19 STD  (not used in PAM)
    _pof_noop,  # 20 XD   (not used in PAM)
    _pof_ce,    # 21 CE
    _pof_noop,  # 22 IPFX (not used in PAM)
    _pof_noop,  # 23 IPFL (not used in PAM)
    _pof_noop,  # 24 NOP  (padding)
]

_STF_TABLE: list[Any] = [
    _stf_ad,    # 0  AD
    _stf_ied,   # 1  IED
    _stf_md,    # 2  MD
    _stf_noop,  # 3  PR
    _stf_noop,  # 4  PI
    _stf_pp,    # 5  PP
    _stf_py,    # 6  PY
    _stf_noop,  # 7  PRF
    _stf_fp,    # 8  FP
    _stf_prd,   # 9  PRD
    _stf_td,    # 10 TD
    _stf_ip,    # 11 IP
    _stf_ipci,  # 12 IPCI
    _stf_noop,  # 13 IPCB
    _stf_rr,    # 14 RR
    _stf_rrf,   # 15 RRF
    _stf_noop,  # 16 DV
    _stf_noop,  # 17 DVF
    _stf_sc,    # 18 SC
    _stf_noop,  # 19 STD
    _stf_noop,  # 20 XD
    _stf_ce,    # 21 CE
    _stf_noop,  # 22 IPFX
    _stf_noop,  # 23 IPFL
    _stf_noop,  # 24 NOP
]
# fmt: on

assert len(_POF_TABLE) == NOP_EVENT_IDX + 1
assert len(_STF_TABLE) == NOP_EVENT_IDX + 1


# ============================================================================
# JIT-compiled simulation kernel
# ============================================================================


def simulate_pam_array(
    initial_state: PAMArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: PAMArrayParams,
) -> tuple[PAMArrayState, jnp.ndarray]:
    """Run a PAM simulation as a pure JAX function.

    This function is JIT-compilable and vmap-able.

    Args:
        initial_state: Starting state (6 scalar fields).
        event_types: ``(num_events,)`` int32 — ``EventType.index`` values.
        year_fractions: ``(num_events,)`` float32 — pre-computed YF per event.
        rf_values: ``(num_events,)`` float32 — pre-computed risk factor values
            (market rate for RR, prepayment amount for PP, 0.0 otherwise).
        params: Static contract parameters.

    Returns:
        ``(final_state, payoffs)`` where payoffs is ``(num_events,)`` float32.
    """

    def step(
        state: PAMArrayState, inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> tuple[PAMArrayState, jnp.ndarray]:
        evt_idx, yf, rf = inputs
        payoff = jax.lax.switch(evt_idx, _POF_TABLE, state, params, yf, rf)
        new_state = jax.lax.switch(evt_idx, _STF_TABLE, state, params, yf, rf)
        return new_state, payoff

    final_state, payoffs = jax.lax.scan(
        step, initial_state, (event_types, year_fractions, rf_values)
    )
    return final_state, payoffs


# JIT-compiled version for single-contract use
simulate_pam_array_jit = jax.jit(simulate_pam_array)

# Vmapped version for batched portfolio simulation
batch_simulate_pam = jax.vmap(simulate_pam_array)


# ============================================================================
# Pre-computation bridge — Python → JAX arrays
# ============================================================================


def _encode_fee_basis(attrs: ContractAttributes) -> int:
    """Encode fee basis as int: 0=A, 1=N, 2=other."""
    from jactus.core.types import FeeBasis

    if attrs.fee_basis == FeeBasis.A:
        return 0
    if attrs.fee_basis == FeeBasis.N:
        return 1
    return 2


def _encode_penalty_type(attrs: ContractAttributes) -> int:
    """Encode penalty type as int: 0=A, 1=N, 2=I."""
    pt = attrs.penalty_type
    if pt == "A":
        return 0
    if pt == "N":
        return 1
    if pt == "I":
        return 2
    return 2  # default


def _get_role_sign(role: ContractRole | None) -> float:
    """Get +1.0 or -1.0 for the contract role."""
    if role in (ContractRole.RPA, ContractRole.RFL):
        return 1.0
    if role in (ContractRole.RPL, ContractRole.PFL):
        return -1.0
    return 1.0


def _extract_params(attrs: ContractAttributes) -> PAMArrayParams:
    """Extract ``PAMArrayParams`` from ``ContractAttributes``."""
    role_sign = _get_role_sign(attrs.contract_role)
    nt = attrs.notional_principal or 0.0
    ipnr = attrs.nominal_interest_rate or 0.0

    # Pre-compute IED accrued interest (same logic as _stf_ied)
    ied_ipac = 0.0
    if attrs.accrued_interest is not None:
        ied_ipac = attrs.accrued_interest
    elif (
        attrs.interest_payment_anchor is not None
        and attrs.initial_exchange_date is not None
        and attrs.interest_payment_anchor < attrs.initial_exchange_date
    ):
        from jactus.core.types import DayCountConvention

        dcc = attrs.day_count_convention or DayCountConvention.A360
        yf = year_fraction(attrs.interest_payment_anchor, attrs.initial_exchange_date, dcc)
        ied_ipac = yf * ipnr * abs(role_sign * nt)

    has_floor = attrs.rate_reset_floor is not None
    has_cap = attrs.rate_reset_cap is not None

    return PAMArrayParams(
        role_sign=jnp.array(role_sign, dtype=_F32),
        notional_principal=jnp.array(nt, dtype=_F32),
        nominal_interest_rate=jnp.array(ipnr, dtype=_F32),
        premium_discount_at_ied=jnp.array(attrs.premium_discount_at_ied or 0.0, dtype=_F32),
        accrued_interest=jnp.array(attrs.accrued_interest or 0.0, dtype=_F32),
        fee_rate=jnp.array(attrs.fee_rate or 0.0, dtype=_F32),
        fee_basis=jnp.array(_encode_fee_basis(attrs), dtype=jnp.int32),
        penalty_rate=jnp.array(attrs.penalty_rate or 0.0, dtype=_F32),
        penalty_type=jnp.array(_encode_penalty_type(attrs), dtype=jnp.int32),
        price_at_purchase_date=jnp.array(attrs.price_at_purchase_date or 0.0, dtype=_F32),
        price_at_termination_date=jnp.array(attrs.price_at_termination_date or 0.0, dtype=_F32),
        rate_reset_spread=jnp.array(attrs.rate_reset_spread or 0.0, dtype=_F32),
        rate_reset_multiplier=jnp.array(
            attrs.rate_reset_multiplier if attrs.rate_reset_multiplier is not None else 1.0,
            dtype=_F32,
        ),
        rate_reset_floor=jnp.array(attrs.rate_reset_floor or 0.0, dtype=_F32),
        rate_reset_cap=jnp.array(attrs.rate_reset_cap or 1.0, dtype=_F32),
        rate_reset_next=jnp.array(
            attrs.rate_reset_next if attrs.rate_reset_next is not None else ipnr,
            dtype=_F32,
        ),
        has_rate_floor=jnp.array(1.0 if has_floor else 0.0, dtype=_F32),
        has_rate_cap=jnp.array(1.0 if has_cap else 0.0, dtype=_F32),
        ied_ipac=jnp.array(ied_ipac, dtype=_F32),
    )


def precompute_pam_arrays(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> tuple[PAMArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, PAMArrayParams]:
    """Pre-compute JAX arrays for array-mode PAM simulation.

    Uses the existing ``PrincipalAtMaturityContract`` to generate the event
    schedule and initial state, then converts to JAX arrays suitable for
    ``simulate_pam_array``.

    Args:
        attrs: Contract attributes (must be PAM type).
        rf_observer: Risk factor observer (queried for RR/PP events).

    Returns:
        ``(initial_state, event_types, year_fractions, rf_values, params)``
    """
    from jactus.contracts.pam import PrincipalAtMaturityContract
    from jactus.core.types import DayCountConvention

    # Use existing contract to generate schedule and initial state
    contract = PrincipalAtMaturityContract(attrs, rf_observer)
    schedule = contract.generate_event_schedule()
    init_state = contract.initialize_state()

    dcc = attrs.day_count_convention or DayCountConvention.A360

    # Convert to arrays
    event_type_list: list[int] = []
    yf_list: list[float] = []
    rf_list: list[float] = []

    # Track sd for year fraction computation
    current_sd: ActusDateTime = init_state.sd

    for event in schedule.events:
        # Event type index
        event_type_list.append(event.event_type.index)

        # Year fraction from current sd to event time
        calc_time = event.calculation_time or event.event_time
        yf = year_fraction(current_sd, calc_time, dcc)
        yf_list.append(yf)

        # Pre-query risk factor for RR events
        rf_val = 0.0
        if event.event_type == EventType.RR:
            market_object = attrs.rate_reset_market_object or ""
            try:
                rf_val = float(rf_observer.observe_risk_factor(market_object, event.event_time))
            except (KeyError, NotImplementedError, TypeError):
                rf_val = 0.0
        elif event.event_type == EventType.PP:
            try:
                rf_val = float(
                    rf_observer.observe_event(
                        attrs.contract_id or "",
                        EventType.PP,
                        event.event_time,
                    )
                )
            except (KeyError, NotImplementedError, TypeError):
                rf_val = 0.0
        rf_list.append(rf_val)

        # All PAM STFs set sd = time
        current_sd = event.event_time

    # Convert ContractState → PAMArrayState
    initial_array_state = PAMArrayState(
        nt=jnp.array(float(init_state.nt), dtype=_F32),
        ipnr=jnp.array(float(init_state.ipnr), dtype=_F32),
        ipac=jnp.array(float(init_state.ipac), dtype=_F32),
        feac=jnp.array(float(init_state.feac), dtype=_F32),
        nsc=jnp.array(float(init_state.nsc), dtype=_F32),
        isc=jnp.array(float(init_state.isc), dtype=_F32),
    )

    params = _extract_params(attrs)

    return (
        initial_array_state,
        jnp.array(event_type_list, dtype=jnp.int32),
        jnp.array(yf_list, dtype=_F32),
        jnp.array(rf_list, dtype=_F32),
        params,
    )


# ============================================================================
# Batch / portfolio API
# ============================================================================


def _pad_arrays(
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
    mask = jnp.concatenate([jnp.ones(n, dtype=_F32), jnp.zeros(pad_n, dtype=_F32)])
    event_types = jnp.concatenate([event_types, jnp.full(pad_n, NOP_EVENT_IDX, dtype=jnp.int32)])
    year_fractions = jnp.concatenate([year_fractions, jnp.zeros(pad_n, dtype=_F32)])
    rf_values = jnp.concatenate([rf_values, jnp.zeros(pad_n, dtype=_F32)])
    return event_types, year_fractions, rf_values, mask


def prepare_pam_batch(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
) -> tuple[PAMArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, PAMArrayParams, jnp.ndarray]:
    """Pre-compute and pad arrays for a batch of PAM contracts.

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.

    Returns:
        ``(initial_states, event_types, year_fractions, rf_values, params, masks)``
        where each array has a leading batch dimension.
    """
    precomputed = [precompute_pam_arrays(attrs, obs) for attrs, obs in contracts]
    max_events = max(et.shape[0] for _, et, _, _, _ in precomputed)

    # Pad and stack
    states_list = []
    et_list = []
    yf_list = []
    rf_list = []
    params_list = []
    mask_list = []

    for init_state, event_types, year_fracs, rf_vals, params in precomputed:
        et_padded, yf_padded, rf_padded, mask = _pad_arrays(
            event_types, year_fracs, rf_vals, max_events
        )
        states_list.append(init_state)
        et_list.append(et_padded)
        yf_list.append(yf_padded)
        rf_list.append(rf_padded)
        params_list.append(params)
        mask_list.append(mask)

    # Stack into batched arrays
    def _stack_named_tuples(tuples: list[Any], cls: Any) -> Any:
        """Stack a list of NamedTuples into a single NamedTuple of batched arrays."""
        fields: dict[str, jnp.ndarray] = {}
        for field_name in cls._fields:
            fields[field_name] = jnp.stack([getattr(t, field_name) for t in tuples])
        return cls(**fields)

    batched_states = _stack_named_tuples(states_list, PAMArrayState)
    batched_params = _stack_named_tuples(params_list, PAMArrayParams)
    batched_et = jnp.stack(et_list)
    batched_yf = jnp.stack(yf_list)
    batched_rf = jnp.stack(rf_list)
    batched_masks = jnp.stack(mask_list)

    return batched_states, batched_et, batched_yf, batched_rf, batched_params, batched_masks


def simulate_pam_portfolio(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
    discount_rate: float | None = None,
    year_fractions_from_valuation: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """End-to-end portfolio simulation with optional PV.

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.
        discount_rate: If provided, compute present values.
        year_fractions_from_valuation: ``(batch, max_events)`` year fractions
            from valuation date for PV discounting.  If ``None`` and
            ``discount_rate`` is set, year fractions are computed from each
            contract's ``status_date``.

    Returns:
        Dict with ``payoffs``, ``masks``, ``final_states``, and optionally
        ``present_values`` and ``total_pv``.
    """
    (
        batched_states,
        batched_et,
        batched_yf,
        batched_rf,
        batched_params,
        batched_masks,
    ) = prepare_pam_batch(contracts)

    # Run batched simulation
    final_states, payoffs = batch_simulate_pam(
        batched_states, batched_et, batched_yf, batched_rf, batched_params
    )

    # Mask padding
    masked_payoffs = payoffs * batched_masks
    total_cashflows = jnp.sum(masked_payoffs, axis=1)

    result = {
        "payoffs": masked_payoffs,
        "masks": batched_masks,
        "final_states": final_states,
        "total_cashflows": total_cashflows,
        "num_contracts": len(contracts),
    }

    if discount_rate is not None:
        # Use cumulative year fractions for discounting
        if year_fractions_from_valuation is not None:
            disc_yfs = year_fractions_from_valuation
        else:
            # Approximate: use cumulative sum of per-event year fractions
            disc_yfs = jnp.cumsum(batched_yf, axis=1)
        discount_factors = 1.0 / (1.0 + discount_rate * disc_yfs)
        pvs = jnp.sum(masked_payoffs * discount_factors, axis=1)
        result["present_values"] = pvs
        result["total_pv"] = jnp.sum(pvs)

    return result
