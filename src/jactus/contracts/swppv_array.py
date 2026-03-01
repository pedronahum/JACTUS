"""Array-mode SWPPV simulation — JIT-compiled, vmap-able pure JAX.

This module provides a high-performance simulation path for SWPPV (Plain Vanilla
Interest Rate Swap) contracts using ``jax.lax.scan`` for the event loop and
``jax.lax.switch`` for payoff/state-transition dispatch.  The entire simulation
kernel is JIT-compilable and can be vectorized across a portfolio with
``jax.vmap``.

Architecture:
    Pre-computation (Python) -> Pure JAX kernel (jit + vmap)

    The existing ``PlainVanillaSwapContract`` generates event schedules and
    initializes state (Python-level, runs once per contract).  This module
    converts the results to JAX arrays and runs the numerical simulation as
    a pure function.

Key SWPPV specifics:
    - Two legs: fixed (uses ``fixed_rate`` from params) and floating (uses
      ``ipnr`` from state, updated by RR events).
    - State tracks dual accruals: ``ipac1`` (fixed leg) and ``ipac2`` (floating).
    - Net IP payoff: ``role_sign * nsc * isc * ((ipac1 + yf*fixed_rate*nt) -
      (ipac2 + yf*ipnr*nt))``
    - IPFX payoff (separate): ``role_sign * (ipac1 + yf*fixed_rate*nt)``
    - IPFL payoff (separate): ``-role_sign * (ipac2 + yf*ipnr*nt)``
    - No notional exchange at IED or MD (payoff = 0).
    - RR: accrues both legs, then updates floating rate.

Example::

    from jactus.contracts.swppv_array import precompute_swppv_arrays, simulate_swppv_array

    arrays = precompute_swppv_arrays(attrs, rf_observer)
    final_state, payoffs = simulate_swppv_array(*arrays)

    # Portfolio:
    from jactus.contracts.swppv_array import simulate_swppv_portfolio
    result = simulate_swppv_portfolio(contracts, discount_rate=0.05)
"""

from __future__ import annotations

from datetime import datetime as _datetime
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jactus.core import (
    ActusDateTime,
    ContractAttributes,
    ContractRole,
    EventType,
)
from jactus.core.types import NUM_EVENT_TYPES
from jactus.observers import RiskFactorObserver
from jactus.utilities.conventions import year_fraction

# Import shared infrastructure from array_common
from jactus.contracts.array_common import (
    NOP_EVENT_IDX,
    F32 as _F32,
    USE_DATE_ARRAY as _USE_DATE_ARRAY,
    # Cached EventType indices
    AD_IDX as _AD_IDX,
    IED_IDX as _IED_IDX,
    MD_IDX as _MD_IDX,
    PRD_IDX as _PRD_IDX,
    TD_IDX as _TD_IDX,
    IP_IDX as _IP_IDX,
    RR_IDX as _RR_IDX,
    RRF_IDX as _RRF_IDX,
    CE_IDX as _CE_IDX,
    # Encoding helpers
    get_role_sign as _get_role_sign,
    # Date helpers
    adt_to_dt as _adt_to_dt,
    dt_to_adt as _dt_to_adt,
    # Schedule helpers
    CYCLE_MONTHS_MAP as _CYCLE_MONTHS_MAP,
    parse_cycle_fast as _parse_cycle_fast,
    fast_schedule as _fast_schedule,
    get_evt_priority as _get_evt_priority,
    get_yf_fn,
    # Batch infrastructure
    RawPrecomputed as _RawPrecomputed,
    pad_arrays as _pad_arrays,
    compute_vectorised_year_fractions as _compute_vectorised_year_fractions,
    prequery_risk_factors as _prequery_risk_factors,
)

# IPFX/IPFL indices (not in array_common, define locally)
_IPFX_IDX = EventType.IPFX.index  # 22
_IPFL_IDX = EventType.IPFL.index  # 23


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class SWPPVArrayState(NamedTuple):
    """Minimal scan-loop state for SWPPV simulation.

    All fields are scalar ``jnp.ndarray`` (float32). ``sd`` (status date) is
    omitted because year fractions are pre-computed before the JIT boundary.
    """

    nt: jnp.ndarray  # Notional principal
    ipnr: jnp.ndarray  # Floating interest rate (updated by RR)
    ipac1: jnp.ndarray  # Fixed leg accrued interest
    ipac2: jnp.ndarray  # Floating leg accrued interest
    nsc: jnp.ndarray  # Notional scaling multiplier
    isc: jnp.ndarray  # Interest scaling multiplier


class SWPPVArrayParams(NamedTuple):
    """Static contract parameters extracted from ``ContractAttributes``.

    These do not change during the scan loop.
    """

    role_sign: jnp.ndarray  # +1.0 or -1.0
    notional_principal: jnp.ndarray
    fixed_rate: jnp.ndarray  # nominal_interest_rate (fixed leg)
    rate_reset_spread: jnp.ndarray
    rate_reset_multiplier: jnp.ndarray
    rate_reset_floor: jnp.ndarray
    rate_reset_cap: jnp.ndarray
    has_rate_floor: jnp.ndarray  # 1.0 if floor is active, else 0.0
    has_rate_cap: jnp.ndarray  # 1.0 if cap is active, else 0.0
    price_at_purchase_date: jnp.ndarray
    price_at_termination_date: jnp.ndarray


# ============================================================================
# Pure JAX payoff functions  (state, params, yf, rf) -> scalar payoff
# ============================================================================


def _pof_ad(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_AD_SWPPV: Analysis Date — zero payoff."""
    return jnp.array(0.0, dtype=_F32)


def _pof_ied(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_IED_SWPPV: No notional exchange for swaps."""
    return jnp.array(0.0, dtype=_F32)


def _pof_md(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_MD_SWPPV: No notional exchange at maturity."""
    return jnp.array(0.0, dtype=_F32)


def _pof_prd(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_PRD_SWPPV: Purchase Date payoff.

    Formula: POF_PRD = role_sign * (-PPRD)
    """
    return params.role_sign * (-params.price_at_purchase_date)


def _pof_td(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_TD_SWPPV: Termination Date payoff.

    For SWPPV, PTD is the mark-to-market settlement amount.
    """
    return params.price_at_termination_date


def _pof_ip(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_IP_SWPPV: Net Interest Payment.

    Net settlement: payoff = role_sign * ((ipac1 + yf*fixed*nt) - (ipac2 + yf*float*nt))
    """
    total_fixed = state.ipac1 + yf * params.fixed_rate * state.nt
    total_float = state.ipac2 + yf * state.ipnr * state.nt
    net_accrual = total_fixed - total_float
    return params.role_sign * net_accrual


def _pof_ipfx(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_IPFX_SWPPV: Fixed Leg Interest Payment.

    Formula: role_sign * (ipac1 + yf * fixed_rate * nt)
    """
    total_fixed = state.ipac1 + yf * params.fixed_rate * state.nt
    return params.role_sign * total_fixed


def _pof_ipfl(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_IPFL_SWPPV: Floating Leg Interest Payment.

    Formula: -role_sign * (ipac2 + yf * ipnr * nt)
    """
    total_float = state.ipac2 + yf * state.ipnr * state.nt
    return -params.role_sign * total_float


def _pof_rr(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_RR_SWPPV: Rate Reset — zero payoff."""
    return jnp.array(0.0, dtype=_F32)


def _pof_ce(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_CE_SWPPV: Credit Event — zero payoff."""
    return jnp.array(0.0, dtype=_F32)


def _pof_noop(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """No-op payoff for unused event types and padding."""
    return jnp.array(0.0, dtype=_F32)


# ============================================================================
# Pure JAX state transition functions  (state, params, yf, rf) -> new state
# ============================================================================


def _accrue_both_legs(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Accrue interest for both fixed and floating legs.

    Returns (new_ipac1, new_ipac2).
    """
    new_ipac1 = state.ipac1 + yf * params.fixed_rate * state.nt
    new_ipac2 = state.ipac2 + yf * state.ipnr * state.nt
    return new_ipac1, new_ipac2


def _stf_ad(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> SWPPVArrayState:
    """STF_AD_SWPPV: Accrue both legs."""
    new_ipac1, new_ipac2 = _accrue_both_legs(state, params, yf)
    return state._replace(ipac1=new_ipac1, ipac2=new_ipac2)


def _stf_ied(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> SWPPVArrayState:
    """STF_IED_SWPPV: Initialize state at IED.

    Sets nt, initialises ipnr to the initial floating rate (from state, set at
    pre-computation), resets accruals, sets nsc=1, isc=1.
    """
    return SWPPVArrayState(
        nt=params.notional_principal,
        ipnr=state.ipnr,  # Preserve initial floating rate from pre-computation
        ipac1=jnp.array(0.0, dtype=_F32),
        ipac2=jnp.array(0.0, dtype=_F32),
        nsc=jnp.array(1.0, dtype=_F32),
        isc=jnp.array(1.0, dtype=_F32),
    )


def _stf_prd(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> SWPPVArrayState:
    """STF_PRD_SWPPV: Accrue both legs at purchase date."""
    new_ipac1, new_ipac2 = _accrue_both_legs(state, params, yf)
    return state._replace(ipac1=new_ipac1, ipac2=new_ipac2)


def _stf_md(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> SWPPVArrayState:
    """STF_MD_SWPPV: Maturity — zero notional and accruals."""
    return state._replace(
        nt=jnp.array(0.0, dtype=_F32),
        ipac1=jnp.array(0.0, dtype=_F32),
        ipac2=jnp.array(0.0, dtype=_F32),
    )


def _stf_td(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> SWPPVArrayState:
    """STF_TD_SWPPV: Termination — zero notional and accruals."""
    return state._replace(
        nt=jnp.array(0.0, dtype=_F32),
        ipac1=jnp.array(0.0, dtype=_F32),
        ipac2=jnp.array(0.0, dtype=_F32),
    )


def _stf_ip(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> SWPPVArrayState:
    """STF_IP_SWPPV: Net IP — reset both accruals to zero."""
    return state._replace(
        ipac1=jnp.array(0.0, dtype=_F32),
        ipac2=jnp.array(0.0, dtype=_F32),
    )


def _stf_ipfx(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> SWPPVArrayState:
    """STF_IPFX_SWPPV: Fixed leg payment — accrue floating, reset fixed accrual.

    The floating leg continues to accrue; the fixed leg resets.
    """
    new_ipac2 = state.ipac2 + yf * state.ipnr * state.nt
    return state._replace(
        ipac1=jnp.array(0.0, dtype=_F32),
        ipac2=new_ipac2,
    )


def _stf_ipfl(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> SWPPVArrayState:
    """STF_IPFL_SWPPV: Floating leg payment — reset floating accrual.

    IPFL follows IPFX on same date, so fixed accrual is already reset.
    """
    return state._replace(
        ipac2=jnp.array(0.0, dtype=_F32),
    )


def _stf_rr(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> SWPPVArrayState:
    """STF_RR_SWPPV: Rate Reset — accrue both legs, then update floating rate.

    Formula: ipnr = clamp(multiplier * rf + spread, floor, cap)
    """
    new_ipac1, new_ipac2 = _accrue_both_legs(state, params, yf)
    raw_rate = params.rate_reset_multiplier * rf + params.rate_reset_spread
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
    return state._replace(ipnr=clamped, ipac1=new_ipac1, ipac2=new_ipac2)


def _stf_ce(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> SWPPVArrayState:
    """STF_CE_SWPPV: Credit Event — no state change."""
    return state


def _stf_noop(
    state: SWPPVArrayState, params: SWPPVArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> SWPPVArrayState:
    """No-op state transition for unused event types and padding."""
    return state


# ============================================================================
# Dispatch tables — indexed by EventType.index (0..23) + NOP (24)
# ============================================================================

# fmt: off
_POF_TABLE: list[Any] = [
    _pof_ad,    # 0  AD
    _pof_ied,   # 1  IED
    _pof_md,    # 2  MD
    _pof_noop,  # 3  PR   (not used in SWPPV)
    _pof_noop,  # 4  PI   (not used in SWPPV)
    _pof_noop,  # 5  PP   (not used in SWPPV)
    _pof_noop,  # 6  PY   (not used in SWPPV)
    _pof_noop,  # 7  PRF  (not used in SWPPV)
    _pof_noop,  # 8  FP   (not used in SWPPV)
    _pof_prd,   # 9  PRD
    _pof_td,    # 10 TD
    _pof_ip,    # 11 IP
    _pof_noop,  # 12 IPCI (not used in SWPPV)
    _pof_noop,  # 13 IPCB (not used in SWPPV)
    _pof_rr,    # 14 RR
    _pof_noop,  # 15 RRF  (not used in SWPPV)
    _pof_noop,  # 16 DV   (not used in SWPPV)
    _pof_noop,  # 17 DVF  (not used in SWPPV)
    _pof_noop,  # 18 SC   (not used in SWPPV)
    _pof_noop,  # 19 STD  (not used in SWPPV)
    _pof_noop,  # 20 XD   (not used in SWPPV)
    _pof_ce,    # 21 CE
    _pof_ipfx,  # 22 IPFX
    _pof_ipfl,  # 23 IPFL
    _pof_noop,  # 24 NOP  (padding)
]

_STF_TABLE: list[Any] = [
    _stf_ad,    # 0  AD
    _stf_ied,   # 1  IED
    _stf_md,    # 2  MD
    _stf_noop,  # 3  PR
    _stf_noop,  # 4  PI
    _stf_noop,  # 5  PP
    _stf_noop,  # 6  PY
    _stf_noop,  # 7  PRF
    _stf_noop,  # 8  FP
    _stf_prd,   # 9  PRD
    _stf_td,    # 10 TD
    _stf_ip,    # 11 IP
    _stf_noop,  # 12 IPCI
    _stf_noop,  # 13 IPCB
    _stf_rr,    # 14 RR
    _stf_noop,  # 15 RRF
    _stf_noop,  # 16 DV
    _stf_noop,  # 17 DVF
    _stf_noop,  # 18 SC
    _stf_noop,  # 19 STD
    _stf_noop,  # 20 XD
    _stf_ce,    # 21 CE
    _stf_ipfx,  # 22 IPFX
    _stf_ipfl,  # 23 IPFL
    _stf_noop,  # 24 NOP
]
# fmt: on

assert len(_POF_TABLE) == NOP_EVENT_IDX + 1
assert len(_STF_TABLE) == NOP_EVENT_IDX + 1


# ============================================================================
# JIT-compiled simulation kernel
# ============================================================================


def simulate_swppv_array(
    initial_state: SWPPVArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: SWPPVArrayParams,
) -> tuple[SWPPVArrayState, jnp.ndarray]:
    """Run a SWPPV simulation as a pure JAX function.

    This function is JIT-compilable and vmap-able.

    Args:
        initial_state: Starting state (6 scalar fields).
        event_types: ``(num_events,)`` int32 -- ``EventType.index`` values.
        year_fractions: ``(num_events,)`` float32 -- pre-computed YF per event.
        rf_values: ``(num_events,)`` float32 -- pre-computed risk factor values
            (market rate for RR, 0.0 otherwise).
        params: Static contract parameters.

    Returns:
        ``(final_state, payoffs)`` where payoffs is ``(num_events,)`` float32.
    """

    def step(
        state: SWPPVArrayState, inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> tuple[SWPPVArrayState, jnp.ndarray]:
        evt_idx, yf, rf = inputs
        payoff = jax.lax.switch(evt_idx, _POF_TABLE, state, params, yf, rf)
        new_state = jax.lax.switch(evt_idx, _STF_TABLE, state, params, yf, rf)
        return new_state, payoff

    final_state, payoffs = jax.lax.scan(
        step, initial_state, (event_types, year_fractions, rf_values), unroll=8
    )
    return final_state, payoffs


# JIT-compiled version for single-contract use
simulate_swppv_array_jit = jax.jit(simulate_swppv_array)

# Vmapped version (kept as fallback)
batch_simulate_swppv_vmap = jax.vmap(simulate_swppv_array)


def batch_simulate_swppv_auto(
    initial_states: SWPPVArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: SWPPVArrayParams,
) -> tuple[SWPPVArrayState, jnp.ndarray]:
    """Batched simulation using the optimal strategy for all backends.

    Uses the single-scan batch approach (``batch_simulate_swppv``) which
    processes all contracts in shaped ``[B, T]`` arrays via a single
    ``lax.scan``.
    """
    return batch_simulate_swppv(initial_states, event_types, year_fractions, rf_values, params)  # type: ignore[no-any-return]


# ============================================================================
# Manually-batched simulation — eliminates vmap dispatch overhead on CPU
# ============================================================================


@jax.jit
def batch_simulate_swppv(
    initial_states: SWPPVArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: SWPPVArrayParams,
) -> tuple[SWPPVArrayState, jnp.ndarray]:
    """Batched SWPPV simulation without vmap — single scan over ``[B]`` arrays.

    Eliminates JAX vmap CPU dispatch overhead by operating directly on
    batch-dimensioned arrays.

    Args:
        initial_states: ``SWPPVArrayState`` with each field shape ``[B]``.
        event_types: ``[B, T]`` int32 -- event type indices per contract.
        year_fractions: ``[B, T]`` float32.
        rf_values: ``[B, T]`` float32.
        params: ``SWPPVArrayParams`` with each field shape ``[B]``.

    Returns:
        ``(final_states, payoffs)`` where ``payoffs`` is ``[B, T]``.
    """
    # Transpose to [T, B] so scan iterates over time steps
    et_t = event_types.T
    yf_t = year_fractions.T
    rf_t = rf_values.T

    def step(
        states: SWPPVArrayState,
        inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[SWPPVArrayState, jnp.ndarray]:
        et, yf, rf = inputs  # each [B]

        # Common sub-expressions: accrual for both legs
        accrue_fixed = states.ipac1 + yf * params.fixed_rate * states.nt
        accrue_float = states.ipac2 + yf * states.ipnr * states.nt

        # ---- Payoffs (branchless jnp.where dispatch) ----
        payoff = jnp.zeros_like(states.nt)

        # IED: zero (no notional exchange)
        # MD: zero (no notional exchange at maturity)
        # PRD: role_sign * (-PPRD)
        payoff = jnp.where(
            et == _PRD_IDX,
            params.role_sign * (-params.price_at_purchase_date),
            payoff,
        )
        # TD: PTD (mark-to-market)
        payoff = jnp.where(et == _TD_IDX, params.price_at_termination_date, payoff)
        # IP (net settlement): role_sign * (fixed_accrual - floating_accrual)
        payoff = jnp.where(
            et == _IP_IDX,
            params.role_sign * (accrue_fixed - accrue_float),
            payoff,
        )
        # IPFX (separate settlement): role_sign * fixed_accrual
        payoff = jnp.where(
            et == _IPFX_IDX,
            params.role_sign * accrue_fixed,
            payoff,
        )
        # IPFL (separate settlement): -role_sign * floating_accrual
        payoff = jnp.where(
            et == _IPFL_IDX,
            -params.role_sign * accrue_float,
            payoff,
        )

        # ---- State transitions (branchless) ----

        # nt: set at IED, zero at MD/TD
        new_nt = states.nt
        new_nt = jnp.where(et == _IED_IDX, params.notional_principal, new_nt)
        new_nt = jnp.where((et == _MD_IDX) | (et == _TD_IDX), 0.0, new_nt)

        # ipnr (floating rate): default unchanged
        new_ipnr = states.ipnr
        # RR: clamp(multiplier * rf + spread, floor, cap)
        raw_rate = params.rate_reset_multiplier * rf + params.rate_reset_spread
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
        new_ipnr = jnp.where(et == _RR_IDX, clamped, new_ipnr)

        # ipac1 (fixed leg accrual):
        #   accrue group: AD, PRD, RR, CE  -> accrue_fixed
        #   zero group:   IP, IPFX, MD, TD -> 0
        #   IED:          0
        #   default:      unchanged
        is_accrue_ipac1 = (
            (et == _AD_IDX) | (et == _PRD_IDX) | (et == _RR_IDX) | (et == _CE_IDX)
        )
        is_zero_ipac1 = (
            (et == _IP_IDX) | (et == _IPFX_IDX) | (et == _MD_IDX) | (et == _TD_IDX)
        )
        new_ipac1 = states.ipac1
        new_ipac1 = jnp.where(is_accrue_ipac1, accrue_fixed, new_ipac1)
        new_ipac1 = jnp.where(is_zero_ipac1, 0.0, new_ipac1)
        new_ipac1 = jnp.where(et == _IED_IDX, 0.0, new_ipac1)

        # ipac2 (floating leg accrual):
        #   accrue group: AD, PRD, RR, CE, IPFX -> accrue_float
        #   zero group:   IP, IPFL, MD, TD      -> 0
        #   IED:          0
        #   default:      unchanged
        is_accrue_ipac2 = (
            (et == _AD_IDX)
            | (et == _PRD_IDX)
            | (et == _RR_IDX)
            | (et == _CE_IDX)
            | (et == _IPFX_IDX)
        )
        is_zero_ipac2 = (
            (et == _IP_IDX) | (et == _IPFL_IDX) | (et == _MD_IDX) | (et == _TD_IDX)
        )
        new_ipac2 = states.ipac2
        new_ipac2 = jnp.where(is_accrue_ipac2, accrue_float, new_ipac2)
        new_ipac2 = jnp.where(is_zero_ipac2, 0.0, new_ipac2)
        new_ipac2 = jnp.where(et == _IED_IDX, 0.0, new_ipac2)

        # nsc, isc: only change at IED (set to 1.0)
        new_nsc = jnp.where(et == _IED_IDX, 1.0, states.nsc)
        new_isc = jnp.where(et == _IED_IDX, 1.0, states.isc)

        new_state = SWPPVArrayState(
            nt=new_nt,
            ipnr=new_ipnr,
            ipac1=new_ipac1,
            ipac2=new_ipac2,
            nsc=new_nsc,
            isc=new_isc,
        )
        return new_state, payoff

    final_states, payoffs_t = jax.lax.scan(step, initial_states, (et_t, yf_t, rf_t), unroll=8)
    # payoffs_t is [T, B]; transpose back to [B, T]
    return final_states, payoffs_t.T


# ============================================================================
# Pre-computation bridge — Python -> JAX arrays
# ============================================================================


def _extract_params(attrs: ContractAttributes) -> SWPPVArrayParams:
    """Extract ``SWPPVArrayParams`` from ``ContractAttributes``."""
    role_sign = _get_role_sign(attrs.contract_role)
    fixed_rate = attrs.nominal_interest_rate or 0.0

    has_floor = attrs.rate_reset_floor is not None
    has_cap = attrs.rate_reset_cap is not None

    return SWPPVArrayParams(
        role_sign=jnp.array(role_sign, dtype=_F32),
        notional_principal=jnp.array(attrs.notional_principal or 0.0, dtype=_F32),
        fixed_rate=jnp.array(fixed_rate, dtype=_F32),
        rate_reset_spread=jnp.array(attrs.rate_reset_spread or 0.0, dtype=_F32),
        rate_reset_multiplier=jnp.array(
            attrs.rate_reset_multiplier if attrs.rate_reset_multiplier is not None else 1.0,
            dtype=_F32,
        ),
        rate_reset_floor=jnp.array(attrs.rate_reset_floor or 0.0, dtype=_F32),
        rate_reset_cap=jnp.array(attrs.rate_reset_cap or 1.0, dtype=_F32),
        has_rate_floor=jnp.array(1.0 if has_floor else 0.0, dtype=_F32),
        has_rate_cap=jnp.array(1.0 if has_cap else 0.0, dtype=_F32),
        price_at_purchase_date=jnp.array(attrs.price_at_purchase_date or 0.0, dtype=_F32),
        price_at_termination_date=jnp.array(attrs.price_at_termination_date or 0.0, dtype=_F32),
    )


def _extract_params_raw(attrs: ContractAttributes) -> dict[str, float | int]:
    """Extract params as plain Python floats/ints (no jnp.array overhead)."""
    role_sign = _get_role_sign(attrs.contract_role)

    return {
        "role_sign": role_sign,
        "notional_principal": attrs.notional_principal or 0.0,
        "fixed_rate": attrs.nominal_interest_rate or 0.0,
        "rate_reset_spread": attrs.rate_reset_spread or 0.0,
        "rate_reset_multiplier": (
            attrs.rate_reset_multiplier if attrs.rate_reset_multiplier is not None else 1.0
        ),
        "rate_reset_floor": attrs.rate_reset_floor or 0.0,
        "rate_reset_cap": attrs.rate_reset_cap or 1.0,
        "has_rate_floor": 1.0 if attrs.rate_reset_floor is not None else 0.0,
        "has_rate_cap": 1.0 if attrs.rate_reset_cap is not None else 0.0,
        "price_at_purchase_date": attrs.price_at_purchase_date or 0.0,
        "price_at_termination_date": attrs.price_at_termination_date or 0.0,
    }


def _params_raw_to_jax(raw: dict[str, float | int]) -> SWPPVArrayParams:
    """Convert raw Python params to JAX SWPPVArrayParams."""
    return SWPPVArrayParams(
        **{k: jnp.array(raw[k], dtype=_F32) for k in SWPPVArrayParams._fields}
    )


# ---------------------------------------------------------------------------
# Fast schedule generation — bypasses PlainVanillaSwapContract entirely
# ---------------------------------------------------------------------------


def _fast_swppv_schedule(
    attrs: ContractAttributes,
) -> list[tuple[int, _datetime, _datetime]]:
    """Generate SWPPV schedule as lightweight (evt_idx, evt_dt, calc_dt) tuples.

    Replicates the logic of ``PlainVanillaSwapContract.generate_event_schedule``
    without creating ``ContractEvent`` objects.
    """
    from jactus.core.types import BusinessDayConvention

    ied = attrs.initial_exchange_date
    md = attrs.maturity_date
    sd = attrs.status_date
    assert ied is not None
    assert md is not None

    bdc = attrs.business_day_convention

    # For non-NULL BDC or non-SD EOMC, fall back to the full path
    has_bdc = bdc is not None and bdc != BusinessDayConvention.NULL
    has_eomc = (
        attrs.end_of_month_convention is not None and attrs.end_of_month_convention.value != "SD"
    )
    if has_bdc or has_eomc:
        return _fallback_swppv_schedule(attrs)

    ied_dt = _adt_to_dt(ied)
    md_dt = _adt_to_dt(md)
    sd_dt = _adt_to_dt(sd)

    events: list[tuple[int, _datetime, _datetime]] = []

    # Determine settlement mode: D = separate (IPFX/IPFL), S = net (IP)
    ds = attrs.delivery_settlement or "D"
    use_separate = ds == "D"

    # IED
    if ied_dt >= sd_dt:
        events.append((_IED_IDX, ied_dt, ied_dt))

    # RR: Rate Reset schedule (exclude maturity date)
    if attrs.rate_reset_cycle and attrs.rate_reset_anchor:
        rr_dates = _fast_schedule(attrs.rate_reset_anchor, attrs.rate_reset_cycle, md)
        for dt in rr_dates:
            if dt >= md_dt:
                break
            events.append((_RR_IDX, dt, dt))

    # IP / IPFX+IPFL: Interest Payment schedule
    if attrs.interest_payment_cycle:
        ip_anchor = (
            attrs.interest_payment_anchor
            or attrs.interest_calculation_base_anchor
            or ied
        )
        ip_dates = _fast_schedule(ip_anchor, attrs.interest_payment_cycle, md)

        # Add maturity date as final payment if not already included
        if md_dt not in ip_dates:
            ip_dates.append(md_dt)
            ip_dates = sorted(set(ip_dates))

        for dt in ip_dates:
            if dt < ied_dt:
                continue
            if use_separate:
                events.append((_IPFX_IDX, dt, dt))
                events.append((_IPFL_IDX, dt, dt))
            else:
                events.append((_IP_IDX, dt, dt))

    # PRD: Purchase Date
    if attrs.purchase_date:
        events.append(
            (_PRD_IDX, _adt_to_dt(attrs.purchase_date), _adt_to_dt(attrs.purchase_date))
        )

    # TD: Termination Date
    if attrs.termination_date:
        events.append(
            (_TD_IDX, _adt_to_dt(attrs.termination_date), _adt_to_dt(attrs.termination_date))
        )

    # MD: Maturity Date
    events.append((_MD_IDX, md_dt, md_dt))

    # Filter: remove events before SD
    events = [(ei, et, ct) for ei, et, ct in events if et >= sd_dt]

    # If PRD exists, remove IED and events before PRD
    if attrs.purchase_date:
        prd_dt = _adt_to_dt(attrs.purchase_date)
        events = [(ei, et, ct) for ei, et, ct in events if ei != _IED_IDX and et >= prd_dt]

    # Sort by (event_time, priority)
    # Use SWPPV-specific ordering: IPFX before IPFL before IP before RR
    events.sort(key=lambda e: (e[1], _get_swppv_priority(e[0])))

    # If TD exists, remove all events after TD
    if attrs.termination_date:
        td_dt = _adt_to_dt(attrs.termination_date)
        events = [(ei, et, ct) for ei, et, ct in events if et <= td_dt]

    return events


# SWPPV-specific event priority (matches PlainVanillaSwapContract._EVENT_ORDER)
_SWPPV_PRIORITY: dict[int, int] = {
    _IED_IDX: 0,
    _PRD_IDX: 1,
    _IPFX_IDX: 2,
    _IPFL_IDX: 3,
    _IP_IDX: 4,
    _RR_IDX: 5,
    _MD_IDX: 10,
    _TD_IDX: 11,
    _AD_IDX: 12,
}


def _get_swppv_priority(evt_idx: int) -> int:
    """Get sort priority for a SWPPV event type index."""
    return _SWPPV_PRIORITY.get(evt_idx, 99)


def _fallback_swppv_schedule(
    attrs: ContractAttributes,
) -> list[tuple[int, _datetime, _datetime]]:
    """Fall back to the full PlainVanillaSwapContract for BDC/EOMC cases."""
    from jactus.contracts.swppv import PlainVanillaSwapContract
    from jactus.observers import ConstantRiskFactorObserver

    rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
    contract = PlainVanillaSwapContract(attrs, rf_obs)
    schedule = contract.generate_event_schedule()
    result: list[tuple[int, _datetime, _datetime]] = []
    for event in schedule.events:
        evt_dt = _adt_to_dt(event.event_time)
        calc_dt = _adt_to_dt(event.calculation_time) if event.calculation_time else evt_dt
        result.append((event.event_type.index, evt_dt, calc_dt))
    return result


def _fast_swppv_init_state(
    attrs: ContractAttributes,
) -> tuple[float, float, float, float, float, float, _datetime]:
    """Compute initial SWPPV state as Python floats.

    Returns ``(nt, ipnr, ipac1, ipac2, nsc, isc, sd_datetime)``.

    For mid-life contracts (IED < SD), falls back to the scalar
    ``PlainVanillaSwapContract`` to handle pre-simulation.
    """
    sd = attrs.status_date
    ied = attrs.initial_exchange_date
    sd_dt = _adt_to_dt(sd)

    needs_post_ied = (ied and ied < sd) or attrs.purchase_date
    if needs_post_ied:
        from jactus.contracts.swppv import PlainVanillaSwapContract
        from jactus.observers import ConstantRiskFactorObserver

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
        contract = PlainVanillaSwapContract(attrs, rf_obs)
        state = contract.initialize_state()

        nt = float(state.nt)
        ipnr = float(state.ipnr)
        ipac1 = float(state.ipac1) if state.ipac1 is not None else 0.0
        ipac2 = float(state.ipac2) if state.ipac2 is not None else 0.0
        nsc = float(state.nsc)
        isc = float(state.isc)

        init_sd_dt = _adt_to_dt(state.sd) if state.sd else sd_dt
        return (nt, ipnr, ipac1, ipac2, nsc, isc, init_sd_dt)

    # Pre-IED: initial floating rate from nominal_interest_rate_2
    ipnr = attrs.nominal_interest_rate_2 or 0.0
    # nt starts as notional_principal (will be set at IED)
    nt = attrs.notional_principal or 0.0
    return (nt, ipnr, 0.0, 0.0, 1.0, 1.0, sd_dt)


# ---------------------------------------------------------------------------
# Core pre-computation (Python scalar path)
# ---------------------------------------------------------------------------


def _precompute_raw(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> _RawPrecomputed:
    """Pre-compute all data as pure Python types (no JAX arrays).

    This is the core pre-computation that can be batched efficiently —
    all JAX array creation is deferred to the caller.
    """
    if _USE_DATE_ARRAY:
        return _precompute_raw_da(attrs, rf_observer)

    from jactus.core.types import DayCountConvention

    # 1. Fast schedule generation (no contract object)
    schedule = _fast_swppv_schedule(attrs)

    # 2. Fast state initialization
    nt, ipnr, ipac1, ipac2, nsc, isc, init_sd_dt = _fast_swppv_init_state(attrs)

    # 3. Compute year fractions and risk factors
    dcc = attrs.day_count_convention or DayCountConvention.A360
    yf_fn = get_yf_fn(dcc)

    event_type_list: list[int] = []
    yf_list: list[float] = []
    current_sd_dt = init_sd_dt

    for evt_idx, evt_dt, calc_dt in schedule:
        event_type_list.append(evt_idx)
        if yf_fn is not None:
            yf_list.append(yf_fn(current_sd_dt, calc_dt))
        else:
            yf_list.append(year_fraction(_dt_to_adt(current_sd_dt), _dt_to_adt(calc_dt), dcc))
        current_sd_dt = evt_dt

    # Risk factor pre-query
    rf_list = _prequery_risk_factors(schedule, attrs, rf_observer)

    # 4. Extract params as raw Python dict
    params_raw = _extract_params_raw(attrs)

    return _RawPrecomputed(
        state=(nt, ipnr, ipac1, ipac2, nsc, isc),
        event_types=event_type_list,
        year_fractions=yf_list,
        rf_values=rf_list,
        params=params_raw,
    )


# ---------------------------------------------------------------------------
# DateArray-based pre-computation (vectorised year fractions)
# ---------------------------------------------------------------------------


def _precompute_raw_da(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> _RawPrecomputed:
    """Pre-compute using vectorised year fractions (NumPy, no JAX overhead).

    Schedule generation reuses ``_fast_swppv_schedule`` (Python business logic),
    but year fractions are computed in a single vectorised NumPy pass.
    """
    from jactus.core.types import DayCountConvention

    # 1. Schedule
    schedule = _fast_swppv_schedule(attrs)

    # 2. State initialisation
    nt, ipnr, ipac1, ipac2, nsc, isc, init_sd_dt = _fast_swppv_init_state(attrs)

    if not schedule:
        params_raw = _extract_params_raw(attrs)
        return _RawPrecomputed(
            state=(nt, ipnr, ipac1, ipac2, nsc, isc),
            event_types=[],
            year_fractions=[],
            rf_values=[],
            params=params_raw,
        )

    # 3. Event types + vectorised year fractions
    event_type_list = [evt_idx for evt_idx, _, _ in schedule]
    dcc = attrs.day_count_convention or DayCountConvention.A360
    yf_list = _compute_vectorised_year_fractions(schedule, init_sd_dt, dcc)

    # 4. Risk factor pre-query
    rf_list = _prequery_risk_factors(schedule, attrs, rf_observer)

    # 5. Extract params
    params_raw = _extract_params_raw(attrs)

    return _RawPrecomputed(
        state=(nt, ipnr, ipac1, ipac2, nsc, isc),
        event_types=event_type_list,
        year_fractions=yf_list,
        rf_values=rf_list,
        params=params_raw,
    )


def _raw_to_jax(
    raw: _RawPrecomputed,
) -> tuple[SWPPVArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, SWPPVArrayParams]:
    """Convert raw pre-computed data to JAX arrays."""
    nt, ipnr, ipac1, ipac2, nsc, isc = raw.state
    return (
        SWPPVArrayState(
            nt=jnp.array(nt, dtype=_F32),
            ipnr=jnp.array(ipnr, dtype=_F32),
            ipac1=jnp.array(ipac1, dtype=_F32),
            ipac2=jnp.array(ipac2, dtype=_F32),
            nsc=jnp.array(nsc, dtype=_F32),
            isc=jnp.array(isc, dtype=_F32),
        ),
        jnp.array(raw.event_types, dtype=jnp.int32),
        jnp.array(raw.year_fractions, dtype=_F32),
        jnp.array(raw.rf_values, dtype=_F32),
        _params_raw_to_jax(raw.params),
    )


def precompute_swppv_arrays(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> tuple[SWPPVArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, SWPPVArrayParams]:
    """Pre-compute JAX arrays for array-mode SWPPV simulation.

    Generates the event schedule and initial state directly from attributes
    (bypassing ``PlainVanillaSwapContract``), then converts to JAX arrays
    suitable for ``simulate_swppv_array``.

    Args:
        attrs: Contract attributes (must be SWPPV type).
        rf_observer: Risk factor observer (queried for RR events).

    Returns:
        ``(initial_state, event_types, year_fractions, rf_values, params)``
    """
    return _raw_to_jax(_precompute_raw(attrs, rf_observer))


# ============================================================================
# Batch / portfolio API
# ============================================================================


def _raw_list_to_jax_batch(
    raw_list: list[_RawPrecomputed],
) -> tuple[SWPPVArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, SWPPVArrayParams, jnp.ndarray]:
    """Convert a list of ``_RawPrecomputed`` to padded JAX batch arrays.

    Pads shorter contracts with NOP events and builds NumPy arrays first
    (fast C-level construction) then transfers to JAX via ``jnp.asarray``.
    """
    max_events = max(len(r.event_types) for r in raw_list)

    # State fields: (batch,) each — SWPPV has 6 state fields
    state_nt = [r.state[0] for r in raw_list]
    state_ipnr = [r.state[1] for r in raw_list]
    state_ipac1 = [r.state[2] for r in raw_list]
    state_ipac2 = [r.state[3] for r in raw_list]
    state_nsc = [r.state[4] for r in raw_list]
    state_isc = [r.state[5] for r in raw_list]

    # Event arrays: (batch, max_events) each, with padding
    et_batch: list[list[int]] = []
    yf_batch: list[list[float]] = []
    rf_batch: list[list[float]] = []
    mask_batch: list[list[float]] = []

    for r in raw_list:
        n_events = len(r.event_types)
        pad_n = max_events - n_events
        et_batch.append(r.event_types + [NOP_EVENT_IDX] * pad_n)
        yf_batch.append(r.year_fractions + [0.0] * pad_n)
        rf_batch.append(r.rf_values + [0.0] * pad_n)
        mask_batch.append([1.0] * n_events + [0.0] * pad_n)

    # Param fields: (batch,) each
    param_fields: dict[str, list[float | int]] = {k: [] for k in SWPPVArrayParams._fields}
    for r in raw_list:
        for k in SWPPVArrayParams._fields:
            param_fields[k].append(r.params[k])

    # Build NumPy arrays first (fast C-level), then transfer to JAX
    batched_states = SWPPVArrayState(
        nt=jnp.asarray(np.array(state_nt, dtype=np.float32)),
        ipnr=jnp.asarray(np.array(state_ipnr, dtype=np.float32)),
        ipac1=jnp.asarray(np.array(state_ipac1, dtype=np.float32)),
        ipac2=jnp.asarray(np.array(state_ipac2, dtype=np.float32)),
        nsc=jnp.asarray(np.array(state_nsc, dtype=np.float32)),
        isc=jnp.asarray(np.array(state_isc, dtype=np.float32)),
    )

    batched_et = jnp.asarray(np.array(et_batch, dtype=np.int32))
    batched_yf = jnp.asarray(np.array(yf_batch, dtype=np.float32))
    batched_rf = jnp.asarray(np.array(rf_batch, dtype=np.float32))
    batched_masks = jnp.asarray(np.array(mask_batch, dtype=np.float32))

    batched_params = SWPPVArrayParams(
        **{k: jnp.asarray(np.array(param_fields[k], dtype=np.float32)) for k in SWPPVArrayParams._fields}
    )

    return batched_states, batched_et, batched_yf, batched_rf, batched_params, batched_masks


def prepare_swppv_batch(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
) -> tuple[SWPPVArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, SWPPVArrayParams, jnp.ndarray]:
    """Pre-compute and pad arrays for a batch of SWPPV contracts.

    Uses per-contract Python pre-computation (sequential path).

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.

    Returns:
        ``(initial_states, event_types, year_fractions, rf_values, params, masks)``
        where each array has a leading batch dimension.
    """
    raw_list = [_precompute_raw(attrs, obs) for attrs, obs in contracts]
    return _raw_list_to_jax_batch(raw_list)


def simulate_swppv_portfolio(
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
    ) = prepare_swppv_batch(contracts)

    # Run batched simulation
    final_states, payoffs = batch_simulate_swppv_auto(
        batched_states, batched_et, batched_yf, batched_rf, batched_params
    )

    # Mask padding
    masked_payoffs = payoffs * batched_masks
    total_cashflows = jnp.sum(masked_payoffs, axis=1)

    result: dict[str, Any] = {
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
