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

import re as _re
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

# ---------------------------------------------------------------------------
# NOP event index — used to pad shorter contracts in batched simulation.
# ---------------------------------------------------------------------------
NOP_EVENT_IDX: int = NUM_EVENT_TYPES  # 24 (one past the last valid EventType.index)

# ---------------------------------------------------------------------------
# DateArray feature flag — enables vectorised year fraction pre-computation.
# Set to False to fall back to the original per-event Python loop.
# ---------------------------------------------------------------------------
_USE_DATE_ARRAY: bool = True

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

# Vmapped version (kept as fallback, e.g. for GPU where vmap is efficient)
batch_simulate_pam_vmap = jax.vmap(simulate_pam_array)


# ============================================================================
# Manually-batched simulation — eliminates vmap dispatch overhead on CPU
# ============================================================================


@jax.jit
def batch_simulate_pam(
    initial_states: PAMArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: PAMArrayParams,
) -> tuple[PAMArrayState, jnp.ndarray]:
    """Batched PAM simulation without vmap — single scan over ``[B]`` arrays.

    Eliminates JAX vmap CPU dispatch overhead by operating directly on
    batch-dimensioned arrays.  Each scan step computes all event-type
    outcomes for all contracts simultaneously using branchless
    ``jnp.where`` dispatch.

    Args:
        initial_states: ``PAMArrayState`` with each field shape ``[B]``.
        event_types: ``[B, T]`` int32 — event type indices per contract.
        year_fractions: ``[B, T]`` float32.
        rf_values: ``[B, T]`` float32.
        params: ``PAMArrayParams`` with each field shape ``[B]``.

    Returns:
        ``(final_states, payoffs)`` where ``payoffs`` is ``[B, T]``.
    """
    # Transpose to [T, B] so scan iterates over time steps
    et_t = event_types.T
    yf_t = year_fractions.T
    rf_t = rf_values.T

    def step(
        states: PAMArrayState,
        inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[PAMArrayState, jnp.ndarray]:
        et, yf, rf = inputs  # each [B]

        # Common sub-expression: interest accrual
        accrue = states.ipac + yf * states.ipnr * states.nt

        # ---- Payoffs (branchless jnp.where dispatch) ----
        payoff = jnp.zeros_like(states.nt)

        # IED: role_sign * (-1) * (notional + premium)
        payoff = jnp.where(
            et == _IED_IDX,
            params.role_sign
            * (-1.0)
            * (params.notional_principal + params.premium_discount_at_ied),
            payoff,
        )
        # MD: nsc * nt + isc * ipac + feac  (uses state.ipac, NOT accrue)
        payoff = jnp.where(
            et == _MD_IDX,
            states.nsc * states.nt + states.isc * states.ipac + states.feac,
            payoff,
        )
        # PP: rf (pre-computed prepayment amount)
        payoff = jnp.where(et == _PP_IDX, rf, payoff)
        # PY: penalty (type-dependent)
        payoff = jnp.where(
            et == _PY_IDX,
            jnp.where(
                params.penalty_type == 0,
                params.penalty_rate,
                yf * states.nt * params.penalty_rate,
            ),
            payoff,
        )
        # FP: fee payment (basis-dependent)
        payoff = jnp.where(
            et == _FP_IDX,
            jnp.where(
                params.fee_basis == 0,
                params.fee_rate,
                jnp.where(
                    params.fee_basis == 1,
                    yf * states.nt * params.fee_rate + states.feac,
                    states.feac,
                ),
            ),
            payoff,
        )
        # PRD: -(price + accrue)
        payoff = jnp.where(
            et == _PRD_IDX,
            (-1.0) * (params.price_at_purchase_date + accrue),
            payoff,
        )
        # TD: price + accrue
        payoff = jnp.where(
            et == _TD_IDX,
            params.price_at_termination_date + accrue,
            payoff,
        )
        # IP: isc * accrue
        payoff = jnp.where(et == _IP_IDX, states.isc * accrue, payoff)

        # ---- State transitions (branchless) ----

        # nt: default unchanged
        new_nt = states.nt
        new_nt = jnp.where(
            et == _IED_IDX,
            params.role_sign * params.notional_principal,
            new_nt,
        )
        new_nt = jnp.where((et == _MD_IDX) | (et == _TD_IDX), 0.0, new_nt)
        new_nt = jnp.where(et == _PP_IDX, states.nt - rf, new_nt)
        new_nt = jnp.where(et == _IPCI_IDX, states.nt + accrue, new_nt)

        # ipnr: default unchanged
        new_ipnr = states.ipnr
        new_ipnr = jnp.where(et == _IED_IDX, params.nominal_interest_rate, new_ipnr)
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
        new_ipnr = jnp.where(et == _RRF_IDX, params.rate_reset_next, new_ipnr)

        # ipac: three distinct behaviours
        #   accrue group: AD, PP, PY, FP, PRD, RR, RRF, SC, CE
        #   zero group:   MD, TD, IP, IPCI
        #   special:      IED → ied_ipac
        #   default:      unchanged (NOP and unused event types)
        is_accrue = (
            (et == _AD_IDX)
            | (et == _PP_IDX)
            | (et == _PY_IDX)
            | (et == _FP_IDX)
            | (et == _PRD_IDX)
            | (et == _RR_IDX)
            | (et == _RRF_IDX)
            | (et == _SC_IDX)
            | (et == _CE_IDX)
        )
        is_zero_ipac = (et == _MD_IDX) | (et == _TD_IDX) | (et == _IP_IDX) | (et == _IPCI_IDX)
        new_ipac = states.ipac
        new_ipac = jnp.where(is_accrue, accrue, new_ipac)
        new_ipac = jnp.where(is_zero_ipac, 0.0, new_ipac)
        new_ipac = jnp.where(et == _IED_IDX, params.ied_ipac, new_ipac)

        # feac: zero at IED, MD, FP, TD; unchanged otherwise
        new_feac = jnp.where(
            (et == _IED_IDX) | (et == _MD_IDX) | (et == _FP_IDX) | (et == _TD_IDX),
            0.0,
            states.feac,
        )

        # nsc, isc: only change at IED (set to 1.0)
        new_nsc = jnp.where(et == _IED_IDX, 1.0, states.nsc)
        new_isc = jnp.where(et == _IED_IDX, 1.0, states.isc)

        new_state = PAMArrayState(
            nt=new_nt,
            ipnr=new_ipnr,
            ipac=new_ipac,
            feac=new_feac,
            nsc=new_nsc,
            isc=new_isc,
        )
        return new_state, payoff

    final_states, payoffs_t = jax.lax.scan(step, initial_states, (et_t, yf_t, rf_t))
    # payoffs_t is [T, B]; transpose back to [B, T]
    return final_states, payoffs_t.T


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


def _extract_params_raw(attrs: ContractAttributes) -> dict[str, float | int]:
    """Extract params as plain Python floats/ints (no jnp.array overhead)."""
    role_sign = _get_role_sign(attrs.contract_role)
    nt = attrs.notional_principal or 0.0
    ipnr = attrs.nominal_interest_rate or 0.0

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

    return {
        "role_sign": role_sign,
        "notional_principal": nt,
        "nominal_interest_rate": ipnr,
        "premium_discount_at_ied": attrs.premium_discount_at_ied or 0.0,
        "accrued_interest": attrs.accrued_interest or 0.0,
        "fee_rate": attrs.fee_rate or 0.0,
        "fee_basis": _encode_fee_basis(attrs),
        "penalty_rate": attrs.penalty_rate or 0.0,
        "penalty_type": _encode_penalty_type(attrs),
        "price_at_purchase_date": attrs.price_at_purchase_date or 0.0,
        "price_at_termination_date": attrs.price_at_termination_date or 0.0,
        "rate_reset_spread": attrs.rate_reset_spread or 0.0,
        "rate_reset_multiplier": (
            attrs.rate_reset_multiplier if attrs.rate_reset_multiplier is not None else 1.0
        ),
        "rate_reset_floor": attrs.rate_reset_floor or 0.0,
        "rate_reset_cap": attrs.rate_reset_cap or 1.0,
        "rate_reset_next": attrs.rate_reset_next if attrs.rate_reset_next is not None else ipnr,
        "has_rate_floor": 1.0 if attrs.rate_reset_floor is not None else 0.0,
        "has_rate_cap": 1.0 if attrs.rate_reset_cap is not None else 0.0,
        "ied_ipac": ied_ipac,
    }


def _params_raw_to_jax(raw: dict[str, float | int]) -> PAMArrayParams:
    """Convert raw Python params to JAX PAMArrayParams."""
    return PAMArrayParams(
        role_sign=jnp.array(raw["role_sign"], dtype=_F32),
        notional_principal=jnp.array(raw["notional_principal"], dtype=_F32),
        nominal_interest_rate=jnp.array(raw["nominal_interest_rate"], dtype=_F32),
        premium_discount_at_ied=jnp.array(raw["premium_discount_at_ied"], dtype=_F32),
        accrued_interest=jnp.array(raw["accrued_interest"], dtype=_F32),
        fee_rate=jnp.array(raw["fee_rate"], dtype=_F32),
        fee_basis=jnp.array(raw["fee_basis"], dtype=jnp.int32),
        penalty_rate=jnp.array(raw["penalty_rate"], dtype=_F32),
        penalty_type=jnp.array(raw["penalty_type"], dtype=jnp.int32),
        price_at_purchase_date=jnp.array(raw["price_at_purchase_date"], dtype=_F32),
        price_at_termination_date=jnp.array(raw["price_at_termination_date"], dtype=_F32),
        rate_reset_spread=jnp.array(raw["rate_reset_spread"], dtype=_F32),
        rate_reset_multiplier=jnp.array(raw["rate_reset_multiplier"], dtype=_F32),
        rate_reset_floor=jnp.array(raw["rate_reset_floor"], dtype=_F32),
        rate_reset_cap=jnp.array(raw["rate_reset_cap"], dtype=_F32),
        rate_reset_next=jnp.array(raw["rate_reset_next"], dtype=_F32),
        has_rate_floor=jnp.array(raw["has_rate_floor"], dtype=_F32),
        has_rate_cap=jnp.array(raw["has_rate_cap"], dtype=_F32),
        ied_ipac=jnp.array(raw["ied_ipac"], dtype=_F32),
    )


# ---------------------------------------------------------------------------
# Fast schedule generation — bypasses PrincipalAtMaturityContract entirely
# ---------------------------------------------------------------------------

# Cache for EVENT_SCHEDULE_PRIORITY lookups
_EVT_PRIORITY: dict[int, int] = {}


def _get_evt_priority(evt_idx: int) -> int:
    """Get sort priority for an event type index (cached)."""
    if not _EVT_PRIORITY:
        from jactus.core.types import EVENT_SCHEDULE_PRIORITY

        for et, pri in EVENT_SCHEDULE_PRIORITY.items():
            _EVT_PRIORITY[et.index] = pri
    return _EVT_PRIORITY.get(evt_idx, 99)


def _adt_to_dt(adt: ActusDateTime) -> _datetime:
    """Convert ActusDateTime to Python datetime (fast path)."""
    if adt.hour == 24:
        from datetime import timedelta

        return _datetime(adt.year, adt.month, adt.day) + timedelta(days=1)  # noqa: DTZ001
    return _datetime(adt.year, adt.month, adt.day, adt.hour, adt.minute, adt.second)  # noqa: DTZ001


def _dt_to_adt(dt: _datetime) -> ActusDateTime:
    """Convert Python datetime to ActusDateTime."""
    return ActusDateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


# Days-in-month lookup (index 0 unused). Avoids calendar.monthrange() overhead.
_DAYS_IN_MONTH = (0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def _days_in_month(y: int, m: int) -> int:
    """Return number of days in month ``m`` of year ``y``."""
    if m == 2 and (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)):
        return 29
    return _DAYS_IN_MONTH[m]


def _fast_month_schedule(
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
        d = min(start_d, _days_in_month(y, m))
        current = _datetime(y, m, d)  # noqa: DTZ001
        if current > end_dt:
            break
        dates.append(current)
        n += 1
    return dates


_CYCLE_MONTHS_MAP = {"M": 1, "Q": 3, "H": 6, "Y": 12}

# Pre-compiled regex for cycle parsing (avoid re-compiling per call)
_CYCLE_RE = _re.compile(r"^(\d+)([DWMQHY])([-+]?)$")


def _parse_cycle_fast(cycle: str) -> tuple[int, str, str]:
    """Parse cycle string without repeated re.match overhead."""
    m = _CYCLE_RE.match(cycle.upper())
    if not m:
        from jactus.core.time import parse_cycle

        return parse_cycle(cycle)
    return int(m.group(1)), m.group(2), m.group(3)


def _fast_schedule(
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
        return [_adt_to_dt(start)]

    multiplier, period, _stub = _parse_cycle_fast(cycle)
    end_dt = _adt_to_dt(end)

    if period in _CYCLE_MONTHS_MAP:
        cycle_months = multiplier * _CYCLE_MONTHS_MAP[period]
        return _fast_month_schedule(start.year, start.month, start.day, cycle_months, end_dt)

    # Day/week-based — use timedelta
    from datetime import timedelta

    start_dt = _adt_to_dt(start)
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


# Cached EventType index values for fast comparison
_AD_IDX = EventType.AD.index
_IED_IDX = EventType.IED.index
_MD_IDX = EventType.MD.index
_PP_IDX = EventType.PP.index
_PY_IDX = EventType.PY.index
_FP_IDX = EventType.FP.index
_PRD_IDX = EventType.PRD.index
_TD_IDX = EventType.TD.index
_IP_IDX = EventType.IP.index
_IPCI_IDX = EventType.IPCI.index
_RR_IDX = EventType.RR.index
_RRF_IDX = EventType.RRF.index
_SC_IDX = EventType.SC.index
_CE_IDX = EventType.CE.index


def _fast_pam_schedule(
    attrs: ContractAttributes,
) -> list[tuple[int, _datetime, _datetime]]:
    """Generate PAM schedule as lightweight (evt_idx, evt_dt, calc_dt) tuples.

    Replicates the logic of ``PrincipalAtMaturityContract.generate_event_schedule``
    without creating ``ContractEvent`` objects or a ``PrincipalAtMaturityContract``.
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
        return _fallback_pam_schedule(attrs)

    ied_dt = _adt_to_dt(ied)
    md_dt = _adt_to_dt(md)
    sd_dt = _adt_to_dt(sd)

    # events: (evt_type_idx, event_time_dt, calc_time_dt)
    events: list[tuple[int, _datetime, _datetime]] = []

    # IED
    if ied_dt >= sd_dt:
        events.append((_IED_IDX, ied_dt, ied_dt))

    # IP / IPCI
    if attrs.interest_payment_cycle:
        ip_anchor = attrs.interest_payment_anchor or ied
        ipced = attrs.interest_capitalization_end_date
        ip_dates = _fast_schedule(ip_anchor, attrs.interest_payment_cycle, md)
        ipced_dt = _adt_to_dt(ipced) if ipced else None

        # Add IPCED if not already on a cycle date
        if ipced_dt and ipced_dt not in ip_dates:
            ip_dates = sorted(set(ip_dates + [ipced_dt]))

        # Stub handling
        if md_dt not in ip_dates and ip_dates:
            ip_cycle_str = attrs.interest_payment_cycle or ""
            if ip_cycle_str.endswith("+"):
                ip_dates[-1] = md_dt
            else:
                ip_dates.append(md_dt)
            ip_dates = sorted(set(ip_dates))

        for dt in ip_dates:
            if dt < ied_dt:
                continue
            if ipced_dt and dt <= ipced_dt:
                events.append((_IPCI_IDX, dt, dt))
            else:
                events.append((_IP_IDX, dt, dt))

    # RR / RRF
    if attrs.rate_reset_cycle and attrs.rate_reset_anchor:
        rr_dates = _fast_schedule(attrs.rate_reset_anchor, attrs.rate_reset_cycle, md)
        rr_cycle_str = attrs.rate_reset_cycle or ""
        if rr_cycle_str.endswith("+") and rr_dates and rr_dates[-1] != md_dt:
            rr_dates = rr_dates[:-1]
        first_rr = True
        for dt in rr_dates:
            if dt >= md_dt:
                break
            if first_rr and attrs.rate_reset_next is not None:
                events.append((_RRF_IDX, dt, dt))
                first_rr = False
            else:
                events.append((_RR_IDX, dt, dt))
                first_rr = False

    # FP
    if attrs.fee_payment_cycle:
        fp_anchor = attrs.fee_payment_anchor or ied
        fp_dates = _fast_schedule(fp_anchor, attrs.fee_payment_cycle, md)
        for dt in fp_dates:
            if dt > ied_dt:
                events.append((_FP_IDX, dt, dt))

    # SC
    if attrs.scaling_index_cycle:
        sc_anchor = attrs.scaling_index_anchor or ied
        sc_dates = _fast_schedule(sc_anchor, attrs.scaling_index_cycle, md)
        for dt in sc_dates:
            if dt > ied_dt:
                events.append((_SC_IDX, dt, dt))

    # PRD
    if attrs.purchase_date:
        events.append((_PRD_IDX, _adt_to_dt(attrs.purchase_date), _adt_to_dt(attrs.purchase_date)))

    # TD
    if attrs.termination_date:
        events.append(
            (_TD_IDX, _adt_to_dt(attrs.termination_date), _adt_to_dt(attrs.termination_date))
        )

    # MD
    events.append((_MD_IDX, md_dt, md_dt))

    # Filter: remove events before SD
    events = [(ei, et, ct) for ei, et, ct in events if et >= sd_dt]

    # If PRD exists, remove IED and events before PRD
    if attrs.purchase_date:
        prd_dt = _adt_to_dt(attrs.purchase_date)
        events = [(ei, et, ct) for ei, et, ct in events if ei != _IED_IDX and et >= prd_dt]

    # Sort by (event_time, priority)
    events.sort(key=lambda e: (e[1], _get_evt_priority(e[0])))

    # If TD exists, remove all events after TD
    if attrs.termination_date:
        td_dt = _adt_to_dt(attrs.termination_date)
        events = [(ei, et, ct) for ei, et, ct in events if et <= td_dt]

    return events


def _fallback_pam_schedule(
    attrs: ContractAttributes,
) -> list[tuple[int, _datetime, _datetime]]:
    """Fall back to the full PrincipalAtMaturityContract for BDC/EOMC cases."""
    from jactus.contracts.pam import PrincipalAtMaturityContract
    from jactus.observers import ConstantRiskFactorObserver

    rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
    contract = PrincipalAtMaturityContract(attrs, rf_obs)
    schedule = contract.generate_event_schedule()
    result: list[tuple[int, _datetime, _datetime]] = []
    for event in schedule.events:
        evt_dt = _adt_to_dt(event.event_time)
        calc_dt = _adt_to_dt(event.calculation_time) if event.calculation_time else evt_dt
        result.append((event.event_type.index, evt_dt, calc_dt))
    return result


def _fast_pam_init_state(
    attrs: ContractAttributes,
) -> tuple[float, float, float, float, float, float, _datetime]:
    """Compute initial PAM state as Python floats.

    Returns ``(nt, ipnr, ipac, feac, nsc, isc, sd_datetime)``.
    """
    sd = attrs.status_date
    ied = attrs.initial_exchange_date
    sd_dt = _adt_to_dt(sd)

    needs_post_ied = (ied and ied < sd) or attrs.purchase_date
    if needs_post_ied:
        role_sign = _get_role_sign(attrs.contract_role)
        nt = role_sign * (attrs.notional_principal or 0.0)
        ipnr = attrs.nominal_interest_rate or 0.0

        if ied and ied >= sd and attrs.purchase_date:
            init_sd_dt = _adt_to_dt(ied)
            ipac = 0.0
        else:
            init_sd_dt = sd_dt
            accrual_start = attrs.interest_payment_anchor or ied
            if attrs.accrued_interest is not None:
                ipac = attrs.accrued_interest
            elif accrual_start and accrual_start < sd:
                from jactus.core.types import DayCountConvention

                dcc = attrs.day_count_convention or DayCountConvention.A360
                ipac = year_fraction(accrual_start, sd, dcc) * ipnr * abs(nt)
            else:
                ipac = 0.0

        return (nt, ipnr, ipac, 0.0, 1.0, 1.0, init_sd_dt)

    return (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, sd_dt)


# Year fraction fast paths for common day count conventions
def _yf_a360(d1: _datetime, d2: _datetime) -> float:
    return (d2 - d1).days / 360.0


def _yf_a365(d1: _datetime, d2: _datetime) -> float:
    return (d2 - d1).days / 365.0


def _yf_30e360(d1: _datetime, d2: _datetime) -> float:
    y1, m1, dd1 = d1.year, d1.month, d1.day
    y2, m2, dd2 = d2.year, d2.month, d2.day
    if dd1 == 31:
        dd1 = 30
    if dd2 == 31:
        dd2 = 30
    return ((y2 - y1) * 360 + (m2 - m1) * 30 + (dd2 - dd1)) / 360.0


def _yf_b30360(d1: _datetime, d2: _datetime) -> float:
    y1, m1, dd1 = d1.year, d1.month, d1.day
    y2, m2, dd2 = d2.year, d2.month, d2.day
    if dd1 == 31:
        dd1 = 30
    if dd1 >= 30 and dd2 == 31:
        dd2 = 30
    return ((y2 - y1) * 360 + (m2 - m1) * 30 + (dd2 - dd1)) / 360.0


class _RawPrecomputed(NamedTuple):
    """Pre-computed data as Python types (no JAX overhead)."""

    state: tuple[float, float, float, float, float, float]  # nt, ipnr, ipac, feac, nsc, isc
    event_types: list[int]
    year_fractions: list[float]
    rf_values: list[float]
    params: dict[str, float | int]


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
    schedule = _fast_pam_schedule(attrs)

    # 2. Fast state initialization (no contract object)
    nt, ipnr, ipac, feac, nsc, isc, init_sd_dt = _fast_pam_init_state(attrs)

    # 3. Compute year fractions and risk factors
    dcc = attrs.day_count_convention or DayCountConvention.A360

    # Pick fast YF function for common DCCs
    if dcc == DayCountConvention.A360:
        yf_fn = _yf_a360
    elif dcc == DayCountConvention.A365:
        yf_fn = _yf_a365
    elif dcc == DayCountConvention.E30360:
        yf_fn = _yf_30e360
    elif dcc == DayCountConvention.B30360:
        yf_fn = _yf_b30360
    else:
        yf_fn = None  # fall back to full year_fraction

    event_type_list: list[int] = []
    yf_list: list[float] = []
    rf_list: list[float] = []
    current_sd_dt = init_sd_dt

    market_object = attrs.rate_reset_market_object or ""
    contract_id = attrs.contract_id or ""

    for evt_idx, evt_dt, calc_dt in schedule:
        event_type_list.append(evt_idx)

        # Year fraction
        if yf_fn is not None:
            yf_list.append(yf_fn(current_sd_dt, calc_dt))
        else:
            yf_list.append(year_fraction(_dt_to_adt(current_sd_dt), _dt_to_adt(calc_dt), dcc))

        # Risk factor pre-query
        rf_val = 0.0
        if evt_idx == _RR_IDX:
            try:
                rf_val = float(rf_observer.observe_risk_factor(market_object, _dt_to_adt(evt_dt)))
            except (KeyError, NotImplementedError, TypeError):
                rf_val = 0.0
        elif evt_idx == _PP_IDX:
            try:
                rf_val = float(
                    rf_observer.observe_event(contract_id, EventType.PP, _dt_to_adt(evt_dt))
                )
            except (KeyError, NotImplementedError, TypeError):
                rf_val = 0.0
        rf_list.append(rf_val)

        current_sd_dt = evt_dt

    # 4. Extract params as raw Python dict
    params_raw = _extract_params_raw(attrs)

    return _RawPrecomputed(
        state=(nt, ipnr, ipac, feac, nsc, isc),
        event_types=event_type_list,
        year_fractions=yf_list,
        rf_values=rf_list,
        params=params_raw,
    )


# ---------------------------------------------------------------------------
# DateArray-based pre-computation  (vectorised year fractions)
# ---------------------------------------------------------------------------


def _precompute_raw_da(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> _RawPrecomputed:
    """Pre-compute using vectorised year fractions (NumPy, no JAX overhead).

    Schedule generation reuses ``_fast_pam_schedule`` (Python business logic),
    but year fractions are computed in a single vectorised NumPy pass using
    the Hinnant ordinal algorithm from :mod:`jactus.utilities.date_array`.
    """
    from jactus.core.types import DayCountConvention

    # 1. Schedule (same as before — Python business logic)
    schedule = _fast_pam_schedule(attrs)

    # 2. State initialisation (same as before)
    nt, ipnr, ipac, feac, nsc, isc, init_sd_dt = _fast_pam_init_state(attrs)

    if not schedule:
        params_raw = _extract_params_raw(attrs)
        return _RawPrecomputed(
            state=(nt, ipnr, ipac, feac, nsc, isc),
            event_types=[],
            year_fractions=[],
            rf_values=[],
            params=params_raw,
        )

    # 3. Build parallel NumPy arrays from schedule tuples
    n_events = len(schedule)
    event_type_list = [evt_idx for evt_idx, _, _ in schedule]

    # Status-date chain: [init_sd, evt_0, ..., evt_{n-2}]
    # Calc-date array:   [calc_0, calc_1, ..., calc_{n-1}]
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

    # 4. Vectorised year fraction (pure NumPy — no JAX dispatch overhead)
    dcc = attrs.day_count_convention or DayCountConvention.A360

    if dcc in (DayCountConvention.A360, DayCountConvention.A365):
        # Ordinal-based: compute ordinals with Hinnant algorithm (NumPy)
        sd_ord = _np_ymd_to_ordinal(sd_y, sd_m, sd_d)
        calc_ord = _np_ymd_to_ordinal(calc_y, calc_m, calc_d)
        delta = (calc_ord - sd_ord).astype(np.float64)
        divisor = 360.0 if dcc == DayCountConvention.A360 else 365.0
        yf_list = (delta / divisor).tolist()
    elif dcc == DayCountConvention.E30360:
        yf_list = _np_yf_30e360(sd_y, sd_m, sd_d, calc_y, calc_m, calc_d).tolist()
    elif dcc == DayCountConvention.B30360:
        yf_list = _np_yf_b30360(sd_y, sd_m, sd_d, calc_y, calc_m, calc_d).tolist()
    else:
        # Fallback to scalar for AA, E30360ISDA, BUS252
        yf_list = []
        current_sd_dt = init_sd_dt
        for _ei, evt_dt, calc_dt in schedule:
            yf_list.append(
                year_fraction(_dt_to_adt(current_sd_dt), _dt_to_adt(calc_dt), dcc)
            )
            current_sd_dt = evt_dt

    # 5. Risk factor pre-query (per-event, observer is Python)
    rf_list: list[float] = []
    market_object = attrs.rate_reset_market_object or ""
    contract_id = attrs.contract_id or ""
    for evt_idx, evt_dt, _calc_dt in schedule:
        rf_val = 0.0
        if evt_idx == _RR_IDX:
            try:
                rf_val = float(
                    rf_observer.observe_risk_factor(market_object, _dt_to_adt(evt_dt))
                )
            except (KeyError, NotImplementedError, TypeError):
                rf_val = 0.0
        elif evt_idx == _PP_IDX:
            try:
                rf_val = float(
                    rf_observer.observe_event(contract_id, EventType.PP, _dt_to_adt(evt_dt))
                )
            except (KeyError, NotImplementedError, TypeError):
                rf_val = 0.0
        rf_list.append(rf_val)

    # 6. Extract params
    params_raw = _extract_params_raw(attrs)

    return _RawPrecomputed(
        state=(nt, ipnr, ipac, feac, nsc, isc),
        event_types=event_type_list,
        year_fractions=yf_list,
        rf_values=rf_list,
        params=params_raw,
    )


# ---------------------------------------------------------------------------
# NumPy-backed ordinal & year-fraction helpers (zero JAX overhead)
# ---------------------------------------------------------------------------


def _np_ymd_to_ordinal(
    y: np.ndarray, m: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Hinnant Y/M/D→ordinal, NumPy version (for pre-computation path)."""
    a = np.where(m <= 2, 1, 0).astype(np.int64)
    y_adj = y.astype(np.int64) - a
    m_adj = m.astype(np.int64) + 12 * a - 3

    doy = (153 * m_adj + 2) // 5 + d.astype(np.int64) - 1

    era = np.where(y_adj >= 0, y_adj // 400, (y_adj - 399) // 400)
    yoe = y_adj - era * 400

    doe = 365 * yoe + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 305


def _np_yf_30e360(
    y1: np.ndarray, m1: np.ndarray, d1: np.ndarray,
    y2: np.ndarray, m2: np.ndarray, d2: np.ndarray,
) -> np.ndarray:
    """30E/360 year fraction, vectorised NumPy."""
    dd1 = np.where(d1 == 31, 30, d1)
    dd2 = np.where(d2 == 31, 30, d2)
    days = (y2 - y1) * 360 + (m2 - m1) * 30 + (dd2 - dd1)
    return days.astype(np.float64) / 360.0


def _np_yf_b30360(
    y1: np.ndarray, m1: np.ndarray, d1: np.ndarray,
    y2: np.ndarray, m2: np.ndarray, d2: np.ndarray,
) -> np.ndarray:
    """30/360 US (Bond Basis) year fraction, vectorised NumPy."""
    dd1 = np.where(d1 == 31, 30, d1)
    dd2 = np.where((dd1 >= 30) & (d2 == 31), 30, d2)
    days = (y2 - y1) * 360 + (m2 - m1) * 30 + (dd2 - dd1)
    return days.astype(np.float64) / 360.0


def _raw_to_jax(
    raw: _RawPrecomputed,
) -> tuple[PAMArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, PAMArrayParams]:
    """Convert raw pre-computed data to JAX arrays."""
    nt, ipnr, ipac, feac, nsc, isc = raw.state
    return (
        PAMArrayState(
            nt=jnp.array(nt, dtype=_F32),
            ipnr=jnp.array(ipnr, dtype=_F32),
            ipac=jnp.array(ipac, dtype=_F32),
            feac=jnp.array(feac, dtype=_F32),
            nsc=jnp.array(nsc, dtype=_F32),
            isc=jnp.array(isc, dtype=_F32),
        ),
        jnp.array(raw.event_types, dtype=jnp.int32),
        jnp.array(raw.year_fractions, dtype=_F32),
        jnp.array(raw.rf_values, dtype=_F32),
        _params_raw_to_jax(raw.params),
    )


def precompute_pam_arrays(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> tuple[PAMArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, PAMArrayParams]:
    """Pre-compute JAX arrays for array-mode PAM simulation.

    Generates the event schedule and initial state directly from attributes
    (bypassing ``PrincipalAtMaturityContract``), then converts to JAX arrays
    suitable for ``simulate_pam_array``.

    Args:
        attrs: Contract attributes (must be PAM type).
        rf_observer: Risk factor observer (queried for RR/PP events).

    Returns:
        ``(initial_state, event_types, year_fractions, rf_values, params)``
    """
    return _raw_to_jax(_precompute_raw(attrs, rf_observer))


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

    Uses a two-phase approach for efficiency:
    1. Pre-compute all contracts as Python types (no JAX overhead)
    2. Batch-convert to JAX arrays in a single pass

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.

    Returns:
        ``(initial_states, event_types, year_fractions, rf_values, params, masks)``
        where each array has a leading batch dimension.
    """
    # Phase 1: Pre-compute everything as Python types
    raw_list = [_precompute_raw(attrs, obs) for attrs, obs in contracts]
    max_events = max(len(r.event_types) for r in raw_list)

    # Phase 2: Pad and batch-convert to JAX arrays
    # Build Python lists for batch conversion (one jnp.array per field)

    # State fields: (batch,) each
    state_nt = [r.state[0] for r in raw_list]
    state_ipnr = [r.state[1] for r in raw_list]
    state_ipac = [r.state[2] for r in raw_list]
    state_feac = [r.state[3] for r in raw_list]
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
    param_fields: dict[str, list[float | int]] = {k: [] for k in PAMArrayParams._fields}
    for r in raw_list:
        for k in PAMArrayParams._fields:
            param_fields[k].append(r.params[k])

    # Single batch conversion: build NumPy arrays first (fast C-level construction
    # from Python lists), then transfer to JAX via jnp.asarray (near-zero copy).
    # This avoids jnp.array()'s tracing/dispatch overhead which dominated prep time.
    batched_states = PAMArrayState(
        nt=jnp.asarray(np.array(state_nt, dtype=np.float32)),
        ipnr=jnp.asarray(np.array(state_ipnr, dtype=np.float32)),
        ipac=jnp.asarray(np.array(state_ipac, dtype=np.float32)),
        feac=jnp.asarray(np.array(state_feac, dtype=np.float32)),
        nsc=jnp.asarray(np.array(state_nsc, dtype=np.float32)),
        isc=jnp.asarray(np.array(state_isc, dtype=np.float32)),
    )

    batched_et = jnp.asarray(np.array(et_batch, dtype=np.int32))
    batched_yf = jnp.asarray(np.array(yf_batch, dtype=np.float32))
    batched_rf = jnp.asarray(np.array(rf_batch, dtype=np.float32))
    batched_masks = jnp.asarray(np.array(mask_batch, dtype=np.float32))

    # Params: determine dtype per field (int32 for fee_basis/penalty_type, float32 otherwise)
    _int_fields = {"fee_basis", "penalty_type"}
    batched_params = PAMArrayParams(
        **{
            k: jnp.asarray(
                np.array(
                    param_fields[k],
                    dtype=np.int32 if k in _int_fields else np.float32,
                )
            )
            for k in PAMArrayParams._fields
        }
    )

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
