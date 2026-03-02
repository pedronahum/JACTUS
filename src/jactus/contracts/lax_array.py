"""Array-mode LAX simulation -- JIT-compiled, vmap-able pure JAX.

This module provides a high-performance simulation path for LAX (Exotic
Linear Amortizer) contracts using ``jax.lax.scan`` for the event loop and
``jax.lax.switch`` for payoff/state-transition dispatch.  The entire
simulation kernel is JIT-compilable and can be vectorized across a
portfolio with ``jax.vmap``.

Architecture:
    Pre-computation (Python) -> Pure JAX kernel (jit + vmap)

    The existing ``ExoticLinearAmortizerContract`` generates event schedules
    and initializes state (Python-level, runs once per contract).  This
    module converts the results to JAX arrays and runs the numerical
    simulation as a pure function.

Key differences from LAM:
    - ``prnxt`` varies per event (from ARPRANX/ARPRCL/ARPRNXT arrays)
      instead of being constant.  A per-event ``prnxt_schedule`` array
      is passed to the kernel alongside ``event_types``, ``year_fractions``,
      and ``rf_values``.
    - PI (Principal Increase) events increase notional (opposite of PR).
    - The kernel reads ``prnxt`` from the per-event schedule at each
      PR/PI event, overriding the state's ``prnxt`` field.

Example::

    from jactus.contracts.lax_array import precompute_lax_arrays, simulate_lax_array

    arrays = precompute_lax_arrays(attrs, rf_observer)
    final_state, payoffs = simulate_lax_array(*arrays)

    # Portfolio:
    from jactus.contracts.lax_array import simulate_lax_portfolio
    result = simulate_lax_portfolio(contracts, discount_rate=0.05)
"""

from __future__ import annotations

from datetime import datetime as _datetime
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jactus.contracts.array_common import (
    # Cached EventType indices
    AD_IDX as _AD_IDX,
)
from jactus.contracts.array_common import (
    CE_IDX as _CE_IDX,
)
from jactus.contracts.array_common import (
    F32 as _F32,
)
from jactus.contracts.array_common import (
    FP_IDX as _FP_IDX,
)
from jactus.contracts.array_common import (
    IED_IDX as _IED_IDX,
)
from jactus.contracts.array_common import (
    IP_IDX as _IP_IDX,
)
from jactus.contracts.array_common import (
    IPCB_IDX as _IPCB_IDX,
)
from jactus.contracts.array_common import (
    IPCI_IDX as _IPCI_IDX,
)
from jactus.contracts.array_common import (
    MD_IDX as _MD_IDX,
)

# Import shared infrastructure from array_common
from jactus.contracts.array_common import (
    NOP_EVENT_IDX,
)
from jactus.contracts.array_common import (
    PI_IDX as _PI_IDX,
)
from jactus.contracts.array_common import (
    PP_IDX as _PP_IDX,
)
from jactus.contracts.array_common import (
    PR_IDX as _PR_IDX,
)
from jactus.contracts.array_common import (
    PRD_IDX as _PRD_IDX,
)
from jactus.contracts.array_common import (
    PRF_IDX as _PRF_IDX,
)
from jactus.contracts.array_common import (
    PY_IDX as _PY_IDX,
)
from jactus.contracts.array_common import (
    RR_IDX as _RR_IDX,
)
from jactus.contracts.array_common import (
    RRF_IDX as _RRF_IDX,
)
from jactus.contracts.array_common import (
    SC_IDX as _SC_IDX,
)
from jactus.contracts.array_common import (
    TD_IDX as _TD_IDX,
)
from jactus.contracts.array_common import (
    # Date helpers
    adt_to_dt as _adt_to_dt,
)
from jactus.contracts.array_common import (
    compute_vectorised_year_fractions as _compute_vectorised_year_fractions,
)
from jactus.contracts.array_common import (
    dt_to_adt as _dt_to_adt,
)
from jactus.contracts.array_common import (
    # Encoding helpers
    encode_fee_basis as _encode_fee_basis,
)
from jactus.contracts.array_common import (
    encode_penalty_type as _encode_penalty_type,
)
from jactus.contracts.array_common import (
    get_role_sign as _get_role_sign,
)
from jactus.contracts.array_common import (
    prequery_risk_factors as _prequery_risk_factors,
)

# Import IPCB mode constants from lam_array
from jactus.contracts.lam_array import (
    IPCB_NT,
    IPCB_NTL,
    _encode_ipcb_mode,
)
from jactus.core import (
    ActusDateTime,
    ContractAttributes,
)
from jactus.observers import RiskFactorObserver
from jactus.utilities.conventions import year_fraction

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class LAXArrayState(NamedTuple):
    """Minimal scan-loop state for LAX simulation.

    Same structure as ``LAMArrayState``: 8 scalar ``jnp.ndarray`` fields.
    The key difference is that ``prnxt`` is updated from a per-event
    schedule at PR/PI events instead of remaining constant.
    """

    nt: jnp.ndarray  # Notional principal (signed)
    ipnr: jnp.ndarray  # Nominal interest rate
    ipac: jnp.ndarray  # Accrued interest
    feac: jnp.ndarray  # Accrued fees
    nsc: jnp.ndarray  # Notional scaling multiplier
    isc: jnp.ndarray  # Interest scaling multiplier
    prnxt: jnp.ndarray  # Next principal redemption amount (signed)
    ipcb: jnp.ndarray  # Interest calculation base


class LAXArrayParams(NamedTuple):
    """Static contract parameters for LAX simulation.

    Same structure as ``LAMArrayParams``.
    """

    role_sign: jnp.ndarray  # +1.0 or -1.0
    notional_principal: jnp.ndarray
    nominal_interest_rate: jnp.ndarray
    premium_discount_at_ied: jnp.ndarray
    accrued_interest: jnp.ndarray  # IPAC attribute
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
    has_rate_floor: jnp.ndarray  # 1.0 if floor active, else 0.0
    has_rate_cap: jnp.ndarray  # 1.0 if cap active, else 0.0
    ied_ipac: jnp.ndarray  # Pre-computed accrued interest at IED
    # LAX-specific params (same fields as LAM)
    next_principal_redemption_amount: jnp.ndarray  # unsigned PRNXT (initial)
    ipcb_mode: jnp.ndarray  # 0=NT, 1=NTIED, 2=NTL (int32)
    interest_calculation_base_amount: jnp.ndarray  # IPCBA (unsigned)


# ============================================================================
# Pure JAX payoff functions  (state, params, yf, rf) -> scalar payoff
# ============================================================================


def _accrue_interest_lax(state: LAXArrayState, yf: jnp.ndarray) -> jnp.ndarray:
    """Common sub-expression: ipac + yf * ipnr * ipcb (LAX uses ipcb)."""
    return state.ipac + yf * state.ipnr * state.ipcb


def _pof_ad(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_ied(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_IED: R(CNTRL) * (-1) * Nsc * NT."""
    return params.role_sign * (-1.0) * state.nsc * params.notional_principal


def _pof_pr(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_PR: Nsc * effective_prnxt.

    Prnxt is capped at remaining notional to prevent overshoot.
    Note: state.prnxt is already updated from the per-event schedule
    before this function is called.
    """
    effective_prnxt = jnp.sign(state.prnxt) * jnp.minimum(jnp.abs(state.prnxt), jnp.abs(state.nt))
    return state.nsc * effective_prnxt


def _pof_pi(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_PI: Principal Increase -- negative of PR.

    PI payoff = -Nsc * prnxt (receives additional principal).
    """
    return -state.nsc * state.prnxt


def _pof_md(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_MD: Nsc * Nt + Isc * Ipac + Feac."""
    return state.nsc * state.nt + state.isc * state.ipac + state.feac


def _pof_pp(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_PP: rf (pre-computed prepayment amount from observer)."""
    return rf


def _pof_py(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_PY: Penalty payment (type-dependent)."""
    pof_a = params.penalty_rate
    pof_ni = yf * state.nt * params.penalty_rate
    return jnp.where(params.penalty_type == 0, pof_a, pof_ni)


def _pof_prf(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_PRF: Principal Redemption Amount Fixing -- no payoff."""
    return jnp.array(0.0, dtype=_F32)


def _pof_fp(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_FP: Fee payment (basis-dependent)."""
    pof_a = params.fee_rate
    pof_n = yf * state.nt * params.fee_rate + state.feac
    return jnp.where(
        params.fee_basis == 0,
        pof_a,
        jnp.where(params.fee_basis == 1, pof_n, state.feac),
    )


def _pof_prd(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_PRD: -(PPRD + Ipac + Y * Ipnr * Ipcb)."""
    return (-1.0) * (params.price_at_purchase_date + _accrue_interest_lax(state, yf))


def _pof_td(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_TD: PTD + Ipac + Y * Ipnr * Ipcb."""
    return params.price_at_termination_date + _accrue_interest_lax(state, yf)


def _pof_ip(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_IP: Isc * (Ipac + Y * Ipnr * Ipcb)."""
    return state.isc * _accrue_interest_lax(state, yf)


def _pof_ipci(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_ipcb(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_rr(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_rrf(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_sc(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_ce(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_noop(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


# ============================================================================
# Pure JAX state transition functions  (state, params, yf, rf) -> new state
# ============================================================================


def _stf_ad(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    return state._replace(ipac=_accrue_interest_lax(state, yf))


def _stf_ied(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    """STF_IED: Initialize all state variables."""
    nt = params.role_sign * params.notional_principal
    prnxt = params.role_sign * params.next_principal_redemption_amount
    ipcb = jnp.where(
        params.interest_calculation_base_amount > 0.0,
        params.role_sign * params.interest_calculation_base_amount,
        nt,
    )
    return LAXArrayState(
        nt=nt,
        ipnr=params.nominal_interest_rate,
        ipac=params.ied_ipac,
        feac=jnp.array(0.0, dtype=_F32),
        nsc=jnp.array(1.0, dtype=_F32),
        isc=jnp.array(1.0, dtype=_F32),
        prnxt=prnxt,
        ipcb=ipcb,
    )


def _stf_pr(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    """STF_PR: Principal Redemption.

    Note: state.prnxt is already updated from the per-event schedule
    before this function is called.
    """
    new_ipac = _accrue_interest_lax(state, yf)
    effective_prnxt = jnp.sign(state.prnxt) * jnp.minimum(jnp.abs(state.prnxt), jnp.abs(state.nt))
    new_nt = state.nt - effective_prnxt
    new_ipcb = jnp.where(
        params.ipcb_mode == IPCB_NTL,
        state.ipcb,
        new_nt,
    )
    return state._replace(nt=new_nt, ipac=new_ipac, ipcb=new_ipcb)


def _stf_pi(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    """STF_PI: Principal Increase -- opposite of PR.

    Nt += prnxt (increase notional).
    """
    new_ipac = _accrue_interest_lax(state, yf)
    new_nt = state.nt + state.prnxt
    new_ipcb = jnp.where(
        params.ipcb_mode == IPCB_NTL,
        state.ipcb,
        new_nt,
    )
    return state._replace(nt=new_nt, ipac=new_ipac, ipcb=new_ipcb)


def _stf_md(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    return state._replace(
        nt=jnp.array(0.0, dtype=_F32),
        ipac=jnp.array(0.0, dtype=_F32),
        feac=jnp.array(0.0, dtype=_F32),
        ipcb=jnp.array(0.0, dtype=_F32),
    )


def _stf_pp(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    """STF_PP: Prepayment."""
    new_ipac = _accrue_interest_lax(state, yf)
    new_nt = state.nt - rf
    new_ipcb = jnp.where(
        params.ipcb_mode == IPCB_NTL,
        state.ipcb,
        new_nt,
    )
    return state._replace(nt=new_nt, ipac=new_ipac, ipcb=new_ipcb)


def _stf_py(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    return state._replace(ipac=_accrue_interest_lax(state, yf))


def _stf_prf(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    """STF_PRF: Principal Redemption Amount Fixing -- no-op in array mode.

    The per-event prnxt is already injected via the prnxt_schedule.
    """
    return state._replace(ipac=_accrue_interest_lax(state, yf))


def _stf_fp(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    return state._replace(
        ipac=_accrue_interest_lax(state, yf),
        feac=jnp.array(0.0, dtype=_F32),
    )


def _stf_prd(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    return state._replace(ipac=_accrue_interest_lax(state, yf))


def _stf_td(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    return state._replace(
        nt=jnp.array(0.0, dtype=_F32),
        ipac=jnp.array(0.0, dtype=_F32),
        feac=jnp.array(0.0, dtype=_F32),
        ipcb=jnp.array(0.0, dtype=_F32),
    )


def _stf_ip(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    return state._replace(ipac=jnp.array(0.0, dtype=_F32))


def _stf_ipci(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    """STF_IPCI: Interest Capitalization."""
    total_accrued = _accrue_interest_lax(state, yf)
    new_nt = state.nt + total_accrued
    new_ipcb = jnp.where(params.ipcb_mode == IPCB_NT, new_nt, state.ipcb)
    return state._replace(
        nt=new_nt,
        ipac=jnp.array(0.0, dtype=_F32),
        ipcb=new_ipcb,
    )


def _stf_ipcb(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    """STF_IPCB: Interest Calculation Base fixing (NTL mode)."""
    new_ipac = _accrue_interest_lax(state, yf)
    return state._replace(ipcb=state.nt, ipac=new_ipac)


def _stf_rr(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    """STF_RR: Rate Reset."""
    new_ipac = _accrue_interest_lax(state, yf)
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
    return state._replace(ipnr=clamped, ipac=new_ipac)


def _stf_rrf(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    new_ipac = _accrue_interest_lax(state, yf)
    return state._replace(ipnr=params.rate_reset_next, ipac=new_ipac)


def _stf_sc(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    return state._replace(ipac=_accrue_interest_lax(state, yf))


def _stf_ce(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    return state._replace(ipac=_accrue_interest_lax(state, yf))


def _stf_noop(
    state: LAXArrayState, params: LAXArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> LAXArrayState:
    return state


# ============================================================================
# Dispatch tables -- indexed by EventType.index (0..23) + NOP (24)
# ============================================================================

# fmt: off
_POF_TABLE: list[Any] = [
    _pof_ad,    # 0  AD
    _pof_ied,   # 1  IED
    _pof_md,    # 2  MD
    _pof_pr,    # 3  PR
    _pof_pi,    # 4  PI   (LAX supports PI)
    _pof_pp,    # 5  PP
    _pof_py,    # 6  PY
    _pof_prf,   # 7  PRF  (LAX supports PRF)
    _pof_fp,    # 8  FP
    _pof_prd,   # 9  PRD
    _pof_td,    # 10 TD
    _pof_ip,    # 11 IP
    _pof_ipci,  # 12 IPCI
    _pof_ipcb,  # 13 IPCB
    _pof_rr,    # 14 RR
    _pof_rrf,   # 15 RRF
    _pof_noop,  # 16 DV   (not used in LAX)
    _pof_noop,  # 17 DVF  (not used in LAX)
    _pof_sc,    # 18 SC
    _pof_noop,  # 19 STD  (not used in LAX)
    _pof_noop,  # 20 XD   (not used in LAX)
    _pof_ce,    # 21 CE
    _pof_noop,  # 22 IPFX (not used in LAX)
    _pof_noop,  # 23 IPFL (not used in LAX)
    _pof_noop,  # 24 NOP  (padding)
]

_STF_TABLE: list[Any] = [
    _stf_ad,    # 0  AD
    _stf_ied,   # 1  IED
    _stf_md,    # 2  MD
    _stf_pr,    # 3  PR
    _stf_pi,    # 4  PI   (LAX supports PI)
    _stf_pp,    # 5  PP
    _stf_py,    # 6  PY
    _stf_prf,   # 7  PRF  (LAX supports PRF)
    _stf_fp,    # 8  FP
    _stf_prd,   # 9  PRD
    _stf_td,    # 10 TD
    _stf_ip,    # 11 IP
    _stf_ipci,  # 12 IPCI
    _stf_ipcb,  # 13 IPCB
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


def simulate_lax_array(
    initial_state: LAXArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    prnxt_schedule: jnp.ndarray,
    params: LAXArrayParams,
) -> tuple[LAXArrayState, jnp.ndarray]:
    """Run a LAX simulation as a pure JAX function.

    This function is JIT-compilable and vmap-able.

    Args:
        initial_state: Starting state (8 scalar fields).
        event_types: ``(num_events,)`` int32 -- ``EventType.index`` values.
        year_fractions: ``(num_events,)`` float32 -- pre-computed YF per event.
        rf_values: ``(num_events,)`` float32 -- pre-computed risk factor values.
        prnxt_schedule: ``(num_events,)`` float32 -- per-event prnxt values
            (signed, only meaningful at PR/PI events; 0.0 elsewhere).
        params: Static contract parameters.

    Returns:
        ``(final_state, payoffs)`` where payoffs is ``(num_events,)`` float32.
    """

    def step(
        state: LAXArrayState,
        inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[LAXArrayState, jnp.ndarray]:
        evt_idx, yf, rf, prnxt_val = inputs

        # Inject per-event prnxt at PR/PI/PRF events
        is_prnxt_event = (evt_idx == _PR_IDX) | (evt_idx == _PI_IDX) | (evt_idx == _PRF_IDX)
        new_prnxt = jnp.where(is_prnxt_event, prnxt_val, state.prnxt)
        state = state._replace(prnxt=new_prnxt)

        payoff = jax.lax.switch(evt_idx, _POF_TABLE, state, params, yf, rf)
        new_state = jax.lax.switch(evt_idx, _STF_TABLE, state, params, yf, rf)
        return new_state, payoff

    final_state, payoffs = jax.lax.scan(
        step, initial_state, (event_types, year_fractions, rf_values, prnxt_schedule), unroll=8
    )
    return final_state, payoffs


# JIT-compiled version for single-contract use
simulate_lax_array_jit = jax.jit(simulate_lax_array)

# Vmapped version
batch_simulate_lax_vmap = jax.vmap(simulate_lax_array)


def batch_simulate_lax_auto(
    initial_states: LAXArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    prnxt_schedule: jnp.ndarray,
    params: LAXArrayParams,
) -> tuple[LAXArrayState, jnp.ndarray]:
    """Batched simulation using the optimal strategy for all backends.

    Uses the single-scan batch approach (``batch_simulate_lax``) which
    processes all contracts in shaped ``[B, T]`` arrays via a single
    ``lax.scan``.
    """
    return batch_simulate_lax(  # type: ignore[no-any-return]
        initial_states, event_types, year_fractions, rf_values, prnxt_schedule, params
    )


# ============================================================================
# Manually-batched simulation -- eliminates vmap dispatch overhead on CPU
# ============================================================================


@jax.jit
def batch_simulate_lax(
    initial_states: LAXArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    prnxt_schedule: jnp.ndarray,
    params: LAXArrayParams,
) -> tuple[LAXArrayState, jnp.ndarray]:
    """Batched LAX simulation without vmap -- single scan over ``[B]`` arrays.

    Args:
        initial_states: ``LAXArrayState`` with each field shape ``[B]``.
        event_types: ``[B, T]`` int32.
        year_fractions: ``[B, T]`` float32.
        rf_values: ``[B, T]`` float32.
        prnxt_schedule: ``[B, T]`` float32 -- per-event prnxt values.
        params: ``LAXArrayParams`` with each field shape ``[B]``.

    Returns:
        ``(final_states, payoffs)`` where ``payoffs`` is ``[B, T]``.
    """
    # Transpose to [T, B] so scan iterates over time steps
    et_t = event_types.T
    yf_t = year_fractions.T
    rf_t = rf_values.T
    prnxt_t = prnxt_schedule.T

    def step(
        states: LAXArrayState,
        inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[LAXArrayState, jnp.ndarray]:
        et, yf, rf, prnxt_val = inputs  # each [B]

        # Inject per-event prnxt at PR/PI/PRF events
        is_prnxt_event = (et == _PR_IDX) | (et == _PI_IDX) | (et == _PRF_IDX)
        current_prnxt = jnp.where(is_prnxt_event, prnxt_val, states.prnxt)
        states = states._replace(prnxt=current_prnxt)

        # Common sub-expression: interest accrual using ipcb
        accrue = states.ipac + yf * states.ipnr * states.ipcb

        # ---- Payoffs (branchless jnp.where dispatch) ----
        payoff = jnp.zeros_like(states.nt)

        # IED: role_sign * (-1) * nsc * notional_principal
        payoff = jnp.where(
            et == _IED_IDX,
            params.role_sign * (-1.0) * states.nsc * params.notional_principal,
            payoff,
        )
        # PR: nsc * effective_prnxt
        effective_prnxt = jnp.sign(states.prnxt) * jnp.minimum(
            jnp.abs(states.prnxt), jnp.abs(states.nt)
        )
        payoff = jnp.where(
            et == _PR_IDX,
            states.nsc * effective_prnxt,
            payoff,
        )
        # PI: -nsc * prnxt (receive additional principal)
        payoff = jnp.where(
            et == _PI_IDX,
            -states.nsc * states.prnxt,
            payoff,
        )
        # MD: nsc * nt + isc * ipac + feac
        payoff = jnp.where(
            et == _MD_IDX,
            states.nsc * states.nt + states.isc * states.ipac + states.feac,
            payoff,
        )
        # PP: rf
        payoff = jnp.where(et == _PP_IDX, rf, payoff)
        # PY: penalty
        payoff = jnp.where(
            et == _PY_IDX,
            jnp.where(
                params.penalty_type == 0,
                params.penalty_rate,
                yf * states.nt * params.penalty_rate,
            ),
            payoff,
        )
        # FP: fee payment
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
        # PR: nt -= effective_prnxt
        new_nt = jnp.where(et == _PR_IDX, states.nt - effective_prnxt, new_nt)
        # PI: nt += prnxt
        new_nt = jnp.where(et == _PI_IDX, states.nt + states.prnxt, new_nt)
        new_nt = jnp.where((et == _MD_IDX) | (et == _TD_IDX), 0.0, new_nt)
        new_nt = jnp.where(et == _PP_IDX, states.nt - rf, new_nt)
        new_nt = jnp.where(et == _IPCI_IDX, states.nt + accrue, new_nt)

        # ipnr: default unchanged
        new_ipnr = states.ipnr
        new_ipnr = jnp.where(et == _IED_IDX, params.nominal_interest_rate, new_ipnr)
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

        # ipac: accrue group, zero group, special IED
        is_accrue = (
            (et == _AD_IDX)
            | (et == _PR_IDX)
            | (et == _PI_IDX)
            | (et == _PP_IDX)
            | (et == _PY_IDX)
            | (et == _FP_IDX)
            | (et == _PRD_IDX)
            | (et == _PRF_IDX)
            | (et == _RR_IDX)
            | (et == _RRF_IDX)
            | (et == _SC_IDX)
            | (et == _CE_IDX)
            | (et == _IPCB_IDX)
        )
        is_zero_ipac = (et == _MD_IDX) | (et == _TD_IDX) | (et == _IP_IDX) | (et == _IPCI_IDX)
        new_ipac = states.ipac
        new_ipac = jnp.where(is_accrue, accrue, new_ipac)
        new_ipac = jnp.where(is_zero_ipac, 0.0, new_ipac)
        new_ipac = jnp.where(et == _IED_IDX, params.ied_ipac, new_ipac)

        # feac: zero at IED, MD, FP, TD
        new_feac = jnp.where(
            (et == _IED_IDX) | (et == _MD_IDX) | (et == _FP_IDX) | (et == _TD_IDX),
            0.0,
            states.feac,
        )

        # nsc, isc: only change at IED
        new_nsc = jnp.where(et == _IED_IDX, 1.0, states.nsc)
        new_isc = jnp.where(et == _IED_IDX, 1.0, states.isc)

        # prnxt: already updated from prnxt_schedule above; set at IED
        new_prnxt = jnp.where(
            et == _IED_IDX,
            params.role_sign * params.next_principal_redemption_amount,
            states.prnxt,
        )

        # ipcb: complex rules
        ied_ipcb = jnp.where(
            params.interest_calculation_base_amount > 0.0,
            params.role_sign * params.interest_calculation_base_amount,
            params.role_sign * params.notional_principal,
        )

        new_ipcb = states.ipcb
        new_ipcb = jnp.where(et == _IED_IDX, ied_ipcb, new_ipcb)
        # PR: ipcb = new_nt if mode NT/NTIED, unchanged if NTL
        pr_ipcb = jnp.where(params.ipcb_mode == IPCB_NTL, states.ipcb, new_nt)
        new_ipcb = jnp.where(et == _PR_IDX, pr_ipcb, new_ipcb)
        # PI: same ipcb update rule
        pi_nt = states.nt + states.prnxt
        pi_ipcb = jnp.where(params.ipcb_mode == IPCB_NTL, states.ipcb, pi_nt)
        new_ipcb = jnp.where(et == _PI_IDX, pi_ipcb, new_ipcb)
        # PP
        pp_nt = states.nt - rf
        pp_ipcb = jnp.where(params.ipcb_mode == IPCB_NTL, states.ipcb, pp_nt)
        new_ipcb = jnp.where(et == _PP_IDX, pp_ipcb, new_ipcb)
        # IPCI
        ipci_nt = states.nt + accrue
        ipci_ipcb = jnp.where(params.ipcb_mode == IPCB_NT, ipci_nt, states.ipcb)
        new_ipcb = jnp.where(et == _IPCI_IDX, ipci_ipcb, new_ipcb)
        # IPCB event
        new_ipcb = jnp.where(et == _IPCB_IDX, states.nt, new_ipcb)
        # MD/TD
        new_ipcb = jnp.where((et == _MD_IDX) | (et == _TD_IDX), 0.0, new_ipcb)

        new_state = LAXArrayState(
            nt=new_nt,
            ipnr=new_ipnr,
            ipac=new_ipac,
            feac=new_feac,
            nsc=new_nsc,
            isc=new_isc,
            prnxt=new_prnxt,
            ipcb=new_ipcb,
        )
        return new_state, payoff

    final_states, payoffs_t = jax.lax.scan(
        step, initial_states, (et_t, yf_t, rf_t, prnxt_t), unroll=8
    )
    return final_states, payoffs_t.T


# ============================================================================
# Pre-computation bridge -- Python -> JAX arrays
# ============================================================================


class _LAXRawPrecomputed(NamedTuple):
    """Pre-computed data for LAX, extending RawPrecomputed with prnxt schedule."""

    state: tuple[float, ...]
    event_types: list[int]
    year_fractions: list[float]
    rf_values: list[float]
    prnxt_values: list[float]  # per-event signed prnxt
    params: dict[str, float | int]


def _extract_params(attrs: ContractAttributes) -> LAXArrayParams:
    """Extract ``LAXArrayParams`` from ``ContractAttributes``."""
    role_sign = _get_role_sign(attrs.contract_role)
    nt = attrs.notional_principal or 0.0
    ipnr = attrs.nominal_interest_rate or 0.0

    # Pre-compute IED accrued interest
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

    # Initial PRNXT (first value from array or explicit)
    prnxt = _get_initial_prnxt(attrs)
    ipcba = attrs.interest_calculation_base_amount or 0.0

    return LAXArrayParams(
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
        next_principal_redemption_amount=jnp.array(prnxt, dtype=_F32),
        ipcb_mode=jnp.array(_encode_ipcb_mode(attrs), dtype=jnp.int32),
        interest_calculation_base_amount=jnp.array(ipcba, dtype=_F32),
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

    prnxt = _get_initial_prnxt(attrs)
    ipcba = attrs.interest_calculation_base_amount or 0.0

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
        "next_principal_redemption_amount": prnxt,
        "ipcb_mode": _encode_ipcb_mode(attrs),
        "interest_calculation_base_amount": ipcba,
    }


def _params_raw_to_jax(raw: dict[str, float | int]) -> LAXArrayParams:
    """Convert raw Python params to JAX LAXArrayParams."""
    _int_fields = {"fee_basis", "penalty_type", "ipcb_mode"}
    return LAXArrayParams(
        **{
            k: jnp.array(raw[k], dtype=jnp.int32 if k in _int_fields else _F32)
            for k in LAXArrayParams._fields
        }
    )


# ---------------------------------------------------------------------------
# LAX-specific helpers
# ---------------------------------------------------------------------------


def _get_initial_prnxt(attrs: ContractAttributes) -> float:
    """Get the initial (unsigned) prnxt for a LAX contract.

    Uses the first value from ``array_pr_next`` if available, otherwise
    falls back to ``next_principal_redemption_amount`` or 0.0.
    """
    if attrs.next_principal_redemption_amount is not None:
        return attrs.next_principal_redemption_amount
    if attrs.array_pr_next:
        return attrs.array_pr_next[0]
    return 0.0


def _get_prnxt_for_time(attrs: ContractAttributes, time: ActusDateTime) -> float:
    """Look up the unsigned prnxt value from array for a given event time.

    Returns the prnxt from the most recent array segment anchor at or before time.
    Returns the initial prnxt if no array is defined.
    """
    if not attrs.array_pr_anchor or not attrs.array_pr_next:
        return _get_initial_prnxt(attrs)
    prnxt_val = attrs.array_pr_next[0]
    for i, anchor in enumerate(attrs.array_pr_anchor):
        if time >= anchor and i < len(attrs.array_pr_next):
            prnxt_val = attrs.array_pr_next[i]
    return prnxt_val


# ---------------------------------------------------------------------------
# Schedule generation via scalar LAX contract
# ---------------------------------------------------------------------------


def _lax_schedule_via_scalar(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> list[tuple[int, _datetime, _datetime]]:
    """Generate LAX schedule using the scalar ExoticLinearAmortizerContract.

    This handles all the ARPRANX/ARPRCL/ARINCDEC complexity correctly.
    Returns lightweight ``(evt_idx, evt_dt, calc_dt)`` tuples.
    """
    from jactus.contracts.lax import ExoticLinearAmortizerContract

    contract = ExoticLinearAmortizerContract(attrs, rf_observer)
    schedule = contract.generate_event_schedule()

    result: list[tuple[int, _datetime, _datetime]] = []
    for event in schedule.events:
        evt_dt = _adt_to_dt(event.event_time)
        calc_dt = _adt_to_dt(event.calculation_time) if event.calculation_time else evt_dt
        result.append((event.event_type.index, evt_dt, calc_dt))
    return result


def _lax_init_state(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> tuple[float, float, float, float, float, float, float, float, _datetime]:
    """Compute initial LAX state as Python floats.

    Returns ``(nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb, sd_datetime)``.

    For mid-life contracts (IED < SD), uses the scalar LAX contract to
    handle the pre-simulation of events that already occurred.
    """
    from jactus.contracts.lax import ExoticLinearAmortizerContract

    sd = attrs.status_date
    ied = attrs.initial_exchange_date
    sd_dt = _adt_to_dt(sd)

    role_sign = _get_role_sign(attrs.contract_role)
    prnxt_unsigned = _get_initial_prnxt(attrs)
    prnxt = role_sign * prnxt_unsigned

    needs_post_ied = (ied and ied < sd) or attrs.purchase_date
    if needs_post_ied:
        contract = ExoticLinearAmortizerContract(attrs, rf_observer)
        state = contract.initialize_state()

        nt = float(state.nt)
        ipnr = float(state.ipnr)
        ipac = float(state.ipac)
        feac = float(state.feac)
        nsc = float(state.nsc)
        isc = float(state.isc)
        prnxt_val = float(state.prnxt) if state.prnxt is not None else prnxt
        ipcb = float(state.ipcb) if state.ipcb is not None else nt

        init_sd_dt = _adt_to_dt(state.sd) if state.sd else sd_dt
        return (nt, ipnr, ipac, feac, nsc, isc, prnxt_val, ipcb, init_sd_dt)

    # Pre-IED: all zeros except nsc=1, isc=1, prnxt
    ipcb = 0.0
    return (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, prnxt, ipcb, sd_dt)


# ---------------------------------------------------------------------------
# Core pre-computation
# ---------------------------------------------------------------------------


def _precompute_raw(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> _LAXRawPrecomputed:
    """Pre-compute all data as pure Python types (no JAX arrays).

    Uses the scalar ExoticLinearAmortizerContract for schedule generation
    to correctly handle ARPRANX/ARPRCL/ARPRNXT arrays, then extracts
    per-event prnxt values for the array-mode kernel.
    """
    from jactus.core.types import DayCountConvention

    # 1. Schedule via scalar contract
    schedule = _lax_schedule_via_scalar(attrs, rf_observer)

    # 2. State initialisation
    nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb, init_sd_dt = _lax_init_state(attrs, rf_observer)

    if not schedule:
        params_raw = _extract_params_raw(attrs)
        return _LAXRawPrecomputed(
            state=(nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb),
            event_types=[],
            year_fractions=[],
            rf_values=[],
            prnxt_values=[],
            params=params_raw,
        )

    # 3. Event types + vectorised year fractions
    event_type_list = [evt_idx for evt_idx, _, _ in schedule]
    dcc = attrs.day_count_convention or DayCountConvention.A360
    yf_list = _compute_vectorised_year_fractions(schedule, init_sd_dt, dcc)

    # 4. Risk factor pre-query
    rf_list = _prequery_risk_factors(schedule, attrs, rf_observer)

    # 5. Per-event prnxt values (signed)
    role_sign = _get_role_sign(attrs.contract_role)
    prnxt_list: list[float] = []
    for evt_idx, _evt_dt, calc_dt in schedule:
        if evt_idx in (_PR_IDX, _PI_IDX, _PRF_IDX):
            prnxt_unsigned = _get_prnxt_for_time(attrs, _dt_to_adt(calc_dt))
            prnxt_list.append(role_sign * prnxt_unsigned)
        else:
            prnxt_list.append(0.0)

    # 6. Extract params
    params_raw = _extract_params_raw(attrs)

    return _LAXRawPrecomputed(
        state=(nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb),
        event_types=event_type_list,
        year_fractions=yf_list,
        rf_values=rf_list,
        prnxt_values=prnxt_list,
        params=params_raw,
    )


def _raw_to_jax(
    raw: _LAXRawPrecomputed,
) -> tuple[LAXArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, LAXArrayParams]:
    """Convert raw pre-computed data to JAX arrays."""
    nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb = raw.state
    return (
        LAXArrayState(
            nt=jnp.array(nt, dtype=_F32),
            ipnr=jnp.array(ipnr, dtype=_F32),
            ipac=jnp.array(ipac, dtype=_F32),
            feac=jnp.array(feac, dtype=_F32),
            nsc=jnp.array(nsc, dtype=_F32),
            isc=jnp.array(isc, dtype=_F32),
            prnxt=jnp.array(prnxt, dtype=_F32),
            ipcb=jnp.array(ipcb, dtype=_F32),
        ),
        jnp.array(raw.event_types, dtype=jnp.int32),
        jnp.array(raw.year_fractions, dtype=_F32),
        jnp.array(raw.rf_values, dtype=_F32),
        jnp.array(raw.prnxt_values, dtype=_F32),
        _params_raw_to_jax(raw.params),
    )


def precompute_lax_arrays(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> tuple[LAXArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, LAXArrayParams]:
    """Pre-compute JAX arrays for array-mode LAX simulation.

    Uses the scalar ``ExoticLinearAmortizerContract`` for schedule generation
    to correctly handle ARPRANX/ARPRCL/ARPRNXT array schedules.

    Args:
        attrs: Contract attributes (must be LAX type).
        rf_observer: Risk factor observer (queried for RR/PP events).

    Returns:
        ``(initial_state, event_types, year_fractions, rf_values, prnxt_schedule, params)``
    """
    return _raw_to_jax(_precompute_raw(attrs, rf_observer))


# ============================================================================
# Batch / portfolio API
# ============================================================================


def _raw_list_to_jax_batch(
    raw_list: list[_LAXRawPrecomputed],
) -> tuple[
    LAXArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, LAXArrayParams, jnp.ndarray
]:
    """Convert a list of ``_LAXRawPrecomputed`` to padded JAX batch arrays.

    Returns:
        ``(states, event_types, year_fractions, rf_values, prnxt_schedule, params, masks)``
    """
    max_events = max(len(r.event_types) for r in raw_list)

    # State fields
    state_nt = [r.state[0] for r in raw_list]
    state_ipnr = [r.state[1] for r in raw_list]
    state_ipac = [r.state[2] for r in raw_list]
    state_feac = [r.state[3] for r in raw_list]
    state_nsc = [r.state[4] for r in raw_list]
    state_isc = [r.state[5] for r in raw_list]
    state_prnxt = [r.state[6] for r in raw_list]
    state_ipcb = [r.state[7] for r in raw_list]

    # Event arrays with padding
    et_batch: list[list[int]] = []
    yf_batch: list[list[float]] = []
    rf_batch: list[list[float]] = []
    prnxt_batch: list[list[float]] = []
    mask_batch: list[list[float]] = []

    for r in raw_list:
        n_events = len(r.event_types)
        pad_n = max_events - n_events
        et_batch.append(r.event_types + [NOP_EVENT_IDX] * pad_n)
        yf_batch.append(r.year_fractions + [0.0] * pad_n)
        rf_batch.append(r.rf_values + [0.0] * pad_n)
        prnxt_batch.append(r.prnxt_values + [0.0] * pad_n)
        mask_batch.append([1.0] * n_events + [0.0] * pad_n)

    # Param fields
    param_fields: dict[str, list[float | int]] = {k: [] for k in LAXArrayParams._fields}
    for r in raw_list:
        for k in LAXArrayParams._fields:
            param_fields[k].append(r.params[k])

    # Build NumPy arrays, then transfer to JAX
    batched_states = LAXArrayState(
        nt=jnp.asarray(np.array(state_nt, dtype=np.float32)),
        ipnr=jnp.asarray(np.array(state_ipnr, dtype=np.float32)),
        ipac=jnp.asarray(np.array(state_ipac, dtype=np.float32)),
        feac=jnp.asarray(np.array(state_feac, dtype=np.float32)),
        nsc=jnp.asarray(np.array(state_nsc, dtype=np.float32)),
        isc=jnp.asarray(np.array(state_isc, dtype=np.float32)),
        prnxt=jnp.asarray(np.array(state_prnxt, dtype=np.float32)),
        ipcb=jnp.asarray(np.array(state_ipcb, dtype=np.float32)),
    )

    batched_et = jnp.asarray(np.array(et_batch, dtype=np.int32))
    batched_yf = jnp.asarray(np.array(yf_batch, dtype=np.float32))
    batched_rf = jnp.asarray(np.array(rf_batch, dtype=np.float32))
    batched_prnxt = jnp.asarray(np.array(prnxt_batch, dtype=np.float32))
    batched_masks = jnp.asarray(np.array(mask_batch, dtype=np.float32))

    _int_fields = {"fee_basis", "penalty_type", "ipcb_mode"}
    batched_params = LAXArrayParams(
        **{
            k: jnp.asarray(
                np.array(
                    param_fields[k],
                    dtype=np.int32 if k in _int_fields else np.float32,
                )
            )
            for k in LAXArrayParams._fields
        }
    )

    return (
        batched_states,
        batched_et,
        batched_yf,
        batched_rf,
        batched_prnxt,
        batched_params,
        batched_masks,
    )


def prepare_lax_batch(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
) -> tuple[
    LAXArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, LAXArrayParams, jnp.ndarray
]:
    """Pre-compute and pad arrays for a batch of LAX contracts.

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.

    Returns:
        ``(initial_states, event_types, year_fractions, rf_values, prnxt_schedule, params, masks)``
    """
    raw_list = [_precompute_raw(attrs, obs) for attrs, obs in contracts]
    return _raw_list_to_jax_batch(raw_list)


def simulate_lax_portfolio(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
    discount_rate: float | None = None,
    year_fractions_from_valuation: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """End-to-end portfolio simulation with optional PV.

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.
        discount_rate: If provided, compute present values.
        year_fractions_from_valuation: ``(batch, max_events)`` year fractions
            from valuation date for PV discounting.

    Returns:
        Dict with ``payoffs``, ``masks``, ``final_states``, and optionally
        ``present_values`` and ``total_pv``.
    """
    (
        batched_states,
        batched_et,
        batched_yf,
        batched_rf,
        batched_prnxt,
        batched_params,
        batched_masks,
    ) = prepare_lax_batch(contracts)

    # Run batched simulation
    final_states, payoffs = batch_simulate_lax_auto(
        batched_states, batched_et, batched_yf, batched_rf, batched_prnxt, batched_params
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
        if year_fractions_from_valuation is not None:
            disc_yfs = year_fractions_from_valuation
        else:
            disc_yfs = jnp.cumsum(batched_yf, axis=1)
        discount_factors = 1.0 / (1.0 + discount_rate * disc_yfs)
        pvs = jnp.sum(masked_payoffs * discount_factors, axis=1)
        result["present_values"] = pvs
        result["total_pv"] = jnp.sum(pvs)

    return result
