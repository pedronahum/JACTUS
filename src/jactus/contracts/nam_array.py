"""Array-mode NAM simulation — JIT-compiled, vmap-able pure JAX.

This module provides a high-performance simulation path for NAM (Negative
Amortizer) contracts using ``jax.lax.scan`` for the event loop and
``jax.lax.switch`` for payoff/state-transition dispatch.  The entire
simulation kernel is JIT-compilable and can be vectorized across a
portfolio with ``jax.vmap``.

Architecture:
    Pre-computation (Python) -> Pure JAX kernel (jit + vmap)

    The existing ``NegativeAmortizerContract`` generates event schedules and
    initializes state (Python-level, runs once per contract).  This module
    converts the results to JAX arrays and runs the numerical simulation as
    a pure function.

Key differences from LAM:
    - PR (Principal Redemption) payoff is net of accrued interest:
      ``Nsc * (Prnxt - Ipac - Y * Ipnr * Ipcb)`` (can be negative for
      negative amortization).
    - PR state transition updates notional by the net principal portion:
      ``Nt -= (net_principal - total_accrued)``, and keeps ``Ipac`` at
      the total accrued value (NOT reset to zero).
    - All other events (IED, MD, IP, IPCI, IPCB, RR, etc.) are identical
      to LAM.

Example::

    from jactus.contracts.nam_array import precompute_nam_arrays, simulate_nam_array

    arrays = precompute_nam_arrays(attrs, rf_observer)
    final_state, payoffs = simulate_nam_array(*arrays)

    # Portfolio:
    from jactus.contracts.nam_array import simulate_nam_portfolio
    result = simulate_nam_portfolio(contracts, discount_rate=0.05)
"""

from __future__ import annotations

import math
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
    get_yf_fn,
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
    USE_DATE_ARRAY as _USE_DATE_ARRAY,
)
from jactus.contracts.array_common import (
    # Batch infrastructure
    RawPrecomputed as _RawPrecomputed,
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
    fast_schedule as _fast_schedule,
)
from jactus.contracts.array_common import (
    get_evt_priority as _get_evt_priority,
)
from jactus.contracts.array_common import (
    get_role_sign as _get_role_sign,
)
from jactus.contracts.array_common import (
    prequery_risk_factors as _prequery_risk_factors,
)
from jactus.core import (
    ContractAttributes,
)
from jactus.observers import RiskFactorObserver
from jactus.utilities.conventions import year_fraction

# ---------------------------------------------------------------------------
# IPCB mode encoding
# ---------------------------------------------------------------------------
IPCB_NT: int = 0  # Interest on current notional (tracks nt)
IPCB_NTIED: int = 1  # Interest on initial notional (fixed at IED value)
IPCB_NTL: int = 2  # Interest on lagged notional (updated at IPCB events)


def _encode_ipcb_mode(attrs: ContractAttributes) -> int:
    """Encode interest calculation base mode as integer."""
    ipcb = attrs.interest_calculation_base
    if ipcb is None or str(ipcb) == "NT":
        return IPCB_NT
    if str(ipcb) == "NTIED":
        return IPCB_NTIED
    if str(ipcb) == "NTL":
        return IPCB_NTL
    return IPCB_NT  # default


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class NAMArrayState(NamedTuple):
    """Minimal scan-loop state for NAM simulation.

    All fields are scalar ``jnp.ndarray`` (float32).  Same structure as
    ``LAMArrayState``: ``prnxt`` (next principal redemption amount, signed)
    and ``ipcb`` (interest calculation base).
    """

    nt: jnp.ndarray  # Notional principal (signed)
    ipnr: jnp.ndarray  # Nominal interest rate
    ipac: jnp.ndarray  # Accrued interest
    feac: jnp.ndarray  # Accrued fees
    nsc: jnp.ndarray  # Notional scaling multiplier
    isc: jnp.ndarray  # Interest scaling multiplier
    prnxt: jnp.ndarray  # Next principal redemption amount (signed)
    ipcb: jnp.ndarray  # Interest calculation base


class NAMArrayParams(NamedTuple):
    """Static contract parameters for NAM simulation.

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
    # NAM-specific params (same as LAM)
    next_principal_redemption_amount: jnp.ndarray  # unsigned PRNXT
    ipcb_mode: jnp.ndarray  # 0=NT, 1=NTIED, 2=NTL (int32)
    interest_calculation_base_amount: jnp.ndarray  # IPCBA (unsigned)


# ============================================================================
# Pure JAX payoff functions  (state, params, yf, rf) -> scalar payoff
# ============================================================================


def _accrue_interest_nam(state: NAMArrayState, yf: jnp.ndarray) -> jnp.ndarray:
    """Common sub-expression: ipac + yf * ipnr * ipcb (NAM uses ipcb)."""
    return state.ipac + yf * state.ipnr * state.ipcb


def _pof_ad(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_ied(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_IED: R(CNTRL) * (-1) * Nsc * NT.

    At IED, nsc=1.0, so effectively: role_sign * (-1) * NT.
    """
    return params.role_sign * (-1.0) * state.nsc * params.notional_principal


def _pof_pr(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_PR_NAM: Nsc * (Prnxt - Ipac - Y * Ipnr * Ipcb).

    Net of accrued interest.  If interest exceeds Prnxt, payoff is negative
    (borrower pays less than interest, remainder is capitalised).

    Clamped so that |net_payment| <= |Nt| (don't overshoot remaining notional).
    """
    total_accrued = _accrue_interest_nam(state, yf)
    net_payment = state.prnxt - total_accrued
    # Clamp: don't pay more principal than remaining notional
    net_payment = jnp.where(
        jnp.abs(net_payment) > jnp.abs(state.nt),
        state.nt,
        net_payment,
    )
    return state.nsc * net_payment


def _pof_md(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_MD: Nsc * Nt + Isc * Ipac + Feac."""
    return state.nsc * state.nt + state.isc * state.ipac + state.feac


def _pof_pp(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_PP: rf (pre-computed prepayment amount from observer)."""
    return rf


def _pof_py(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_PY: Penalty payment (type-dependent)."""
    pof_a = params.penalty_rate
    pof_ni = yf * state.nt * params.penalty_rate
    return jnp.where(params.penalty_type == 0, pof_a, pof_ni)


def _pof_fp(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
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
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_PRD: -(PPRD + Ipac + Y * Ipnr * Ipcb).

    Uses ipcb for interest accrual, not nt.
    """
    return (-1.0) * (params.price_at_purchase_date + _accrue_interest_nam(state, yf))


def _pof_td(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_TD: PTD + Ipac + Y * Ipnr * Ipcb.

    Uses ipcb for interest accrual, not nt.
    """
    return params.price_at_termination_date + _accrue_interest_nam(state, yf)


def _pof_ip(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    """POF_IP: Isc * (Ipac + Y * Ipnr * Ipcb).

    Uses ipcb for interest accrual, not nt.
    """
    return state.isc * _accrue_interest_nam(state, yf)


def _pof_ipci(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_ipcb(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_rr(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_rrf(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_sc(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_ce(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


def _pof_noop(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> jnp.ndarray:
    return jnp.array(0.0, dtype=_F32)


# ============================================================================
# Pure JAX state transition functions  (state, params, yf, rf) -> new state
# ============================================================================


def _stf_ad(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    return state._replace(ipac=_accrue_interest_nam(state, yf))


def _stf_ied(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    """STF_IED: Initialize all state variables.

    Sets nt, ipnr, ipac, feac=0, nsc=1, isc=1.
    Sets prnxt = role_sign * PRNXT.
    Sets ipcb = role_sign * NT (or role_sign * IPCBA if specified).
    """
    nt = params.role_sign * params.notional_principal
    prnxt = params.role_sign * params.next_principal_redemption_amount
    # IPCB: use IPCBA if specified (non-zero), otherwise use NT
    ipcb = jnp.where(
        params.interest_calculation_base_amount > 0.0,
        params.role_sign * params.interest_calculation_base_amount,
        nt,
    )
    return NAMArrayState(
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
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    """STF_PR_NAM: Principal Redemption with negative amortization.

    Total accrued = Ipac + Y * Ipnr * Ipcb
    Net principal reduction = Prnxt - total_accrued  (can be negative)
    Nt = Nt - clamp(net_principal_reduction)
    Ipac = total_accrued (kept, NOT reset to zero)
    Ipcb = Nt (if mode NT or NTIED), unchanged if NTL
    """
    total_accrued = _accrue_interest_nam(state, yf)
    net_principal_reduction = state.prnxt - total_accrued
    # Clamp: don't reduce notional past zero
    net_principal_reduction = jnp.where(
        jnp.abs(net_principal_reduction) > jnp.abs(state.nt),
        state.nt,
        net_principal_reduction,
    )
    new_nt = state.nt - net_principal_reduction
    # IPCB update: NT/NTIED track current notional, NTL unchanged
    new_ipcb = jnp.where(
        params.ipcb_mode == IPCB_NTL,
        state.ipcb,
        new_nt,
    )
    return state._replace(nt=new_nt, ipac=total_accrued, ipcb=new_ipcb)


def _stf_md(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    return state._replace(
        nt=jnp.array(0.0, dtype=_F32),
        ipac=jnp.array(0.0, dtype=_F32),
        feac=jnp.array(0.0, dtype=_F32),
        ipcb=jnp.array(0.0, dtype=_F32),
    )


def _stf_pp(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    """STF_PP: Prepayment - accrue interest, reduce notional, update IPCB."""
    new_ipac = _accrue_interest_nam(state, yf)
    new_nt = state.nt - rf  # rf = prepayment amount
    # IPCB update: NT/NTIED track current notional, NTL unchanged
    new_ipcb = jnp.where(
        params.ipcb_mode == IPCB_NTL,
        state.ipcb,
        new_nt,
    )
    return state._replace(nt=new_nt, ipac=new_ipac, ipcb=new_ipcb)


def _stf_py(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    return state._replace(ipac=_accrue_interest_nam(state, yf))


def _stf_fp(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    return state._replace(
        ipac=_accrue_interest_nam(state, yf),
        feac=jnp.array(0.0, dtype=_F32),
    )


def _stf_prd(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    return state._replace(ipac=_accrue_interest_nam(state, yf))


def _stf_td(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    return state._replace(
        nt=jnp.array(0.0, dtype=_F32),
        ipac=jnp.array(0.0, dtype=_F32),
        feac=jnp.array(0.0, dtype=_F32),
        ipcb=jnp.array(0.0, dtype=_F32),
    )


def _stf_ip(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    return state._replace(ipac=jnp.array(0.0, dtype=_F32))


def _stf_ipci(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    """STF_IPCI: Interest Capitalization.

    Nt = Nt + total_accrued
    Ipac = 0
    Ipcb = Nt if mode NT, otherwise unchanged
    """
    total_accrued = _accrue_interest_nam(state, yf)
    new_nt = state.nt + total_accrued
    new_ipcb = jnp.where(params.ipcb_mode == IPCB_NT, new_nt, state.ipcb)
    return state._replace(
        nt=new_nt,
        ipac=jnp.array(0.0, dtype=_F32),
        ipcb=new_ipcb,
    )


def _stf_ipcb(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    """STF_IPCB: Interest Calculation Base fixing (NTL mode).

    Ipcb = Nt
    Ipac = accrue (using old ipcb)
    """
    new_ipac = _accrue_interest_nam(state, yf)
    return state._replace(ipcb=state.nt, ipac=new_ipac)


def _stf_rr(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    """STF_RR: Rate Reset - same as LAM but accrual uses ipcb."""
    new_ipac = _accrue_interest_nam(state, yf)
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
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    new_ipac = _accrue_interest_nam(state, yf)
    return state._replace(ipnr=params.rate_reset_next, ipac=new_ipac)


def _stf_sc(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    return state._replace(ipac=_accrue_interest_nam(state, yf))


def _stf_ce(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
    return state._replace(ipac=_accrue_interest_nam(state, yf))


def _stf_noop(
    state: NAMArrayState, params: NAMArrayParams, yf: jnp.ndarray, rf: jnp.ndarray
) -> NAMArrayState:
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
    _pof_noop,  # 4  PI   (not used in NAM)
    _pof_pp,    # 5  PP
    _pof_py,    # 6  PY
    _pof_noop,  # 7  PRF  (not used in NAM)
    _pof_fp,    # 8  FP
    _pof_prd,   # 9  PRD
    _pof_td,    # 10 TD
    _pof_ip,    # 11 IP
    _pof_ipci,  # 12 IPCI
    _pof_ipcb,  # 13 IPCB
    _pof_rr,    # 14 RR
    _pof_rrf,   # 15 RRF
    _pof_noop,  # 16 DV   (not used in NAM)
    _pof_noop,  # 17 DVF  (not used in NAM)
    _pof_sc,    # 18 SC
    _pof_noop,  # 19 STD  (not used in NAM)
    _pof_noop,  # 20 XD   (not used in NAM)
    _pof_ce,    # 21 CE
    _pof_noop,  # 22 IPFX (not used in NAM)
    _pof_noop,  # 23 IPFL (not used in NAM)
    _pof_noop,  # 24 NOP  (padding)
]

_STF_TABLE: list[Any] = [
    _stf_ad,    # 0  AD
    _stf_ied,   # 1  IED
    _stf_md,    # 2  MD
    _stf_pr,    # 3  PR
    _stf_noop,  # 4  PI
    _stf_pp,    # 5  PP
    _stf_py,    # 6  PY
    _stf_noop,  # 7  PRF
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


def simulate_nam_array(
    initial_state: NAMArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: NAMArrayParams,
) -> tuple[NAMArrayState, jnp.ndarray]:
    """Run a NAM simulation as a pure JAX function.

    This function is JIT-compilable and vmap-able.

    Args:
        initial_state: Starting state (8 scalar fields).
        event_types: ``(num_events,)`` int32 -- ``EventType.index`` values.
        year_fractions: ``(num_events,)`` float32 -- pre-computed YF per event.
        rf_values: ``(num_events,)`` float32 -- pre-computed risk factor values
            (market rate for RR, prepayment amount for PP, 0.0 otherwise).
        params: Static contract parameters.

    Returns:
        ``(final_state, payoffs)`` where payoffs is ``(num_events,)`` float32.
    """

    def step(
        state: NAMArrayState, inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> tuple[NAMArrayState, jnp.ndarray]:
        evt_idx, yf, rf = inputs
        payoff = jax.lax.switch(evt_idx, _POF_TABLE, state, params, yf, rf)
        new_state = jax.lax.switch(evt_idx, _STF_TABLE, state, params, yf, rf)
        return new_state, payoff

    final_state, payoffs = jax.lax.scan(
        step, initial_state, (event_types, year_fractions, rf_values), unroll=8
    )
    return final_state, payoffs


# JIT-compiled version for single-contract use
simulate_nam_array_jit = jax.jit(simulate_nam_array)

# Vmapped version (kept as fallback, e.g. for GPU where vmap is efficient)
batch_simulate_nam_vmap = jax.vmap(simulate_nam_array)


def batch_simulate_nam_auto(
    initial_states: NAMArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: NAMArrayParams,
) -> tuple[NAMArrayState, jnp.ndarray]:
    """Batched simulation using the optimal strategy for all backends.

    Uses the single-scan batch approach (``batch_simulate_nam``) which
    processes all contracts in shaped ``[B, T]`` arrays via a single
    ``lax.scan``.  This is faster than ``vmap`` on CPU, GPU, *and* TPU
    because it avoids per-contract dispatch overhead.
    """
    return batch_simulate_nam(initial_states, event_types, year_fractions, rf_values, params)  # type: ignore[no-any-return]


# ============================================================================
# Manually-batched simulation -- eliminates vmap dispatch overhead on CPU
# ============================================================================


@jax.jit
def batch_simulate_nam(
    initial_states: NAMArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: NAMArrayParams,
) -> tuple[NAMArrayState, jnp.ndarray]:
    """Batched NAM simulation without vmap -- single scan over ``[B]`` arrays.

    Eliminates JAX vmap CPU dispatch overhead by operating directly on
    batch-dimensioned arrays.  Each scan step computes all event-type
    outcomes for all contracts simultaneously using branchless
    ``jnp.where`` dispatch.

    Args:
        initial_states: ``NAMArrayState`` with each field shape ``[B]``.
        event_types: ``[B, T]`` int32 -- event type indices per contract.
        year_fractions: ``[B, T]`` float32.
        rf_values: ``[B, T]`` float32.
        params: ``NAMArrayParams`` with each field shape ``[B]``.

    Returns:
        ``(final_states, payoffs)`` where ``payoffs`` is ``[B, T]``.
    """
    # Transpose to [T, B] so scan iterates over time steps
    et_t = event_types.T
    yf_t = year_fractions.T
    rf_t = rf_values.T

    def step(
        states: NAMArrayState,
        inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[NAMArrayState, jnp.ndarray]:
        et, yf, rf = inputs  # each [B]

        # Common sub-expression: interest accrual using ipcb (not nt)
        accrue = states.ipac + yf * states.ipnr * states.ipcb

        # ---- Payoffs (branchless jnp.where dispatch) ----
        payoff = jnp.zeros_like(states.nt)

        # IED: role_sign * (-1) * nsc * notional_principal
        payoff = jnp.where(
            et == _IED_IDX,
            params.role_sign * (-1.0) * states.nsc * params.notional_principal,
            payoff,
        )
        # PR (NAM): nsc * (prnxt - total_accrued), clamped
        pr_net_payment = states.prnxt - accrue
        pr_net_payment = jnp.where(
            jnp.abs(pr_net_payment) > jnp.abs(states.nt),
            states.nt,
            pr_net_payment,
        )
        payoff = jnp.where(
            et == _PR_IDX,
            states.nsc * pr_net_payment,
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
        # PRD: -(price + accrue)  -- uses ipcb via accrue
        payoff = jnp.where(
            et == _PRD_IDX,
            (-1.0) * (params.price_at_purchase_date + accrue),
            payoff,
        )
        # TD: price + accrue  -- uses ipcb via accrue
        payoff = jnp.where(
            et == _TD_IDX,
            params.price_at_termination_date + accrue,
            payoff,
        )
        # IP: isc * accrue  -- uses ipcb via accrue
        payoff = jnp.where(et == _IP_IDX, states.isc * accrue, payoff)

        # ---- State transitions (branchless) ----

        # nt: default unchanged
        new_nt = states.nt
        new_nt = jnp.where(
            et == _IED_IDX,
            params.role_sign * params.notional_principal,
            new_nt,
        )
        # PR (NAM): nt -= clamped net_principal_reduction
        # pr_net_payment already clamped above (same value used for payoff)
        new_nt = jnp.where(et == _PR_IDX, states.nt - pr_net_payment, new_nt)
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
        #   accrue group: AD, PP, PY, FP, PRD, RR, RRF, SC, CE, IPCB
        #   accrue group (NAM PR): PR sets ipac = total_accrued (NOT zero)
        #   zero group:   MD, TD, IP, IPCI
        #   special:      IED -> ied_ipac
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
            | (et == _IPCB_IDX)
        )
        is_zero_ipac = (et == _MD_IDX) | (et == _TD_IDX) | (et == _IP_IDX) | (et == _IPCI_IDX)
        new_ipac = states.ipac
        new_ipac = jnp.where(is_accrue, accrue, new_ipac)
        # NAM PR: ipac = total_accrued (same as accrue)
        new_ipac = jnp.where(et == _PR_IDX, accrue, new_ipac)
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

        # prnxt: set at IED, otherwise unchanged
        new_prnxt = jnp.where(
            et == _IED_IDX,
            params.role_sign * params.next_principal_redemption_amount,
            states.prnxt,
        )

        # ipcb: complex rules depending on event type and mode
        # IED: use IPCBA if specified, else NT
        ied_ipcb = jnp.where(
            params.interest_calculation_base_amount > 0.0,
            params.role_sign * params.interest_calculation_base_amount,
            params.role_sign * params.notional_principal,
        )

        new_ipcb = states.ipcb
        new_ipcb = jnp.where(et == _IED_IDX, ied_ipcb, new_ipcb)
        # PR (NAM): ipcb = new_nt if mode NT/NTIED, unchanged if NTL
        # Note: new_nt for PR is states.nt - pr_net_payment
        pr_new_nt = states.nt - pr_net_payment
        pr_ipcb = jnp.where(params.ipcb_mode == IPCB_NTL, states.ipcb, pr_new_nt)
        new_ipcb = jnp.where(et == _PR_IDX, pr_ipcb, new_ipcb)
        # PP: same as PR for ipcb update
        pp_nt = states.nt - rf  # new_nt after PP
        pp_ipcb = jnp.where(params.ipcb_mode == IPCB_NTL, states.ipcb, pp_nt)
        new_ipcb = jnp.where(et == _PP_IDX, pp_ipcb, new_ipcb)
        # IPCI: ipcb = new_nt if mode NT, otherwise unchanged
        ipci_nt = states.nt + accrue  # new_nt after IPCI
        ipci_ipcb = jnp.where(params.ipcb_mode == IPCB_NT, ipci_nt, states.ipcb)
        new_ipcb = jnp.where(et == _IPCI_IDX, ipci_ipcb, new_ipcb)
        # IPCB event: always set ipcb = nt (current notional before event)
        new_ipcb = jnp.where(et == _IPCB_IDX, states.nt, new_ipcb)
        # MD/TD: ipcb = 0
        new_ipcb = jnp.where((et == _MD_IDX) | (et == _TD_IDX), 0.0, new_ipcb)

        new_state = NAMArrayState(
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

    final_states, payoffs_t = jax.lax.scan(step, initial_states, (et_t, yf_t, rf_t), unroll=8)
    # payoffs_t is [T, B]; transpose back to [B, T]
    return final_states, payoffs_t.T


# ============================================================================
# Pre-computation bridge -- Python -> JAX arrays
# ============================================================================


def _extract_params(attrs: ContractAttributes) -> NAMArrayParams:
    """Extract ``NAMArrayParams`` from ``ContractAttributes``."""
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

    # Use auto-calculated PRNXT if not explicitly specified
    prnxt = _compute_prnxt(attrs)
    ipcba = attrs.interest_calculation_base_amount or 0.0

    return NAMArrayParams(
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

    # Use auto-calculated PRNXT if not explicitly specified
    prnxt = _compute_prnxt(attrs)
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


def _params_raw_to_jax(raw: dict[str, float | int]) -> NAMArrayParams:
    """Convert raw Python params to JAX NAMArrayParams."""
    _int_fields = {"fee_basis", "penalty_type", "ipcb_mode"}
    return NAMArrayParams(
        **{
            k: jnp.array(raw[k], dtype=jnp.int32 if k in _int_fields else _F32)
            for k in NAMArrayParams._fields
        }
    )


# ---------------------------------------------------------------------------
# Fast schedule generation -- bypasses NegativeAmortizerContract entirely
# ---------------------------------------------------------------------------


def _fast_nam_schedule(
    attrs: ContractAttributes,
) -> list[tuple[int, _datetime, _datetime]]:
    """Generate NAM schedule as lightweight (evt_idx, evt_dt, calc_dt) tuples.

    Replicates the logic of ``NegativeAmortizerContract.generate_event_schedule``
    without creating ``ContractEvent`` objects or a ``NegativeAmortizerContract``.

    NAM schedule generation is identical to LAM.
    """
    from jactus.core.types import BusinessDayConvention

    ied = attrs.initial_exchange_date
    md = attrs.maturity_date
    sd = attrs.status_date
    assert ied is not None

    bdc = attrs.business_day_convention

    # For non-NULL BDC or non-SD EOMC, fall back to the full path
    has_bdc = bdc is not None and bdc != BusinessDayConvention.NULL
    has_eomc = (
        attrs.end_of_month_convention is not None and attrs.end_of_month_convention.value != "SD"
    )
    if has_bdc or has_eomc:
        return _fallback_nam_schedule(attrs)

    # Calculate MD if not provided
    if md is None and attrs.principal_redemption_cycle:
        prnxt = attrs.next_principal_redemption_amount or 0.0
        nt = attrs.notional_principal or 0.0
        if prnxt > 0:
            n_periods = math.ceil(nt / prnxt)
            pr_anchor = attrs.principal_redemption_anchor or ied
            pr_dates = _fast_schedule(pr_anchor, attrs.principal_redemption_cycle, None)
            if not pr_dates:
                # Generate enough dates via far_end
                from jactus.core.time import add_period

                far_end = ied
                for _ in range(n_periods + 2):
                    far_end = add_period(far_end, attrs.principal_redemption_cycle)
                pr_dates = _fast_schedule(pr_anchor, attrs.principal_redemption_cycle, far_end)
            pr_dates_from_ied = [d for d in pr_dates if d >= _adt_to_dt(ied)]
            if len(pr_dates_from_ied) >= n_periods:
                md_dt = pr_dates_from_ied[n_periods - 1]
                md = _dt_to_adt(md_dt)

    if md is None:
        return _fallback_nam_schedule(attrs)

    ied_dt = _adt_to_dt(ied)
    md_dt = _adt_to_dt(md)
    sd_dt = _adt_to_dt(sd)

    events: list[tuple[int, _datetime, _datetime]] = []

    # IED
    if ied_dt >= sd_dt:
        events.append((_IED_IDX, ied_dt, ied_dt))

    # PR: Principal Redemption schedule
    if attrs.principal_redemption_cycle:
        pr_anchor = attrs.principal_redemption_anchor or ied
        pr_dates = _fast_schedule(pr_anchor, attrs.principal_redemption_cycle, md)
        pr_cycle_str = attrs.principal_redemption_cycle or ""
        if pr_cycle_str.endswith("+") and pr_dates and pr_dates[-1] != md_dt:
            pr_dates = pr_dates[:-1]
        for dt in pr_dates:
            if dt >= md_dt:
                break
            if dt >= ied_dt:
                events.append((_PR_IDX, dt, dt))

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

    # IPCB: only if mode is NTL and cycle is specified
    if _encode_ipcb_mode(attrs) == IPCB_NTL and attrs.interest_calculation_base_cycle:
        ipcb_anchor = attrs.interest_calculation_base_anchor or ied
        ipcb_dates = _fast_schedule(ipcb_anchor, attrs.interest_calculation_base_cycle, md)
        ipcb_cycle_str = attrs.interest_calculation_base_cycle or ""
        if ipcb_cycle_str.endswith("+") and ipcb_dates and ipcb_dates[-1] != md_dt:
            ipcb_dates = ipcb_dates[:-1]
        for dt in ipcb_dates:
            if dt > ied_dt and dt < md_dt:
                events.append((_IPCB_IDX, dt, dt))

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


def _fallback_nam_schedule(
    attrs: ContractAttributes,
) -> list[tuple[int, _datetime, _datetime]]:
    """Fall back to the full NegativeAmortizerContract for BDC/EOMC cases."""
    from jactus.contracts.nam import NegativeAmortizerContract
    from jactus.observers import ConstantRiskFactorObserver

    rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
    contract = NegativeAmortizerContract(attrs, rf_obs)
    schedule = contract.generate_event_schedule()
    result: list[tuple[int, _datetime, _datetime]] = []
    for event in schedule.events:
        evt_dt = _adt_to_dt(event.event_time)
        calc_dt = _adt_to_dt(event.calculation_time) if event.calculation_time else evt_dt
        result.append((event.event_type.index, evt_dt, calc_dt))
    return result


def _compute_prnxt(attrs: ContractAttributes) -> float:
    """Compute the next principal redemption amount (unsigned).

    If PRNXT is provided in attributes, returns it directly.
    Otherwise auto-calculates as NT / n_periods.
    """
    if attrs.next_principal_redemption_amount is not None:
        return attrs.next_principal_redemption_amount

    ied = attrs.initial_exchange_date
    md = attrs.maturity_date
    nt = attrs.notional_principal or 0.0
    if not (nt and attrs.principal_redemption_cycle and md and ied):
        return 0.0

    # Count PR periods
    pr_anchor = attrs.principal_redemption_anchor or ied
    pr_dates = _fast_schedule(pr_anchor, attrs.principal_redemption_cycle, md)
    pr_dates = [d for d in pr_dates if d <= _adt_to_dt(md)]
    if _adt_to_dt(md) not in pr_dates:
        pr_dates.append(_adt_to_dt(md))
    n_periods = len(pr_dates)
    return nt / n_periods if n_periods > 0 else 0.0


def _fast_nam_init_state(
    attrs: ContractAttributes,
) -> tuple[float, float, float, float, float, float, float, float, _datetime]:
    """Compute initial NAM state as Python floats.

    Returns ``(nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb, sd_datetime)``.

    For mid-life contracts (IED < SD), falls back to the scalar
    ``NegativeAmortizerContract`` to handle the pre-simulation of PR
    events that already occurred.
    """
    sd = attrs.status_date
    ied = attrs.initial_exchange_date
    sd_dt = _adt_to_dt(sd)

    role_sign = _get_role_sign(attrs.contract_role)
    prnxt_unsigned = _compute_prnxt(attrs)
    prnxt = role_sign * prnxt_unsigned

    needs_post_ied = (ied and ied < sd) or attrs.purchase_date
    if needs_post_ied:
        # Fall back to scalar NAM contract to handle mid-life initialization,
        # since PR events between IED and SD reduce notional and ipcb.
        from jactus.contracts.nam import NegativeAmortizerContract
        from jactus.observers import ConstantRiskFactorObserver

        rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
        contract = NegativeAmortizerContract(attrs, rf_obs)
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
# Core pre-computation (Python scalar path)
# ---------------------------------------------------------------------------


def _precompute_raw(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> _RawPrecomputed:
    """Pre-compute all data as pure Python types (no JAX arrays).

    This is the core pre-computation that can be batched efficiently --
    all JAX array creation is deferred to the caller.
    """
    if _USE_DATE_ARRAY:
        return _precompute_raw_da(attrs, rf_observer)

    from jactus.core.types import DayCountConvention

    # 1. Fast schedule generation (no contract object)
    schedule = _fast_nam_schedule(attrs)

    # 2. Fast state initialization
    nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb, init_sd_dt = _fast_nam_init_state(attrs)

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
        state=(nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb),
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

    Schedule generation reuses ``_fast_nam_schedule`` (Python business logic),
    but year fractions are computed in a single vectorised NumPy pass.
    """
    from jactus.core.types import DayCountConvention

    # 1. Schedule
    schedule = _fast_nam_schedule(attrs)

    # 2. State initialisation
    nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb, init_sd_dt = _fast_nam_init_state(attrs)

    if not schedule:
        params_raw = _extract_params_raw(attrs)
        return _RawPrecomputed(
            state=(nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb),
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
        state=(nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb),
        event_types=event_type_list,
        year_fractions=yf_list,
        rf_values=rf_list,
        params=params_raw,
    )


def _raw_to_jax(
    raw: _RawPrecomputed,
) -> tuple[NAMArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, NAMArrayParams]:
    """Convert raw pre-computed data to JAX arrays."""
    nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb = raw.state
    return (
        NAMArrayState(
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
        _params_raw_to_jax(raw.params),
    )


def precompute_nam_arrays(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> tuple[NAMArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, NAMArrayParams]:
    """Pre-compute JAX arrays for array-mode NAM simulation.

    Generates the event schedule and initial state directly from attributes
    (bypassing ``NegativeAmortizerContract``), then converts to JAX arrays
    suitable for ``simulate_nam_array``.

    Args:
        attrs: Contract attributes (must be NAM type).
        rf_observer: Risk factor observer (queried for RR/PP events).

    Returns:
        ``(initial_state, event_types, year_fractions, rf_values, params)``
    """
    return _raw_to_jax(_precompute_raw(attrs, rf_observer))


# ============================================================================
# Batch / portfolio API
# ============================================================================


def _raw_list_to_jax_batch(
    raw_list: list[_RawPrecomputed],
) -> tuple[NAMArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, NAMArrayParams, jnp.ndarray]:
    """Convert a list of ``_RawPrecomputed`` to padded JAX batch arrays.

    Pads shorter contracts with NOP events and builds NumPy arrays first
    (fast C-level construction) then transfers to JAX via ``jnp.asarray``.
    """
    max_events = max(len(r.event_types) for r in raw_list)

    # State fields: (batch,) each -- NAM has 8 state fields
    state_nt = [r.state[0] for r in raw_list]
    state_ipnr = [r.state[1] for r in raw_list]
    state_ipac = [r.state[2] for r in raw_list]
    state_feac = [r.state[3] for r in raw_list]
    state_nsc = [r.state[4] for r in raw_list]
    state_isc = [r.state[5] for r in raw_list]
    state_prnxt = [r.state[6] for r in raw_list]
    state_ipcb = [r.state[7] for r in raw_list]

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
    param_fields: dict[str, list[float | int]] = {k: [] for k in NAMArrayParams._fields}
    for r in raw_list:
        for k in NAMArrayParams._fields:
            param_fields[k].append(r.params[k])

    # Build NumPy arrays first (fast C-level), then transfer to JAX
    batched_states = NAMArrayState(
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
    batched_masks = jnp.asarray(np.array(mask_batch, dtype=np.float32))

    _int_fields = {"fee_basis", "penalty_type", "ipcb_mode"}
    batched_params = NAMArrayParams(
        **{
            k: jnp.asarray(
                np.array(
                    param_fields[k],
                    dtype=np.int32 if k in _int_fields else np.float32,
                )
            )
            for k in NAMArrayParams._fields
        }
    )

    return batched_states, batched_et, batched_yf, batched_rf, batched_params, batched_masks


def prepare_nam_batch(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
) -> tuple[NAMArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, NAMArrayParams, jnp.ndarray]:
    """Pre-compute and pad arrays for a batch of NAM contracts.

    Uses per-contract Python pre-computation (sequential path).
    NAM contracts are not batch-schedule-eligible due to the
    more complex schedule structure (PR + IP + IPCB events).

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.

    Returns:
        ``(initial_states, event_types, year_fractions, rf_values, params, masks)``
        where each array has a leading batch dimension.
    """
    raw_list = [_precompute_raw(attrs, obs) for attrs, obs in contracts]
    return _raw_list_to_jax_batch(raw_list)


def simulate_nam_portfolio(
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
    ) = prepare_nam_batch(contracts)

    # Run batched simulation
    final_states, payoffs = batch_simulate_nam_auto(
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
