"""Array-mode ANN simulation -- JIT-compiled, vmap-able pure JAX.

This module provides a high-performance simulation path for ANN (Annuity)
contracts using the NAM array kernel.  ANN is identical to NAM at the
simulation level -- the only difference is how ``prnxt`` (the next
principal redemption amount) is computed:

* **NAM/LAM**: ``prnxt = NT / n_periods`` (uniform principal repayment).
* **ANN**: ``prnxt`` is calculated via the ACTUS annuity formula so that
  the *total* payment (principal + interest) stays constant across
  periods.

Architecture:
    Pre-computation (Python, annuity formula) -> NAM Pure JAX kernel (jit + vmap)

    The pre-computation phase leverages the scalar ``AnnuityContract`` to
    compute ``prnxt`` via the ACTUS annuity formula and to generate the
    event schedule (which includes PRF events).  The numerical simulation
    kernel is identical to NAM -- we simply reuse ``simulate_nam_array``,
    ``batch_simulate_nam``, etc.

Example::

    from jactus.contracts.ann_array import precompute_ann_arrays, simulate_ann_array

    arrays = precompute_ann_arrays(attrs, rf_observer)
    final_state, payoffs = simulate_ann_array(*arrays)

    # Portfolio:
    from jactus.contracts.ann_array import simulate_ann_portfolio
    result = simulate_ann_portfolio(contracts, discount_rate=0.05)
"""

from __future__ import annotations

from datetime import datetime as _datetime
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jactus.contracts.array_common import (
    F32 as _F32,
)

# Import shared infrastructure from array_common
from jactus.contracts.array_common import (
    NOP_EVENT_IDX,
    # Schedule helpers
    get_yf_fn,
)
from jactus.contracts.array_common import (
    PRF_IDX as _PRF_IDX,
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
    get_role_sign as _get_role_sign,
)
from jactus.contracts.array_common import (
    prequery_risk_factors as _prequery_risk_factors,
)

# ---------------------------------------------------------------------------
# Reuse NAM's kernel, state, and params -- ANN is numerically identical
# ---------------------------------------------------------------------------
from jactus.contracts.nam_array import (
    NAMArrayParams,
    NAMArrayState,
    _encode_ipcb_mode,
    _params_raw_to_jax,
    batch_simulate_nam,
    batch_simulate_nam_auto,
    simulate_nam_array,
    simulate_nam_array_jit,
)
from jactus.core import (
    ContractAttributes,
)
from jactus.observers import RiskFactorObserver
from jactus.utilities.conventions import year_fraction

# Type aliases for public API clarity
ANNArrayState = NAMArrayState
ANNArrayParams = NAMArrayParams


# ============================================================================
# Simulation entry points (thin wrappers around NAM)
# ============================================================================


def simulate_ann_array(
    initial_state: ANNArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: ANNArrayParams,
) -> tuple[ANNArrayState, jnp.ndarray]:
    """Run an ANN simulation as a pure JAX function.

    Delegates to ``simulate_nam_array`` -- the kernel is identical.

    Args:
        initial_state: Starting state (8 scalar fields).
        event_types: ``(num_events,)`` int32 -- ``EventType.index`` values.
        year_fractions: ``(num_events,)`` float32 -- pre-computed YF per event.
        rf_values: ``(num_events,)`` float32 -- pre-computed risk factor values.
        params: Static contract parameters.

    Returns:
        ``(final_state, payoffs)`` where payoffs is ``(num_events,)`` float32.
    """
    return simulate_nam_array(initial_state, event_types, year_fractions, rf_values, params)


simulate_ann_array_jit = simulate_nam_array_jit

batch_simulate_ann = batch_simulate_nam
batch_simulate_ann_auto = batch_simulate_nam_auto
batch_simulate_ann_vmap = jax.vmap(simulate_ann_array)


# ============================================================================
# Pre-computation bridge -- Python -> JAX arrays
# ============================================================================


def _ann_compute_prnxt(attrs: ContractAttributes, rf_observer: RiskFactorObserver) -> float:
    """Compute the ANN ``prnxt`` using the scalar AnnuityContract.

    Instantiates the scalar ``AnnuityContract``, calls ``initialize_state()``,
    and extracts the auto-calculated ``prnxt`` value.  This leverages the
    existing (well-tested) annuity formula implementation.

    Returns:
        Unsigned ``prnxt`` value (positive).
    """
    from jactus.contracts.ann import AnnuityContract

    # If PRNXT is explicitly provided, just return it
    if attrs.next_principal_redemption_amount is not None:
        return attrs.next_principal_redemption_amount

    # Use the scalar AnnuityContract to compute prnxt via the annuity formula
    contract = AnnuityContract(attrs, rf_observer)
    state = contract.initialize_state()

    prnxt_signed = float(state.prnxt) if state.prnxt is not None else 0.0
    # Return unsigned value
    return abs(prnxt_signed)


def _extract_params(attrs: ContractAttributes, prnxt_unsigned: float) -> ANNArrayParams:
    """Extract ``ANNArrayParams`` from ``ContractAttributes`` with pre-computed prnxt."""
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
    ipcba = attrs.interest_calculation_base_amount or 0.0

    return ANNArrayParams(
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
        next_principal_redemption_amount=jnp.array(prnxt_unsigned, dtype=_F32),
        ipcb_mode=jnp.array(_encode_ipcb_mode(attrs), dtype=jnp.int32),
        interest_calculation_base_amount=jnp.array(ipcba, dtype=_F32),
    )


def _extract_params_raw(attrs: ContractAttributes, prnxt_unsigned: float) -> dict[str, float | int]:
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
        "next_principal_redemption_amount": prnxt_unsigned,
        "ipcb_mode": _encode_ipcb_mode(attrs),
        "interest_calculation_base_amount": attrs.interest_calculation_base_amount or 0.0,
    }


# ---------------------------------------------------------------------------
# Schedule generation -- uses AnnuityContract fallback
# ---------------------------------------------------------------------------


def _fallback_ann_schedule(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> list[tuple[int, _datetime, _datetime]]:
    """Generate ANN schedule via scalar AnnuityContract.

    ANN schedules include PRF (Principal Redemption Fixing) events that
    are not present in NAM/LAM.  In the array path, PRF events are NOPs
    (zero payoff, no state change) because prnxt is pre-computed during
    the Python phase.  We convert PRF to NOP to avoid needing PRF-specific
    JAX handlers.
    """
    from jactus.contracts.ann import AnnuityContract

    contract = AnnuityContract(attrs, rf_observer)
    schedule = contract.generate_event_schedule()
    result: list[tuple[int, _datetime, _datetime]] = []
    for event in schedule.events:
        evt_dt = _adt_to_dt(event.event_time)
        calc_dt = _adt_to_dt(event.calculation_time) if event.calculation_time else evt_dt
        evt_idx = event.event_type.index
        # Convert PRF events to NOP -- prnxt is pre-computed, so PRF has
        # no effect in the array kernel
        if evt_idx == _PRF_IDX:
            evt_idx = NOP_EVENT_IDX
        result.append((evt_idx, evt_dt, calc_dt))
    return result


# ---------------------------------------------------------------------------
# Init state -- uses AnnuityContract for prnxt calculation
# ---------------------------------------------------------------------------


def _fast_ann_init_state(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> tuple[float, float, float, float, float, float, float, float, _datetime]:
    """Compute initial ANN state as Python floats.

    Returns ``(nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb, sd_datetime)``.

    Uses the scalar ``AnnuityContract`` to handle the annuity formula
    calculation and mid-life initialization (IED < SD, PRD).
    """
    from jactus.contracts.ann import AnnuityContract

    sd = attrs.status_date
    sd_dt = _adt_to_dt(sd)

    contract = AnnuityContract(attrs, rf_observer)
    state = contract.initialize_state()

    nt = float(state.nt)
    ipnr = float(state.ipnr)
    ipac = float(state.ipac)
    feac = float(state.feac)
    nsc = float(state.nsc)
    isc = float(state.isc)
    prnxt_val = float(state.prnxt) if state.prnxt is not None else 0.0
    ipcb = float(state.ipcb) if state.ipcb is not None else nt

    init_sd_dt = _adt_to_dt(state.sd) if state.sd else sd_dt
    return (nt, ipnr, ipac, feac, nsc, isc, prnxt_val, ipcb, init_sd_dt)


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

    # 1. Schedule generation via AnnuityContract (handles PRF events)
    schedule = _fallback_ann_schedule(attrs, rf_observer)

    # 2. State initialization via AnnuityContract (handles annuity formula)
    nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb, init_sd_dt = _fast_ann_init_state(
        attrs, rf_observer
    )

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

    # 4. Compute prnxt (unsigned) for params
    prnxt_unsigned = abs(prnxt)  # prnxt is signed; NAMArrayParams wants unsigned

    # 5. Extract params
    params_raw = _extract_params_raw(attrs, prnxt_unsigned)

    return _RawPrecomputed(
        state=(nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb),
        event_types=event_type_list,
        year_fractions=yf_list,
        rf_values=rf_list,
        params=params_raw,
    )


def _precompute_raw_da(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> _RawPrecomputed:
    """Pre-compute using vectorised year fractions (NumPy, no JAX overhead).

    Schedule generation reuses ``_fallback_ann_schedule`` (Python business
    logic), but year fractions are computed in a single vectorised NumPy pass.
    """
    from jactus.core.types import DayCountConvention

    # 1. Schedule
    schedule = _fallback_ann_schedule(attrs, rf_observer)

    # 2. State initialization
    nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb, init_sd_dt = _fast_ann_init_state(
        attrs, rf_observer
    )

    if not schedule:
        prnxt_unsigned = abs(prnxt)
        params_raw = _extract_params_raw(attrs, prnxt_unsigned)
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

    # 5. Extract params with annuity-computed prnxt
    prnxt_unsigned = abs(prnxt)
    params_raw = _extract_params_raw(attrs, prnxt_unsigned)

    return _RawPrecomputed(
        state=(nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb),
        event_types=event_type_list,
        year_fractions=yf_list,
        rf_values=rf_list,
        params=params_raw,
    )


def _raw_to_jax(
    raw: _RawPrecomputed,
) -> tuple[ANNArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, ANNArrayParams]:
    """Convert raw pre-computed data to JAX arrays."""
    nt, ipnr, ipac, feac, nsc, isc, prnxt, ipcb = raw.state
    return (
        ANNArrayState(
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


def precompute_ann_arrays(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> tuple[ANNArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, ANNArrayParams]:
    """Pre-compute JAX arrays for array-mode ANN simulation.

    Generates the event schedule and initial state using the scalar
    ``AnnuityContract`` (which computes ``prnxt`` via the ACTUS annuity
    formula), then converts to JAX arrays suitable for
    ``simulate_ann_array``.

    PRF events in the schedule are converted to NOPs since the annuity
    formula is evaluated once during this pre-computation phase.

    Args:
        attrs: Contract attributes (must be ANN type).
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
) -> tuple[ANNArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, ANNArrayParams, jnp.ndarray]:
    """Convert a list of ``_RawPrecomputed`` to padded JAX batch arrays.

    Pads shorter contracts with NOP events and builds NumPy arrays first
    (fast C-level construction) then transfers to JAX via ``jnp.asarray``.
    """
    max_events = max(len(r.event_types) for r in raw_list)

    # State fields: (batch,) each -- ANN has 8 state fields (same as NAM)
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
    param_fields: dict[str, list[float | int]] = {k: [] for k in ANNArrayParams._fields}
    for r in raw_list:
        for k in ANNArrayParams._fields:
            param_fields[k].append(r.params[k])

    # Build NumPy arrays first (fast C-level), then transfer to JAX
    batched_states = ANNArrayState(
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
    batched_params = ANNArrayParams(
        **{
            k: jnp.asarray(
                np.array(
                    param_fields[k],
                    dtype=np.int32 if k in _int_fields else np.float32,
                )
            )
            for k in ANNArrayParams._fields
        }
    )

    return batched_states, batched_et, batched_yf, batched_rf, batched_params, batched_masks


def prepare_ann_batch(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
) -> tuple[ANNArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, ANNArrayParams, jnp.ndarray]:
    """Pre-compute and pad arrays for a batch of ANN contracts.

    Uses per-contract Python pre-computation (sequential path).

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.

    Returns:
        ``(initial_states, event_types, year_fractions, rf_values, params, masks)``
        where each array has a leading batch dimension.
    """
    raw_list = [_precompute_raw(attrs, obs) for attrs, obs in contracts]
    return _raw_list_to_jax_batch(raw_list)


def simulate_ann_portfolio(
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
    ) = prepare_ann_batch(contracts)

    # Run batched simulation
    final_states, payoffs = batch_simulate_ann_auto(
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
