"""Array-mode STK (Stock) simulation — JIT-compiled, vmap-able pure JAX.

STK is a simple equity position contract with events: AD, PRD, TD, DV, CE.
State is minimal (constant throughout). No ``lax.scan`` needed — payoffs are
computed directly from event types and parameters.

Example::

    from jactus.contracts.stk_array import precompute_stk_arrays, simulate_stk_array

    arrays = precompute_stk_arrays(attrs, rf_observer)
    final_state, payoffs = simulate_stk_array(*arrays)

    # Portfolio:
    from jactus.contracts.stk_array import simulate_stk_portfolio
    result = simulate_stk_portfolio(contracts)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jactus.contracts.array_common import (
    AD_IDX,
    DV_IDX,
    F32,
    NOP_EVENT_IDX,
    PRD_IDX,
    TD_IDX,
    adt_to_dt,
    dt_to_adt,
    fast_schedule,
    get_role_sign,
)
from jactus.core import ContractAttributes
from jactus.observers import RiskFactorObserver

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class STKArrayState(NamedTuple):
    """Minimal state for STK simulation.

    STK only tracks notional principal (constant, always 0.0).
    """

    nt: jnp.ndarray  # Notional principal (always 0.0)


class STKArrayParams(NamedTuple):
    """Static contract parameters for STK."""

    role_sign: jnp.ndarray  # +1.0 or -1.0
    pprd: jnp.ndarray  # Price at purchase date
    ptd: jnp.ndarray  # Price at termination date


# ---------------------------------------------------------------------------
# Simulation kernel (no scan — direct vectorised payoff computation)
# ---------------------------------------------------------------------------


def simulate_stk_array(
    initial_state: STKArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: STKArrayParams,
) -> tuple[STKArrayState, jnp.ndarray]:
    """Run a STK simulation as a pure JAX function.

    STK payoffs depend only on event type and static params (except DV which
    uses rf_values for the observed dividend amount).

    Args:
        initial_state: Starting state (nt only).
        event_types: ``(num_events,)`` int32.
        year_fractions: ``(num_events,)`` float32 (unused).
        rf_values: ``(num_events,)`` float32 — dividend amounts for DV events.
        params: Static contract parameters.

    Returns:
        ``(final_state, payoffs)`` where state is unchanged.
    """
    payoffs = jnp.where(
        event_types == PRD_IDX,
        params.role_sign * (-params.pprd),
        jnp.where(
            event_types == TD_IDX,
            params.role_sign * params.ptd,
            jnp.where(
                event_types == DV_IDX,
                params.role_sign * rf_values,
                0.0,  # AD, CE, NOP → 0
            ),
        ),
    )
    return initial_state, payoffs


simulate_stk_array_jit = jax.jit(simulate_stk_array)

batch_simulate_stk_vmap = jax.vmap(simulate_stk_array)


@jax.jit
def batch_simulate_stk(
    initial_states: STKArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: STKArrayParams,
) -> tuple[STKArrayState, jnp.ndarray]:
    """Batched STK simulation — vectorised payoff computation.

    Args:
        initial_states: ``STKArrayState`` with each field shape ``[B]``.
        event_types: ``[B, T]`` int32.
        year_fractions: ``[B, T]`` float32 (unused).
        rf_values: ``[B, T]`` float32 — dividend amounts for DV events.
        params: ``STKArrayParams`` with each field shape ``[B]``.

    Returns:
        ``(final_states, payoffs)`` where ``payoffs`` is ``[B, T]``.
    """
    rs = params.role_sign.reshape(-1, 1)
    pprd = params.pprd.reshape(-1, 1)
    ptd = params.ptd.reshape(-1, 1)

    payoffs = jnp.where(
        event_types == PRD_IDX,
        rs * (-pprd),
        jnp.where(
            event_types == TD_IDX,
            rs * ptd,
            jnp.where(
                event_types == DV_IDX,
                rs * rf_values,
                0.0,
            ),
        ),
    )
    return initial_states, payoffs


def batch_simulate_stk_auto(
    initial_states: STKArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: STKArrayParams,
) -> tuple[STKArrayState, jnp.ndarray]:
    """Batched simulation using the optimal strategy."""
    return batch_simulate_stk(initial_states, event_types, year_fractions, rf_values, params)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------


def precompute_stk_arrays(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> tuple[STKArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, STKArrayParams]:
    """Pre-compute JAX arrays for array-mode STK simulation.

    Args:
        attrs: Contract attributes (must be STK type).
        rf_observer: Risk factor observer for dividend observation.

    Returns:
        ``(initial_state, event_types, year_fractions, rf_values, params)``
    """
    role_sign = get_role_sign(attrs.contract_role)

    initial_state = STKArrayState(nt=jnp.array(0.0, dtype=F32))
    stk_params = STKArrayParams(
        role_sign=jnp.array(role_sign, dtype=F32),
        pprd=jnp.array(attrs.price_at_purchase_date or 0.0, dtype=F32),
        ptd=jnp.array(attrs.price_at_termination_date or 0.0, dtype=F32),
    )

    # Build schedule: PRD, DV cycle events, TD
    schedule: list[tuple[int, float]] = []  # (event_idx, rf_value)

    sd_dt = adt_to_dt(attrs.status_date)

    # PRD
    if attrs.purchase_date:
        prd_dt = adt_to_dt(attrs.purchase_date)
        if prd_dt > sd_dt:
            schedule.append((PRD_IDX, 0.0))

    # DV: dividend cycle events
    if attrs.dividend_cycle:
        dv_start = attrs.dividend_anchor or attrs.purchase_date or attrs.status_date
        dv_end = attrs.termination_date or attrs.maturity_date
        if dv_end:
            dv_dates = fast_schedule(dv_start, attrs.dividend_cycle, dv_end)
            dvmo = attrs.market_object_code_of_dividends or ""
            for dv_dt in dv_dates:
                if dv_dt > sd_dt:
                    # Observe dividend amount
                    dv_amount = 0.0
                    if dvmo:
                        try:
                            dv_amount = float(
                                rf_observer.observe_risk_factor(dvmo, dt_to_adt(dv_dt))
                            )
                        except (KeyError, NotImplementedError, TypeError):
                            dv_amount = 0.0
                    schedule.append((DV_IDX, dv_amount))

    # TD
    if attrs.termination_date:
        td_dt = adt_to_dt(attrs.termination_date)
        if td_dt > sd_dt:
            schedule.append((TD_IDX, 0.0))

    # If empty schedule, add a single AD event
    if not schedule:
        schedule.append((AD_IDX, 0.0))

    n_events = len(schedule)
    et_list = [s[0] for s in schedule]
    rf_list = [s[1] for s in schedule]

    event_types = jnp.array(et_list, dtype=jnp.int32)
    year_fractions = jnp.zeros(n_events, dtype=F32)
    rf_values = jnp.array(rf_list, dtype=F32)

    return initial_state, event_types, year_fractions, rf_values, stk_params


def prepare_stk_batch(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
) -> tuple[STKArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, STKArrayParams, jnp.ndarray]:
    """Pre-compute and pad arrays for a batch of STK contracts.

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.

    Returns:
        ``(initial_states, event_types, year_fractions, rf_values, params, masks)``
    """
    n = len(contracts)
    precomputed = [precompute_stk_arrays(a, o) for a, o in contracts]

    # Find max events across all contracts
    max_events = max(p[1].shape[0] for p in precomputed)

    nt_arr = np.zeros(n, dtype=np.float32)
    rs_arr = np.zeros(n, dtype=np.float32)
    pprd_arr = np.zeros(n, dtype=np.float32)
    ptd_arr = np.zeros(n, dtype=np.float32)

    et_batch = np.full((n, max_events), NOP_EVENT_IDX, dtype=np.int32)
    yf_batch = np.zeros((n, max_events), dtype=np.float32)
    rf_batch = np.zeros((n, max_events), dtype=np.float32)
    mask_batch = np.zeros((n, max_events), dtype=np.float32)

    for i, (state, et, yf, rf, params) in enumerate(precomputed):
        k = et.shape[0]
        nt_arr[i] = float(state.nt)
        rs_arr[i] = float(params.role_sign)
        pprd_arr[i] = float(params.pprd)
        ptd_arr[i] = float(params.ptd)
        et_batch[i, :k] = np.asarray(et)
        yf_batch[i, :k] = np.asarray(yf)
        rf_batch[i, :k] = np.asarray(rf)
        mask_batch[i, :k] = 1.0

    initial_states = STKArrayState(nt=jnp.asarray(nt_arr))
    stk_params = STKArrayParams(
        role_sign=jnp.asarray(rs_arr),
        pprd=jnp.asarray(pprd_arr),
        ptd=jnp.asarray(ptd_arr),
    )

    return (
        initial_states,
        jnp.asarray(et_batch),
        jnp.asarray(yf_batch),
        jnp.asarray(rf_batch),
        stk_params,
        jnp.asarray(mask_batch),
    )


def simulate_stk_portfolio(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
    discount_rate: float | None = None,
) -> dict[str, Any]:
    """End-to-end STK portfolio simulation.

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.
        discount_rate: If provided, compute present values.

    Returns:
        Dict with ``payoffs``, ``masks``, ``final_states``, ``total_cashflows``.
    """
    (
        batched_states,
        batched_et,
        batched_yf,
        batched_rf,
        batched_params,
        batched_masks,
    ) = prepare_stk_batch(contracts)

    final_states, payoffs = batch_simulate_stk_auto(
        batched_states, batched_et, batched_yf, batched_rf, batched_params
    )

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
        disc_yfs = jnp.cumsum(batched_yf, axis=1)
        discount_factors = 1.0 / (1.0 + discount_rate * disc_yfs)
        pvs = jnp.sum(masked_payoffs * discount_factors, axis=1)
        result["present_values"] = pvs
        result["total_pv"] = jnp.sum(pvs)

    return result
