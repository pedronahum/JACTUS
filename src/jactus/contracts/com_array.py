"""Array-mode COM (Commodity) simulation — JIT-compiled, vmap-able pure JAX.

COM is a simple commodity position contract with events: AD, PRD, TD, CE.
State is minimal (constant throughout). No ``lax.scan`` needed — payoffs are
computed directly from event types and parameters.

Example::

    from jactus.contracts.com_array import precompute_com_arrays, simulate_com_array

    arrays = precompute_com_arrays(attrs, rf_observer)
    final_state, payoffs = simulate_com_array(*arrays)

    # Portfolio:
    from jactus.contracts.com_array import simulate_com_portfolio
    result = simulate_com_portfolio(contracts)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jactus.contracts.array_common import (
    F32,
    NOP_EVENT_IDX,
    AD_IDX,
    PRD_IDX,
    TD_IDX,
    adt_to_dt,
    get_role_sign,
)
from jactus.core import ContractAttributes, EventType
from jactus.observers import RiskFactorObserver


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class COMArrayState(NamedTuple):
    """Minimal state for COM simulation.

    COM only tracks notional principal (constant, always 0.0).
    """

    nt: jnp.ndarray  # Notional principal (always 0.0)


class COMArrayParams(NamedTuple):
    """Static contract parameters for COM."""

    role_sign: jnp.ndarray  # +1.0 or -1.0
    pprd: jnp.ndarray  # Price at purchase date
    ptd: jnp.ndarray  # Price at termination date
    quantity: jnp.ndarray  # Quantity


# ---------------------------------------------------------------------------
# Simulation kernel (no scan — direct vectorised payoff computation)
# ---------------------------------------------------------------------------


def simulate_com_array(
    initial_state: COMArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: COMArrayParams,
) -> tuple[COMArrayState, jnp.ndarray]:
    """Run a COM simulation as a pure JAX function.

    COM payoffs depend only on event type and static params.

    Args:
        initial_state: Starting state (nt only).
        event_types: ``(num_events,)`` int32.
        year_fractions: ``(num_events,)`` float32 (unused).
        rf_values: ``(num_events,)`` float32 (unused).
        params: Static contract parameters.

    Returns:
        ``(final_state, payoffs)`` where state is unchanged.
    """
    payoffs = jnp.where(
        event_types == PRD_IDX,
        params.role_sign * (-params.pprd) * params.quantity,
        jnp.where(
            event_types == TD_IDX,
            params.role_sign * params.ptd * params.quantity,
            0.0,  # AD, CE, NOP → 0
        ),
    )
    return initial_state, payoffs


simulate_com_array_jit = jax.jit(simulate_com_array)

batch_simulate_com_vmap = jax.vmap(simulate_com_array)


@jax.jit
def batch_simulate_com(
    initial_states: COMArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: COMArrayParams,
) -> tuple[COMArrayState, jnp.ndarray]:
    """Batched COM simulation — vectorised payoff computation.

    Args:
        initial_states: ``COMArrayState`` with each field shape ``[B]``.
        event_types: ``[B, T]`` int32.
        year_fractions: ``[B, T]`` float32 (unused).
        rf_values: ``[B, T]`` float32 (unused).
        params: ``COMArrayParams`` with each field shape ``[B]``.

    Returns:
        ``(final_states, payoffs)`` where ``payoffs`` is ``[B, T]``.
    """
    rs = params.role_sign.reshape(-1, 1)
    pprd = params.pprd.reshape(-1, 1)
    ptd = params.ptd.reshape(-1, 1)
    qty = params.quantity.reshape(-1, 1)

    payoffs = jnp.where(
        event_types == PRD_IDX,
        rs * (-pprd) * qty,
        jnp.where(
            event_types == TD_IDX,
            rs * ptd * qty,
            0.0,
        ),
    )
    return initial_states, payoffs


def batch_simulate_com_auto(
    initial_states: COMArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: COMArrayParams,
) -> tuple[COMArrayState, jnp.ndarray]:
    """Batched simulation using the optimal strategy."""
    return batch_simulate_com(initial_states, event_types, year_fractions, rf_values, params)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------


def precompute_com_arrays(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> tuple[COMArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, COMArrayParams]:
    """Pre-compute JAX arrays for array-mode COM simulation.

    Args:
        attrs: Contract attributes (must be COM type).
        rf_observer: Risk factor observer (not used for COM).

    Returns:
        ``(initial_state, event_types, year_fractions, rf_values, params)``
    """
    role_sign = get_role_sign(attrs.contract_role)

    initial_state = COMArrayState(nt=jnp.array(0.0, dtype=F32))
    com_params = COMArrayParams(
        role_sign=jnp.array(role_sign, dtype=F32),
        pprd=jnp.array(attrs.price_at_purchase_date or 0.0, dtype=F32),
        ptd=jnp.array(attrs.price_at_termination_date or 0.0, dtype=F32),
        quantity=jnp.array(attrs.quantity or 1.0, dtype=F32),
    )

    # Build schedule: PRD, TD
    et_list: list[int] = []
    sd_dt = adt_to_dt(attrs.status_date)

    # PRD
    if attrs.purchase_date:
        prd_dt = adt_to_dt(attrs.purchase_date)
        if prd_dt > sd_dt:
            et_list.append(PRD_IDX)

    # TD
    if attrs.termination_date:
        td_dt = adt_to_dt(attrs.termination_date)
        if td_dt > sd_dt:
            et_list.append(TD_IDX)

    # If empty schedule, add a single AD event
    if not et_list:
        et_list.append(AD_IDX)

    n_events = len(et_list)
    event_types = jnp.array(et_list, dtype=jnp.int32)
    year_fractions = jnp.zeros(n_events, dtype=F32)
    rf_values = jnp.zeros(n_events, dtype=F32)

    return initial_state, event_types, year_fractions, rf_values, com_params


def prepare_com_batch(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
) -> tuple[COMArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, COMArrayParams, jnp.ndarray]:
    """Pre-compute and pad arrays for a batch of COM contracts.

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.

    Returns:
        ``(initial_states, event_types, year_fractions, rf_values, params, masks)``
    """
    n = len(contracts)
    precomputed = [precompute_com_arrays(a, o) for a, o in contracts]

    # Find max events across all contracts
    max_events = max(p[1].shape[0] for p in precomputed)

    nt_arr = np.zeros(n, dtype=np.float32)
    rs_arr = np.zeros(n, dtype=np.float32)
    pprd_arr = np.zeros(n, dtype=np.float32)
    ptd_arr = np.zeros(n, dtype=np.float32)
    qty_arr = np.zeros(n, dtype=np.float32)

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
        qty_arr[i] = float(params.quantity)
        et_batch[i, :k] = np.asarray(et)
        yf_batch[i, :k] = np.asarray(yf)
        rf_batch[i, :k] = np.asarray(rf)
        mask_batch[i, :k] = 1.0

    initial_states = COMArrayState(nt=jnp.asarray(nt_arr))
    com_params = COMArrayParams(
        role_sign=jnp.asarray(rs_arr),
        pprd=jnp.asarray(pprd_arr),
        ptd=jnp.asarray(ptd_arr),
        quantity=jnp.asarray(qty_arr),
    )

    return (
        initial_states,
        jnp.asarray(et_batch),
        jnp.asarray(yf_batch),
        jnp.asarray(rf_batch),
        com_params,
        jnp.asarray(mask_batch),
    )


def simulate_com_portfolio(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
    discount_rate: float | None = None,
) -> dict[str, Any]:
    """End-to-end COM portfolio simulation.

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
    ) = prepare_com_batch(contracts)

    final_states, payoffs = batch_simulate_com_auto(
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
