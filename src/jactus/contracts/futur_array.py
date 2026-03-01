"""Array-mode FUTUR (Futures) simulation -- JIT-compiled, vmap-able pure JAX.

FUTUR is a futures contract with linear payoff based on the difference
between spot and futures price.  Events: PRD, TD, MD, XD, STD.
State is minimal.  No ``lax.scan`` needed -- the settlement amount
(``Xa = spot - futures_price``) is pre-computed at ``XD`` during the
pre-computation phase and baked into ``rf_values``.

Example::

    from jactus.contracts.futur_array import precompute_futur_arrays, simulate_futur_array

    arrays = precompute_futur_arrays(attrs, rf_observer)
    final_state, payoffs = simulate_futur_array(*arrays)

    # Portfolio:
    from jactus.contracts.futur_array import simulate_futur_portfolio
    result = simulate_futur_portfolio(contracts)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jactus.contracts.array_common import (
    AD_IDX,
    F32,
    MD_IDX,
    NOP_EVENT_IDX,
    PRD_IDX,
    STD_IDX,
    TD_IDX,
    XD_IDX,
    adt_to_dt,
    dt_to_adt,
    get_role_sign,
)
from jactus.core import ContractAttributes, EventType
from jactus.observers import RiskFactorObserver


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class FUTURArrayState(NamedTuple):
    """Minimal state for FUTUR simulation.

    FUTUR state is constant throughout (xa is pre-computed).
    """

    nt: jnp.ndarray  # Notional principal (always 0.0)


class FUTURArrayParams(NamedTuple):
    """Static contract parameters for FUTUR."""

    role_sign: jnp.ndarray  # +1.0 or -1.0
    pprd: jnp.ndarray  # Price at purchase date
    ptd: jnp.ndarray  # Price at termination date
    nt: jnp.ndarray  # Notional principal (number of units)


# ---------------------------------------------------------------------------
# Simulation kernel (no scan -- direct vectorised payoff computation)
# ---------------------------------------------------------------------------


def simulate_futur_array(
    initial_state: FUTURArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: FUTURArrayParams,
) -> tuple[FUTURArrayState, jnp.ndarray]:
    """Run a FUTUR simulation as a pure JAX function.

    Event semantics encoded in ``event_types``:

    * **PRD**: ``role_sign * (-pprd)``
    * **TD**: ``ptd * nt``
    * **MD**: 0.0 (settlement amount is calculated at XD, paid at STD)
    * **XD**: 0.0 (settlement amount calculation, no cashflow)
    * **STD**: ``role_sign * rf_values``  where ``rf_values`` carries
      the pre-computed settlement amount ``Xa = spot - futures_price``.

    Args:
        initial_state: Starting state (nt only).
        event_types: ``(num_events,)`` int32.
        year_fractions: ``(num_events,)`` float32 (unused).
        rf_values: ``(num_events,)`` float32 -- settlement amount for STD events.
        params: Static contract parameters.

    Returns:
        ``(final_state, payoffs)`` where state is unchanged.
    """
    payoffs = jnp.where(
        event_types == PRD_IDX,
        params.role_sign * (-params.pprd),
        jnp.where(
            event_types == TD_IDX,
            params.ptd * params.nt,
            jnp.where(
                event_types == STD_IDX,
                params.role_sign * rf_values,
                0.0,  # AD, MD, XD, CE, NOP -> 0
            ),
        ),
    )
    return initial_state, payoffs


simulate_futur_array_jit = jax.jit(simulate_futur_array)

batch_simulate_futur_vmap = jax.vmap(simulate_futur_array)


@jax.jit
def batch_simulate_futur(
    initial_states: FUTURArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: FUTURArrayParams,
) -> tuple[FUTURArrayState, jnp.ndarray]:
    """Batched FUTUR simulation -- vectorised payoff computation.

    Args:
        initial_states: ``FUTURArrayState`` with each field shape ``[B]``.
        event_types: ``[B, T]`` int32.
        year_fractions: ``[B, T]`` float32 (unused).
        rf_values: ``[B, T]`` float32.
        params: ``FUTURArrayParams`` with each field shape ``[B]``.

    Returns:
        ``(final_states, payoffs)`` where ``payoffs`` is ``[B, T]``.
    """
    rs = params.role_sign.reshape(-1, 1)
    pprd = params.pprd.reshape(-1, 1)
    ptd = params.ptd.reshape(-1, 1)
    nt = params.nt.reshape(-1, 1)

    payoffs = jnp.where(
        event_types == PRD_IDX,
        rs * (-pprd),
        jnp.where(
            event_types == TD_IDX,
            ptd * nt,
            jnp.where(
                event_types == STD_IDX,
                rs * rf_values,
                0.0,
            ),
        ),
    )
    return initial_states, payoffs


def batch_simulate_futur_auto(
    initial_states: FUTURArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: FUTURArrayParams,
) -> tuple[FUTURArrayState, jnp.ndarray]:
    """Batched simulation using the optimal strategy."""
    return batch_simulate_futur(  # type: ignore[no-any-return]
        initial_states, event_types, year_fractions, rf_values, params
    )


# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------


def precompute_futur_arrays(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> tuple[FUTURArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, FUTURArrayParams]:
    """Pre-compute JAX arrays for array-mode FUTUR simulation.

    Mirrors the scalar ``FutureContract.simulate()`` logic:

    * XD at maturity: observe spot price, compute ``Xa = spot - futures_price``
    * STD: pays ``role_sign * Xa``
    * PRD: pays ``role_sign * (-pprd)``
    * TD: pays ``ptd * nt``

    The settlement amount ``Xa`` is pre-computed and stored in ``rf_values``
    at the STD event position.

    Args:
        attrs: Contract attributes (must be FUTUR type).
        rf_observer: Risk factor observer for spot price observation.

    Returns:
        ``(initial_state, event_types, year_fractions, rf_values, params)``
    """
    role_sign = get_role_sign(attrs.contract_role)
    nt_val = attrs.notional_principal or 1.0
    futures_price = attrs.future_price or 0.0

    initial_state = FUTURArrayState(nt=jnp.array(0.0, dtype=F32))
    futur_params = FUTURArrayParams(
        role_sign=jnp.array(role_sign, dtype=F32),
        pprd=jnp.array(attrs.price_at_purchase_date or 0.0, dtype=F32),
        ptd=jnp.array(attrs.price_at_termination_date or 0.0, dtype=F32),
        nt=jnp.array(nt_val, dtype=F32),
    )

    # Build schedule: (event_idx, rf_value)
    schedule: list[tuple[int, float]] = []
    sd_dt = adt_to_dt(attrs.status_date)

    # Check for pre-exercised state
    pre_exercised = (
        attrs.exercise_date is not None and attrs.exercise_amount is not None
    )

    if pre_exercised:
        # Pre-exercised: only STD with known exercise amount
        xa = attrs.exercise_amount or 0.0
        schedule.append((STD_IDX, xa))
    else:
        # PRD
        if attrs.purchase_date:
            prd_dt = adt_to_dt(attrs.purchase_date)
            if prd_dt > sd_dt:
                schedule.append((PRD_IDX, 0.0))

        # TD
        if attrs.termination_date:
            td_dt = adt_to_dt(attrs.termination_date)
            if td_dt > sd_dt:
                schedule.append((TD_IDX, 0.0))

        # MD at maturity
        assert attrs.maturity_date is not None
        schedule.append((MD_IDX, 0.0))

        # XD at maturity: observe spot, compute Xa
        underlier_ref = attrs.contract_structure or ""
        try:
            spot_price = float(
                rf_observer.observe_risk_factor(underlier_ref, attrs.maturity_date)
            )
        except (KeyError, NotImplementedError, TypeError):
            spot_price = 0.0

        xa = spot_price - futures_price
        schedule.append((XD_IDX, 0.0))

        # STD: settlement amount = Xa (pre-computed)
        schedule.append((STD_IDX, xa))

    # If empty schedule, add a single AD event
    if not schedule:
        schedule.append((AD_IDX, 0.0))

    n_events = len(schedule)
    et_list = [s[0] for s in schedule]
    rf_list = [s[1] for s in schedule]

    event_types = jnp.array(et_list, dtype=jnp.int32)
    year_fractions = jnp.zeros(n_events, dtype=F32)
    rf_values = jnp.array(rf_list, dtype=F32)

    return initial_state, event_types, year_fractions, rf_values, futur_params


def prepare_futur_batch(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
) -> tuple[
    FUTURArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, FUTURArrayParams, jnp.ndarray
]:
    """Pre-compute and pad arrays for a batch of FUTUR contracts.

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.

    Returns:
        ``(initial_states, event_types, year_fractions, rf_values, params, masks)``
    """
    n = len(contracts)
    precomputed = [precompute_futur_arrays(a, o) for a, o in contracts]

    # Find max events across all contracts
    max_events = max(p[1].shape[0] for p in precomputed)

    nt_arr = np.zeros(n, dtype=np.float32)
    rs_arr = np.zeros(n, dtype=np.float32)
    pprd_arr = np.zeros(n, dtype=np.float32)
    ptd_arr = np.zeros(n, dtype=np.float32)
    nt_param_arr = np.zeros(n, dtype=np.float32)

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
        nt_param_arr[i] = float(params.nt)
        et_batch[i, :k] = np.asarray(et)
        yf_batch[i, :k] = np.asarray(yf)
        rf_batch[i, :k] = np.asarray(rf)
        mask_batch[i, :k] = 1.0

    initial_states = FUTURArrayState(nt=jnp.asarray(nt_arr))
    futur_params = FUTURArrayParams(
        role_sign=jnp.asarray(rs_arr),
        pprd=jnp.asarray(pprd_arr),
        ptd=jnp.asarray(ptd_arr),
        nt=jnp.asarray(nt_param_arr),
    )

    return (
        initial_states,
        jnp.asarray(et_batch),
        jnp.asarray(yf_batch),
        jnp.asarray(rf_batch),
        futur_params,
        jnp.asarray(mask_batch),
    )


def simulate_futur_portfolio(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
    discount_rate: float | None = None,
) -> dict[str, Any]:
    """End-to-end FUTUR portfolio simulation.

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
    ) = prepare_futur_batch(contracts)

    final_states, payoffs = batch_simulate_futur_auto(
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
