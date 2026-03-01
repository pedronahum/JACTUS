"""Array-mode FXOUT (FX Outright) simulation -- JIT-compiled, vmap-able pure JAX.

FXOUT is an FX forward/spot contract with settlement in two currencies.
Events: PRD, TD, MD (gross/dual), STD (net/delivery).  State is minimal
(constant throughout).  No ``lax.scan`` needed -- payoffs are computed
directly from event types and parameters.

Example::

    from jactus.contracts.fxout_array import precompute_fxout_arrays, simulate_fxout_array

    arrays = precompute_fxout_arrays(attrs, rf_observer)
    final_state, payoffs = simulate_fxout_array(*arrays)

    # Portfolio:
    from jactus.contracts.fxout_array import simulate_fxout_portfolio
    result = simulate_fxout_portfolio(contracts)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jactus.contracts.array_common import (
    F32,
    MD_IDX,
    NOP_EVENT_IDX,
    PRD_IDX,
    STD_IDX,
    TD_IDX,
    adt_to_dt,
    dt_to_adt,
    get_role_sign,
)
from jactus.core import ContractAttributes, EventType
from jactus.observers import RiskFactorObserver


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class FXOUTArrayState(NamedTuple):
    """Minimal state for FXOUT simulation.

    FXOUT only tracks notional principal (constant, always 0.0).
    """

    nt: jnp.ndarray  # Notional principal (always 0.0)


class FXOUTArrayParams(NamedTuple):
    """Static contract parameters for FXOUT."""

    role_sign: jnp.ndarray  # +1.0 or -1.0
    pprd: jnp.ndarray  # Price at purchase date
    ptd: jnp.ndarray  # Price at termination date
    notional_1: jnp.ndarray  # NT  (first currency amount)
    notional_2: jnp.ndarray  # NT2 (second currency amount)


# ---------------------------------------------------------------------------
# Simulation kernel (no scan -- direct vectorised payoff computation)
# ---------------------------------------------------------------------------


def simulate_fxout_array(
    initial_state: FXOUTArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: FXOUTArrayParams,
) -> tuple[FXOUTArrayState, jnp.ndarray]:
    """Run a FXOUT simulation as a pure JAX function.

    Event semantics encoded in ``event_types``:

    * **PRD**: ``role_sign * (-pprd)``
    * **TD**: ``ptd``  (already directional in scalar impl)
    * **MD**: Two consecutive MD events for gross settlement.
      First MD (rf_values == 0): ``role_sign * notional_1``
      Second MD (rf_values == 1): ``-role_sign * notional_2``
    * **STD**: Net settlement: ``role_sign * (notional_1 - fx_rate * notional_2)``
      where ``rf_values`` carries the observed FX rate.

    Args:
        initial_state: Starting state (nt only).
        event_types: ``(num_events,)`` int32.
        year_fractions: ``(num_events,)`` float32 (unused).
        rf_values: ``(num_events,)`` float32 -- FX rate for STD events,
            leg indicator (0.0 = first currency, 1.0 = second) for MD events.
        params: Static contract parameters.

    Returns:
        ``(final_state, payoffs)`` where state is unchanged.
    """
    rs = params.role_sign
    nt1 = params.notional_1
    nt2 = params.notional_2

    # MD events: first leg (rf_values==0) -> +role_sign * NT1
    #            second leg (rf_values==1) -> -role_sign * NT2
    md_payoff = jnp.where(
        rf_values < 0.5,
        rs * nt1,
        -rs * nt2,
    )

    # STD event: net settlement  role_sign * (NT1 - fx_rate * NT2)
    std_payoff = rs * (nt1 - rf_values * nt2)

    payoffs = jnp.where(
        event_types == PRD_IDX,
        rs * (-params.pprd),
        jnp.where(
            event_types == TD_IDX,
            params.ptd,
            jnp.where(
                event_types == MD_IDX,
                md_payoff,
                jnp.where(
                    event_types == STD_IDX,
                    std_payoff,
                    0.0,  # AD, CE, NOP -> 0
                ),
            ),
        ),
    )
    return initial_state, payoffs


simulate_fxout_array_jit = jax.jit(simulate_fxout_array)

batch_simulate_fxout_vmap = jax.vmap(simulate_fxout_array)


@jax.jit
def batch_simulate_fxout(
    initial_states: FXOUTArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: FXOUTArrayParams,
) -> tuple[FXOUTArrayState, jnp.ndarray]:
    """Batched FXOUT simulation -- vectorised payoff computation.

    Args:
        initial_states: ``FXOUTArrayState`` with each field shape ``[B]``.
        event_types: ``[B, T]`` int32.
        year_fractions: ``[B, T]`` float32 (unused).
        rf_values: ``[B, T]`` float32.
        params: ``FXOUTArrayParams`` with each field shape ``[B]``.

    Returns:
        ``(final_states, payoffs)`` where ``payoffs`` is ``[B, T]``.
    """
    rs = params.role_sign.reshape(-1, 1)
    pprd = params.pprd.reshape(-1, 1)
    ptd = params.ptd.reshape(-1, 1)
    nt1 = params.notional_1.reshape(-1, 1)
    nt2 = params.notional_2.reshape(-1, 1)

    md_payoff = jnp.where(rf_values < 0.5, rs * nt1, -rs * nt2)
    std_payoff = rs * (nt1 - rf_values * nt2)

    payoffs = jnp.where(
        event_types == PRD_IDX,
        rs * (-pprd),
        jnp.where(
            event_types == TD_IDX,
            ptd,
            jnp.where(
                event_types == MD_IDX,
                md_payoff,
                jnp.where(
                    event_types == STD_IDX,
                    std_payoff,
                    0.0,
                ),
            ),
        ),
    )
    return initial_states, payoffs


def batch_simulate_fxout_auto(
    initial_states: FXOUTArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: FXOUTArrayParams,
) -> tuple[FXOUTArrayState, jnp.ndarray]:
    """Batched simulation using the optimal strategy."""
    return batch_simulate_fxout(  # type: ignore[no-any-return]
        initial_states, event_types, year_fractions, rf_values, params
    )


# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------


def precompute_fxout_arrays(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> tuple[FXOUTArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, FXOUTArrayParams]:
    """Pre-compute JAX arrays for array-mode FXOUT simulation.

    Mirrors the scalar ``FXOutrightContract.simulate()`` logic: gross
    settlement produces two MD events (one per currency leg); net/delivery
    settlement produces a single STD event with the FX rate baked in.

    Args:
        attrs: Contract attributes (must be FXOUT type).
        rf_observer: Risk factor observer for FX rate observation.

    Returns:
        ``(initial_state, event_types, year_fractions, rf_values, params)``
    """
    role_sign = get_role_sign(attrs.contract_role)

    initial_state = FXOUTArrayState(nt=jnp.array(0.0, dtype=F32))
    fxout_params = FXOUTArrayParams(
        role_sign=jnp.array(role_sign, dtype=F32),
        pprd=jnp.array(attrs.price_at_purchase_date or 0.0, dtype=F32),
        ptd=jnp.array(attrs.price_at_termination_date or 0.0, dtype=F32),
        notional_1=jnp.array(attrs.notional_principal or 0.0, dtype=F32),
        notional_2=jnp.array(attrs.notional_principal_2 or 0.0, dtype=F32),
    )

    # Build schedule: (event_idx, rf_value)
    schedule: list[tuple[int, float]] = []
    sd_dt = adt_to_dt(attrs.status_date)

    maturity_date = attrs.settlement_date or attrs.maturity_date

    # Check for early termination
    early_term = False
    if attrs.termination_date and maturity_date:
        early_term = attrs.termination_date.to_iso() < maturity_date.to_iso()

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

    # Settlement/maturity events (suppressed if early termination)
    if maturity_date and not early_term:
        ds = attrs.delivery_settlement or "D"

        # Determine if we use net STD or gross MD events
        # Scalar logic: DS='S' with non-zero settlement_period -> STD (net)
        # Otherwise -> two MD events (gross, one per currency leg)
        sp = attrs.settlement_period
        has_sp = bool(sp) and sp != "P0D"

        if ds == "S" and has_sp:
            # Net cash settlement: single STD event
            # Observe FX rate at maturity
            cur = attrs.currency or "XXX"
            cur2 = attrs.currency_2 or "YYY"
            rate_id = f"{cur2}/{cur}"
            try:
                fx_rate = float(
                    rf_observer.observe_risk_factor(rate_id, maturity_date)
                )
            except (KeyError, NotImplementedError, TypeError):
                fx_rate = 0.0
            schedule.append((STD_IDX, fx_rate))
        else:
            # Gross settlement: two MD events
            # First MD (leg 1, first currency): rf_values=0.0 -> +role_sign*NT1
            schedule.append((MD_IDX, 0.0))
            # Second MD (leg 2, second currency): rf_values=1.0 -> -role_sign*NT2
            schedule.append((MD_IDX, 1.0))

    # If empty schedule, add a single AD event
    if not schedule:
        from jactus.contracts.array_common import AD_IDX

        schedule.append((AD_IDX, 0.0))

    n_events = len(schedule)
    et_list = [s[0] for s in schedule]
    rf_list = [s[1] for s in schedule]

    event_types = jnp.array(et_list, dtype=jnp.int32)
    year_fractions = jnp.zeros(n_events, dtype=F32)
    rf_values = jnp.array(rf_list, dtype=F32)

    return initial_state, event_types, year_fractions, rf_values, fxout_params


def prepare_fxout_batch(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
) -> tuple[
    FXOUTArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, FXOUTArrayParams, jnp.ndarray
]:
    """Pre-compute and pad arrays for a batch of FXOUT contracts.

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.

    Returns:
        ``(initial_states, event_types, year_fractions, rf_values, params, masks)``
    """
    n = len(contracts)
    precomputed = [precompute_fxout_arrays(a, o) for a, o in contracts]

    # Find max events across all contracts
    max_events = max(p[1].shape[0] for p in precomputed)

    nt_arr = np.zeros(n, dtype=np.float32)
    rs_arr = np.zeros(n, dtype=np.float32)
    pprd_arr = np.zeros(n, dtype=np.float32)
    ptd_arr = np.zeros(n, dtype=np.float32)
    nt1_arr = np.zeros(n, dtype=np.float32)
    nt2_arr = np.zeros(n, dtype=np.float32)

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
        nt1_arr[i] = float(params.notional_1)
        nt2_arr[i] = float(params.notional_2)
        et_batch[i, :k] = np.asarray(et)
        yf_batch[i, :k] = np.asarray(yf)
        rf_batch[i, :k] = np.asarray(rf)
        mask_batch[i, :k] = 1.0

    initial_states = FXOUTArrayState(nt=jnp.asarray(nt_arr))
    fxout_params = FXOUTArrayParams(
        role_sign=jnp.asarray(rs_arr),
        pprd=jnp.asarray(pprd_arr),
        ptd=jnp.asarray(ptd_arr),
        notional_1=jnp.asarray(nt1_arr),
        notional_2=jnp.asarray(nt2_arr),
    )

    return (
        initial_states,
        jnp.asarray(et_batch),
        jnp.asarray(yf_batch),
        jnp.asarray(rf_batch),
        fxout_params,
        jnp.asarray(mask_batch),
    )


def simulate_fxout_portfolio(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
    discount_rate: float | None = None,
) -> dict[str, Any]:
    """End-to-end FXOUT portfolio simulation.

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
    ) = prepare_fxout_batch(contracts)

    final_states, payoffs = batch_simulate_fxout_auto(
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
