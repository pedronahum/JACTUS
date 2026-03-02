"""Array-mode OPTNS (Options) simulation -- JIT-compiled, vmap-able pure JAX.

OPTNS is a vanilla option contract (call/put/collar) with European, American,
or Bermudan exercise.  Events: PRD, TD, MD, XD, STD.  State is minimal.
No ``lax.scan`` needed -- the exercise amount (intrinsic value) is
pre-computed at ``XD`` during the pre-computation phase and baked into
``rf_values``.

Example::

    from jactus.contracts.optns_array import precompute_optns_arrays, simulate_optns_array

    arrays = precompute_optns_arrays(attrs, rf_observer)
    final_state, payoffs = simulate_optns_array(*arrays)

    # Portfolio:
    from jactus.contracts.optns_array import simulate_optns_portfolio
    result = simulate_optns_portfolio(contracts)
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
from jactus.core import ContractAttributes
from jactus.observers import RiskFactorObserver

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class OPTNSArrayState(NamedTuple):
    """Minimal state for OPTNS simulation.

    OPTNS state is constant throughout (xa is pre-computed).
    """

    nt: jnp.ndarray  # Notional principal (always 0.0)


class OPTNSArrayParams(NamedTuple):
    """Static contract parameters for OPTNS."""

    role_sign: jnp.ndarray  # +1.0 or -1.0
    pprd: jnp.ndarray  # Price at purchase date (premium)
    ptd: jnp.ndarray  # Price at termination date


# ---------------------------------------------------------------------------
# Simulation kernel (no scan -- direct vectorised payoff computation)
# ---------------------------------------------------------------------------


def simulate_optns_array(
    initial_state: OPTNSArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: OPTNSArrayParams,
) -> tuple[OPTNSArrayState, jnp.ndarray]:
    """Run an OPTNS simulation as a pure JAX function.

    Event semantics encoded in ``event_types``:

    * **PRD**: ``role_sign * (-pprd)``  (premium payment)
    * **TD**: ``role_sign * ptd``  (termination)
    * **MD**: 0.0 (maturity, no cashflow)
    * **XD**: 0.0 (exercise calculation, no cashflow)
    * **STD**: ``role_sign * rf_values``  where ``rf_values`` carries
      the pre-computed exercise amount ``Xa = max(spot - strike, 0)``
      for calls or ``max(strike - spot, 0)`` for puts.

    Args:
        initial_state: Starting state (nt only).
        event_types: ``(num_events,)`` int32.
        year_fractions: ``(num_events,)`` float32 (unused).
        rf_values: ``(num_events,)`` float32 -- exercise amount for STD events.
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
                event_types == STD_IDX,
                params.role_sign * rf_values,
                0.0,  # AD, MD, XD, CE, NOP -> 0
            ),
        ),
    )
    return initial_state, payoffs


simulate_optns_array_jit = jax.jit(simulate_optns_array)

batch_simulate_optns_vmap = jax.vmap(simulate_optns_array)


@jax.jit
def batch_simulate_optns(
    initial_states: OPTNSArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: OPTNSArrayParams,
) -> tuple[OPTNSArrayState, jnp.ndarray]:
    """Batched OPTNS simulation -- vectorised payoff computation.

    Args:
        initial_states: ``OPTNSArrayState`` with each field shape ``[B]``.
        event_types: ``[B, T]`` int32.
        year_fractions: ``[B, T]`` float32 (unused).
        rf_values: ``[B, T]`` float32.
        params: ``OPTNSArrayParams`` with each field shape ``[B]``.

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
                event_types == STD_IDX,
                rs * rf_values,
                0.0,
            ),
        ),
    )
    return initial_states, payoffs


def batch_simulate_optns_auto(
    initial_states: OPTNSArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: OPTNSArrayParams,
) -> tuple[OPTNSArrayState, jnp.ndarray]:
    """Batched simulation using the optimal strategy."""
    return batch_simulate_optns(  # type: ignore[no-any-return]
        initial_states, event_types, year_fractions, rf_values, params
    )


# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------


def _compute_intrinsic_value(
    option_type: str,
    spot_price: float,
    strike_1: float,
    strike_2: float | None = None,
) -> float:
    """Compute option intrinsic value (pure Python, for pre-computation).

    Args:
        option_type: 'C' (call), 'P' (put), or 'CP' (collar).
        spot_price: Current spot price of the underlying.
        strike_1: Primary strike price.
        strike_2: Secondary strike price (collar only).

    Returns:
        Intrinsic value (always >= 0).
    """
    opt = option_type.upper()
    if opt in ("C", "CALL"):
        return max(spot_price - strike_1, 0.0)
    if opt in ("P", "PUT"):
        return max(strike_1 - spot_price, 0.0)
    if opt in ("CP", "COLLAR"):
        call_val = max(spot_price - strike_1, 0.0)
        put_val = max((strike_2 or 0.0) - spot_price, 0.0)
        return call_val + put_val
    return 0.0


def precompute_optns_arrays(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> tuple[OPTNSArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, OPTNSArrayParams]:
    """Pre-compute JAX arrays for array-mode OPTNS simulation.

    Mirrors the scalar ``OptionContract.simulate()`` logic:

    * PRD: premium payment ``role_sign * (-pprd)``
    * TD: termination ``role_sign * ptd``
    * XD at maturity (European): observe spot, compute intrinsic value
    * STD: pays ``role_sign * Xa`` where Xa is the intrinsic value

    The exercise amount ``Xa`` is pre-computed and stored in ``rf_values``
    at the STD event position.

    Args:
        attrs: Contract attributes (must be OPTNS type).
        rf_observer: Risk factor observer for spot price observation.

    Returns:
        ``(initial_state, event_types, year_fractions, rf_values, params)``
    """
    role_sign = get_role_sign(attrs.contract_role)

    initial_state = OPTNSArrayState(nt=jnp.array(0.0, dtype=F32))
    optns_params = OPTNSArrayParams(
        role_sign=jnp.array(role_sign, dtype=F32),
        pprd=jnp.array(attrs.price_at_purchase_date or 0.0, dtype=F32),
        ptd=jnp.array(attrs.price_at_termination_date or 0.0, dtype=F32),
    )

    # Build schedule: (event_idx, rf_value)
    schedule: list[tuple[int, float]] = []
    sd_dt = adt_to_dt(attrs.status_date)

    option_type = attrs.option_type or "C"
    strike_1 = attrs.option_strike_1 or 0.0
    strike_2 = attrs.option_strike_2
    underlier_ref = attrs.contract_structure or ""

    # Check for pre-exercised state
    pre_exercised = attrs.exercise_date is not None and attrs.exercise_amount is not None

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

        # XD: exercise date(s)
        # For European: single XD at maturity
        # For American: monthly XD events (we compute best intrinsic)
        # For Bermudan: specific dates
        # In all cases, we pre-compute the exercise amount at XD and
        # carry it through to STD.
        exercise_type = (attrs.option_exercise_type or "E").upper()

        if exercise_type == "E":
            # European: XD at maturity
            try:
                spot = float(rf_observer.observe_risk_factor(underlier_ref, attrs.maturity_date))
            except (KeyError, NotImplementedError, TypeError):
                spot = 0.0
            xa = _compute_intrinsic_value(option_type, spot, strike_1, strike_2)
            schedule.append((XD_IDX, 0.0))
        elif exercise_type == "A":
            # American: monthly XD dates; use last XD intrinsic value
            # (matches Python path where each XD overwrites xa, and STD
            # uses the final xa from the last XD before settlement)
            from jactus.contracts.array_common import fast_schedule

            xd_start = attrs.purchase_date or attrs.status_date
            xd_dates = fast_schedule(xd_start, "1M", attrs.maturity_date)
            last_xa = 0.0
            for xd_dt in xd_dates[1:]:
                if xd_dt > sd_dt:
                    try:
                        spot = float(
                            rf_observer.observe_risk_factor(underlier_ref, dt_to_adt(xd_dt))
                        )
                    except (KeyError, NotImplementedError, TypeError):
                        spot = 0.0
                    last_xa = _compute_intrinsic_value(option_type, spot, strike_1, strike_2)
                    schedule.append((XD_IDX, 0.0))
            xa = last_xa
        elif exercise_type == "B":
            # Bermudan: exercise on specific date(s)
            xa = 0.0
            if attrs.option_exercise_end_date:
                try:
                    spot = float(
                        rf_observer.observe_risk_factor(
                            underlier_ref, attrs.option_exercise_end_date
                        )
                    )
                except (KeyError, NotImplementedError, TypeError):
                    spot = 0.0
                xa = _compute_intrinsic_value(option_type, spot, strike_1, strike_2)
                schedule.append((XD_IDX, 0.0))
        else:
            xa = 0.0

        # STD: settlement with pre-computed exercise amount
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

    return initial_state, event_types, year_fractions, rf_values, optns_params


def prepare_optns_batch(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
) -> tuple[OPTNSArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, OPTNSArrayParams, jnp.ndarray]:
    """Pre-compute and pad arrays for a batch of OPTNS contracts.

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.

    Returns:
        ``(initial_states, event_types, year_fractions, rf_values, params, masks)``
    """
    n = len(contracts)
    precomputed = [precompute_optns_arrays(a, o) for a, o in contracts]

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

    initial_states = OPTNSArrayState(nt=jnp.asarray(nt_arr))
    optns_params = OPTNSArrayParams(
        role_sign=jnp.asarray(rs_arr),
        pprd=jnp.asarray(pprd_arr),
        ptd=jnp.asarray(ptd_arr),
    )

    return (
        initial_states,
        jnp.asarray(et_batch),
        jnp.asarray(yf_batch),
        jnp.asarray(rf_batch),
        optns_params,
        jnp.asarray(mask_batch),
    )


def simulate_optns_portfolio(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
    discount_rate: float | None = None,
) -> dict[str, Any]:
    """End-to-end OPTNS portfolio simulation.

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
    ) = prepare_optns_batch(contracts)

    final_states, payoffs = batch_simulate_optns_auto(
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
