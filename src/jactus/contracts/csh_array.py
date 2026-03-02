"""Array-mode CSH (Cash) simulation — JIT-compiled, vmap-able pure JAX.

CSH is the simplest contract type: a single AD event with zero payoff.
The batch version is trivial — no scan needed, just returns zeros.

Example::

    from jactus.contracts.csh_array import precompute_csh_arrays, simulate_csh_array

    arrays = precompute_csh_arrays(attrs, rf_observer)
    final_state, payoffs = simulate_csh_array(*arrays)

    # Portfolio:
    from jactus.contracts.csh_array import simulate_csh_portfolio
    result = simulate_csh_portfolio(contracts)
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jactus.contracts.array_common import (
    AD_IDX,
    F32,
    get_role_sign,
)
from jactus.core import ContractAttributes
from jactus.observers import RiskFactorObserver

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class CSHArrayState(NamedTuple):
    """Minimal scan-loop state for CSH simulation.

    CSH only tracks notional principal (constant throughout).
    """

    nt: jnp.ndarray  # Notional principal (signed)


class CSHArrayParams(NamedTuple):
    """Static contract parameters for CSH."""

    role_sign: jnp.ndarray  # +1.0 or -1.0
    notional_principal: jnp.ndarray


# ---------------------------------------------------------------------------
# Simulation kernel (trivial — CSH has only AD events with zero payoff)
# ---------------------------------------------------------------------------


def simulate_csh_array(
    initial_state: CSHArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: CSHArrayParams,
) -> tuple[CSHArrayState, jnp.ndarray]:
    """Run a CSH simulation as a pure JAX function.

    CSH is trivial: all payoffs are 0.0, state never changes.

    Args:
        initial_state: Starting state (nt only).
        event_types: ``(num_events,)`` int32.
        year_fractions: ``(num_events,)`` float32 (unused).
        rf_values: ``(num_events,)`` float32 (unused).
        params: Static contract parameters (unused).

    Returns:
        ``(final_state, payoffs)`` where payoffs is all zeros.
    """
    payoffs = jnp.zeros_like(year_fractions)
    return initial_state, payoffs


simulate_csh_array_jit = jax.jit(simulate_csh_array)

batch_simulate_csh_vmap = jax.vmap(simulate_csh_array)


@jax.jit
def batch_simulate_csh(
    initial_states: CSHArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: CSHArrayParams,
) -> tuple[CSHArrayState, jnp.ndarray]:
    """Batched CSH simulation — trivially returns zeros.

    Args:
        initial_states: ``CSHArrayState`` with each field shape ``[B]``.
        event_types: ``[B, T]`` int32.
        year_fractions: ``[B, T]`` float32.
        rf_values: ``[B, T]`` float32.
        params: ``CSHArrayParams`` with each field shape ``[B]``.

    Returns:
        ``(final_states, payoffs)`` where ``payoffs`` is ``[B, T]`` of zeros.
    """
    payoffs = jnp.zeros_like(year_fractions)
    return initial_states, payoffs


def batch_simulate_csh_auto(
    initial_states: CSHArrayState,
    event_types: jnp.ndarray,
    year_fractions: jnp.ndarray,
    rf_values: jnp.ndarray,
    params: CSHArrayParams,
) -> tuple[CSHArrayState, jnp.ndarray]:
    """Batched simulation using the optimal strategy."""
    return batch_simulate_csh(initial_states, event_types, year_fractions, rf_values, params)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Pre-computation
# ---------------------------------------------------------------------------


def precompute_csh_arrays(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> tuple[CSHArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, CSHArrayParams]:
    """Pre-compute JAX arrays for array-mode CSH simulation.

    Args:
        attrs: Contract attributes (must be CSH type).
        rf_observer: Risk factor observer (not used for CSH).

    Returns:
        ``(initial_state, event_types, year_fractions, rf_values, params)``
    """
    role_sign = get_role_sign(attrs.contract_role)
    nt = role_sign * (attrs.notional_principal or 0.0)

    initial_state = CSHArrayState(nt=jnp.array(nt, dtype=F32))
    params = CSHArrayParams(
        role_sign=jnp.array(role_sign, dtype=F32),
        notional_principal=jnp.array(attrs.notional_principal or 0.0, dtype=F32),
    )

    # CSH has a single AD event
    event_types = jnp.array([AD_IDX], dtype=jnp.int32)
    year_fractions = jnp.array([0.0], dtype=F32)
    rf_values = jnp.array([0.0], dtype=F32)

    return initial_state, event_types, year_fractions, rf_values, params


def prepare_csh_batch(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
) -> tuple[CSHArrayState, jnp.ndarray, jnp.ndarray, jnp.ndarray, CSHArrayParams, jnp.ndarray]:
    """Pre-compute and pad arrays for a batch of CSH contracts.

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.

    Returns:
        ``(initial_states, event_types, year_fractions, rf_values, params, masks)``
    """
    n = len(contracts)

    nt_arr = np.zeros(n, dtype=np.float32)
    rs_arr = np.zeros(n, dtype=np.float32)
    np_arr = np.zeros(n, dtype=np.float32)

    for i, (attrs, _obs) in enumerate(contracts):
        role_sign = get_role_sign(attrs.contract_role)
        nt_arr[i] = role_sign * (attrs.notional_principal or 0.0)
        rs_arr[i] = role_sign
        np_arr[i] = attrs.notional_principal or 0.0

    # CSH: single AD event per contract
    event_types = jnp.full((n, 1), AD_IDX, dtype=jnp.int32)
    year_fractions = jnp.zeros((n, 1), dtype=F32)
    rf_values = jnp.zeros((n, 1), dtype=F32)
    masks = jnp.ones((n, 1), dtype=F32)

    initial_states = CSHArrayState(nt=jnp.asarray(nt_arr))
    params = CSHArrayParams(
        role_sign=jnp.asarray(rs_arr),
        notional_principal=jnp.asarray(np_arr),
    )

    return initial_states, event_types, year_fractions, rf_values, params, masks


def simulate_csh_portfolio(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
    discount_rate: float | None = None,
) -> dict[str, Any]:
    """End-to-end CSH portfolio simulation.

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
    ) = prepare_csh_batch(contracts)

    final_states, payoffs = batch_simulate_csh_auto(
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
