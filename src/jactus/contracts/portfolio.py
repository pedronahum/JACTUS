"""Unified portfolio simulation API for mixed contract types.

Accepts a portfolio of contracts with mixed types, groups them by type,
dispatches to each type's batch kernel, and returns per-contract results.

Contracts without a dedicated array-mode implementation (CLM, UMP, SWAPS,
CAPFL, CEG, CEC) fall back to the scalar Python simulation path.

Example::

    from jactus.contracts.portfolio import simulate_portfolio

    contracts = [
        (pam_attrs, rf_obs),
        (lam_attrs, rf_obs),
        (csh_attrs, rf_obs),
        (optns_attrs, rf_obs),
    ]
    result = simulate_portfolio(contracts)
    # result["total_cashflows"]  -> jnp.array of shape (4,)
    # result["payoffs"]          -> list of per-contract payoff arrays
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from jactus.core import ContractAttributes, ContractType
from jactus.observers import RiskFactorObserver

# ---------------------------------------------------------------------------
# Type -> portfolio function registry
# ---------------------------------------------------------------------------

_PORTFOLIO_FN_REGISTRY: dict[ContractType, Any] = {}


def _get_portfolio_fn(ct: ContractType) -> Any | None:
    """Lazy-load portfolio function for a contract type."""
    if ct in _PORTFOLIO_FN_REGISTRY:
        return _PORTFOLIO_FN_REGISTRY[ct]

    fn: Any = None
    try:
        if ct == ContractType.PAM:
            from jactus.contracts.pam_array import simulate_pam_portfolio

            fn = simulate_pam_portfolio
        elif ct == ContractType.LAM:
            from jactus.contracts.lam_array import simulate_lam_portfolio

            fn = simulate_lam_portfolio
        elif ct == ContractType.NAM:
            from jactus.contracts.nam_array import simulate_nam_portfolio

            fn = simulate_nam_portfolio
        elif ct == ContractType.ANN:
            from jactus.contracts.ann_array import simulate_ann_portfolio

            fn = simulate_ann_portfolio
        elif ct == ContractType.LAX:
            from jactus.contracts.lax_array import simulate_lax_portfolio

            fn = simulate_lax_portfolio
        elif ct == ContractType.CSH:
            from jactus.contracts.csh_array import simulate_csh_portfolio

            fn = simulate_csh_portfolio
        elif ct == ContractType.STK:
            from jactus.contracts.stk_array import simulate_stk_portfolio

            fn = simulate_stk_portfolio
        elif ct == ContractType.COM:
            from jactus.contracts.com_array import simulate_com_portfolio

            fn = simulate_com_portfolio
        elif ct == ContractType.FXOUT:
            from jactus.contracts.fxout_array import simulate_fxout_portfolio

            fn = simulate_fxout_portfolio
        elif ct == ContractType.FUTUR:
            from jactus.contracts.futur_array import simulate_futur_portfolio

            fn = simulate_futur_portfolio
        elif ct == ContractType.OPTNS:
            from jactus.contracts.optns_array import simulate_optns_portfolio

            fn = simulate_optns_portfolio
        elif ct == ContractType.SWPPV:
            from jactus.contracts.swppv_array import simulate_swppv_portfolio

            fn = simulate_swppv_portfolio
    except ImportError:
        fn = None

    _PORTFOLIO_FN_REGISTRY[ct] = fn
    return fn


# Contract types that use the scalar Python fallback path
_FALLBACK_TYPES = frozenset(
    {
        ContractType.CLM,
        ContractType.UMP,
        ContractType.SWAPS,
        ContractType.CAPFL,
        ContractType.CEG,
        ContractType.CEC,
    }
)

# All types with dedicated array-mode batch kernels
BATCH_SUPPORTED_TYPES = frozenset(
    {
        ContractType.PAM,
        ContractType.LAM,
        ContractType.NAM,
        ContractType.ANN,
        ContractType.LAX,
        ContractType.CSH,
        ContractType.STK,
        ContractType.COM,
        ContractType.FXOUT,
        ContractType.FUTUR,
        ContractType.OPTNS,
        ContractType.SWPPV,
    }
)


def _simulate_scalar_fallback(
    attrs: ContractAttributes,
    rf_observer: RiskFactorObserver,
) -> float:
    """Simulate a single contract via the scalar Python path.

    Returns total cashflow (sum of all event payoffs).
    """
    from jactus.contracts import create_contract

    contract = create_contract(attrs, rf_observer)
    result = contract.simulate()
    return sum(float(e.payoff) for e in result.events)


def simulate_portfolio(
    contracts: list[tuple[ContractAttributes, RiskFactorObserver]],
    discount_rate: float | None = None,
) -> dict[str, Any]:
    """Simulate a mixed-type portfolio using optimal batch strategies.

    Groups contracts by type, dispatches to each type's batch kernel,
    and falls back to the scalar Python path for unsupported types.

    Args:
        contracts: List of ``(attributes, rf_observer)`` pairs.
            Contract types may be mixed.
        discount_rate: If provided, compute present values (passed to
            each type's portfolio function where supported).

    Returns:
        Dict with:
            - ``total_cashflows``: ``jnp.ndarray`` of shape ``(N,)`` —
              total cashflow for each contract in input order.
            - ``num_contracts``: Total number of contracts.
            - ``batch_contracts``: Number of contracts simulated via
              batch kernels.
            - ``fallback_contracts``: Number of contracts simulated via
              the scalar Python path.
            - ``types_used``: Set of ``ContractType`` values present.
            - ``per_type_results``: Dict mapping ``ContractType`` to the
              raw result dict from each type's portfolio function
              (only for batch-simulated types).
    """
    n = len(contracts)
    if n == 0:
        return {
            "total_cashflows": jnp.zeros(0),
            "num_contracts": 0,
            "batch_contracts": 0,
            "fallback_contracts": 0,
            "types_used": set(),
            "per_type_results": {},
        }

    # Group contracts by type, preserving original indices
    type_groups: dict[ContractType, list[tuple[int, ContractAttributes, RiskFactorObserver]]] = {}
    for i, (attrs, rf_obs) in enumerate(contracts):
        ct = attrs.contract_type
        if ct not in type_groups:
            type_groups[ct] = []
        type_groups[ct].append((i, attrs, rf_obs))

    # Output array
    total_cashflows = np.zeros(n, dtype=np.float32)
    per_type_results: dict[ContractType, dict[str, Any]] = {}
    batch_count = 0
    fallback_count = 0

    for ct, group in type_groups.items():
        indices = [g[0] for g in group]
        group_contracts = [(g[1], g[2]) for g in group]

        portfolio_fn = _get_portfolio_fn(ct)

        if portfolio_fn is not None:
            # Batch simulation path
            kwargs: dict[str, Any] = {}
            if discount_rate is not None:
                kwargs["discount_rate"] = discount_rate

            result = portfolio_fn(group_contracts, **kwargs)
            per_type_results[ct] = result

            group_totals = result["total_cashflows"]
            for j, idx in enumerate(indices):
                total_cashflows[idx] = float(group_totals[j])

            batch_count += len(group)
        else:
            # Scalar fallback path
            for _j, (idx, attrs, rf_obs) in enumerate(group):
                total_cashflows[idx] = _simulate_scalar_fallback(attrs, rf_obs)

            fallback_count += len(group)

    return {
        "total_cashflows": jnp.asarray(total_cashflows),
        "num_contracts": n,
        "batch_contracts": batch_count,
        "fallback_contracts": fallback_count,
        "types_used": set(type_groups.keys()),
        "per_type_results": per_type_results,
    }
