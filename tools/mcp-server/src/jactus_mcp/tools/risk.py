"""Risk analytics tools for computing DV01, delta, gamma, and PV01."""

import logging
from typing import Any

from jactus.contracts import create_contract
from jactus.core import ContractAttributes
from jactus.observers import ConstantRiskFactorObserver
from pydantic import ValidationError

from jactus_mcp.tools._utils import prepare_attributes

logger = logging.getLogger(__name__)


def _total_cashflow(attrs: ContractAttributes, rate: float) -> float:
    """Compute the total cashflow (sum of all payoffs) at a given risk factor rate."""
    rf = ConstantRiskFactorObserver(constant_value=rate)
    contract = create_contract(attrs, rf)
    result = contract.simulate()
    return sum(float(e.payoff) for e in result.events)


def _total_cashflow_at_nominal_rate(
    base_attrs: dict[str, Any], rate_override: float
) -> float:
    """Simulate with a modified nominal_interest_rate and return total cashflow."""
    modified = dict(base_attrs)
    modified["nominal_interest_rate"] = rate_override
    valid_fields = set(ContractAttributes.model_fields.keys())
    prepared = prepare_attributes(modified)
    prepared = {k: v for k, v in prepared.items() if k in valid_fields}
    attrs = ContractAttributes(**prepared)
    rf = ConstantRiskFactorObserver(constant_value=0.0)
    contract = create_contract(attrs, rf)
    result = contract.simulate()
    return sum(float(e.payoff) for e in result.events)


def compute_risk(
    attributes: dict[str, Any],
    risk_metric: str = "dv01",
    base_rate: float = 0.05,
    bump_size: float = 0.0001,
) -> dict[str, Any]:
    """Compute a risk metric for a contract using finite differences.

    Supports DV01, delta, gamma, and PV01 risk metrics. Uses finite
    difference approximation on the nominal_interest_rate.

    Args:
        attributes: Contract attributes dictionary.
        risk_metric: One of "dv01", "delta", "gamma", "pv01".
        base_rate: The base nominal interest rate (default 0.05).
        bump_size: The bump size for finite difference (default 0.0001 = 1bp).

    Returns:
        Dictionary with metric name, value, base rate, and bump size.
    """
    valid_metrics = {"dv01", "delta", "gamma", "pv01"}
    metric = risk_metric.lower()
    if metric not in valid_metrics:
        return {
            "success": False,
            "error": f"Invalid risk_metric '{risk_metric}'. Valid: {', '.join(sorted(valid_metrics))}",
            "error_code": "invalid_metric",
            "suggestion": "Use one of: dv01, delta, gamma, pv01",
        }

    try:
        # Ensure nominal_interest_rate is set to base_rate
        attrs_with_rate = dict(attributes)
        attrs_with_rate["nominal_interest_rate"] = base_rate

        # Compute base PV
        pv_base = _total_cashflow_at_nominal_rate(attrs_with_rate, base_rate)

        if metric in ("dv01", "delta", "pv01"):
            # First-order finite difference: dPV/dr
            pv_up = _total_cashflow_at_nominal_rate(attrs_with_rate, base_rate + bump_size)
            value = (pv_up - pv_base) / bump_size

            if metric == "dv01":
                # DV01: change in PV per 1bp move (absolute)
                value = value * 0.0001
            elif metric == "pv01":
                # PV01: change in PV for a 1bp parallel shift (same as DV01 here)
                value = value * 0.0001

        elif metric == "gamma":
            # Second-order finite difference: d²PV/dr²
            pv_up = _total_cashflow_at_nominal_rate(attrs_with_rate, base_rate + bump_size)
            pv_down = _total_cashflow_at_nominal_rate(attrs_with_rate, base_rate - bump_size)
            value = (pv_up - 2 * pv_base + pv_down) / (bump_size ** 2)

        return {
            "success": True,
            "metric": metric,
            "value": value,
            "base_rate": base_rate,
            "bump_size": bump_size,
            "base_pv": pv_base,
            "contract_type": attributes.get("contract_type", "unknown"),
            "contract_id": attributes.get("contract_id", "unknown"),
        }

    except KeyError as e:
        return {
            "success": False,
            "error": f"Invalid attribute value: {e!s}",
            "error_code": "invalid_attribute",
            "suggestion": "Use jactus_get_contract_schema to see valid enum values.",
        }
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
            errors.append(f"{field}: {error['msg']}")
        return {
            "success": False,
            "error": f"Validation failed: {'; '.join(errors)}",
            "error_code": "validation_error",
            "suggestion": "Use jactus_validate_attributes to check attributes first.",
        }
    except Exception as e:
        logger.error(f"Risk computation error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "error_code": "computation_error",
            "suggestion": "Check attributes and try jactus_simulate_contract first.",
        }


def simulate_portfolio(
    contracts: list[dict[str, Any]],
    risk_factor_rate: float = 0.05,
) -> dict[str, Any]:
    """Simulate a portfolio of contracts and return aggregate results.

    Simulates each contract individually and aggregates the results into
    a portfolio summary with total inflows, outflows, and net cashflow.

    Args:
        contracts: List of contract attribute dictionaries.
        risk_factor_rate: Flat risk factor rate applied to all contracts.

    Returns:
        Dictionary with portfolio size, per-contract summaries, and aggregates.
    """
    if not contracts:
        return {
            "success": False,
            "error": "Empty contracts list",
            "error_code": "empty_portfolio",
            "suggestion": "Provide at least one contract in the contracts array.",
        }

    valid_fields = set(ContractAttributes.model_fields.keys())
    rf = ConstantRiskFactorObserver(constant_value=risk_factor_rate)

    contract_results = []
    total_inflows = 0.0
    total_outflows = 0.0
    errors = []

    for i, raw_attrs in enumerate(contracts):
        cid = raw_attrs.get("contract_id", f"contract-{i}")
        try:
            prepared = prepare_attributes(raw_attrs)
            prepared = {k: v for k, v in prepared.items() if k in valid_fields}
            attrs = ContractAttributes(**prepared)
            contract = create_contract(attrs, rf)
            result = contract.simulate()

            payoffs = [float(e.payoff) for e in result.events]
            non_zero = [p for p in payoffs if abs(p) > 1e-10]
            inflows = sum(p for p in non_zero if p > 0)
            outflows = sum(p for p in non_zero if p < 0)
            net = inflows + outflows

            total_inflows += inflows
            total_outflows += outflows

            contract_results.append({
                "contract_id": cid,
                "contract_type": attrs.contract_type.name,
                "summary": {
                    "total_inflows": inflows,
                    "total_outflows": outflows,
                    "net_cashflow": net,
                    "event_count": len(result.events),
                    "non_zero_events": len(non_zero),
                },
            })
        except Exception as e:
            errors.append({
                "contract_id": cid,
                "error": str(e),
            })
            contract_results.append({
                "contract_id": cid,
                "error": str(e),
            })

    response: dict[str, Any] = {
        "success": len(errors) == 0,
        "portfolio_size": len(contracts),
        "contracts": contract_results,
        "aggregate": {
            "total_inflows": total_inflows,
            "total_outflows": total_outflows,
            "total_net_cashflow": total_inflows + total_outflows,
            "successful": len(contracts) - len(errors),
            "failed": len(errors),
        },
    }

    if errors:
        response["errors"] = errors

    return response
