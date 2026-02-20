"""Map between official ACTUS test JSON format and JACTUS internal types.

Handles the conversion of:
- camelCase ACTUS JSON term names -> snake_case JACTUS attribute names
- String values -> proper Python/JACTUS types (enums, dates, floats)
- Market data (dataObserved) -> TimeSeriesRiskFactorObserver
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from jactus.core import ActusDateTime, ContractAttributes, ContractRole, ContractType
from jactus.core.types import Calendar, DayCountConvention, EndOfMonthConvention
from jactus.observers.risk_factor import BaseRiskFactorObserver

# ============================================================================
# ACTUS JSON camelCase -> JACTUS attribute name mapping
# ============================================================================

# Maps the camelCase keys used in actus-tests JSON to JACTUS ContractAttributes fields
TERM_NAME_MAP: dict[str, str] = {
    "contractType": "contract_type",
    "contractID": "contract_id",
    "statusDate": "status_date",
    "contractDealDate": "contract_deal_date",
    "currency": "currency",
    "notionalPrincipal": "notional_principal",
    "initialExchangeDate": "initial_exchange_date",
    "maturityDate": "maturity_date",
    "nominalInterestRate": "nominal_interest_rate",
    "dayCountConvention": "day_count_convention",
    "endOfMonthConvention": "end_of_month_convention",
    "businessDayConvention": "business_day_convention",
    "contractRole": "contract_role",
    "premiumDiscountAtIED": "premium_discount_at_ied",
    "rateMultiplier": "rate_reset_multiplier",
    "rateSpread": "rate_reset_spread",
    "marketObjectCodeOfRateReset": "rate_reset_market_object",
    "cycleOfInterestPayment": "interest_payment_cycle",
    "cycleAnchorDateOfInterestPayment": "interest_payment_anchor",
    "cycleOfRateReset": "rate_reset_cycle",
    "cycleAnchorDateOfRateReset": "rate_reset_anchor",
    "cycleOfPrincipalRedemption": "principal_redemption_cycle",
    "cycleAnchorDateOfPrincipalRedemption": "principal_redemption_anchor",
    "nextPrincipalRedemptionPayment": "next_principal_redemption_amount",
    "accruedInterest": "accrued_interest",
    "capitalizationEndDate": "interest_capitalization_end_date",
    "cycleOfFee": "fee_payment_cycle",
    "cycleAnchorDateOfFee": "fee_payment_anchor",
    "feeRate": "fee_rate",
    "feeBasis": "fee_basis",
    "feeAccrued": "fee_accrued",
    "purchaseDate": "purchase_date",
    "priceAtPurchaseDate": "price_at_purchase_date",
    "terminationDate": "termination_date",
    "priceAtTerminationDate": "price_at_termination_date",
    "scalingEffect": "scaling_effect",
    "scalingIndexAtStatusDate": "scaling_index_at_status_date",
    "scalingIndexAtContractDealDate": "scaling_index_at_contract_deal_date",
    "marketObjectCodeOfScalingIndex": "scaling_market_object",
    "cycleOfScalingIndex": "scaling_index_cycle",
    "cycleAnchorDateOfScalingIndex": "scaling_index_anchor",
    "interestCalculationBase": "interest_calculation_base",
    "interestCalculationBaseAmount": "interest_calculation_base_amount",
    "cycleOfInterestCalculationBase": "interest_calculation_base_cycle",
    "cycleAnchorDateOfInterestCalculationBase": "interest_calculation_base_anchor",
    "prepaymentEffect": "prepayment_effect",
    "penaltyType": "penalty_type",
    "penaltyRate": "penalty_rate",
    "rateFloor": "rate_reset_floor",
    "rateCap": "rate_reset_cap",
    "lifeCap": "rate_reset_cap",
    "lifeFloor": "rate_reset_floor",
    "periodCap": "rate_reset_cap",
    "periodFloor": "rate_reset_floor",
    "nextResetRate": "rate_reset_next",
    "contractPerformance": "contract_performance",
    "calendar": "calendar",
    # Derivative-specific
    "settlementPeriod": "settlement_period",
    "deliverySettlement": "delivery_type",
    "optionType": "option_type",
    "optionStrike1": "option_strike_1",
    "optionExerciseType": "option_exercise_type",
    "optionExerciseEndDate": "option_exercise_end_date",
    "futuresPrice": "future_price",
    "contractStructure": "contract_structure",
    # Coverage / credit enhancement
    "coverageOfCreditEnhancement": "coverage",
    "creditEventTypeCovered": "credit_event_type",
    "guaranteedExposure": "credit_enhancement_guarantee_extent",
    # ANN-specific: amortizationDate is the horizon for annuity calculation
    "amortizationDate": "amortization_date",
}

# Day count convention string mapping
DCC_MAP: dict[str, DayCountConvention] = {
    "AA": DayCountConvention.AA,
    "A360": DayCountConvention.A360,
    "A365": DayCountConvention.A365,
    "30E360": DayCountConvention.E30360,
    "30E360ISDA": DayCountConvention.E30360ISDA,
    "_30E360": DayCountConvention.E30360,
    "30360": DayCountConvention.B30360,
    "BUS252": DayCountConvention.BUS252,
}

# Contract type mapping
CT_MAP: dict[str, ContractType] = {
    "PAM": ContractType.PAM,
    "LAM": ContractType.LAM,
    "NAM": ContractType.NAM,
    "ANN": ContractType.ANN,
    "CLM": ContractType.CLM,
    "LAX": ContractType.LAX,
    "UMP": ContractType.UMP,
    "CSH": ContractType.CSH,
    "STK": ContractType.STK,
    "COM": ContractType.COM,
    "FXOUT": ContractType.FXOUT,
    "OPTNS": ContractType.OPTNS,
    "FUTUR": ContractType.FUTUR,
    "SWPPV": ContractType.SWPPV,
    "SWAPS": ContractType.SWAPS,
    "CAPFL": ContractType.CAPFL,
    "CEG": ContractType.CEG,
    "CEC": ContractType.CEC,
}

# Contract role mapping
ROLE_MAP: dict[str, ContractRole] = {
    "RPA": ContractRole.RPA,
    "RPL": ContractRole.RPL,
    "LG": ContractRole.LG,
    "ST": ContractRole.ST,
    "BUY": ContractRole.BUY,
    "SEL": ContractRole.SEL,
}

# End-of-month convention mapping
EOMC_MAP: dict[str, EndOfMonthConvention] = {
    "EOM": EndOfMonthConvention.EOM,
    "SD": EndOfMonthConvention.SD,
}

# Calendar mapping
CALENDAR_MAP: dict[str, Calendar] = {
    "NC": Calendar.NO_CALENDAR,
    "NoCalendar": Calendar.NO_CALENDAR,
    "MF": Calendar.MONDAY_TO_FRIDAY,
    "MondayToFriday": Calendar.MONDAY_TO_FRIDAY,
    "TARGET": Calendar.TARGET,
    "US-NYSE": Calendar.US_NYSE,
    "UK-SETTLEMENT": Calendar.UK_SETTLEMENT,
}

# Cycle format mapping: ACTUS JSON uses ISO 8601 duration-like format (P1ML0)
# while JACTUS uses shorthand (1M). The trailing L0/L1 indicates stub handling.
def _parse_cycle(cycle_str: str) -> str:
    """Convert ACTUS ISO cycle string to JACTUS format.

    ACTUS format: P{n}{period}L{stub} e.g. "P1ML0", "P3ML1", "P1YL0"
    JACTUS format: {n}{period}{stub} e.g. "1M+", "3M-", "1Y+"

    L0 = long stub  → "+"
    L1 = short stub → "-"
    """
    if not cycle_str:
        return cycle_str
    s = cycle_str
    if s.startswith("P"):
        s = s[1:]
    # Extract stub indicator
    stub = ""
    if "L" in s:
        stub_part = s[s.index("L"):]
        s = s[: s.index("L")]
        if stub_part == "L0":
            stub = "+"  # Long stub
        elif stub_part == "L1":
            stub = "-"  # Short stub
    return s + stub


def _parse_datetime(dt_str: str) -> ActusDateTime:
    """Parse an ACTUS datetime string to ActusDateTime.

    Handles both full format (YYYY-MM-DDTHH:MM:SS) and shortened
    format (YYYY-MM-DDTHH:MM) used in ACTUS test files.
    """
    # Normalize shortened datetime format (e.g., "2013-01-01T00:00")
    if len(dt_str) == 16 and dt_str[10] == "T":
        dt_str = dt_str + ":00"
    return ActusDateTime.from_iso(dt_str)


def _parse_value(key: str, value: str | Any) -> Any:
    """Parse a single ACTUS test term value to the appropriate JACTUS type."""
    if value is None or value == "":
        return None

    jactus_key = TERM_NAME_MAP.get(key, key)

    # Type-specific parsing
    if jactus_key == "contract_type":
        return CT_MAP.get(str(value), str(value))
    if jactus_key == "contract_role":
        return ROLE_MAP.get(str(value), str(value))
    if jactus_key == "day_count_convention":
        return DCC_MAP.get(str(value), str(value))
    if jactus_key == "end_of_month_convention":
        return EOMC_MAP.get(str(value), str(value))
    if jactus_key == "calendar":
        return CALENDAR_MAP.get(str(value), str(value))
    if jactus_key == "scaling_effect":
        # ACTUS tests use 'O' (letter) where JACTUS enum uses '0' (digit)
        # e.g., "IOO" -> "I00", "INO" -> "IN0"
        return str(value).replace("O", "0")
    if jactus_key in (
        "status_date",
        "contract_deal_date",
        "initial_exchange_date",
        "maturity_date",
        "interest_payment_anchor",
        "rate_reset_anchor",
        "principal_redemption_anchor",
        "purchase_date",
        "termination_date",
        "interest_capitalization_end_date",
        "amortization_date",
        "fee_payment_anchor",
        "scaling_index_anchor",
        "interest_calculation_base_anchor",
        "option_exercise_end_date",
    ):
        return _parse_datetime(str(value))
    if jactus_key in (
        "interest_payment_cycle",
        "rate_reset_cycle",
        "principal_redemption_cycle",
        "fee_payment_cycle",
        "scaling_index_cycle",
        "interest_calculation_base_cycle",
    ):
        return _parse_cycle(str(value))
    if jactus_key in (
        "notional_principal",
        "nominal_interest_rate",
        "premium_discount_at_ied",
        "rate_reset_multiplier",
        "rate_reset_spread",
        "rate_reset_floor",
        "rate_reset_cap",
        "rate_reset_next",
        "next_principal_redemption_amount",
        "accrued_interest",
        "fee_rate",
        "fee_accrued",
        "price_at_purchase_date",
        "price_at_termination_date",
        "penalty_rate",
        "scaling_index_at_status_date",
        "scaling_index_at_contract_deal_date",
        "option_strike_1",
        "future_price",
        "coverage",
    ):
        return float(value)

    # Default: return as-is
    return value


def parse_test_terms(terms: dict[str, Any]) -> dict[str, Any]:
    """Convert ACTUS test JSON terms to JACTUS ContractAttributes kwargs.

    Args:
        terms: Dictionary from ACTUS test JSON "terms" field

    Returns:
        Dictionary suitable for ContractAttributes(**kwargs)
    """
    kwargs: dict[str, Any] = {}
    for key, value in terms.items():
        if key not in TERM_NAME_MAP:
            continue  # Skip unknown terms
        jactus_key = TERM_NAME_MAP[key]
        parsed_value = _parse_value(key, value)
        if parsed_value is not None:
            kwargs[jactus_key] = parsed_value
    return kwargs


def load_test_file(path: Path) -> dict[str, Any]:
    """Load an ACTUS test JSON file.

    Args:
        path: Path to the JSON file

    Returns:
        Dictionary of test case ID -> test case data
    """
    with open(path) as f:
        return json.load(f)


# ============================================================================
# Time-series risk factor observer for ACTUS test market data
# ============================================================================


class TimeSeriesRiskFactorObserver(BaseRiskFactorObserver):
    """Risk factor observer with time-series data from ACTUS test cases.

    Supports the dataObserved format from ACTUS test JSON:
    {
        "USD_SWP": {
            "identifier": "USD_SWP",
            "data": [
                {"timestamp": "2013-02-01T00:00:00", "value": "0.01"},
                {"timestamp": "2013-05-01T00:00:00", "value": "0.02"}
            ]
        }
    }

    For a given observation time, returns the most recent data point
    at or before that time (piecewise constant interpolation).
    """

    def __init__(
        self,
        data_observed: dict[str, Any] | None = None,
        default_value: float = 0.0,
    ):
        super().__init__(name="TimeSeriesRiskFactorObserver")
        self.default_value = default_value
        # Parse time series: {identifier: [(ActusDateTime, float), ...]}
        self._series: dict[str, list[tuple[ActusDateTime, float]]] = {}
        if data_observed:
            for identifier, obs_data in data_observed.items():
                points = []
                for point in obs_data.get("data", []):
                    ts = ActusDateTime.from_iso(point["timestamp"])
                    val = float(point["value"])
                    points.append((ts, val))
                # Sort by time
                points.sort(key=lambda p: p[0])
                self._series[identifier] = points

    def _get_risk_factor(
        self,
        identifier: str,
        time: ActusDateTime,
        state: Any = None,
        attributes: Any = None,
    ) -> jnp.ndarray:
        """Get the most recent observation at or before the given time."""
        if identifier not in self._series:
            return jnp.array(self.default_value, dtype=jnp.float32)

        series = self._series[identifier]
        # Find the most recent observation at or before time
        best_value = self.default_value
        for ts, val in series:
            if ts <= time:
                best_value = val
            else:
                break
        return jnp.array(best_value, dtype=jnp.float32)

    def _get_event_data(
        self,
        identifier: str,
        event_type: Any,
        time: ActusDateTime,
        state: Any = None,
        attributes: Any = None,
    ) -> Any:
        """Delegate to risk factor observation."""
        return self._get_risk_factor(identifier, time, state, attributes)
