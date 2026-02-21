"""Contract attributes for ACTUS contracts.

This module provides the ContractAttributes class, which represents all possible
attributes (contract terms) for ACTUS contracts using Pydantic for validation.

References:
    ACTUS Technical Specification v1.1, Sections 4-5 (Contract Attributes)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from jactus.core.time import ActusDateTime
from jactus.core.types import (
    BusinessDayConvention,
    Calendar,
    ContractPerformance,
    ContractRole,
    ContractType,
    Cycle,
    DayCountConvention,
    EndOfMonthConvention,
    FeeBasis,
    InterestCalculationBase,
    PrepaymentEffect,
    ScalingEffect,
)


class ContractAttributes(BaseModel):
    """All possible attributes for an ACTUS contract.

    This class uses Pydantic for automatic validation. Not all attributes
    are required for all contract types - use validate_contract_type_compatibility()
    to ensure a valid configuration.

    Attributes follow ACTUS naming conventions. See ACTUS documentation for
    the meaning and constraints of each attribute.

    Example:
        >>> attrs = ContractAttributes(
        ...     contract_id="LOAN-001",
        ...     contract_type=ContractType.PAM,
        ...     contract_role=ContractRole.RPA,
        ...     status_date=ActusDateTime.from_iso("2024-01-01T00:00:00"),
        ...     initial_exchange_date=ActusDateTime.from_iso("2024-01-15T00:00:00"),
        ...     maturity_date=ActusDateTime.from_iso("2029-01-15T00:00:00"),
        ...     notional_principal=100000.0,
        ...     nominal_interest_rate=0.05,
        ...     currency="USD"
        ... )

    References:
        ACTUS Technical Specification v1.1, Section 4
    """

    # ========== CRITICAL ATTRIBUTES ==========
    contract_id: str = Field(..., description="Unique contract identifier")
    contract_type: ContractType = Field(..., description="ACTUS contract type (CT)")
    contract_role: ContractRole = Field(
        ..., description="Contract role - asset or liability (CNTRL)"
    )
    status_date: ActusDateTime = Field(..., description="Analysis/valuation date (SD)")
    contract_deal_date: ActusDateTime | None = Field(None, description="Contract deal date (CDD)")
    initial_exchange_date: ActusDateTime | None = Field(
        None, description="Initial exchange date - contract start (IED)"
    )
    maturity_date: ActusDateTime | None = Field(
        None, description="Maturity date - contract end (MD)"
    )
    purchase_date: ActusDateTime | None = Field(
        None, description="Purchase date for secondary market (PRD)"
    )
    termination_date: ActusDateTime | None = Field(None, description="Early termination date (TD)")
    analysis_dates: list[ActusDateTime] | None = Field(
        None, description="Array of analysis dates for valuation (AD)"
    )

    # ========== NOTIONAL AND RATES ==========
    notional_principal: float | None = Field(None, description="Notional principal amount (NT)")
    nominal_interest_rate: float | None = Field(
        None, description="Nominal interest rate as decimal (IPNR)"
    )
    nominal_interest_rate_2: float | None = Field(
        None, description="Second nominal interest rate for swaps (IPNR2)"
    )
    currency: str = Field(default="USD", description="Currency ISO code (CUR)")
    currency_2: str | None = Field(None, description="Second currency for FX contracts (CUR2)")
    notional_principal_2: float | None = Field(
        None, description="Second notional principal for FX/swaps (NT2)"
    )

    # ========== SETTLEMENT AND DELIVERY ==========
    delivery_settlement: str | None = Field(
        None,
        description="Delivery/settlement mode: 'D' (delivery/net) or 'S' (settlement/gross) (DS)",
    )
    settlement_date: ActusDateTime | None = Field(
        None, description="Settlement date for derivatives (STD)"
    )

    # ========== OPTIONS AND DERIVATIVES ==========
    option_type: str | None = Field(
        None,
        description="Option type: 'C' (call), 'P' (put), 'CP' (collar) (OPTP)",
    )
    option_strike_1: float | None = Field(
        None, description="Primary strike price for options (OPS1)"
    )
    option_strike_2: float | None = Field(
        None, description="Secondary strike price for collar options (OPS2)"
    )
    option_exercise_type: str | None = Field(
        None,
        description="Exercise type: 'E' (European), 'A' (American), 'B' (Bermudan) (OPXT)",
    )
    option_exercise_end_date: ActusDateTime | None = Field(
        None, description="Last date option can be exercised (OPXED)"
    )
    exercise_date: ActusDateTime | None = Field(
        None, description="Date of option/derivative exercise (XD)"
    )
    exercise_amount: float | None = Field(
        None, description="Amount determined at exercise (XA)"
    )
    settlement_period: str | None = Field(
        None, description="Period between exercise and settlement (STPD)"
    )
    delivery_type: str | None = Field(
        None,
        description="Delivery type: 'P' (physical), 'C' (cash) (DVTP)",
    )
    contract_structure: str | None = Field(
        None,
        description="Reference to underlier or child contracts (CTST)",
    )
    future_price: float | None = Field(
        None,
        description="Agreed futures price (PFUT)",
    )
    settlement_currency: str | None = Field(
        None,
        description="Settlement currency for cross-currency contracts (CURS)",
    )
    fixing_period: str | None = Field(
        None,
        description="Period between rate observation and reset (RRFIX)",
    )

    # ========== COMMODITY AND EQUITY ==========
    quantity: float | None = Field(None, description="Quantity of commodity/stock (QT)")
    unit: str | None = Field(None, description="Unit of measurement (UNIT)")
    market_object_code: str | None = Field(
        None, description="Market object code for price observation (MOC)"
    )
    market_object_code_of_dividends: str | None = Field(
        None, description="Market object code for dividend observation (DVMO)"
    )
    dividend_cycle: Cycle | None = Field(
        None, description="Dividend payment cycle (DVCL)"
    )
    dividend_anchor: ActusDateTime | None = Field(
        None, description="Dividend payment anchor date (DVANX)"
    )

    # ========== DAY COUNT AND BUSINESS DAY CONVENTIONS ==========
    day_count_convention: DayCountConvention | None = Field(
        None, description="Day count convention for interest (DCC)"
    )
    business_day_convention: BusinessDayConvention = Field(
        default=BusinessDayConvention.NULL, description="Business day convention (BDC)"
    )
    end_of_month_convention: EndOfMonthConvention = Field(
        default=EndOfMonthConvention.SD, description="End of month convention (EOMC)"
    )
    calendar: Calendar = Field(
        default=Calendar.NO_CALENDAR, description="Business day calendar (CLDR)"
    )

    # ========== SCHEDULE ATTRIBUTES ==========
    interest_payment_cycle: Cycle | None = Field(None, description="Interest payment cycle (IPCL)")
    interest_payment_anchor: ActusDateTime | None = Field(
        None, description="Interest payment schedule anchor (IPANX)"
    )
    interest_capitalization_end_date: ActusDateTime | None = Field(
        None, description="Interest capitalization end date (IPCED)"
    )
    principal_redemption_cycle: Cycle | None = Field(
        None, description="Principal redemption cycle (PRCL)"
    )
    principal_redemption_anchor: ActusDateTime | None = Field(
        None, description="Principal redemption anchor (PRANX)"
    )
    fee_payment_cycle: Cycle | None = Field(None, description="Fee payment cycle (FECL)")
    fee_payment_anchor: ActusDateTime | None = Field(None, description="Fee payment anchor (FEANX)")
    rate_reset_cycle: Cycle | None = Field(None, description="Rate reset cycle (RRCL)")
    rate_reset_anchor: ActusDateTime | None = Field(None, description="Rate reset anchor (RRANX)")
    scaling_index_cycle: Cycle | None = Field(None, description="Scaling index cycle (SCCL)")
    scaling_index_anchor: ActusDateTime | None = Field(
        None, description="Scaling index anchor (SCANX)"
    )
    next_principal_redemption_amount: float | None = Field(
        None, description="Next principal redemption amount (PRNXT)"
    )
    interest_calculation_base_cycle: Cycle | None = Field(
        None, description="Interest calculation base cycle (IPCBCL)"
    )
    interest_calculation_base_anchor: ActusDateTime | None = Field(
        None, description="Interest calculation base anchor (IPCBANX)"
    )

    # ========== ARRAY SCHEDULE ATTRIBUTES ==========
    array_pr_anchor: list[ActusDateTime] | None = Field(
        None, description="Array of PR anchors (ARPRANX)"
    )
    array_pr_cycle: list[Cycle] | None = Field(None, description="Array of PR cycles (ARPRCL)")
    array_pr_next: list[float] | None = Field(
        None, description="Array of next PR amounts (ARPRNXT)"
    )
    array_increase_decrease: list[str] | None = Field(
        None, description="Array of increase/decrease indicators (ARINCDEC): 'INC' or 'DEC'"
    )
    array_ip_anchor: list[ActusDateTime] | None = Field(
        None, description="Array of IP anchors (ARIPANX)"
    )
    array_ip_cycle: list[Cycle] | None = Field(None, description="Array of IP cycles (ARIPCL)")
    array_rr_anchor: list[ActusDateTime] | None = Field(
        None, description="Array of RR anchors (ARRRANX)"
    )
    array_rr_cycle: list[Cycle] | None = Field(None, description="Array of RR cycles (ARRRCL)")
    array_rate: list[float] | None = Field(None, description="Array of interest rates (ARRATE)")
    array_fixed_variable: list[str] | None = Field(
        None, description="Array of fixed/variable rate indicators (ARFIXVAR): 'F' or 'V'"
    )

    # ========== RATE RESET ATTRIBUTES ==========
    rate_reset_market_object: str | None = Field(
        None, description="Rate reset market reference (RRMO)"
    )
    rate_reset_multiplier: float | None = Field(None, description="Rate reset multiplier (RRMLT)")
    rate_reset_spread: float | None = Field(None, description="Rate reset spread (RRSP)")
    rate_reset_floor: float | None = Field(None, description="Rate reset floor (RRLF)")
    rate_reset_cap: float | None = Field(None, description="Rate reset cap (RRLC)")
    rate_reset_next: float | None = Field(None, description="Next rate reset value (RRNXT)")

    # ========== FEE ATTRIBUTES ==========
    fee_rate: float | None = Field(None, description="Fee rate (FER)")
    fee_basis: FeeBasis | None = Field(None, description="Fee calculation basis (FEB)")
    fee_accrued: float | None = Field(None, description="Accrued fees (FEAC)")

    # ========== PREPAYMENT ATTRIBUTES ==========
    prepayment_effect: PrepaymentEffect = Field(
        default=PrepaymentEffect.N, description="Prepayment effect (PPEF)"
    )
    penalty_type: str | None = Field(None, description="Penalty type (PYTP)")
    penalty_rate: float | None = Field(None, description="Penalty rate (PYRT)")

    # ========== SCALING ATTRIBUTES ==========
    scaling_effect: ScalingEffect = Field(
        default=ScalingEffect.S000, description="Scaling effect (SCEF)"
    )
    scaling_index_at_status_date: float | None = Field(
        None, description="Scaling index at SD (SCIXSD)"
    )
    scaling_index_at_contract_deal_date: float | None = Field(
        None, description="Scaling index at CDD (SCIXCDD)"
    )
    scaling_market_object: str | None = Field(None, description="Scaling market reference (SCMO)")

    # ========== CREDIT ENHANCEMENT ATTRIBUTES ==========
    coverage: float | None = Field(None, description="Coverage ratio (CECV)")
    credit_event_type: ContractPerformance | None = Field(
        None, description="Credit event type (CET)"
    )
    credit_enhancement_guarantee_extent: str | None = Field(
        None, description="Guarantee extent: NO, NI, or MV (CEGE)"
    )

    # ========== OTHER ATTRIBUTES ==========
    accrued_interest: float | None = Field(None, description="Accrued interest (IPAC)")
    interest_calculation_base: InterestCalculationBase | None = Field(
        None, description="Interest calculation base (IPCB)"
    )
    interest_calculation_base_amount: float | None = Field(
        None, description="Interest calculation base amount (IPCBA)"
    )
    amortization_date: ActusDateTime | None = Field(
        None, description="Amortization end date for ANN contracts (AMD)"
    )
    contract_performance: ContractPerformance = Field(
        default=ContractPerformance.PF, description="Contract performance status (PRF)"
    )
    premium_discount_at_ied: float | None = Field(
        None, description="Premium/discount at IED (PDIED)"
    )
    price_at_purchase_date: float | None = Field(None, description="Price at purchase date (PPRD)")
    price_at_termination_date: float | None = Field(None, description="Price at termination (PTD)")

    # Pydantic v2 config
    model_config = {
        "arbitrary_types_allowed": True,  # Allow ActusDateTime
        "validate_assignment": True,  # Validate on attribute assignment
    }

    @field_validator("nominal_interest_rate")
    @classmethod
    def validate_interest_rate(cls, v: float | None) -> float | None:
        """Validate that interest rate is greater than -1 (can be negative)."""
        if v is not None and v <= -1.0:
            raise ValueError(f"Interest rate must be > -1, got {v}")
        return v

    @field_validator("notional_principal")
    @classmethod
    def validate_notional(cls, v: float | None) -> float | None:
        """Validate that notional is non-zero if defined."""
        if v is not None and v == 0.0:
            raise ValueError("Notional principal must be non-zero")
        return v

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code is 3 uppercase letters."""
        if not v:
            raise ValueError("Currency code cannot be empty")
        if len(v) != 3:
            raise ValueError(f"Currency code must be 3 characters, got '{v}'")
        if not v.isupper():
            raise ValueError(f"Currency code must be uppercase, got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_dates(self) -> ContractAttributes:
        """Validate date ordering constraints."""
        # Note: IED < SD is allowed per ACTUS spec (contract already existed
        # before the status/observation date). When IED < SD, the IED event
        # is not generated but state is initialized as if IED already occurred.

        # MD must be after IED
        if self.initial_exchange_date and self.maturity_date:
            if self.maturity_date <= self.initial_exchange_date:
                raise ValueError(
                    f"Maturity date {self.maturity_date.to_iso()} "
                    f"must be > initial exchange date {self.initial_exchange_date.to_iso()}"
                )

        # TD must be after IED
        if self.initial_exchange_date and self.termination_date:
            if self.termination_date <= self.initial_exchange_date:
                raise ValueError(
                    f"Termination date {self.termination_date.to_iso()} "
                    f"must be > initial exchange date {self.initial_exchange_date.to_iso()}"
                )

        return self

    @model_validator(mode="after")
    def validate_array_schedules(self) -> ContractAttributes:
        """Validate array schedule consistency."""
        # Validate PR arrays
        pr_arrays = [
            self.array_pr_anchor,
            self.array_pr_cycle,
            self.array_pr_next,
            self.array_increase_decrease,
        ]
        pr_defined = [arr for arr in pr_arrays if arr is not None]
        if pr_defined:
            lengths = [len(arr) for arr in pr_defined]
            if len(set(lengths)) > 1:
                raise ValueError(f"PR array schedules must have same length, got {lengths}")

            # Validate ARINCDEC values
            if self.array_increase_decrease is not None:
                for val in self.array_increase_decrease:
                    if val not in ("INC", "DEC"):
                        raise ValueError(f"ARINCDEC values must be 'INC' or 'DEC', got '{val}'")

        # Validate IP arrays
        ip_arrays = [self.array_ip_anchor, self.array_ip_cycle]
        ip_defined = [arr for arr in ip_arrays if arr is not None]
        if ip_defined:
            lengths = [len(arr) for arr in ip_defined]
            if len(set(lengths)) > 1:
                raise ValueError(f"IP array schedules must have same length, got {lengths}")

        # Validate RR arrays
        rr_arrays = [
            self.array_rr_anchor,
            self.array_rr_cycle,
            self.array_rate,
            self.array_fixed_variable,
        ]
        rr_defined = [arr for arr in rr_arrays if arr is not None]
        if rr_defined:
            lengths = [len(arr) for arr in rr_defined]
            if len(set(lengths)) > 1:
                raise ValueError(f"RR array schedules must have same length, got {lengths}")

            # Validate ARFIXVAR values
            if self.array_fixed_variable is not None:
                for val in self.array_fixed_variable:
                    if val not in ("F", "V"):
                        raise ValueError(f"ARFIXVAR values must be 'F' or 'V', got '{val}'")

        return self

    def get_attribute(self, actus_name: str) -> Any:
        """Get attribute value by ACTUS short name.

        Args:
            actus_name: ACTUS short name (e.g., 'IPNR', 'NT', 'MD')

        Returns:
            Attribute value

        Raises:
            KeyError: If ACTUS name not recognized

        Example:
            >>> attrs.get_attribute('NT')
            100000.0
        """
        if actus_name not in ATTRIBUTE_MAP:
            raise KeyError(f"Unknown ACTUS attribute name: {actus_name}")

        python_name = ATTRIBUTE_MAP[actus_name]
        return getattr(self, python_name)

    def set_attribute(self, actus_name: str, value: Any) -> None:
        """Set attribute value by ACTUS short name.

        Args:
            actus_name: ACTUS short name (e.g., 'IPNR', 'NT', 'MD')
            value: New value (will be validated)

        Raises:
            KeyError: If ACTUS name not recognized
            ValidationError: If value is invalid

        Example:
            >>> attrs.set_attribute('NT', 150000.0)
        """
        if actus_name not in ATTRIBUTE_MAP:
            raise KeyError(f"Unknown ACTUS attribute name: {actus_name}")

        python_name = ATTRIBUTE_MAP[actus_name]
        setattr(self, python_name, value)

    def is_attribute_defined(self, actus_name: str) -> bool:
        """Check if an attribute has a non-None value.

        Args:
            actus_name: ACTUS short name

        Returns:
            True if attribute is defined (not None)

        Example:
            >>> attrs.is_attribute_defined('NT')
            True
        """
        try:
            value = self.get_attribute(actus_name)
            return value is not None
        except KeyError:
            return False


# Mapping from ACTUS short names to Python attribute names
ATTRIBUTE_MAP: dict[str, str] = {
    # Critical attributes
    "CT": "contract_type",
    "CNTRL": "contract_role",
    "SD": "status_date",
    "CDD": "contract_deal_date",
    "IED": "initial_exchange_date",
    "MD": "maturity_date",
    "PRD": "purchase_date",
    "TD": "termination_date",
    # Notional and rates
    "NT": "notional_principal",
    "IPNR": "nominal_interest_rate",
    "CUR": "currency",
    # Conventions
    "DCC": "day_count_convention",
    "BDC": "business_day_convention",
    "EOMC": "end_of_month_convention",
    "CLDR": "calendar",
    # Schedule attributes
    "IPCL": "interest_payment_cycle",
    "IPANX": "interest_payment_anchor",
    "IPCED": "interest_capitalization_end_date",
    "PRCL": "principal_redemption_cycle",
    "PRANX": "principal_redemption_anchor",
    "FECL": "fee_payment_cycle",
    "FEANX": "fee_payment_anchor",
    "RRCL": "rate_reset_cycle",
    "RRANX": "rate_reset_anchor",
    "SCCL": "scaling_index_cycle",
    "SCANX": "scaling_index_anchor",
    "PRNXT": "next_principal_redemption_amount",
    "IPCBCL": "interest_calculation_base_cycle",
    "IPCBANX": "interest_calculation_base_anchor",
    # Array schedules
    "ARPRANX": "array_pr_anchor",
    "ARPRCL": "array_pr_cycle",
    "ARPRNXT": "array_pr_next",
    "ARINCDEC": "array_increase_decrease",
    "ARIPANX": "array_ip_anchor",
    "ARIPCL": "array_ip_cycle",
    "ARRRANX": "array_rr_anchor",
    "ARRRCL": "array_rr_cycle",
    "ARRATE": "array_rate",
    "ARFIXVAR": "array_fixed_variable",
    # Rate reset
    "RRMO": "rate_reset_market_object",
    "RRMLT": "rate_reset_multiplier",
    "RRSP": "rate_reset_spread",
    "RRLF": "rate_reset_floor",
    "RRLC": "rate_reset_cap",
    "RRNXT": "rate_reset_next",
    # Fees
    "FER": "fee_rate",
    "FEB": "fee_basis",
    "FEAC": "fee_accrued",
    # Prepayment
    "PPEF": "prepayment_effect",
    "PYTP": "penalty_type",
    "PYRT": "penalty_rate",
    # Scaling
    "SCEF": "scaling_effect",
    "SCIXSD": "scaling_index_at_status_date",
    "SCMO": "scaling_market_object",
    # Credit enhancement
    "CECV": "coverage",
    "CET": "credit_event_type",
    "CEGE": "credit_enhancement_guarantee_extent",
    # Other
    "IPAC": "accrued_interest",
    "IPCB": "interest_calculation_base",
    "PRF": "contract_performance",
    "PDIED": "premium_discount_at_ied",
    "PPRD": "price_at_purchase_date",
    "PTD": "price_at_termination_date",
    # Options
    "OPTP": "option_type",
    "OPS1": "option_strike_1",
    "OPS2": "option_strike_2",
    "OPXT": "option_exercise_type",
    "OPXED": "option_exercise_end_date",
    "STPD": "settlement_period",
    "DVTP": "delivery_type",
    "CTST": "contract_structure",
    "PFUT": "future_price",
}
