"""Type definitions and enumerations for ACTUS contract standard.

This module defines all enumerations and type aliases used throughout JACTUS.
All enumerations inherit from str for JSON serializability and easy comparison.

References:
    ACTUS Technical Specification v1.1, Section 2 (Notations)
"""

from enum import Enum
from typing import TypeAlias

# Type aliases for clarity
Timestamp: TypeAlias = str  # ISO 8601 format: YYYY-MM-DDTHH:MM:SS
Amount: TypeAlias = float  # Monetary amount
Rate: TypeAlias = float  # Interest rate (decimal, e.g., 0.05 for 5%)
Percentage: TypeAlias = float  # Percentage (decimal, e.g., 0.05 for 5%)
Cycle: TypeAlias = str  # Format: NPS (e.g., '1Y', '3M', '1W-')


class EventType(str, Enum):
    """ACTUS contract event types.

    Event types define the nature of contract events that trigger
    state transitions and cash flows.

    References:
        ACTUS Technical Specification v1.1, Table 4
    """

    AD = "AD"  # Analysis Date
    IED = "IED"  # Initial Exchange Date
    MD = "MD"  # Maturity Date
    PR = "PR"  # Principal Redemption
    PI = "PI"  # Principal Increase
    PP = "PP"  # Principal Prepayment
    PY = "PY"  # Penalty Payment
    PRF = "PRF"  # Principal Redemption Amount Fixing
    FP = "FP"  # Fee Payment
    PRD = "PRD"  # Purchase
    TD = "TD"  # Termination
    IP = "IP"  # Interest Payment
    IPCI = "IPCI"  # Interest Capitalization
    IPCB = "IPCB"  # Interest Calculation Base Fixing
    RR = "RR"  # Rate Reset
    RRF = "RRF"  # Rate Reset Fixing
    DV = "DV"  # Dividend Payment
    DVF = "DVF"  # Dividend Fixing
    SC = "SC"  # Scaling Index Fixing
    STD = "STD"  # Settlement
    XD = "XD"  # Exercise Date
    CE = "CE"  # Credit Event
    IPFX = "IPFX"  # Interest Payment Fixed Leg (swaps)
    IPFL = "IPFL"  # Interest Payment Floating Leg (swaps)

    @property
    def index(self) -> int:
        """Integer index for this event type (for dispatch tables and future jax.lax.switch)."""
        return _EVENT_TYPE_INDEX[self]


# Stable integer mapping for EventType → int (used by dispatch tables).
# Order matches the enum definition. Adding new event types must append to the end.
_EVENT_TYPE_INDEX: dict["EventType", int] = {
    member: i for i, member in enumerate(EventType)
}

# Total number of event types (used to size dispatch tables)
NUM_EVENT_TYPES: int = len(_EVENT_TYPE_INDEX)

# ACTUS event scheduling priority: determines processing order when multiple
# events fall on the same date. Lower number = processed first.
# Reference: ACTUS v1.1 event sequence conventions.
EVENT_SCHEDULE_PRIORITY: dict["EventType", int] = {
    EventType.AD: 0,
    EventType.IED: 1,
    EventType.PR: 4,
    EventType.PI: 5,
    EventType.PP: 6,
    EventType.PY: 7,
    EventType.IP: 8,
    EventType.IPCI: 9,
    EventType.RR: 10,
    EventType.RRF: 11,
    EventType.IPCB: 12,
    EventType.SC: 13,
    EventType.PRF: 14,
    EventType.FP: 15,
    EventType.PRD: 16,
    EventType.TD: 17,
    EventType.MD: 18,
    EventType.STD: 19,
    EventType.XD: 20,
    EventType.DV: 21,
    EventType.DVF: 22,
    EventType.CE: 23,
    EventType.IPFX: 24,
    EventType.IPFL: 25,
}


class ContractType(str, Enum):
    """ACTUS contract types.

    Defines all standardized contract types in the ACTUS taxonomy.

    References:
        ACTUS Technical Specification v1.1, Table 3
    """

    PAM = "PAM"  # Principal At Maturity
    LAM = "LAM"  # Linear Amortizer
    LAX = "LAX"  # Exotic Linear Amortizer
    NAM = "NAM"  # Negative Amortizer
    ANN = "ANN"  # Annuity
    CLM = "CLM"  # Call Money
    UMP = "UMP"  # Undefined Maturity Profile
    CSH = "CSH"  # Cash
    STK = "STK"  # Stock
    COM = "COM"  # Commodity
    FXOUT = "FXOUT"  # Foreign Exchange Outright
    SWPPV = "SWPPV"  # Plain Vanilla Interest Rate Swap
    SWAPS = "SWAPS"  # Swap
    CAPFL = "CAPFL"  # Cap Floor
    OPTNS = "OPTNS"  # Option
    FUTUR = "FUTUR"  # Future
    CEG = "CEG"  # Credit Enhancement Guarantee
    CEC = "CEC"  # Credit Enhancement Collateral


class ContractRole(str, Enum):
    """Contract party role definition.

    Defines the role of the contract creator, which determines
    the sign of cash flows (+1 for asset positions, -1 for liability).

    References:
        ACTUS Technical Specification v1.1, Table 1
    """

    RPA = "RPA"  # Real Position Asset
    RPL = "RPL"  # Real Position Liability
    LG = "LG"  # Long Position
    ST = "ST"  # Short Position
    BUY = "BUY"  # Protection Buyer
    SEL = "SEL"  # Protection Seller
    RFL = "RFL"  # Receive First Leg
    PFL = "PFL"  # Pay First Leg
    COL = "COL"  # Collateral Instrument
    CNO = "CNO"  # Close-out Netting Instrument
    GUA = "GUA"  # Guarantor
    OBL = "OBL"  # Obligee
    UDL = "UDL"  # Underlying
    UDLP = "UDLP"  # Underlying Positive
    UDLM = "UDLM"  # Underlying Negative

    def get_sign(self) -> int:
        """Get the sign convention for this role.

        Returns +1 for asset/long/receive positions,
        -1 for liability/short/pay positions.

        Returns:
            +1 or -1 according to ACTUS Table 1

        Example:
            >>> ContractRole.RPA.get_sign()
            1
            >>> ContractRole.RPL.get_sign()
            -1

        References:
            ACTUS Technical Specification v1.1, Table 1, Section 3.7
        """
        role_signs = {
            ContractRole.RPA: 1,
            ContractRole.RPL: -1,
            ContractRole.LG: 1,
            ContractRole.ST: -1,
            ContractRole.BUY: 1,
            ContractRole.SEL: -1,
            ContractRole.RFL: 1,
            ContractRole.PFL: -1,
            ContractRole.COL: 1,
            ContractRole.CNO: 1,
            ContractRole.GUA: -1,
            ContractRole.OBL: 1,
            ContractRole.UDL: 1,
            ContractRole.UDLP: 1,
            ContractRole.UDLM: -1,
        }
        return role_signs[self]


class DayCountConvention(str, Enum):
    """Day count conventions for year fraction calculation.

    Determines how time periods are calculated for interest accrual.

    References:
        ACTUS Technical Specification v1.1, Section 3.6
        ISDA Definitions
    """

    AA = "AA"  # Actual/Actual ISDA
    A360 = "A360"  # Actual/360
    A365 = "A365"  # Actual/365
    E30360ISDA = "30E360ISDA"  # 30E/360 ISDA
    E30360 = "30E360"  # 30E/360
    B30360 = "30360"  # 30/360 US (Bond Basis)
    BUS252 = "BUS252"  # Business/252


class BusinessDayConvention(str, Enum):
    """Business day adjustment conventions.

    Defines how dates are adjusted when they fall on non-business days.
    Convention format: S/C + Direction + Modified
    - S = Shift, C = Calculate
    - F = Following, P = Preceding
    - M prefix = Modified (don't cross month boundary)

    References:
        ACTUS Technical Specification v1.1, Section 3.4
    """

    NULL = "NULL"  # No adjustment
    SCF = "SCF"  # Shift/Calculate Following
    SCMF = "SCMF"  # Shift/Calculate Modified Following
    CSF = "CSF"  # Calculate/Shift Following
    CSMF = "CSMF"  # Calculate/Shift Modified Following
    SCP = "SCP"  # Shift/Calculate Preceding
    SCMP = "SCMP"  # Shift/Calculate Modified Preceding
    CSP = "CSP"  # Calculate/Shift Preceding
    CSMP = "CSMP"  # Calculate/Shift Modified Preceding


class EndOfMonthConvention(str, Enum):
    """End of month adjustment convention.

    Determines whether dates stay at month-end or maintain day number.

    References:
        ACTUS Technical Specification v1.1, Section 3.3
    """

    EOM = "EOM"  # End of Month - move to last day of month
    SD = "SD"  # Same Day - keep same day number (default)


class Calendar(str, Enum):
    """Business day calendar definitions.

    Defines which days are considered business days (non-holidays).

    Note:
        Additional calendars (TARGET, NYSE, etc.) can be added as needed.
    """

    NO_CALENDAR = "NO_CALENDAR"  # No holidays (all days are business days)
    MONDAY_TO_FRIDAY = "MONDAY_TO_FRIDAY"  # Weekends only
    TARGET = "TARGET"  # European Central Bank calendar
    US_NYSE = "US_NYSE"  # New York Stock Exchange
    UK_SETTLEMENT = "UK_SETTLEMENT"  # UK settlement calendar
    CUSTOM = "CUSTOM"  # User-defined calendar


class ContractPerformance(str, Enum):
    """Contract performance status.

    Indicates the payment performance status of the contract.

    References:
        ACTUS Technical Specification v1.1
    """

    PF = "PF"  # Performant - payments on time
    DL = "DL"  # Delayed - minor delays
    DQ = "DQ"  # Delinquent - significant delays
    DF = "DF"  # Default - major payment failure


class FeeBasis(str, Enum):
    """Fee calculation basis.

    Determines whether fees are absolute amounts or percentages of notional.
    """

    A = "A"  # Absolute amount
    N = "N"  # Notional percentage


class InterestCalculationBase(str, Enum):
    """Base for interest calculation.

    Defines which notional value to use for interest calculations.

    References:
        ACTUS Technical Specification v1.1
    """

    NT = "NT"  # Notional principal (current)
    NTIED = "NTIED"  # Notional at IED (initial)
    NTL = "NTL"  # Notional lagged (previous period)


class CyclePointOfInterestPayment(str, Enum):
    """When IP is calculated/paid in cycle.

    Determines whether interest is paid at the beginning or end of each period.
    """

    B = "B"  # Beginning of cycle
    E = "E"  # End of cycle (default)


class PrepaymentEffect(str, Enum):
    """Effect of prepayment on schedule.

    Determines how unscheduled principal payments affect the contract.
    """

    N = "N"  # No effect on schedule
    A = "A"  # Adjust maturity date (shorten contract)
    M = "M"  # Adjust next principal redemption amount


class ScalingEffect(str, Enum):
    """Scaling index effect on contract.

    Three-character code indicating which contract elements are scaled:
    - Position 1: Interest (I = scale, 0 = no scale)
    - Position 2: Notional (N = scale, 0 = no scale)
    - Position 3: Maturity (M = scale, 0 = no scale)

    Example:
        'IN0' = Scale Interest and Notional, but not Maturity

    References:
        ACTUS Technical Specification v1.1
    """

    S000 = "000"  # No scaling
    I00 = "I00"  # Interest scaling only
    S0N0 = "0N0"  # Notional scaling only
    IN0 = "IN0"  # Interest and notional scaling
    S00M = "00M"  # Maturity scaling only
    I0M = "I0M"  # Interest and maturity scaling
    S0NM = "0NM"  # Notional and maturity scaling
    INM = "INM"  # All scaling (Interest, Notional, Maturity)
