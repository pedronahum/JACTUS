# PAM Contract: A Complete Architecture Walkthrough

**Author**: JACTUS Development Team
**Date**: 2025-10-20
**Purpose**: Guide developers through the JACTUS architecture by explaining all building blocks required to implement a PAM (Principal at Maturity) contract.

---

## Table of Contents

1. [Introduction](#introduction)
2. [What is PAM?](#what-is-pam)
3. [Architecture Overview](#architecture-overview)
4. [Building Blocks](#building-blocks)
5. [Step-by-Step Implementation](#step-by-step-implementation)
6. [Testing Strategy](#testing-strategy)
7. [Advanced Topics](#advanced-topics)
8. [Summary](#summary)

---

## Introduction

This document provides a **complete walkthrough** of the JACTUS architecture by exploring how the PAM (Principal at Maturity) contract type is implemented. By following this guide, you'll understand:

- How JACTUS organizes contract types
- The role of each architectural layer
- How JAX enables performance and differentiation
- How to extend JACTUS with new contract types
- Testing strategies for financial contracts

**Target Audience**: Developers who want to understand JACTUS internals, contribute to the project, or implement custom contract types.

**Prerequisites**: Basic understanding of:
- Python and object-oriented programming
- JAX (helpful but not required)
- Financial contracts (loans, bonds)
- ACTUS standard (helpful but not required)

---

## What is PAM?

### ACTUS Definition

PAM (Principal at Maturity) is an **ACTUS contract type** representing loans or bonds where:
- **Principal** is disbursed at the Initial Exchange Date (IED)
- **Interest** accrues continuously according to a day count convention
- **Interest payments** are made periodically (monthly, quarterly, annually, etc.)
- **Principal** is repaid in full at Maturity Date (MD)

### Real-World Examples

- **Bullet loans**: Corporate loans with interest-only payments
- **Interest-only mortgages**: During the interest-only period
- **Zero-coupon bonds**: When configured without interest payments
- **Term loans**: Fixed-term business loans

### PAM Characteristics

| Characteristic | Description |
|----------------|-------------|
| **Complexity** | Medium (more complex than CSH, simpler than ANN) |
| **Events** | IED, IP (periodic), MD, plus optional PRD, TD, PP, PY, FP, RR, RRF, SC |
| **States** | `nt`, `ipnr`, `ipac`, `feac`, `nsc`, `isc`, `sd`, `tmd` |
| **Risk Factors** | Interest rates, FX rates (if multi-currency) |

---

## Architecture Overview

JACTUS follows a **layered architecture** inspired by the ACTUS standard:

```
┌─────────────────────────────────────────────────────────────┐
│                     USER APPLICATION                         │
│  (Create contracts, run simulations, analyze results)       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   CONTRACT LAYER (Phase 3)                   │
│  • PrincipalAtMaturityContract                              │
│  • CashContract, StockContract, CommodityContract           │
│  • Factory pattern: create_contract()                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    ENGINE LAYER (Phase 2)                    │
│  • SimulationEngine: Orchestrates event generation          │
│  • LifecycleEngine: Applies POF/STF to generate events      │
│  • BaseContract: Abstract contract interface                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   FUNCTION LAYER (Phase 2)                   │
│  • PayoffFunction: POF_XX(state, attrs, time, rf_obs)       │
│  • StateTransitionFunction: STF_XX(state, attrs, time)       │
│  • Function composition and event mapping                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 CORE TYPES LAYER (Phase 1)                   │
│  • ContractState: JAX arrays (nt, ipnr, ipac, etc.)         │
│  • ContractAttributes: Pydantic model                       │
│  • ActusDateTime: Date/time handling                        │
│  • EventType, ContractType, DayCountConvention (enums)      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   UTILITIES LAYER (Phase 1)                  │
│  • Schedules: generate_schedule(), generate_array_schedule()│
│  • Conventions: year_fraction(), adjust_to_business_day()   │
│  • Math: discount_factor(), present_value()                 │
│  • Calendars: is_business_day(), next_business_day()        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     JAX FOUNDATION                           │
│  • Arrays: jax.numpy (jnp.array, jnp.float32)               │
│  • Transformations: jax.jit, jax.vmap, jax.grad             │
│  • Performance: XLA compilation, GPU/TPU support            │
└─────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Separation of Concerns**: Each layer has a clear responsibility
2. **JAX-First**: All numerical operations use JAX for performance
3. **ACTUS Compliance**: Follows ACTUS specifications for contract logic
4. **Extensibility**: Easy to add new contract types
5. **Type Safety**: Extensive use of type hints and Pydantic validation

---

## Building Blocks

Let's explore each building block needed to implement PAM, starting from the bottom up.

### 1. Core Types (`src/jactus/core/types.py`)

All enums and type definitions used throughout JACTUS.

```python
# Contract type enumeration
class ContractType(str, Enum):
    PAM = "PAM"  # Principal at Maturity
    CSH = "CSH"  # Cash
    STK = "STK"  # Stock
    COM = "COM"  # Commodity
    # ... more types

# Event type enumeration
class EventType(str, Enum):
    IED = "IED"  # Initial Exchange
    IP = "IP"    # Interest Payment
    MD = "MD"    # Maturity
    # ... 30+ more event types

# Day count conventions
class DayCountConvention(str, Enum):
    A360 = "A/360"  # Actual/360
    AA = "A/A"      # Actual/Actual
    # ... more conventions

# Contract role (perspective)
class ContractRole(str, Enum):
    RPA = "RPA"  # Real Position Asset (Paying/Borrower)
    RPL = "RPL"  # Real Position Liability (Receiving/Lender)
```

**Purpose**: Provide type-safe enumerations for all contract parameters.

**Why it matters**: Type safety prevents bugs like using "PAM-001" when `ContractType.PAM` is expected.

---

### 2. ActusDateTime (`src/jactus/core/time.py`)

JAX-compatible date/time handling for all contract dates.

```python
@dataclass(frozen=True)
class ActusDateTime:
    """Immutable datetime representation compatible with JAX."""
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int

    def to_iso(self) -> str:
        """Convert to ISO 8601 string."""
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}T{self.hour:02d}:{self.minute:02d}:{self.second:02d}"

    def days_between(self, other: "ActusDateTime") -> int:
        """Compute days between two dates."""
        # Uses Gregorian calendar calculations
        ...

    def add_period(self, period: str) -> "ActusDateTime":
        """Add a period like '3M' (3 months) or '1Y' (1 year)."""
        ...
```

**Why not `datetime.datetime`?**:
- JAX requires immutable, hashable types for JIT compilation
- `ActusDateTime` is a frozen dataclass, making it JAX-compatible
- Provides ACTUS-specific operations like period arithmetic

**Usage in PAM**:
```python
ied = ActusDateTime(2024, 1, 15, 0, 0, 0)
md = ActusDateTime(2029, 1, 15, 0, 0, 0)
```

---

### 3. ContractState (`src/jactus/core/states.py`)

JAX-based state representation using Flax NNX Module.

```python
class ContractState(nnx.Module):
    """Contract state using JAX arrays."""

    # Required fields (present in all states)
    sd: ActusDateTime      # Status Date
    tmd: ActusDateTime     # Maturity Date

    # Financial state variables (JAX arrays)
    nt: jnp.ndarray       # Notional Principal
    ipnr: jnp.ndarray     # Nominal Interest Rate
    ipac: jnp.ndarray     # Interest Payment Accrued
    feac: jnp.ndarray     # Fee Accrued
    nsc: jnp.ndarray      # Notional Scaling Multiplier
    isc: jnp.ndarray      # Interest Scaling Multiplier

    def __init__(
        self,
        sd: ActusDateTime,
        tmd: ActusDateTime,
        nt: jnp.ndarray = jnp.array(0.0, dtype=jnp.float32),
        ipnr: jnp.ndarray = jnp.array(0.0, dtype=jnp.float32),
        # ... other fields
    ):
        self.sd = sd
        self.tmd = tmd
        self.nt = nt
        # ...
```

**Why JAX arrays?**:
- **Performance**: JAX can JIT compile operations on arrays
- **Differentiation**: Can compute sensitivities (Greeks) via `jax.grad`
- **Vectorization**: Can process multiple states in parallel with `jax.vmap`

**Why Flax NNX?**:
- Provides a clean Pytree structure that JAX can transform
- Integrates well with JAX's functional programming paradigm
- Enables state to be passed through JIT-compiled functions

**PAM State Example**:
```python
state = ContractState(
    sd=ActusDateTime(2024, 1, 15, 0, 0, 0),
    tmd=ActusDateTime(2029, 1, 15, 0, 0, 0),
    nt=jnp.array(100000.0, dtype=jnp.float32),  # $100k principal
    ipnr=jnp.array(0.05, dtype=jnp.float32),    # 5% interest rate
    ipac=jnp.array(0.0, dtype=jnp.float32),     # No accrued interest yet
    nsc=jnp.array(1.0, dtype=jnp.float32),      # No scaling
    isc=jnp.array(1.0, dtype=jnp.float32),      # No scaling
)
```

---

### 4. ContractAttributes (`src/jactus/core/attributes.py`)

Pydantic model for contract terms and parameters.

```python
class ContractAttributes(BaseModel):
    """All possible attributes for ACTUS contracts."""

    # Identification
    contract_id: str
    contract_type: ContractType
    contract_role: ContractRole

    # Dates
    status_date: ActusDateTime
    initial_exchange_date: Optional[ActusDateTime] = None
    maturity_date: Optional[ActusDateTime] = None

    # Financial terms
    currency: str
    notional_principal: Optional[float] = None
    nominal_interest_rate: Optional[float] = None

    # Schedules and conventions
    day_count_convention: Optional[DayCountConvention] = None
    interest_payment_cycle: Optional[str] = None  # e.g., "3M", "1Y"

    # ... 50+ more optional fields for advanced features

    class Config:
        frozen = True  # Immutable
        arbitrary_types_allowed = True  # Allow ActusDateTime
```

**Why Pydantic?**:
- **Validation**: Ensures all required fields are present
- **Type checking**: Converts and validates field types
- **Serialization**: Easy JSON export/import
- **Documentation**: Self-documenting with field descriptions

**PAM Attributes Example**:
```python
attrs = ContractAttributes(
    contract_id="PAM-001",
    contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
    maturity_date=ActusDateTime(2029, 1, 15, 0, 0, 0),
    currency="USD",
    notional_principal=100000.0,
    nominal_interest_rate=0.05,
    day_count_convention=DayCountConvention.A360,
    interest_payment_cycle="3M",
)
```

---

### 5. Risk Factor Observers (`src/jactus/observers/`)

Provide market data to contracts during simulation.

```python
class RiskFactorObserverProtocol(Protocol):
    """Protocol for risk factor observers."""

    def get(self, index: int) -> jnp.ndarray:
        """Get risk factor at index."""
        ...

class ConstantRiskFactorObserver:
    """Simple observer returning constant value."""

    def __init__(self, constant_value: float):
        self.constant_value = jnp.array(constant_value, dtype=jnp.float32)

    def get(self, index: int) -> jnp.ndarray:
        return self.constant_value

class JaxRiskFactorObserver:
    """JAX-based observer with array of risk factors."""

    def __init__(self, risk_factors: jnp.ndarray):
        self.risk_factors = risk_factors

    def get(self, index: int) -> jnp.ndarray:
        return self.risk_factors[index]
```

**Why observers?**:
- **Separation of concerns**: Contracts don't know where data comes from
- **Testability**: Easy to mock market data for testing
- **Flexibility**: Can plug in different data sources (historical, Monte Carlo, real-time)

**PAM Usage**:
```python
# For fixed-rate loan
rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

# For variable-rate loan with scenarios
rates = jnp.array([0.03, 0.05, 0.07])
rf_obs = JaxRiskFactorObserver(rates)
```

---

### 6. Utility Functions (`src/jactus/utilities/`)

Helper functions for common operations.

#### Schedule Generation (`schedules.py`)

```python
def generate_schedule(
    start: ActusDateTime,
    cycle: str,
    end: ActusDateTime,
    calendar: Calendar = Calendar.NO_CALENDAR,
    end_of_month_convention: EndOfMonthConvention = EndOfMonthConvention.SAME_DAY,
) -> List[ActusDateTime]:
    """Generate schedule of dates from start to end with given cycle.

    Args:
        start: First date in schedule
        cycle: Period between dates (e.g., "1M", "3M", "1Y")
        end: Last date in schedule (inclusive)
        calendar: Business day calendar
        end_of_month_convention: How to handle month-end dates

    Returns:
        List of ActusDateTime objects
    """
    ...
```

**PAM Usage**: Generate interest payment dates
```python
ip_schedule = generate_schedule(
    start=ied,
    cycle="3M",  # Quarterly
    end=md,
)
```

#### Year Fraction Calculation (`conventions.py`)

```python
def year_fraction(
    start: ActusDateTime,
    end: ActusDateTime,
    convention: DayCountConvention,
) -> float:
    """Calculate year fraction between two dates using day count convention.

    This is critical for interest calculation!

    Examples:
        - A/360: days / 360
        - A/A: actual days / actual days in year
        - 30/360: assumes 30-day months
    """
    ...
```

**PAM Usage**: Calculate interest for each period
```python
yf = year_fraction(last_ip_date, current_ip_date, DayCountConvention.A360)
interest = notional * rate * yf
```

---

### 7. Payoff Functions (`src/jactus/functions/payoff.py`)

Calculate cashflows for each event type.

```python
class PayoffFunction(nnx.Module):
    """Base class for payoff calculations."""

    contract_role: ContractRole
    currency: str
    settlement_currency: Optional[str]

    def calculate_payoff(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserverProtocol,
    ) -> jnp.ndarray:
        """Calculate payoff for given event.

        Returns:
            JAX array containing the payoff amount
        """
        raise NotImplementedError
```

**PAM Payoff Function** (`src/jactus/contracts/pam.py`):

```python
class PAMPayoffFunction(PayoffFunction):
    """Payoff function for PAM contracts."""

    def calculate_payoff(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserverProtocol,
    ) -> jnp.ndarray:
        """Calculate PAM payoff."""

        if event_type == EventType.IED:
            # Initial Exchange: Disburse principal
            return self._pof_ied(state, attributes)

        elif event_type == EventType.IP:
            # Interest Payment: Pay accrued interest
            return self._pof_ip(state, attributes)

        elif event_type == EventType.MD:
            # Maturity: Repay principal + final interest
            return self._pof_md(state, attributes)

        # ... more event types

    def _pof_ied(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_IED: Principal disbursement."""
        role_sign = contract_role_sign(self.contract_role)
        return role_sign * state.nsc * attrs.notional_principal

    def _pof_ip(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_IP: Interest payment."""
        role_sign = contract_role_sign(self.contract_role)
        return role_sign * (state.isc * state.ipac + state.feac)

    def _pof_md(self, state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
        """POF_MD: Principal repayment + final interest."""
        role_sign = contract_role_sign(self.contract_role)
        return role_sign * (state.nsc * state.nt + state.isc * state.ipac + state.feac)
```

**Key concepts**:
- **Role sign**: Reverses cashflow direction based on contract role (borrower vs lender)
- **Scaling multipliers**: `nsc` and `isc` allow for contract adjustments
- **JAX arrays**: All calculations return JAX arrays for performance

---

### 8. State Transition Functions (`src/jactus/functions/state.py`)

Update contract state after each event.

```python
class StateTransitionFunction(nnx.Module):
    """Base class for state transitions."""

    def transition_state(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
    ) -> ContractState:
        """Apply state transition for given event.

        Returns:
            New ContractState (states are immutable)
        """
        raise NotImplementedError
```

**PAM State Transition Function** (`src/jactus/contracts/pam.py`):

```python
class PAMStateTransitionFunction(StateTransitionFunction):
    """State transition for PAM contracts."""

    def transition_state(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
    ) -> ContractState:
        """Apply PAM state transition."""

        if event_type == EventType.IED:
            return self._stf_ied(state, attributes, time)

        elif event_type == EventType.IP:
            return self._stf_ip(state, attributes, time)

        elif event_type == EventType.MD:
            return self._stf_md(state, attributes, time)

        # ... more event types

    def _stf_ied(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> ContractState:
        """STF_IED: Initialize state with loan terms."""
        return ContractState(
            sd=time,
            tmd=attrs.maturity_date,
            nt=jnp.array(attrs.notional_principal, dtype=jnp.float32),
            ipnr=jnp.array(attrs.nominal_interest_rate or 0.0, dtype=jnp.float32),
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
        )

    def _stf_ip(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> ContractState:
        """STF_IP: Reset accrued interest to zero after payment."""
        return ContractState(
            sd=time,
            tmd=state.tmd,
            nt=state.nt,
            ipnr=state.ipnr,
            ipac=jnp.array(0.0, dtype=jnp.float32),  # Reset!
            feac=jnp.array(0.0, dtype=jnp.float32),  # Reset!
            nsc=state.nsc,
            isc=state.isc,
        )

    def _stf_md(
        self, state: ContractState, attrs: ContractAttributes, time: ActusDateTime
    ) -> ContractState:
        """STF_MD: Zero out all state variables at maturity."""
        return ContractState(
            sd=time,
            tmd=state.tmd,
            nt=jnp.array(0.0, dtype=jnp.float32),  # Repaid!
            ipnr=state.ipnr,
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=state.nsc,
            isc=state.isc,
        )
```

**Key concepts**:
- **Immutability**: States are never modified, always create new ones
- **Interest accrual**: Between IP events, interest accumulates in `ipac`
- **State lifecycle**: IED → IP (repeated) → MD

---

### 9. BaseContract (`src/jactus/contracts/base.py`)

Abstract contract interface that all contracts implement.

```python
class BaseContract(nnx.Module):
    """Base contract providing simulation framework."""

    attributes: ContractAttributes
    risk_factor_observer: RiskFactorObserverProtocol

    def generate_event_schedule(self) -> List[ContractEvent]:
        """Generate all future events for this contract."""
        raise NotImplementedError

    def initialize_state(self) -> ContractState:
        """Create initial contract state."""
        raise NotImplementedError

    def get_payoff_function(self) -> PayoffFunction:
        """Get payoff function for this contract type."""
        raise NotImplementedError

    def get_state_transition_function(self) -> StateTransitionFunction:
        """Get state transition function for this contract type."""
        raise NotImplementedError

    def simulate(self) -> SimulationResult:
        """Run full contract simulation."""
        # Uses LifecycleEngine internally
        ...
```

**Why abstract?**: Each contract type implements these methods differently.

---

### 10. PrincipalAtMaturityContract (`src/jactus/contracts/pam.py`)

The PAM contract implementation!

```python
class PrincipalAtMaturityContract(BaseContract):
    """PAM contract implementation."""

    def generate_event_schedule(self) -> List[ContractEvent]:
        """Generate PAM event schedule: IED, IP events, MD."""

        events = []

        # 1. Initial Exchange Date
        if self.attributes.initial_exchange_date:
            events.append(
                ContractEvent(
                    event_type=EventType.IED,
                    event_time=self.attributes.initial_exchange_date,
                    sequence=len(events),
                )
            )

        # 2. Interest Payment dates
        if self.attributes.interest_payment_cycle:
            ip_schedule = generate_schedule(
                start=self.attributes.initial_exchange_date,
                cycle=self.attributes.interest_payment_cycle,
                end=self.attributes.maturity_date,
            )

            for ip_date in ip_schedule:
                if ip_date != self.attributes.maturity_date:
                    events.append(
                        ContractEvent(
                            event_type=EventType.IP,
                            event_time=ip_date,
                            sequence=len(events),
                        )
                    )

        # 3. Maturity Date
        if self.attributes.maturity_date:
            events.append(
                ContractEvent(
                    event_type=EventType.MD,
                    event_time=self.attributes.maturity_date,
                    sequence=len(events),
                )
            )

        return events

    def initialize_state(self) -> ContractState:
        """Create initial state before IED."""
        return ContractState(
            sd=self.attributes.status_date,
            tmd=self.attributes.maturity_date,
            nt=jnp.array(0.0, dtype=jnp.float32),  # No principal yet
            ipnr=jnp.array(0.0, dtype=jnp.float32),  # Rate set at IED
            ipac=jnp.array(0.0, dtype=jnp.float32),
            feac=jnp.array(0.0, dtype=jnp.float32),
            nsc=jnp.array(1.0, dtype=jnp.float32),
            isc=jnp.array(1.0, dtype=jnp.float32),
        )

    def get_payoff_function(self) -> PayoffFunction:
        """Return PAM payoff function."""
        return PAMPayoffFunction(
            contract_role=self.attributes.contract_role,
            currency=self.attributes.currency,
            settlement_currency=self.attributes.settlement_currency,
        )

    def get_state_transition_function(self) -> StateTransitionFunction:
        """Return PAM state transition function."""
        return PAMStateTransitionFunction()
```

**That's it!** The contract delegates to POF/STF for all the complex logic.

---

### 11. Factory Pattern (`src/jactus/contracts/__init__.py`)

Simplifies contract creation.

```python
def create_contract(
    attributes: ContractAttributes,
    risk_factor_observer: RiskFactorObserverProtocol,
) -> BaseContract:
    """Factory function to create contracts."""

    contract_map = {
        ContractType.PAM: PrincipalAtMaturityContract,
        ContractType.CSH: CashContract,
        ContractType.STK: StockContract,
        ContractType.COM: CommodityContract,
    }

    contract_class = contract_map.get(attributes.contract_type)
    if not contract_class:
        raise ValueError(f"Unsupported contract type: {attributes.contract_type}")

    return contract_class(
        attributes=attributes,
        risk_factor_observer=risk_factor_observer,
    )
```

**Usage**:
```python
attrs = ContractAttributes(contract_type=ContractType.PAM, ...)
contract = create_contract(attrs, rf_obs)  # Returns PrincipalAtMaturityContract
```

---

## Step-by-Step Implementation

Let's trace a complete PAM simulation from creation to result.

### Step 1: Create Contract

```python
from jactus.contracts import create_contract
from jactus.core import *
from jactus.observers import ConstantRiskFactorObserver

# Define contract terms
attrs = ContractAttributes(
    contract_id="PAM-001",
    contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,
    status_date=ActusDateTime(2024, 1, 1, 0, 0, 0),
    initial_exchange_date=ActusDateTime(2024, 1, 15, 0, 0, 0),
    maturity_date=ActusDateTime(2025, 1, 15, 0, 0, 0),
    currency="USD",
    notional_principal=100000.0,
    nominal_interest_rate=0.05,
    day_count_convention=DayCountConvention.A360,
    interest_payment_cycle="6M",
)

# Create risk factor observer
rf_obs = ConstantRiskFactorObserver(constant_value=0.05)

# Create contract via factory
contract = create_contract(attrs, rf_obs)
# Returns: PrincipalAtMaturityContract instance
```

### Step 2: Generate Event Schedule

```python
# Generate schedule (called internally by simulate())
events = contract.generate_event_schedule()

# Events:
# - IED: 2024-01-15
# - IP:  2024-07-15 (6 months later)
# - MD:  2025-01-15 (combines IP + principal repayment)
```

### Step 3: Initialize State

```python
# Initialize state (called internally by simulate())
state = contract.initialize_state()

# State:
# - sd: 2024-01-01
# - tmd: 2025-01-15
# - nt: 0.0 (principal not yet disbursed)
# - ipnr: 0.0 (rate not yet set)
# - ipac: 0.0 (no accrued interest)
```

### Step 4: Simulate Contract

```python
# Run simulation
result = contract.simulate()

# Internally:
# 1. Get event schedule
# 2. Initialize state
# 3. For each event:
#    a. Apply POF to compute cashflow
#    b. Apply STF to update state
#    c. Store event with payoff and states
```

### Step 5: Lifecycle for Each Event

#### Event 1: IED (2024-01-15)

```python
# Input:
#   event_type: IED
#   state: ContractState(nt=0, ipnr=0, ...)
#   time: 2024-01-15

# 1. Apply POF_IED
payoff = pof.calculate_payoff(EventType.IED, state, attrs, time, rf_obs)
# payoff = -100000.0 (borrower receives principal, negative from their perspective)

# 2. Apply STF_IED
new_state = stf.transition_state(EventType.IED, state, attrs, time)
# new_state: nt=100000, ipnr=0.05, ipac=0, sd=2024-01-15

# 3. Store event
event = Event(
    event_type=IED,
    event_time=2024-01-15,
    payoff=-100000.0,
    state_pre=old_state,
    state_post=new_state,
)
```

#### Event 2: IP (2024-07-15)

```python
# Input:
#   event_type: IP
#   state: ContractState(nt=100000, ipnr=0.05, ipac=?, ...)
#   time: 2024-07-15

# First, accrue interest from last event (IED) to now
# This happens in an IPCI (Interest Payment Calculation) event
# between IED and IP

# Days: 2024-01-15 to 2024-07-15 = 182 days
# Year fraction: 182/360 = 0.505556
# Interest: 100000 * 0.05 * 0.505556 = 2527.78

# State before IP: ipac = 2527.78

# 1. Apply POF_IP
payoff = pof.calculate_payoff(EventType.IP, state, attrs, time, rf_obs)
# payoff = 2527.78 (borrower pays interest)

# 2. Apply STF_IP
new_state = stf.transition_state(EventType.IP, state, attrs, time)
# new_state: nt=100000, ipnr=0.05, ipac=0 (reset!), sd=2024-07-15
```

#### Event 3: MD (2025-01-15)

```python
# Input:
#   event_type: MD
#   state: ContractState(nt=100000, ipnr=0.05, ipac=?, ...)
#   time: 2025-01-15

# Accrue interest from last IP (2024-07-15) to MD (2025-01-15)
# Days: 184 days
# Year fraction: 184/360 = 0.511111
# Interest: 100000 * 0.05 * 0.511111 = 2555.56

# State before MD: ipac = 2555.56

# 1. Apply POF_MD
payoff = pof.calculate_payoff(EventType.MD, state, attrs, time, rf_obs)
# payoff = 100000 + 2555.56 = 102555.56 (principal + final interest)

# 2. Apply STF_MD
new_state = stf.transition_state(EventType.MD, state, attrs, time)
# new_state: nt=0 (repaid!), ipac=0, sd=2025-01-15
```

### Step 6: Analyze Results

```python
# result.events contains all events with:
# - event_type, event_time, sequence
# - payoff (cashflow)
# - state_pre, state_post

# Get cashflows
cashflows = result.get_cashflows()
# [(2024-01-15, -100000.0, 'USD'),
#  (2024-07-15, 2527.78, 'USD'),
#  (2025-01-15, 102555.56, 'USD')]

# Analyze
total_interest = sum(cf[1] for cf in cashflows if cf[1] > 0 and cf[1] < 50000)
# total_interest = 5083.34
```

---

## Testing Strategy

JACTUS uses a comprehensive testing strategy:

### 1. Unit Tests (`tests/unit/contracts/test_pam.py`)

Test individual components:

```python
def test_pof_ied_calculates_principal_disbursement():
    """Test POF_IED returns negative notional (from borrower perspective)."""
    # Arrange
    state = ContractState(nt=0, ...)
    attrs = ContractAttributes(notional_principal=100000, ...)
    pof = PAMPayoffFunction(contract_role=ContractRole.RPA, ...)

    # Act
    payoff = pof.calculate_payoff(EventType.IED, state, attrs, time, rf_obs)

    # Assert
    assert float(payoff) == -100000.0
```

### 2. Integration Tests (`tests/integration/test_schedule_generation.py`)

Test end-to-end workflows:

```python
def test_pam_end_to_end_simulation():
    """Test complete PAM simulation from creation to results."""
    attrs = ContractAttributes(...)
    contract = create_contract(attrs, rf_obs)
    result = contract.simulate()

    assert len(result.events) > 0
    assert result.events[0].event_type == EventType.IED
    assert result.events[-1].event_type == EventType.MD
```

### 3. Property-Based Tests (`tests/property/test_contract_properties.py`)

Test invariants using Hypothesis:

```python
@given(
    notional=st.floats(min_value=1000, max_value=1e6),
    rate=st.floats(min_value=0.01, max_value=0.15),
)
def test_pam_total_interest_positive(notional, rate):
    """Property: Total interest should always be positive."""
    attrs = ContractAttributes(notional_principal=notional, nominal_interest_rate=rate, ...)
    contract = create_contract(attrs, rf_obs)
    result = contract.simulate()

    total_interest = sum(e.payoff for e in result.events if e.event_type == EventType.IP)
    assert total_interest > 0
```

### 4. Performance Tests (`tests/performance/test_performance.py`)

Benchmark performance:

```python
def test_pam_simulation_performance():
    """30-year mortgage should simulate in < 500ms."""
    attrs = ContractAttributes(maturity_date=30_years_from_now, ...)
    contract = create_contract(attrs, rf_obs)

    start = time.perf_counter()
    result = contract.simulate()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5  # 500ms
```

### 5. JAX Compatibility Tests (`tests/integration/test_jax_compatibility.py`)

Verify JAX integration:

```python
def test_pam_payoffs_use_jax_arrays():
    """Verify all payoffs are JAX arrays."""
    contract = create_contract(attrs, JaxRiskFactorObserver(...))
    result = contract.simulate()

    for event in result.events:
        assert isinstance(event.payoff, jnp.ndarray)
        assert event.payoff.dtype == jnp.float32
```

---

## Advanced Topics

### Interest Accrual Between Events

PAM contracts accrue interest continuously. JACTUS handles this via **implicit IPCI events**:

1. **Between IED and first IP**: Interest accrues in `ipac`
2. **Between IP events**: Interest accrues from 0 to next payment
3. **Between last IP and MD**: Final interest accrues

The `LifecycleEngine` automatically computes accrued interest before each IP/MD event.

### Day Count Conventions

Different conventions affect interest calculation:

- **A/360**: Actual days / 360 (common in money markets)
- **A/A**: Actual days / actual days in year (365 or 366)
- **30/360**: Assumes 30-day months (common in bonds)

Example impact on 1-year loan:
```python
# 365 days at 5% on $100k
A360: 100000 * 0.05 * (365/360) = 5069.44
A_A:  100000 * 0.05 * (365/365) = 5000.00
30360: 100000 * 0.05 * (360/360) = 5000.00
```

### Contract Role (Perspective)

- **RPA (Real Position Asset / Paying)**: Borrower perspective
  - IED cashflow: **negative** (receive money)
  - IP/MD cashflows: **positive** (pay money)

- **RPL (Real Position Liability / Receiving)**: Lender perspective
  - IED cashflow: **positive** (lend money)
  - IP/MD cashflows: **negative** (receive money... wait, this seems backwards in ACTUS!)

Actually in ACTUS:
- **Positive payoff** = outflow from contract holder's perspective
- **Negative payoff** = inflow to contract holder

### JAX Transformations

PAM contracts support JAX transformations:

**JIT Compilation**:
```python
@jax.jit
def compute_npv(contract):
    result = contract.simulate()
    # ... compute NPV
    return npv

npv = compute_npv(contract)  # Compiled, fast!
```

**Vectorization** (via vmap):
```python
# Simulate 1000 scenarios in parallel
scenarios = create_scenario_contracts()  # List of 1000 PAM contracts

def simulate_one(contract):
    return contract.simulate()

# This won't work directly because ContractAttributes can't be vmapped
# Instead, vectorize at the computation level:

rates = jnp.array([0.03, 0.04, 0.05, 0.06, 0.07])

def compute_cost(rate):
    attrs = ContractAttributes(..., nominal_interest_rate=rate)
    contract = create_contract(attrs, JaxRiskFactorObserver(jnp.array([rate])))
    result = contract.simulate()
    return sum_cashflows(result)

costs = jax.vmap(compute_cost)(rates)  # Vectorized!
```

**Automatic Differentiation** (Greeks):
```python
def contract_value(rate: float) -> float:
    attrs = ContractAttributes(..., nominal_interest_rate=rate)
    contract = create_contract(attrs, rf_obs)
    result = contract.simulate()
    return compute_npv(result)

# Compute sensitivity to interest rate (Rho)
rho = jax.grad(contract_value)(0.05)
print(f"DV/DR = {rho}")  # Sensitivity of value to rate change
```

### Adding New Event Types

To add support for a new event type (e.g., `PP` for prepayment):

1. Add to `EventType` enum
2. Implement `_pof_pp()` in `PAMPayoffFunction`
3. Implement `_stf_pp()` in `PAMStateTransitionFunction`
4. Update `generate_event_schedule()` to include PP events
5. Write tests

---

## Summary

### What We Learned

1. **JACTUS Architecture**: Layered design with clear separation of concerns
2. **JAX Integration**: Performance, differentiation, and vectorization
3. **ACTUS Implementation**: How PAM maps to POF/STF functions
4. **Testing Strategy**: Unit, integration, property, performance tests
5. **Extensibility**: How to add new contract types and features

### Key Takeaways

- **POF (Payoff Function)**: Computes cashflows for each event type
- **STF (State Transition Function)**: Updates state after each event
- **BaseContract**: Orchestrates POF/STF via LifecycleEngine
- **JAX Arrays**: Enable high performance and automatic differentiation
- **Factory Pattern**: Simplifies contract creation

### Repository Structure

```
src/jactus/
├── core/              # Phase 1: Fundamental types
│   ├── types.py       # Enums (ContractType, EventType, etc.)
│   ├── time.py        # ActusDateTime
│   ├── states.py      # ContractState
│   ├── attributes.py  # ContractAttributes
│   └── events.py      # ContractEvent, SimulationResult
├── utilities/         # Phase 1: Helper functions
│   ├── schedules.py   # generate_schedule()
│   ├── conventions.py # year_fraction()
│   ├── calendars.py   # Business day adjustments
│   └── math.py        # Financial math
├── functions/         # Phase 2: POF/STF
│   ├── payoff.py      # PayoffFunction base
│   ├── state.py       # StateTransitionFunction base
│   └── composition.py # Function composition
├── observers/         # Phase 2: Risk factors
│   ├── risk_factor.py # JaxRiskFactorObserver
│   └── child_contract.py # Child contract observer
├── engine/            # Phase 2: Simulation
│   ├── lifecycle.py   # LifecycleEngine
│   └── simulator.py   # SimulationEngine
└── contracts/         # Phase 3: Contract types
    ├── base.py        # BaseContract
    ├── pam.py         # PrincipalAtMaturityContract (THIS FILE!)
    ├── csh.py         # CashContract
    ├── stk.py         # StockContract
    ├── com.py         # CommodityContract
    └── __init__.py    # Factory: create_contract()
```

### Next Steps

1. **Explore Other Contracts**: Study CSH, STK, COM implementations
2. **Read ACTUS Spec**: Deep dive into ACTUS specifications
3. **Implement Custom Contract**: Try implementing LAM or ANN
4. **Performance Optimization**: Profile and optimize hot paths
5. **Contribute**: Submit PRs for new features or contract types!

---

## References

- **ACTUS Standard**: https://www.actusfrf.org/
- **JAX Documentation**: https://jax.readthedocs.io/
- **JACTUS Repository**: https://github.com/pedronahum/jactus
- **Examples**: `examples/pam_example.py`

---

**Questions?** Open an issue on GitHub or contribute to discussions!

**Happy coding!** 🎉
