# JACTUS Architecture Guide

**Version**: 1.0.0
**Last Updated**: 2025-10-21
**Status**: Complete ACTUS v1.1 Implementation

---

## Introduction

**JACTUS** (JAX-Accelerated ACTUS) is a Python library for simulating financial contracts using the ACTUS standard, built on Google's JAX framework for high-performance computing.

### What is ACTUS?

**ACTUS** (Algorithmic Contract Types Unified Standard) v1.1 is a comprehensive standard for representing financial contracts as state machines with:
- **Contract Types**: 31 standardized contract types - JACTUS implements 18
- **Events**: Life cycle events (IED, IP, MD, etc.)
- **States**: Contract state variables (notional, accrued interest, etc.)
- **Functions**: Payoff (POF) and State Transition (STF) functions

### Why JAX?

JAX provides:
- **Performance**: JIT compilation via XLA
- **Differentiation**: Automatic differentiation for risk analysis (Greeks)
- **Vectorization**: vmap for batch processing
- **GPU/TPU Support**: Scale to massive portfolios

### Project Goals

1. **ACTUS Compliance**: Faithful implementation of ACTUS specifications
2. **High Performance**: Leverage JAX for speed
3. **Type Safety**: Extensive use of type hints and validation
4. **Extensibility**: Easy to add new contract types
5. **Production Ready**: Comprehensive testing and documentation

---

## Design Principles

### 1. Functional Core, Imperative Shell

- **Core Logic**: Pure functions using JAX (POF, STF)
- **Shell**: Imperative Python for orchestration and I/O

```python
# Pure functional core (JAX)
def pof_ip(state: ContractState, attrs: ContractAttributes) -> jnp.ndarray:
    """Pure function: state + attrs → payoff"""
    return state.isc * state.ipac

# Imperative shell (Python)
class BaseContract:
    def simulate(self) -> SimulationResult:
        """Imperative orchestration"""
        events = self.generate_event_schedule()
        state = self.initialize_state()
        for event in events:
            payoff = self.pof(event.type, state, ...)
            state = self.stf(event.type, state, ...)
```

### 2. Separation of Concerns

Each layer has a single responsibility:

- **Core**: Data structures and types
- **Utilities**: Pure functions for common operations
- **Functions**: POF/STF logic
- **Engine**: Event generation and simulation orchestration
- **Contracts**: Contract-specific implementations

### 3. Immutability

- States are **immutable** (frozen dataclasses)
- State transitions create **new states**
- Enables JAX transformations and easier reasoning

```python
# BAD: Mutation
def stf_ip(state):
    state.ipac = 0.0  # ❌ Mutation!
    return state

# GOOD: Immutability
def stf_ip(state):
    return ContractState(
        ...
        ipac=jnp.array(0.0)  # ✅ New state
    )
```

### 4. Type Safety

- Extensive type hints everywhere
- Pydantic models for validation
- Enums for categorical values
- Runtime type checking in critical paths

```python
def year_fraction(
    start: ActusDateTime,  # Type hint
    end: ActusDateTime,
    convention: DayCountConvention,  # Enum, not string!
) -> float:  # Return type
    ...
```

### 5. Testability

- Small, focused functions easy to test
- Dependency injection (risk factor observers)
- Property-based testing for invariants
- Clear separation enables unit testing

---

## System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      USER APPLICATION                         │
│                                                               │
│  - Contract modeling and simulation                          │
│  - Risk analysis and scenario testing                        │
│  - Cash flow projections                                     │
│  - Reporting and analytics                                   │
└──────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼                           ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│      CLI LAYER           │  │     MCP SERVER LAYER         │
│                          │  │                              │
│  jactus simulate         │  │  jactus_simulate_contract    │
│  jactus contract list    │  │  jactus_list_contracts       │
│  jactus risk dv01        │  │  jactus_validate_attributes  │
│  jactus portfolio agg    │  │  jactus_get_contract_schema  │
└──────────────────────────┘  └──────────────────────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                     PUBLIC API LAYER                          │
│                                                               │
│  from jactus.contracts import create_contract                │
│  from jactus.core import ContractAttributes, ActusDateTime   │
│  from jactus.observers import JaxRiskFactorObserver          │
└──────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌────────────────┐  ┌──────────────┐
│   Contracts   │  │    Observers   │  │  Utilities   │
│               │  │                │  │              │
│ • 18 types    │  │ • RiskFactor   │  │ • Schedules  │
│ • Principal   │  │ • ChildContract│  │ • Conventions│
│ • Derivative  │  │ • Behavioral   │  │ • Calendars  │
│ • Exotic      │  │ • Scenario     │  │ • Math       │
│               │  │                │  │ • Surface2D  │
└───────────────┘  └────────────────┘  └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
        ┌─────────────────────────────────────┐
        │         ENGINE LAYER                 │
        │                                      │
        │  • LifecycleEngine                  │
        │  • SimulationEngine                 │
        │  • BaseContract (abstract)          │
        └─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐  ┌────────────────┐  ┌──────────────┐
│   Functions   │  │   Core Types   │  │  Observers   │
│               │  │                │  │  (Protocol)  │
│ • Payoff      │  │ • State        │  │              │
│ • StateTrans  │  │ • Attributes   │  │ • get(index) │
│ • Composition │  │ • Events       │  │              │
└───────────────┘  └────────────────┘  └──────────────┘
        │                   │
        └───────────────────┘
                            ▼
        ┌─────────────────────────────────────┐
        │          JAX FOUNDATION              │
        │                                      │
        │  • jax.numpy (arrays)               │
        │  • jax.jit (compilation)            │
        │  • jax.vmap (vectorization)         │
        │  • jax.grad (differentiation)       │
        │  • Flax NNX (pytree structures)     │
        └─────────────────────────────────────┘
```

### Layered View

| Layer | Purpose | Key Components |
|-------|---------|----------------|
| **Application** | User-facing functionality | Contract modeling, simulation, analytics |
| **CLI** | Terminal interface | `jactus` command: simulate, risk, portfolio, contract, observer, docs |
| **MCP Server** | AI assistant interface | `jactus_*` tools for contract discovery, validation, simulation |
| **Contracts** | 18 ACTUS implementations | PAM, LAM, ANN, STK, COM, FXOUT, SWAPS, etc. |
| **Observers** | Market data + behavioral models | RiskFactor, Behavioral, Scenario |
| **Engine** | Simulation orchestration | LifecycleEngine, SimulationEngine |
| **Functions** | ACTUS logic | PayoffFunction, StateTransitionFunction |
| **Core** | Fundamental types | State, Attributes, Events, DateTime |
| **Utilities** | Helper functions | Schedules, conventions, calendars, Surface2D |
| **Foundation** | JAX framework | Arrays, JIT, vmap, grad |

---

## Key Components

### 1. ContractState

**Purpose**: Represents contract state at a point in time.

**Implementation**: Frozen dataclass registered as JAX pytree

**Key Fields**:
- `sd`: Status Date (ActusDateTime)
- `tmd`: Maturity Date (ActusDateTime)
- `nt`: Notional Principal (jnp.ndarray)
- `ipnr`: Nominal Interest Rate (jnp.ndarray)
- `ipac`: Interest Payment Accrued (jnp.ndarray)
- `feac`: Fee Accrued (jnp.ndarray)
- `nsc`: Notional Scaling Multiplier (jnp.ndarray)
- `isc`: Interest Scaling Multiplier (jnp.ndarray)

**Design Decisions**:
- Immutable (new state created for each transition)
- JAX arrays for numerical values (enables JIT, grad, vmap)
- Registered as JAX pytree for functional transformations

**File**: `src/jactus/core/states.py`

### 2. ContractAttributes

**Purpose**: Contract terms and parameters (legal agreement).

**Implementation**: Pydantic BaseModel

**Key Fields** (subset of ~80):
- Identification: `contract_id`, `contract_type`, `contract_role`
- Dates: `status_date`, `initial_exchange_date`, `maturity_date`
- Financial: `notional_principal`, `nominal_interest_rate`, `currency`
- Conventions: `day_count_convention`, `business_day_convention`
- Schedules: `interest_payment_cycle`, `fee_payment_cycle`

**Design Decisions**:
- Pydantic for validation and serialization
- Immutable (frozen=True)
- Optional fields with sensible defaults

**File**: `src/jactus/core/attributes.py`

### 3. PayoffFunction

**Purpose**: Calculate cashflow for each event type.

**Interface**:
```python
class PayoffFunction(Protocol):
    def __call__(
        self,
        event_type: EventType,
        state: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> jnp.ndarray:
        ...
```

**Implementations**:
- `PAMPayoffFunction`: IED, IP, MD, PRD, TD, PP, PY, FP, RR, RRF, SC
- `CashPayoffFunction`: AD
- `StockPayoffFunction`: PRD, TD, DV
- `CommodityPayoffFunction`: PRD, TD

**File**: `src/jactus/functions/payoff.py` (base), `src/jactus/contracts/*.py` (implementations)

### 4. StateTransitionFunction

**Purpose**: Update contract state after each event.

**Interface**:
```python
class StateTransitionFunction(Protocol):
    def __call__(
        self,
        event_type: EventType,
        state_pre: ContractState,
        attributes: ContractAttributes,
        time: ActusDateTime,
        risk_factor_observer: RiskFactorObserver,
    ) -> ContractState:
        ...
```

**Implementations**:
- `PAMStateTransitionFunction`: IED, IP, IPCI, MD, etc.
- `CashStateTransitionFunction`: AD
- `StockStateTransitionFunction`: PRD, TD, DV
- `CommodityStateTransitionFunction`: PRD, TD

**File**: `src/jactus/functions/state.py` (base), `src/jactus/contracts/*.py` (implementations)

### 5. LifecycleEngine

**Purpose**: Generate contract events by applying POF/STF.

**Algorithm**:
```python
def generate_contract_lifecycle(
    events: List[ContractEvent],
    initial_state: ContractState,
    pof: PayoffFunction,
    stf: StateTransitionFunction,
    attributes: ContractAttributes,
    rf_obs: RiskFactorObserverProtocol,
) -> List[ContractEvent]:
    state = initial_state

    for event in events:
        # Apply payoff function
        payoff = pof.calculate_payoff(event.type, state, attributes, event.time, rf_obs)

        # Apply state transition
        new_state = stf.transition_state(event.type, state, attributes, event.time)

        # Store event with states
        event.payoff = payoff
        event.state_pre = state
        event.state_post = new_state

        state = new_state

    return events
```

**File**: `src/jactus/engine/lifecycle.py`

### 6. BaseContract

**Purpose**: Abstract base class for all contracts.

**Key Methods**:
```python
class BaseContract(nnx.Module):
    def generate_event_schedule(self) -> List[ContractEvent]:
        """Generate future events (contract-specific)."""
        raise NotImplementedError

    def initialize_state(self) -> ContractState:
        """Create initial state (contract-specific)."""
        raise NotImplementedError

    def get_payoff_function(self) -> PayoffFunction:
        """Get POF instance (contract-specific)."""
        raise NotImplementedError

    def get_state_transition_function(self) -> StateTransitionFunction:
        """Get STF instance (contract-specific)."""
        raise NotImplementedError

    def simulate(self) -> SimulationResult:
        """Run full simulation (common for all contracts)."""
        events = self.generate_event_schedule()
        state = self.initialize_state()
        pof = self.get_payoff_function()
        stf = self.get_state_transition_function()

        events = LifecycleEngine.generate_contract_lifecycle(
            events, state, pof, stf, self.attributes, self.rf_obs
        )

        return SimulationResult(events=events)
```

**File**: `src/jactus/contracts/base.py`

### 7. Factory Pattern

**Purpose**: Simplify contract creation.

**Implementation**:
```python
def create_contract(
    attributes: ContractAttributes,
    risk_factor_observer: RiskFactorObserverProtocol,
) -> BaseContract:
    """Create contract instance based on contract_type."""

    contract_map = {
        ContractType.PAM: PrincipalAtMaturityContract,
        ContractType.CSH: CashContract,
        ContractType.STK: StockContract,
        ContractType.COM: CommodityContract,
    }

    contract_class = contract_map.get(attributes.contract_type)
    if not contract_class:
        raise ValueError(f"Unsupported: {attributes.contract_type}")

    return contract_class(attributes, risk_factor_observer)
```

**Usage**:
```python
attrs = ContractAttributes(contract_type=ContractType.PAM, ...)
contract = create_contract(attrs, rf_obs)  # Auto-selects PAM implementation
```

**File**: `src/jactus/contracts/__init__.py`

### 8. Command-Line Interface (CLI)

**Purpose**: Provide a terminal-based interface for contract simulation, validation, risk analytics, and portfolio management — designed for both human operators and automated pipelines.

**Implementation**: Typer-based CLI registered as `jactus` entry point. Shared formatting via `output.py` with automatic TTY detection (rich tables in terminal, JSON when piped).

**Command Tree**:
- `jactus contract list|schema|validate` — Contract type discovery and attribute validation
- `jactus simulate` — Full contract simulation with event filtering
- `jactus risk dv01|duration|convexity|sensitivities` — Risk metrics via finite-difference bumping
- `jactus portfolio simulate|aggregate` — Multi-contract portfolio simulation and cash flow aggregation
- `jactus observer list|describe` — Risk factor observer discovery
- `jactus docs search` — Documentation keyword search

**Design Decisions**:
- Agent-first: JSON output by default when piped (`sys.stdout.isatty()` detection)
- Mirrors the MCP server surface — same capabilities available from terminal
- Shared `prepare_attributes()` converts string values (dates, enums) to JACTUS types
- Exit codes: 0 (success), 1 (user/input error), 2 (simulation error), 3 (validation failure)
- Risk metrics use finite-difference bumping (not JAX grad) for broad compatibility

**Files**: `src/jactus/cli/__init__.py`, `src/jactus/cli/output.py`, `src/jactus/cli/contract.py`, `src/jactus/cli/simulate.py`, `src/jactus/cli/risk.py`, `src/jactus/cli/portfolio.py`, `src/jactus/cli/observer.py`, `src/jactus/cli/docs.py`

### 9. Behavioral Risk Factor Observers

**Purpose**: Model state-dependent risk factors (prepayment, deposit behavior) that depend on the contract's internal state and can inject events into the simulation schedule.

**Key Distinction from Market Observers**: Market risk factor observers return values based solely on an identifier and time (e.g., a yield curve lookup). Behavioral observers are aware of contract state (notional, interest rate, age, performance status) and dynamically inject **callout events** into the simulation timeline.

**Core Types**:

- `CalloutEvent` — Frozen dataclass representing an event a behavioral model requests be added to the simulation schedule. Fields: `model_id`, `time`, `callout_type` (e.g., `"MRD"`, `"AFD"`), and optional `metadata`.
- `BehaviorRiskFactorObserver` — `runtime_checkable` Protocol extending `RiskFactorObserver` with a `contract_start(attributes)` method that returns a list of `CalloutEvent` objects.
- `BaseBehaviorRiskFactorObserver` — Abstract base class extending `BaseRiskFactorObserver` with abstract `contract_start()`. Subclasses implement `_get_risk_factor()` (state-aware), `_get_event_data()`, and `contract_start()`.

**Callout Event Mechanism**:

When `BaseContract.simulate()` is called with behavioral observers (via a `Scenario` or explicitly), the engine:
1. Calls `contract_start(attributes)` on each behavioral observer to collect callout events.
2. Merges the callout events into the scheduled event timeline (sorted by time).
3. During simulation, when a callout event is reached, the behavioral observer is evaluated with the current contract state.

**Callout Types**:
- `"MRD"` (Multiplicative Reduction Delta) — Used by prepayment models. Represents a fraction by which the notional principal is reduced.
- `"AFD"` (Absolute Funded Delta) — Used by deposit transaction models. Represents an absolute change in the deposit balance.

**Concrete Implementations**:

| Observer | Callout Type | Use Case | File |
|----------|-------------|----------|------|
| `PrepaymentSurfaceObserver` | MRD | 2D surface-based prepayment model (spread x loan age) | `observers/prepayment.py` |
| `DepositTransactionObserver` | AFD | Deposit inflows/outflows for UMP contracts | `observers/deposit_transaction.py` |

**Files**: `src/jactus/observers/behavioral.py`, `src/jactus/observers/prepayment.py`, `src/jactus/observers/deposit_transaction.py`

### 10. Scenario

**Purpose**: Bundle market and behavioral observers into named, reusable simulation configurations.

**Implementation**: Dataclass with `scenario_id`, `description`, `market_observers` dict, and `behavior_observers` dict.

**Key Methods**:
- `get_observer()` — Returns a unified `RiskFactorObserver` by composing all market observers via `CompositeRiskFactorObserver`. If only one market observer is present, returns it directly.
- `get_callout_events(attributes)` — Calls `contract_start()` on all behavioral observers and returns the aggregated callout events sorted by time.
- `add_market_observer(id, observer)` / `add_behavior_observer(id, observer)` — Mutators for building scenarios incrementally.

**Usage**:
```python
from jactus.observers.scenario import Scenario
from jactus.observers import TimeSeriesRiskFactorObserver
from jactus.observers.prepayment import PrepaymentSurfaceObserver

scenario = Scenario(
    scenario_id="base-case",
    description="Base case with moderate prepayment",
    market_observers={
        "rates": TimeSeriesRiskFactorObserver(...),
    },
    behavior_observers={
        "prepayment": PrepaymentSurfaceObserver(...),
    },
)

# Pass to simulation — market observer and callout events are handled automatically
history = contract.simulate(scenario=scenario)
```

**Design Decisions**:
- Scenarios separate market data (time series, curves) from behavioral models (prepayment, deposit transactions)
- Enables easy scenario comparison (base case vs. stress) by swapping observers
- The `Scenario` does not own contracts — it is purely an environment configuration

**File**: `src/jactus/observers/scenario.py`

### 11. Surface2D and LabeledSurface2D

**Purpose**: JAX-compatible 2D surface interpolation, used primarily by behavioral risk models.

**`Surface2D`** (frozen dataclass):
- Defined by `x_margins` (sorted 1D array), `y_margins` (sorted 1D array), and `values` (2D array of shape `(len(x_margins), len(y_margins))`)
- `evaluate(x, y)` performs bilinear interpolation within the grid
- Configurable `extrapolation`: `"constant"` (clamp to nearest edge) or `"raise"` (error on out-of-bounds)
- Serializable via `from_dict()` / `to_dict()`

**`LabeledSurface2D`** (dataclass):
- Uses string labels instead of numeric margins (e.g., contract IDs on x-axis, date strings on y-axis)
- `get(x_label, y_label)` for exact label-based lookups
- `get_row(x_label)` / `get_column(y_label)` for slicing
- Used by `DepositTransactionObserver` for contract-indexed transaction schedules

**File**: `src/jactus/utilities/surface.py`

---

## Data Flow

### Contract Simulation Flow

```
1. USER CREATES CONTRACT
   ↓
   attrs = ContractAttributes(...)
   rf_obs = JaxRiskFactorObserver(...)
   contract = create_contract(attrs, rf_obs)

2. GENERATE EVENT SCHEDULE
   ↓
   events = contract.generate_event_schedule()
   # Returns: [IED(2024-01-15), IP(2024-07-15), MD(2025-01-15)]

3. INITIALIZE STATE
   ↓
   state = contract.initialize_state()
   # Returns: ContractState(nt=0, ipnr=0, ...)

4. LIFECYCLE ENGINE PROCESSES EACH EVENT
   ↓
   For each event in events:

     4a. APPLY PAYOFF FUNCTION
         ↓
         payoff = pof.calculate_payoff(event.type, state, attrs, time, rf_obs)
         # Returns: JAX array with cashflow amount

     4b. APPLY STATE TRANSITION
         ↓
         new_state = stf.transition_state(event.type, state, attrs, time)
         # Returns: New ContractState

     4c. STORE EVENT
         ↓
         event.payoff = payoff
         event.state_pre = state
         event.state_post = new_state

     4d. UPDATE STATE
         ↓
         state = new_state

5. RETURN RESULTS
   ↓
   result = SimulationResult(events=events)
   cashflows = result.get_cashflows()
   # Returns: [(time, amount, currency), ...]
```

### Simulation with Behavioral Observers

When behavioral observers are present (via a `Scenario` or passed directly), the simulation flow gains an additional phase before event processing:

```
1. USER CREATES CONTRACT + SCENARIO
   ↓
   scenario = Scenario(
       market_observers={"rates": ts_observer},
       behavior_observers={"prepay": prepayment_observer},
   )
   contract = create_contract(attrs, rf_obs)

2. COLLECT CALLOUT EVENTS (new phase)
   ↓
   For each behavioral observer:
     callout_events = observer.contract_start(attrs)
   # Returns: [MRD(2024-07-15), MRD(2025-01-15), ...]

3. MERGE INTO SCHEDULE
   ↓
   Callout events are inserted into the regular event schedule
   sorted by time alongside IED, IP, MD, etc.
   # Schedule: [IED(2024-01-15), IP(2024-07-15), MRD(2024-07-15), ...]

4. PROCESS ALL EVENTS (standard + callout)
   ↓
   For each event in merged schedule:
     4a. PAYOFF (POF) — behavioral events use state-aware observer
     4b. STATE TRANSITION (STF) — callout events may modify notional
     4c. STORE EVENT + UPDATE STATE
```

### State Evolution Example (PAM)

```
Timeline:  SD──────IED──────────IP──────────MD
           │       │            │           │
State:     nt=0    nt=100k      nt=100k     nt=0
           ipac=0  ipac=0       ipac=2.5k   ipac=0

Events:            IED          IP          MD
Payoff:            -100k        +2.5k       +102.5k

Transitions:
  SD → IED:  STF_IED sets nt=100k, ipnr=0.05
  IED → IP:  IPCI accrues interest (ipac=2.5k)
  IP → MD:   STF_IP resets ipac=0
  MD → END:  STF_MD sets nt=0 (loan repaid)
```

---

## JAX Integration

### Why JAX?

1. **Performance**: XLA compilation → 10-100x speedup
2. **Differentiation**: Automatic Greeks computation
3. **Vectorization**: Batch processing with vmap
4. **Hardware Acceleration**: GPU/TPU support

### JAX Usage in JACTUS

#### 1. Arrays for State

```python
# All state variables are JAX arrays
state = ContractState(
    nt=jnp.array(100000.0, dtype=jnp.float32),  # float32 for efficiency
    ipac=jnp.array(0.0, dtype=jnp.float32),
    ...
)
```

#### 2. JIT Compilation

```python
@jax.jit
def compute_interest(nt, rate, yf):
    """Compute interest (JIT compiled for speed)."""
    return nt * rate * yf

interest = compute_interest(jnp.array(100000.0), jnp.array(0.05), 0.5)
```

#### 3. Automatic Differentiation

```python
def contract_npv(interest_rate: float) -> float:
    """Compute NPV as function of interest rate."""
    attrs = ContractAttributes(..., nominal_interest_rate=interest_rate)
    contract = create_contract(attrs, rf_obs)
    result = contract.simulate()
    return compute_npv(result.events)

# Compute sensitivity (Rho)
rho = jax.grad(contract_npv)(0.05)  # dNPV/dRate
```

#### 4. Vectorization

```python
# Compute NPV for 1000 scenarios
interest_rates = jnp.linspace(0.01, 0.10, 1000)

def npv_for_rate(rate):
    attrs = ContractAttributes(..., nominal_interest_rate=rate)
    contract = create_contract(attrs, rf_obs)
    result = contract.simulate()
    return compute_npv(result)

# Vectorize across scenarios
npvs = jax.vmap(npv_for_rate)(interest_rates)  # 1000 NPVs in parallel
```

### Limitations

**ContractAttributes Cannot Be Traced**:
- Pydantic models don't support JAX tracing
- Can't vmap/grad over contract creation
- **Solution**: Vectorize at computation level, not contract level

**Example (Won't Work)**:
```python
def create_and_simulate(rate):
    attrs = ContractAttributes(..., nominal_interest_rate=rate)  # Pydantic model!
    contract = create_contract(attrs, rf_obs)
    return contract.simulate()

jax.vmap(create_and_simulate)(rates)  # ❌ Error: Can't trace Pydantic
```

**Example (Works)**:
```python
# Create contracts once
contracts = [create_contract(attrs, rf_obs) for rate in rates]

# Vectorize computation only
def simulate_contract(contract):
    return contract.simulate()

results = [simulate_contract(c) for c in contracts]  # ✅ Works
```

---

## Extending JACTUS

### Adding a New Contract Type

**Example**: Implementing LAM (Linear Amortizer)

#### Step 1: Define Events

```python
# LAM events: IED, IP+PR (interest + principal reduction), MD
# Similar to PAM but with partial principal repayment at each IP
```

#### Step 2: Implement Payoff Function

```python
class LAMPayoffFunction(PayoffFunction):
    def calculate_payoff(self, event_type, state, attrs, time, rf_obs):
        if event_type == EventType.IED:
            return self._pof_ied(state, attrs)
        elif event_type == EventType.IPPR:  # Interest + Principal Repayment
            return self._pof_ippr(state, attrs)
        elif event_type == EventType.MD:
            return self._pof_md(state, attrs)

    def _pof_ippr(self, state, attrs):
        """Interest payment + principal reduction."""
        role_sign = contract_role_sign(self.contract_role)
        interest = state.isc * state.ipac
        principal = state.nsc * attrs.next_principal_redemption_amount
        return role_sign * (interest + principal)
```

#### Step 3: Implement State Transition

```python
class LAMStateTransitionFunction(StateTransitionFunction):
    def transition_state(self, event_type, state, attrs, time):
        if event_type == EventType.IPPR:
            return self._stf_ippr(state, attrs, time)

    def _stf_ippr(self, state, attrs, time):
        """Reduce notional by redemption amount."""
        new_nt = state.nt - attrs.next_principal_redemption_amount
        return ContractState(
            sd=time,
            tmd=state.tmd,
            nt=new_nt,  # Reduced!
            ipac=jnp.array(0.0),  # Reset interest
            ...
        )
```

#### Step 4: Implement Contract Class

```python
class LinearAmortizationContract(BaseContract):
    def generate_event_schedule(self):
        events = []
        events.append(ContractEvent(EventType.IED, self.attrs.ied, 0))

        # Generate IPPR schedule
        ippr_dates = generate_schedule(
            self.attrs.ied, self.attrs.cycle, self.attrs.md
        )
        for date in ippr_dates:
            events.append(ContractEvent(EventType.IPPR, date, len(events)))

        events.append(ContractEvent(EventType.MD, self.attrs.md, len(events)))
        return events

    def get_payoff_function(self):
        return LAMPayoffFunction(...)

    def get_state_transition_function(self):
        return LAMStateTransitionFunction()
```

#### Step 5: Register in Factory

```python
def create_contract(attrs, rf_obs):
    contract_map = {
        ContractType.PAM: PrincipalAtMaturityContract,
        ContractType.LAM: LinearAmortizationContract,  # Add here!
        ...
    }
    ...
```

#### Step 6: Write Tests

```python
def test_lam_principal_amortizes():
    """Test LAM principal reduces each period."""
    attrs = ContractAttributes(
        contract_type=ContractType.LAM,
        notional_principal=120000,
        next_principal_redemption_amount=10000,
        ...
    )
    contract = create_contract(attrs, rf_obs)
    result = contract.simulate()

    # Check principal reduces by 10k each period
    assert result.events[1].state_post.nt == 110000
    assert result.events[2].state_post.nt == 100000
```

### Adding a New Behavioral Risk Observer

**Example**: Implementing a credit migration model that adjusts the interest rate based on a credit rating transition matrix.

#### Step 1: Subclass BaseBehaviorRiskFactorObserver

```python
class CreditMigrationObserver(BaseBehaviorRiskFactorObserver):
    def __init__(self, transition_matrix, observation_cycle="1Y", model_id="credit-migration"):
        super().__init__(name=f"CreditMigration({model_id})")
        self.transition_matrix = transition_matrix
        self.observation_cycle = observation_cycle
        self.model_id = model_id
```

#### Step 2: Implement `contract_start()` to Generate Callout Events

```python
    def contract_start(self, attributes):
        from jactus.core.time import add_period
        events = []
        current = add_period(attributes.initial_exchange_date, self.observation_cycle)
        while current < attributes.maturity_date:
            events.append(CalloutEvent(
                model_id=self.model_id,
                time=current,
                callout_type="MRD",  # or a custom type
            ))
            current = add_period(current, self.observation_cycle)
        return events
```

#### Step 3: Implement `_get_risk_factor()` with State Awareness

```python
    def _get_risk_factor(self, identifier, time, state, attributes):
        # Use contract state to compute risk factor
        current_rate = float(state.ipnr)
        credit_adjustment = self.transition_matrix.lookup(current_rate, time)
        return jnp.array(credit_adjustment, dtype=jnp.float32)
```

#### Step 4: Implement `_get_event_data()`

```python
    def _get_event_data(self, identifier, event_type, time, state, attributes):
        raise KeyError("No event data")
```

#### Step 5: Use with a Scenario

```python
scenario = Scenario(
    scenario_id="credit-stress",
    market_observers={"rates": rate_observer},
    behavior_observers={"credit": CreditMigrationObserver(matrix)},
)
history = contract.simulate(scenario=scenario)
```

---

## Performance Considerations

### Performance Benchmarks

| Operation | Time | Details |
|-----------|------|---------|
| Simple contract (CSH) | < 10ms | Single event |
| PAM 5-year quarterly | < 50ms | ~22 events |
| Stock/Commodity | < 10ms | 2 events |
| PAM 30-year monthly | < 500ms | ~362 events |
| 100 contracts batch | < 500ms | Total for batch |
| Factory creation | < 1ms | Per contract |

### Optimization Strategies

#### 1. JIT Compilation

```python
# Compile hot paths
@jax.jit
def compute_all_interests(notionals, rates, year_fractions):
    return notionals * rates * year_fractions

# Use compiled version
interests = compute_all_interests(nt_array, rate_array, yf_array)
```

#### 2. Vectorization

```python
# Instead of loop
results = []
for contract in contracts:
    results.append(simulate(contract))

# Use vectorization (if possible)
results = jax.vmap(simulate)(contracts)
```

#### 3. Memory Efficiency

- Use `float32` instead of `float64` (2x memory savings)
- Avoid creating intermediate arrays
- Reuse allocated arrays when possible

```python
# Current: All arrays are float32
state.nt.dtype  # jnp.float32 ✅
```

#### 4. Batch Processing

```python
# Process portfolio in batches
batch_size = 1000
for i in range(0, len(portfolio), batch_size):
    batch = portfolio[i:i+batch_size]
    results = process_batch(batch)
```

### GPU / TPU Acceleration

12 of 18 contract types have dedicated array-mode simulation paths (PAM, LAM, NAM, ANN, LAX, CSH, STK, COM, FXOUT, FUTUR, OPTNS, SWPPV). These use JIT-compiled JAX kernels operating on `[B, T]` shaped arrays for portfolio-scale simulation. The unified entry point is `simulate_portfolio()` in `jactus.contracts.portfolio`.

- **Batch strategy**: `batch_simulate_<type>_auto()` uses the single-scan approach on CPU and `jax.vmap` on GPU/TPU, processing all contracts together in shaped `[B, T]` arrays.
- **JIT-compiled pre-computation**: `batch_precompute_pam()` generates event schedules and year fractions as pure JAX operations, keeping data on-device.
- **`lax.scan(unroll=8)`**: Reduces GPU kernel launches by 8× for stateful types.

No code changes are needed — install `jax[cuda13]` or `jax[tpu]` and JACTUS uses the accelerator automatically.

> For comprehensive array-mode documentation, see [ARRAY_MODE.md](ARRAY_MODE.md).

#### Float Precision

| Backend | Default dtype | float64 support |
|---------|--------------|-----------------|
| CPU     | float32      | Yes (enable with `jax_enable_x64`) |
| GPU     | float32      | Yes (enable with `jax_enable_x64`) |
| TPU     | float32      | **Not supported** |

The array-mode path uses float32 throughout for performance. The ACTUS
cross-validation tolerance is ±1.0, well within float32 range. For workloads
requiring double precision (e.g., long-dated swaps, high notionals), enable
64-bit mode before importing JACTUS:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

### Potential Optimizations

1. **Array-mode for remaining 6 types**: Extend to CLM, UMP, SWAPS, CAPFL, CEG, CEC (currently these use scalar fallback)
2. **Batch pre-computation for non-PAM types**: Extend the `batch_precompute_pam()` pattern (pure-JAX schedule generation) to other stateful types
3. **Multi-device parallelism**: Use `jax.experimental.shard_map` for 100K+ contract portfolios
4. **Compilation cache**: `jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")` avoids re-JIT across runs
5. **Cache Event Schedules**: Pre-compute and cache schedules for repeated simulations

---

## Contributing

### How to Contribute

1. **Report Bugs**: Open GitHub issue with minimal reproducible example
2. **Suggest Features**: Describe use case and propose API
3. **Submit PRs**: Follow coding standards, add tests, update docs
4. **Improve Docs**: Fix typos, add examples, clarify explanations

### Coding Standards

- **Style**: Black formatter, Ruff linter
- **Type Hints**: Required for all public APIs
- **Docstrings**: NumPy/Sphinx style (`:param`, `Returns:`)
- **Tests**: Aim for 90%+ coverage
- **Performance**: Benchmark before/after changes

### Testing Requirements

- Unit tests for new functions
- Integration tests for new features
- Property tests for invariants
- Performance tests for hot paths

---

## Conclusion

JACTUS provides a **complete, production-ready implementation** of the ACTUS v1.1 standard:

- ✅ **ACTUS v1.1 Compliant**: 18 contract types fully implemented
- ✅ **JAX-Powered**: High-performance with automatic differentiation
- ✅ **Extensible**: Clean architecture for custom contracts
- ✅ **Thoroughly Tested**: 1,192 tests, 95%+ coverage
- ✅ **Production-Ready**: Type-safe, documented, performant

**Getting Started**:
1. Try `jactus contract list` and `jactus simulate --type PAM` from the CLI
2. Read [PAM.md](PAM.md) for a detailed walkthrough of JACTUS internals
3. Explore the Jupyter notebooks in `examples/notebooks/` for hands-on learning
4. Run `examples/pam_example.py` and other Python examples
5. Check out the contract implementations in `src/jactus/contracts/`
6. Review the [derivative contracts guide](derivatives.md) for advanced features

**Questions?** Open an issue on [GitHub](https://github.com/pedronahum/JACTUS/issues) or start a discussion!

---

**Happy coding!** 🚀
