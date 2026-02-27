"""Documentation search tools."""

from typing import Any

from ._utils import get_jactus_root


def search_docs(query: str) -> dict[str, Any]:
    """Search JACTUS documentation for specific topics.

    Args:
        query: Search query (e.g., "day count convention", "state transition")

    Returns:
        Dictionary with relevant documentation sections.
    """
    jactus_root = get_jactus_root()
    docs_dir = jactus_root / "docs"

    # Files to search
    doc_files = [
        docs_dir / "ARCHITECTURE.md",
        docs_dir / "PAM.md",
        docs_dir / "derivatives.md",
        jactus_root / "README.md",
    ]

    results = []
    query_lower = query.lower()
    query_words = query_lower.split()

    for doc_file in doc_files:
        if not doc_file.exists():
            continue

        try:
            content = doc_file.read_text()
            lines = content.split("\n")

            # Search for query in content (word-based: match lines containing any query word)
            matches = []
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if any(word in line_lower for word in query_words):
                    # Get context (2 lines before and after)
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)
                    context = "\n".join(lines[start:end])

                    matches.append({
                        "line_number": i + 1,
                        "line": line.strip(),
                        "context": context,
                    })

            if matches:
                results.append({
                    "file": doc_file.name,
                    "path": str(doc_file.relative_to(jactus_root)),
                    "matches": matches[:5],  # Limit to top 5 matches per file
                    "total_matches": len(matches),
                })

        except Exception:
            continue

    if not results:
        return {
            "query": query,
            "found": False,
            "message": "No documentation matches found",
            "suggestion": "Try broader search terms or check available documentation files",
            "available_docs": [
                "ARCHITECTURE.md - System architecture and design",
                "PAM.md - Principal at Maturity walkthrough",
                "derivatives.md - Derivative contracts guide",
                "README.md - Project overview and quick start",
            ],
        }

    return {
        "query": query,
        "found": True,
        "total_files": len(results),
        "results": results,
    }


def get_doc_structure() -> dict[str, Any]:
    """Get the structure of JACTUS documentation.

    Returns:
        Dictionary with documentation files and their contents outline.
    """
    jactus_root = get_jactus_root()
    docs_dir = jactus_root / "docs"

    structure = {}

    # Main documentation files
    doc_files = {
        "README.md": jactus_root / "README.md",
        "ARCHITECTURE.md": docs_dir / "ARCHITECTURE.md",
        "PAM.md": docs_dir / "PAM.md",
        "derivatives.md": docs_dir / "derivatives.md",
    }

    for name, path in doc_files.items():
        if not path.exists():
            continue

        try:
            content = path.read_text()

            # Extract headers (markdown headers)
            headers = []
            for line in content.split("\n"):
                if line.startswith("#"):
                    # Count # to determine level
                    level = len(line) - len(line.lstrip("#"))
                    title = line.lstrip("#").strip()
                    headers.append({"level": level, "title": title})

            structure[name] = {
                "path": str(path.relative_to(jactus_root)),
                "size_bytes": len(content),
                "lines": len(content.split("\n")),
                "headers": headers[:20],  # First 20 headers
            }

        except Exception:
            continue

    return {
        "documentation_files": len(structure),
        "structure": structure,
    }


def get_topic_guide(topic: str) -> dict[str, Any]:
    """Get a guide for a specific topic.

    Args:
        topic: Topic name (e.g., "contracts", "jax", "events")

    Returns:
        Dictionary with relevant information about the topic.
    """
    guides = {
        "contracts": {
            "title": "Working with Contracts",
            "content": """
# Contract Types in JACTUS

JACTUS implements 18 ACTUS contract types:

1. **Principal Contracts** (6):
   - PAM: Principal at Maturity
   - LAM: Linear Amortizer
   - LAX: Exotic Linear Amortizer
   - NAM: Negative Amortizer
   - ANN: Annuity
   - CLM: Call Money

2. **Non-Principal** (3):
   - UMP: Undefined Maturity Profile
   - CSH: Cash
   - STK: Stock

3. **Exotic** (1):
   - COM: Commodity

4. **Derivatives** (8):
   - FXOUT, OPTNS, FUTUR, SWPPV, SWAPS, CAPFL, CEG, CEC

## Creating a Contract

```python
from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType
from jactus.observers import ConstantRiskFactorObserver

attrs = ContractAttributes(contract_type=ContractType.PAM, ...)
rf_obs = ConstantRiskFactorObserver(constant_value=0.0)
contract = create_contract(attrs, rf_obs)
result = contract.simulate()
```

See examples/ for more detailed examples.

## Behavioral Observers

Behavioral risk factor observers extend the standard observer framework with
state-dependent models that inject **CalloutEvents** into the simulation timeline.
Use `behavior_observers` or `scenario` parameters in `contract.simulate()`.

Available: `PrepaymentSurfaceObserver`, `DepositTransactionObserver`
See `jactus_get_topic_guide("behavioral")` for details.
""",
        },
        "behavioral": {
            "title": "Behavioral Risk Factor Observers",
            "content": """
# Behavioral Risk Factor Observers

Behavioral observers are state-aware risk models that inject **callout events**
into the simulation timeline. Unlike market observers (which only look up data
by identifier and time), behavioral observers use the contract's internal state
(notional, interest rate, age) to compute their output.

## Two-Phase Protocol

1. **Registration** — At simulation start, the engine calls `contract_start(attributes)`
   on each behavioral observer. The observer returns `CalloutEvent` objects specifying
   when it needs to be evaluated.
2. **Evaluation** — When the simulation reaches a callout time, it calls
   `observe_risk_factor(identifier, time, state, attributes)` with the current
   contract state, and the observer returns a value that drives a contract event.

## Callout Types

| Type | Event | Description |
|------|-------|-------------|
| MRD  | PP (Prepayment) | Multiplicative Reduction Delta — fraction of notional to prepay |
| AFD  | AD (Account Debit/Credit) | Absolute Funded Delta — exact transaction amount |

## Available Observers

### PrepaymentSurfaceObserver
2D surface-based prepayment model: spread (contract rate - market rate) x loan age → prepayment rate.

```python
from jactus.observers.prepayment import PrepaymentSurfaceObserver
from jactus.utilities.surface import Surface2D
import jax.numpy as jnp

surface = Surface2D(
    x_margins=jnp.array([0.0, 1.0, 2.0, 3.0]),   # spread %
    y_margins=jnp.array([0.0, 1.0, 3.0, 5.0]),    # loan age (years)
    values=jnp.array([
        [0.00, 0.00, 0.00, 0.00],
        [0.00, 0.01, 0.02, 0.00],
        [0.00, 0.02, 0.05, 0.01],
        [0.01, 0.05, 0.10, 0.02],
    ]),
)

observer = PrepaymentSurfaceObserver(
    surface=surface,
    fixed_market_rate=0.03,
    prepayment_cycle="6M",
)
```

### DepositTransactionObserver
Models deposit inflows/outflows for UMP contracts by contract ID and date.

```python
from jactus.observers.deposit_transaction import DepositTransactionObserver
from jactus.core import ActusDateTime

observer = DepositTransactionObserver(
    transactions={
        "DEP-001": [
            (ActusDateTime(2024, 1, 1), 1000.0),   # deposit
            (ActusDateTime(2024, 6, 1), -500.0),    # withdrawal
        ],
    },
)
```

## Using with simulate()

```python
contract.simulate(behavior_observers=[observer])
# Or via Scenario:
from jactus.observers.scenario import Scenario
scenario = Scenario(
    scenario_id="base",
    market_observers={"rates": market_obs},
    behavior_observers={"prepayment": prepay_obs},
)
contract.simulate(scenario=scenario)
```
""",
        },
        "scenario": {
            "title": "Scenario Management",
            "content": """
# Scenario Management

A `Scenario` bundles market observers and behavioral observers into a named,
reusable simulation configuration. This enables easy scenario comparison
(base case vs. stress) and consistent grouping of market data with behavioral models.

## Creating a Scenario

```python
from jactus.observers.scenario import Scenario
from jactus.observers import TimeSeriesRiskFactorObserver
from jactus.observers.prepayment import PrepaymentSurfaceObserver

scenario = Scenario(
    scenario_id="base-case",
    description="Base case with moderate prepayment",
    market_observers={
        "rates": TimeSeriesRiskFactorObserver({
            "UST-5Y": [
                (ActusDateTime(2024, 1, 1), 0.045),
                (ActusDateTime(2025, 1, 1), 0.035),
            ],
        }),
    },
    behavior_observers={
        "prepayment": PrepaymentSurfaceObserver(...),
    },
)
```

## Using a Scenario

```python
# Pass to simulate — the scenario provides both market and behavioral observers
contract.simulate(scenario=scenario)

# Access the unified market observer
observer = scenario.get_observer()

# Collect callout events
events = scenario.get_callout_events(attributes)
```

## Scenario Comparison

```python
base = Scenario(scenario_id="base", market_observers={...}, behavior_observers={...})
stress = Scenario(scenario_id="stress", market_observers={...}, behavior_observers={...})

base_result = contract.simulate(scenario=base)
stress_result = contract.simulate(scenario=stress)
```

## Key Methods

- `get_observer()` — returns unified `CompositeRiskFactorObserver` for all market observers
- `get_callout_events(attributes)` — collects callout events from all behavioral observers
- `add_market_observer(id, obs)` — add/replace a market observer
- `add_behavior_observer(id, obs)` — add/replace a behavioral observer
- `list_risk_factors()` — list all configured observer sources
""",
        },
        "jax": {
            "title": "JAX Integration",
            "content": """
# JAX in JACTUS

JACTUS uses JAX for high-performance computing:

1. **Arrays**: All state variables use jnp.ndarray
2. **JIT**: Compile hot paths with @jax.jit
3. **Grad**: Automatic differentiation for risk metrics
4. **Vmap**: Vectorize across scenarios

## Example: Computing Sensitivity

```python
import jax

def contract_npv(rate):
    attrs = ContractAttributes(..., nominal_interest_rate=rate)
    contract = create_contract(attrs, rf_obs)
    result = contract.simulate()
    return compute_npv(result.events)

# Compute dNPV/dRate
sensitivity = jax.grad(contract_npv)(0.05)
```

See docs/ARCHITECTURE.md for more details.
""",
        },
        "events": {
            "title": "ACTUS Event Types",
            "content": """
# Contract Events

ACTUS contracts generate events throughout their lifecycle:

**Common Events:**
- IED: Initial Exchange Date (inception)
- IP: Interest Payment
- PR: Principal Redemption
- MD: Maturity Date
- RR: Rate Reset
- FP: Fee Payment

**Event Flow:**
1. Contract generates event schedule
2. Lifecycle engine processes each event
3. Payoff Function (POF) calculates cash flow
4. State Transition Function (STF) updates state

Use `jactus_get_event_types` to see all event types.
""",
        },
        "attributes": {
            "title": "Contract Attributes",
            "content": """
# Contract Attributes

ContractAttributes defines all contract terms:

**Required Fields (Pydantic-enforced):**
- contract_id: str - Unique identifier
- contract_type: ContractType enum
- status_date: ActusDateTime
- contract_role: ContractRole (RPA, RPL, RFL, PFL, BUY, SEL, LG, ST, etc.)

**Common Fields:**
- initial_exchange_date: Contract inception
- maturity_date: Contract maturity
- notional_principal: Principal amount
- nominal_interest_rate: Interest rate
- currency: ISO currency code (default: USD)

**Conventions:**
- day_count_convention: AA, A360, A365, E30360ISDA, E30360, B30360, BUS252
- business_day_convention: NULL, SCF, SCMF, CSF, CSMF, SCP, SCMP, CSP, CSMP
- end_of_month_convention: EOM, SD
- calendar: NO_CALENDAR, MONDAY_TO_FRIDAY, TARGET, US_NYSE, UK_SETTLEMENT

Use `jactus_get_contract_schema(contract_type)` for specific requirements.
""",
        },
    }

    if topic.lower() in guides:
        return guides[topic.lower()]

    return {
        "error": f"No guide available for topic: {topic}",
        "available_topics": list(guides.keys()),
        "hint": "Use jactus_search_docs for custom searches",
    }
