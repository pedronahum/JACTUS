# Risk Factor Observers Reference

Risk factor observers provide market data (interest rates, FX rates, commodity prices)
to the JACTUS simulation engine. Every contract simulation requires an observer.

## Built-in Observers

### ConstantRiskFactorObserver

Returns the same constant value for all risk factor queries regardless of identifier
or time.

```python
from jactus.observers import ConstantRiskFactorObserver

# Fixed rate of 5% for all queries
rf = ConstantRiskFactorObserver(constant_value=0.05)

# Zero rate (default) — use for fixed-rate contracts
rf = ConstantRiskFactorObserver(constant_value=0.0)
```

**Use case:** Fixed-rate contracts, simple testing, contracts where the market rate
is irrelevant (e.g., PAM with no rate resets).

### DictRiskFactorObserver

Maps risk factor identifiers to fixed constant values. Returns a specific value
for each identifier.

```python
from jactus.observers import DictRiskFactorObserver

rf = DictRiskFactorObserver({
    "LIBOR-3M": 0.05,
    "EURIBOR-6M": 0.035,
    "USD/EUR": 1.18,
})
```

**Use case:** Contracts with multiple risk factors needing different values
(e.g., multi-currency portfolios, FX contracts).

### TimeSeriesRiskFactorObserver

Maps identifiers to time series of values with interpolation. Supports step
(most recent known value) and linear interpolation.

```python
from jactus.observers import TimeSeriesRiskFactorObserver
from jactus.core import ActusDateTime

rf = TimeSeriesRiskFactorObserver(
    time_series={
        "LIBOR-3M": [
            (ActusDateTime(2024, 1, 1), 0.04),
            (ActusDateTime(2024, 7, 1), 0.045),
            (ActusDateTime(2025, 1, 1), 0.05),
        ]
    },
    interpolation="step",  # or "linear"
)
```

**Use case:** Floating-rate contracts with rate resets (SWPPV, variable-rate PAM/LAM).

**Note:** Step vs linear interpolation only differ when a query date falls between
data points. If rate reset dates align exactly with data points, both modes produce
identical results.

### CurveRiskFactorObserver

Yield/rate curves keyed by tenor for term structure modeling.

```python
from jactus.observers import CurveRiskFactorObserver

rf = CurveRiskFactorObserver(
    curves={
        "USD-SWAP": {
            0.25: 0.04,   # 3M
            0.5: 0.042,   # 6M
            1.0: 0.045,   # 1Y
            5.0: 0.05,    # 5Y
            10.0: 0.052,  # 10Y
        }
    }
)
```

**Use case:** Term structure-dependent pricing, yield curve analysis.

### CompositeRiskFactorObserver

Chains multiple observers with fallback behavior. Queries observers in order
until one returns a value.

```python
from jactus.observers import CompositeRiskFactorObserver

rf = CompositeRiskFactorObserver([
    TimeSeriesRiskFactorObserver(time_series={"LIBOR-3M": [...]}),
    DictRiskFactorObserver({"EURIBOR-6M": 0.035}),
    ConstantRiskFactorObserver(constant_value=0.0),  # fallback
])
```

**Use case:** Complex portfolios with mixed market data sources.

### JaxRiskFactorObserver

Integer-indexed, fully JAX-compatible observer for JIT compilation, automatic
differentiation, and vmap.

```python
from jactus.observers import JaxRiskFactorObserver
import jax.numpy as jnp

rates = jnp.array([0.04, 0.045, 0.05])
rf = JaxRiskFactorObserver(rates=rates)
```

**Use case:** Sensitivity analysis with `jax.grad()`, batch simulation with
`jax.vmap()`, GPU-accelerated portfolio analytics.

## Custom Observer Implementation

Subclass `BaseRiskFactorObserver` to create custom observers:

```python
from jactus.observers.base import BaseRiskFactorObserver


class StochasticRateObserver(BaseRiskFactorObserver):
    """Observer that returns rates from a pre-generated stochastic path."""

    def __init__(self, rate_path: dict[str, list[tuple[float, float]]]):
        self.rate_path = rate_path

    def get_risk_factor(self, identifier: str, time: float) -> float:
        """Return the rate for the given identifier at the given time.

        Args:
            identifier: Risk factor identifier (e.g., "LIBOR-3M")
            time: Time in years from status_date

        Returns:
            Rate value at the given time
        """
        if identifier not in self.rate_path:
            return 0.0
        path = self.rate_path[identifier]
        # Find most recent observation
        value = path[0][1]
        for t, v in path:
            if t <= time:
                value = v
            else:
                break
        return value
```

### Method Signature

The key method to implement is:

```python
def get_risk_factor(self, identifier: str, time: float) -> float:
```

- `identifier` — the market object code (e.g., `"LIBOR-3M"`, `"USD/EUR"`)
- `time` — time in years from the contract's `status_date`
- Returns a `float` value (rate, FX rate, price, etc.)

## Multi-Currency Observer Pattern

For FX contracts (FXOUT), provide both currency pair rates:

```python
from jactus.observers import DictRiskFactorObserver

rf = DictRiskFactorObserver({
    "USD/EUR": 1.18,     # spot rate
    "USD/GBP": 1.27,     # another pair
    "EURIBOR-6M": 0.035, # EUR interest rate
    "SOFR-3M": 0.05,     # USD interest rate
})
```

For time-varying FX scenarios:

```python
from jactus.observers import TimeSeriesRiskFactorObserver
from jactus.core import ActusDateTime

rf = TimeSeriesRiskFactorObserver(
    time_series={
        "USD/EUR": [
            (ActusDateTime(2024, 1, 1), 1.10),
            (ActusDateTime(2024, 4, 1), 1.12),
            (ActusDateTime(2024, 7, 1), 1.15),
            (ActusDateTime(2024, 10, 1), 1.18),
        ],
    },
    interpolation="linear",
)
```

## Behavioral Observers

Behavioral observers inject state-dependent events (prepayments, deposit transactions)
into the simulation timeline.

### PrepaymentSurfaceObserver

Models prepayment behavior as a 2D surface indexed by rate spread and loan age:

```python
from jactus.observers import PrepaymentSurfaceObserver
from jactus.utilities.surface import LabeledSurface2D
import numpy as np

# Create prepayment surface: spread x seasoning -> annual prepayment rate
spreads = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
ages = np.array([0, 12, 24, 36, 48, 60])
rates = np.array([...])  # 5x6 matrix of prepayment rates

surface = LabeledSurface2D(x_labels=spreads, y_labels=ages, values=rates)
observer = PrepaymentSurfaceObserver(prepayment_surface=surface)
```

### DepositTransactionObserver

Models deposit inflows and outflows for UMP (Undefined Maturity Profile) contracts.

### Scenario Management

Bundle market and behavioral observers into reusable scenarios:

```python
from jactus.observers.scenario import Scenario

scenario = Scenario(
    name="base_case",
    risk_factor_observer=rf_observer,
    behavior_observers=[prepayment_observer],
)
result = contract.simulate(scenario=scenario)
```

## Observer Selection Guide

| Scenario | Recommended Observer |
|---|---|
| Simple fixed-rate contract | `ConstantRiskFactorObserver(0.0)` |
| Multiple static market factors | `DictRiskFactorObserver({...})` |
| Floating-rate with resets | `TimeSeriesRiskFactorObserver({...})` |
| Term structure modeling | `CurveRiskFactorObserver({...})` |
| Mixed data sources | `CompositeRiskFactorObserver([...])` |
| JAX autodiff / gradients | `JaxRiskFactorObserver(rates=...)` |
| Mortgage prepayment | `PrepaymentSurfaceObserver(surface=...)` |
| Deposit behavior | `DepositTransactionObserver(...)` |
