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

    for doc_file in doc_files:
        if not doc_file.exists():
            continue

        try:
            content = doc_file.read_text()
            lines = content.split("\n")

            # Search for query in content
            matches = []
            for i, line in enumerate(lines):
                if query_lower in line.lower():
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
