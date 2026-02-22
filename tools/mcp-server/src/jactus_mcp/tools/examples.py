"""Example retrieval tools."""

import subprocess
import sys
from pathlib import Path
from typing import Any

from ._utils import get_jactus_root


def list_examples() -> dict[str, Any]:
    """List all available code examples in JACTUS.

    Returns:
        Dictionary with Python scripts and Jupyter notebooks.
    """
    jactus_root = get_jactus_root()
    examples_dir = jactus_root / "examples"
    notebooks_dir = examples_dir / "notebooks"

    # Find Python examples
    python_examples = []
    if examples_dir.exists():
        for py_file in examples_dir.glob("*.py"):
            if py_file.name != "__init__.py":
                python_examples.append({
                    "name": py_file.stem,
                    "filename": py_file.name,
                    "path": str(py_file.relative_to(jactus_root)),
                })

    # Find Jupyter notebooks
    notebooks = []
    if notebooks_dir.exists():
        for nb_file in notebooks_dir.glob("*.ipynb"):
            notebooks.append({
                "name": nb_file.stem,
                "filename": nb_file.name,
                "path": str(nb_file.relative_to(jactus_root)),
            })

    return {
        "python_scripts": python_examples,
        "jupyter_notebooks": notebooks,
        "total": len(python_examples) + len(notebooks),
    }


def get_example(example_name: str) -> dict[str, Any]:
    """Retrieve a specific code example.

    Args:
        example_name: Name of the example (with or without .py extension)

    Returns:
        Dictionary with example code and metadata.
    """
    jactus_root = get_jactus_root()
    examples_dir = jactus_root / "examples"

    # Try to find the example file
    example_file = None

    # Try as Python file
    if not example_name.endswith(".py"):
        py_path = examples_dir / f"{example_name}.py"
        if py_path.exists():
            example_file = py_path

    # Try with .py if provided
    if example_file is None:
        py_path = examples_dir / example_name
        if py_path.exists():
            example_file = py_path

    # Try in notebooks directory
    if example_file is None:
        nb_path = examples_dir / "notebooks" / example_name
        if nb_path.exists():
            example_file = nb_path

    if example_file is None:
        available = list_examples()
        return {
            "error": f"Example '{example_name}' not found",
            "available_examples": available,
            "hint": "Use jactus_list_examples to see all available examples",
        }

    try:
        code = example_file.read_text()

        # Extract docstring if available
        docstring = ""
        if code.startswith('"""') or code.startswith("'''"):
            delimiter = '"""' if code.startswith('"""') else "'''"
            end_idx = code.find(delimiter, 3)
            if end_idx != -1:
                docstring = code[3:end_idx].strip()

        return {
            "name": example_name,
            "filename": example_file.name,
            "path": str(example_file.relative_to(jactus_root)),
            "type": "jupyter_notebook" if example_file.suffix == ".ipynb" else "python_script",
            "docstring": docstring,
            "code": code,
            "lines": len(code.split("\n")),
        }

    except Exception as e:
        return {
            "error": f"Failed to read example: {str(e)}",
            "path": str(example_file),
        }


def get_quick_start_example() -> str:
    """Get a simple quick start example for JACTUS.

    Returns:
        Python code for a basic PAM contract.
    """
    return '''"""Quick Start Example - Principal at Maturity (PAM) Loan"""

from jactus.contracts import create_contract
from jactus.core import ContractAttributes, ContractType, ContractRole, ActusDateTime
from jactus.observers import ConstantRiskFactorObserver

# Create a simple PAM loan
# $100,000 loan at 5% interest, 1 year maturity
attrs = ContractAttributes(
    contract_id="LOAN-001",
    contract_type=ContractType.PAM,
    contract_role=ContractRole.RPA,  # We are the lender
    status_date=ActusDateTime(2024, 1, 1),
    initial_exchange_date=ActusDateTime(2024, 1, 15),
    maturity_date=ActusDateTime(2025, 1, 15),
    notional_principal=100_000.0,
    nominal_interest_rate=0.05,  # 5% annual
    interest_payment_cycle="6M",  # Semi-annual interest (IPCL)
    day_count_convention="30E360",
)

# Create risk factor observer
rf_observer = ConstantRiskFactorObserver(constant_value=0.0)

# Create and simulate the contract
contract = create_contract(attrs, rf_observer)
result = contract.simulate()

# Display cash flows
for event in result.events:
    if event.payoff != 0:
        print(f"{event.time}: {event.type.name:4s} ${event.payoff:>10,.2f}")

# Output:
# 2024-01-15: IED  $-100,000.00  (loan disbursement)
# 2024-07-15: IP   $  2,500.00   (6-month interest)
# 2025-01-15: MD   $102,500.00   (principal + final interest)
'''


def run_example(example_name: str) -> dict[str, Any]:
    """Execute a JACTUS example and return its output.

    Args:
        example_name: Name of the example to run

    Returns:
        Dictionary with stdout, stderr, and return code.
    """
    jactus_root = get_jactus_root()
    examples_dir = jactus_root / "examples"

    # Find the example file
    if not example_name.endswith(".py"):
        py_path = examples_dir / f"{example_name}.py"
    else:
        py_path = examples_dir / example_name

    if not py_path.exists():
        return {
            "error": f"Example '{example_name}' not found",
            "available": list_examples(),
        }

    # Security: ensure the resolved path is within the examples directory
    try:
        py_path.resolve().relative_to(examples_dir.resolve())
    except ValueError:
        return {
            "error": "Invalid example path: must be within examples directory",
            "success": False,
        }

    try:
        result = subprocess.run(
            [sys.executable, str(py_path)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(jactus_root),
        )
        return {
            "name": example_name,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr if result.returncode != 0 else None,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "name": example_name,
            "error": "Example execution timed out (30s limit)",
            "success": False,
        }
    except Exception as e:
        return {
            "name": example_name,
            "error": str(e),
            "success": False,
        }
