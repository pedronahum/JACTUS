"""Fixtures for ACTUS cross-validation tests.

Downloads and caches official ACTUS test JSON files from
https://github.com/actusfrf/actus-tests
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

# Cache directory for downloaded test files
CACHE_DIR = Path(__file__).parent / "data"
BASE_URL = "https://raw.githubusercontent.com/actusfrf/actus-tests/master/tests"

# Contract types that have official test files
AVAILABLE_TEST_FILES = [
    "pam",
    "lam",
    "nam",
    "ann",
    "clm",
    "lax",
    "ump",
    "csh",
    "stk",
    "com",
    "fxout",
    "optns",
    "futur",
    "swppv",
    "swaps",
    "capfl",
    "ceg",
    "cec",
]


def _download_test_file(contract_type: str) -> Path:
    """Download an ACTUS test file if not cached.

    Args:
        contract_type: Lowercase contract type (e.g., "pam")

    Returns:
        Path to the cached JSON file
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"actus-tests-{contract_type}.json"
    cache_path = CACHE_DIR / filename

    if not cache_path.exists():
        url = f"{BASE_URL}/{filename}"
        try:
            urllib.request.urlretrieve(url, cache_path)
        except Exception as e:
            pytest.skip(f"Could not download {url}: {e}")

    return cache_path


def _load_test_cases(contract_type: str) -> dict:
    """Load test cases for a contract type.

    Args:
        contract_type: Lowercase contract type (e.g., "pam")

    Returns:
        Dictionary of test case ID -> test case data
    """
    path = _download_test_file(contract_type)
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def pam_test_cases():
    """Official ACTUS test cases for PAM contracts."""
    return _load_test_cases("pam")


@pytest.fixture(scope="session")
def lam_test_cases():
    """Official ACTUS test cases for LAM contracts."""
    return _load_test_cases("lam")


@pytest.fixture(scope="session")
def nam_test_cases():
    """Official ACTUS test cases for NAM contracts."""
    return _load_test_cases("nam")


@pytest.fixture(scope="session")
def ann_test_cases():
    """Official ACTUS test cases for ANN contracts."""
    return _load_test_cases("ann")


@pytest.fixture(scope="session")
def clm_test_cases():
    """Official ACTUS test cases for CLM contracts."""
    return _load_test_cases("clm")


@pytest.fixture(scope="session")
def lax_test_cases():
    """Official ACTUS test cases for LAX contracts."""
    return _load_test_cases("lax")


@pytest.fixture(scope="session")
def ump_test_cases():
    """Official ACTUS test cases for UMP contracts."""
    return _load_test_cases("ump")


@pytest.fixture(scope="session")
def csh_test_cases():
    """Official ACTUS test cases for CSH contracts."""
    return _load_test_cases("csh")


@pytest.fixture(scope="session")
def stk_test_cases():
    """Official ACTUS test cases for STK contracts."""
    return _load_test_cases("stk")


@pytest.fixture(scope="session")
def com_test_cases():
    """Official ACTUS test cases for COM contracts."""
    return _load_test_cases("com")


@pytest.fixture(scope="session")
def fxout_test_cases():
    """Official ACTUS test cases for FXOUT contracts."""
    return _load_test_cases("fxout")


@pytest.fixture(scope="session")
def optns_test_cases():
    """Official ACTUS test cases for OPTNS contracts."""
    return _load_test_cases("optns")


@pytest.fixture(scope="session")
def futur_test_cases():
    """Official ACTUS test cases for FUTUR contracts."""
    return _load_test_cases("futur")


@pytest.fixture(scope="session")
def swppv_test_cases():
    """Official ACTUS test cases for SWPPV contracts."""
    return _load_test_cases("swppv")


@pytest.fixture(scope="session")
def swaps_test_cases():
    """Official ACTUS test cases for SWAPS contracts."""
    return _load_test_cases("swaps")


@pytest.fixture(scope="session")
def capfl_test_cases():
    """Official ACTUS test cases for CAPFL contracts."""
    return _load_test_cases("capfl")


@pytest.fixture(scope="session")
def ceg_test_cases():
    """Official ACTUS test cases for CEG contracts."""
    return _load_test_cases("ceg")


@pytest.fixture(scope="session")
def cec_test_cases():
    """Official ACTUS test cases for CEC contracts."""
    return _load_test_cases("cec")
