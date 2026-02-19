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
    "pam", "lam", "nam", "ann", "clm", "lax",
    "ump", "csh", "stk", "com",
    "fxout", "optns", "futur", "swppv", "swaps", "capfl",
    "ceg", "cec",
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
