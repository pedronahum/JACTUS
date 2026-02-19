"""Cross-validation of principal contract types (LAM, NAM, ANN) against ACTUS tests.

Downloads test cases from https://github.com/actusfrf/actus-tests
and validates JACTUS simulation results against the expected outputs.
"""

from __future__ import annotations

import pytest

from .runner import run_single_test


class TestLAMCrossValidation:
    """Cross-validate LAM contract against official ACTUS test cases."""

    def test_lam_test_cases_available(self, lam_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(lam_test_cases) > 0, "No LAM test cases found"

    def test_lam_all_cases_summary(self, lam_test_cases):
        """Run all LAM test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(lam_test_cases.keys()):
            results[test_id] = run_single_test(test_id, lam_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"LAM Cross-Validation: {passed}/{total} passed"]
        for test_id, errors in sorted(results.items()):
            if errors:
                lines.append(f"  {test_id}: FAIL ({len(errors)} errors)")
                for e in errors[:3]:
                    lines.append(f"    - {e}")
                if len(errors) > 3:
                    lines.append(f"    ... and {len(errors) - 3} more")
            else:
                lines.append(f"  {test_id}: PASS")

        print(f"\n" + "\n".join(lines))


class TestNAMCrossValidation:
    """Cross-validate NAM contract against official ACTUS test cases."""

    def test_nam_test_cases_available(self, nam_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(nam_test_cases) > 0, "No NAM test cases found"

    def test_nam_all_cases_summary(self, nam_test_cases):
        """Run all NAM test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(nam_test_cases.keys()):
            results[test_id] = run_single_test(test_id, nam_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"NAM Cross-Validation: {passed}/{total} passed"]
        for test_id, errors in sorted(results.items()):
            if errors:
                lines.append(f"  {test_id}: FAIL ({len(errors)} errors)")
                for e in errors[:3]:
                    lines.append(f"    - {e}")
                if len(errors) > 3:
                    lines.append(f"    ... and {len(errors) - 3} more")
            else:
                lines.append(f"  {test_id}: PASS")

        print(f"\n" + "\n".join(lines))


class TestANNCrossValidation:
    """Cross-validate ANN contract against official ACTUS test cases."""

    def test_ann_test_cases_available(self, ann_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(ann_test_cases) > 0, "No ANN test cases found"

    def test_ann_all_cases_summary(self, ann_test_cases):
        """Run all ANN test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(ann_test_cases.keys()):
            results[test_id] = run_single_test(test_id, ann_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"ANN Cross-Validation: {passed}/{total} passed"]
        for test_id, errors in sorted(results.items()):
            if errors:
                lines.append(f"  {test_id}: FAIL ({len(errors)} errors)")
                for e in errors[:3]:
                    lines.append(f"    - {e}")
                if len(errors) > 3:
                    lines.append(f"    ... and {len(errors) - 3} more")
            else:
                lines.append(f"  {test_id}: PASS")

        print(f"\n" + "\n".join(lines))
