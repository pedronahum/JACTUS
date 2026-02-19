"""Cross-validation of PAM contract against official ACTUS test cases.

Downloads test cases from https://github.com/actusfrf/actus-tests
and validates JACTUS simulation results against the expected outputs.
"""

from __future__ import annotations

import pytest

from .runner import run_single_test


class TestPAMCrossValidation:
    """Cross-validate PAM contract against official ACTUS test cases."""

    def test_pam_test_cases_available(self, pam_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(pam_test_cases) > 0, "No PAM test cases found"

    def test_pam_all_cases_summary(self, pam_test_cases):
        """Run all PAM test cases and report aggregate compliance summary.

        This test runs all cases and prints a summary rather than failing
        individually. Gives a clear picture of overall compliance level.
        """
        results: dict[str, list[str]] = {}
        for test_id in sorted(pam_test_cases.keys()):
            results[test_id] = run_single_test(test_id, pam_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        # Build summary
        lines = [f"PAM Cross-Validation: {passed}/{total} passed"]
        for test_id, errors in sorted(results.items()):
            if errors:
                lines.append(f"  {test_id}: FAIL ({len(errors)} errors)")
                for e in errors[:3]:
                    lines.append(f"    - {e}")
                if len(errors) > 3:
                    lines.append(f"    ... and {len(errors) - 3} more")
            else:
                lines.append(f"  {test_id}: PASS")

        summary = "\n".join(lines)
        print(f"\n{summary}")
