"""Cross-validation of CLM contract against official ACTUS test cases."""

from __future__ import annotations

from .runner import run_single_test


class TestCLMCrossValidation:
    """Cross-validate CLM (Call Money) contract against official ACTUS test cases."""

    def test_clm_test_cases_available(self, clm_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(clm_test_cases) > 0, "No CLM test cases found"

    def test_clm_all_cases_summary(self, clm_test_cases):
        """Run all CLM test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(clm_test_cases.keys()):
            results[test_id] = run_single_test(test_id, clm_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"CLM Cross-Validation: {passed}/{total} passed"]
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
