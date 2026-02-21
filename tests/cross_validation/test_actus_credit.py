"""Cross-validation of credit enhancement contracts (CEG, CEC) against ACTUS tests."""

from __future__ import annotations

from .runner import run_single_test


class TestCEGCrossValidation:
    """Cross-validate CEG (Credit Enhancement Guarantee) contract against official ACTUS test cases."""

    def test_ceg_test_cases_available(self, ceg_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(ceg_test_cases) > 0, "No CEG test cases found"

    def test_ceg_all_cases_summary(self, ceg_test_cases):
        """Run all CEG test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(ceg_test_cases.keys()):
            results[test_id] = run_single_test(test_id, ceg_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"CEG Cross-Validation: {passed}/{total} passed"]
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


class TestCECCrossValidation:
    """Cross-validate CEC (Credit Enhancement Collateral) contract against official ACTUS test cases."""

    def test_cec_test_cases_available(self, cec_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(cec_test_cases) > 0, "No CEC test cases found"

    def test_cec_all_cases_summary(self, cec_test_cases):
        """Run all CEC test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(cec_test_cases.keys()):
            results[test_id] = run_single_test(test_id, cec_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"CEC Cross-Validation: {passed}/{total} passed"]
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
