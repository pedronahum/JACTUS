"""Cross-validation of swap contracts (SWPPV, SWAPS, CAPFL) against ACTUS tests."""

from __future__ import annotations

from .runner import run_single_test


class TestSWPPVCrossValidation:
    """Cross-validate SWPPV (Plain Vanilla Swap) contract against official ACTUS test cases."""

    def test_swppv_test_cases_available(self, swppv_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(swppv_test_cases) > 0, "No SWPPV test cases found"

    def test_swppv_all_cases_summary(self, swppv_test_cases):
        """Run all SWPPV test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(swppv_test_cases.keys()):
            results[test_id] = run_single_test(test_id, swppv_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"SWPPV Cross-Validation: {passed}/{total} passed"]
        for test_id, errors in sorted(results.items()):
            if errors:
                lines.append(f"  {test_id}: FAIL ({len(errors)} errors)")
                for e in errors[:3]:
                    lines.append(f"    - {e}")
                if len(errors) > 3:
                    lines.append(f"    ... and {len(errors) - 3} more")
            else:
                lines.append(f"  {test_id}: PASS")

        print("\n" + "\n".join(lines))


class TestSWAPSCrossValidation:
    """Cross-validate SWAPS (Swap) contract against official ACTUS test cases."""

    def test_swaps_test_cases_available(self, swaps_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(swaps_test_cases) > 0, "No SWAPS test cases found"

    def test_swaps_all_cases_summary(self, swaps_test_cases):
        """Run all SWAPS test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(swaps_test_cases.keys()):
            results[test_id] = run_single_test(test_id, swaps_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"SWAPS Cross-Validation: {passed}/{total} passed"]
        for test_id, errors in sorted(results.items()):
            if errors:
                lines.append(f"  {test_id}: FAIL ({len(errors)} errors)")
                for e in errors[:3]:
                    lines.append(f"    - {e}")
                if len(errors) > 3:
                    lines.append(f"    ... and {len(errors) - 3} more")
            else:
                lines.append(f"  {test_id}: PASS")

        print("\n" + "\n".join(lines))


class TestCAPFLCrossValidation:
    """Cross-validate CAPFL (Cap/Floor) contract against official ACTUS test cases."""

    def test_capfl_test_cases_available(self, capfl_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(capfl_test_cases) > 0, "No CAPFL test cases found"

    def test_capfl_all_cases_summary(self, capfl_test_cases):
        """Run all CAPFL test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(capfl_test_cases.keys()):
            results[test_id] = run_single_test(test_id, capfl_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"CAPFL Cross-Validation: {passed}/{total} passed"]
        for test_id, errors in sorted(results.items()):
            if errors:
                lines.append(f"  {test_id}: FAIL ({len(errors)} errors)")
                for e in errors[:3]:
                    lines.append(f"    - {e}")
                if len(errors) > 3:
                    lines.append(f"    ... and {len(errors) - 3} more")
            else:
                lines.append(f"  {test_id}: PASS")

        print("\n" + "\n".join(lines))
