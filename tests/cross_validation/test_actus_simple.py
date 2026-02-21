"""Cross-validation of simple asset contracts (CSH, STK, COM) against ACTUS tests."""

from __future__ import annotations

from .runner import run_single_test


class TestCSHCrossValidation:
    """Cross-validate CSH (Cash) contract against official ACTUS test cases."""

    def test_csh_test_cases_available(self, csh_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(csh_test_cases) > 0, "No CSH test cases found"

    def test_csh_all_cases_summary(self, csh_test_cases):
        """Run all CSH test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(csh_test_cases.keys()):
            results[test_id] = run_single_test(test_id, csh_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"CSH Cross-Validation: {passed}/{total} passed"]
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


class TestSTKCrossValidation:
    """Cross-validate STK (Stock) contract against official ACTUS test cases."""

    def test_stk_test_cases_available(self, stk_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(stk_test_cases) > 0, "No STK test cases found"

    def test_stk_all_cases_summary(self, stk_test_cases):
        """Run all STK test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(stk_test_cases.keys()):
            results[test_id] = run_single_test(test_id, stk_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"STK Cross-Validation: {passed}/{total} passed"]
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


class TestCOMCrossValidation:
    """Cross-validate COM (Commodity) contract against official ACTUS test cases."""

    def test_com_test_cases_available(self, com_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(com_test_cases) > 0, "No COM test cases found"

    def test_com_all_cases_summary(self, com_test_cases):
        """Run all COM test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(com_test_cases.keys()):
            results[test_id] = run_single_test(test_id, com_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"COM Cross-Validation: {passed}/{total} passed"]
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
