"""Cross-validation of exotic linear amortization contracts (LAX, UMP) against ACTUS tests."""

from __future__ import annotations

from .runner import run_single_test


class TestLAXCrossValidation:
    """Cross-validate LAX (Exotic Linear Amortizer) contract against official ACTUS test cases."""

    def test_lax_test_cases_available(self, lax_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(lax_test_cases) > 0, "No LAX test cases found"

    def test_lax_all_cases_summary(self, lax_test_cases):
        """Run all LAX test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(lax_test_cases.keys()):
            results[test_id] = run_single_test(test_id, lax_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"LAX Cross-Validation: {passed}/{total} passed"]
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


class TestUMPCrossValidation:
    """Cross-validate UMP (Undefined Maturity Profile) contract against official ACTUS test cases."""

    def test_ump_test_cases_available(self, ump_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(ump_test_cases) > 0, "No UMP test cases found"

    def test_ump_all_cases_summary(self, ump_test_cases):
        """Run all UMP test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(ump_test_cases.keys()):
            results[test_id] = run_single_test(test_id, ump_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"UMP Cross-Validation: {passed}/{total} passed"]
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
