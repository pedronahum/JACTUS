"""Cross-validation of derivative contracts (FXOUT, OPTNS, FUTUR) against ACTUS tests."""

from __future__ import annotations

from .runner import run_single_test


class TestFXOUTCrossValidation:
    """Cross-validate FXOUT (Foreign Exchange Outright) contract against official ACTUS test cases."""

    def test_fxout_test_cases_available(self, fxout_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(fxout_test_cases) > 0, "No FXOUT test cases found"

    def test_fxout_all_cases_summary(self, fxout_test_cases):
        """Run all FXOUT test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(fxout_test_cases.keys()):
            results[test_id] = run_single_test(test_id, fxout_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"FXOUT Cross-Validation: {passed}/{total} passed"]
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


class TestOPTNSCrossValidation:
    """Cross-validate OPTNS (Option) contract against official ACTUS test cases."""

    def test_optns_test_cases_available(self, optns_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(optns_test_cases) > 0, "No OPTNS test cases found"

    def test_optns_all_cases_summary(self, optns_test_cases):
        """Run all OPTNS test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(optns_test_cases.keys()):
            results[test_id] = run_single_test(test_id, optns_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"OPTNS Cross-Validation: {passed}/{total} passed"]
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


class TestFUTURCrossValidation:
    """Cross-validate FUTUR (Future) contract against official ACTUS test cases."""

    def test_futur_test_cases_available(self, futur_test_cases):
        """Verify test cases were downloaded successfully."""
        assert len(futur_test_cases) > 0, "No FUTUR test cases found"

    def test_futur_all_cases_summary(self, futur_test_cases):
        """Run all FUTUR test cases and report aggregate compliance summary."""
        results: dict[str, list[str]] = {}
        for test_id in sorted(futur_test_cases.keys()):
            results[test_id] = run_single_test(test_id, futur_test_cases[test_id])

        passed = sum(1 for errs in results.values() if not errs)
        total = len(results)

        lines = [f"FUTUR Cross-Validation: {passed}/{total} passed"]
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
