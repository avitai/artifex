from __future__ import annotations

from dataclasses import dataclass

import pytest

from artifex.generative_models.core import device_testing as device_testing_module


@dataclass(frozen=True, slots=True)
class _FakeCheck:
    name: str
    severity: device_testing_module.TestSeverity
    passed: bool

    def run(self, device_manager: object) -> tuple[bool, str | None, dict[str, object]]:
        del device_manager
        error = None if self.passed else f"{self.name} failed"
        return self.passed, error, {"check": self.name}


def test_run_device_tests_stops_after_critical_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Critical failures should stop the suite immediately."""
    checks = (
        _FakeCheck("critical", device_testing_module.TestSeverity.CRITICAL, False),
        _FakeCheck("important", device_testing_module.TestSeverity.IMPORTANT, True),
    )
    monkeypatch.setattr(device_testing_module, "_default_checks", lambda: checks)

    suite = device_testing_module.run_device_tests(device_manager=object())

    assert suite.name == "Device Diagnostics"
    assert [result.test_name for result in suite.results] == ["critical"]
    assert suite.is_healthy is False
    assert suite.critical_failures[0].error_message == "critical failed"


def test_run_device_tests_critical_only_filters_optional_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Critical-only mode should run only critical checks."""
    checks = (
        _FakeCheck("critical", device_testing_module.TestSeverity.CRITICAL, True),
        _FakeCheck("important", device_testing_module.TestSeverity.IMPORTANT, True),
    )
    monkeypatch.setattr(device_testing_module, "_default_checks", lambda: checks)

    suite = device_testing_module.run_device_tests(
        device_manager=object(),
        critical_only=True,
    )

    assert [result.test_name for result in suite.results] == ["critical"]
    assert suite.is_healthy is True


def test_test_suite_is_immutable() -> None:
    """Diagnostics results should be returned as immutable tuples."""
    result = device_testing_module.TestResult(
        test_name="basic",
        passed=True,
        severity=device_testing_module.TestSeverity.CRITICAL,
        execution_time=0.1,
        metadata={"kind": "basic"},
    )
    suite = device_testing_module.TestSuite(name="Device Diagnostics", results=(result,))

    assert isinstance(suite.results, tuple)
    with pytest.raises(TypeError):
        result.metadata["extra"] = "value"  # type: ignore[index]
    with pytest.raises(AttributeError):
        suite.results.append(result)  # type: ignore[attr-defined]
