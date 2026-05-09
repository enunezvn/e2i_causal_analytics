"""Unit tests for the CONFIG.brand auto-sync against scenario regimes.

Pins backlog #21.5 (ultrareview bug_006): without an explicit ``--brand``,
``CONFIG.brand`` must follow the regime's declared brand for synthetic_v2
scenarios. Pre-fix symptom: ``--regime scenario_b`` wrote
``df["brand"]="Fabhalta"`` while ``CONFIG.brand`` stayed at its default
(``"Kisqali"``), creating data↔metadata divergence in MLflow tags,
``cohort_name``, scope-spec problem description, and ``state["brand"]``
readers throughout the runner (lines 1842, 2034, 2207, 2360, etc.).

The override block at ``scripts/run_tier0_test.py:6270-6277`` resolves this:
- ``args.brand`` truthy → use it (existing behavior, regression-pinned).
- else regime ∈ scenario_* → ``_SCENARIO_REGIME_TO_BRAND[regime]``.
- else (legacy default/adverse/clean) → preserve CONFIG default (no change).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "run_tier0_test.py"


def _run_dry(*extra: str) -> subprocess.CompletedProcess[str]:
    """Run the runner in --dry-run mode and capture stdout."""
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--dry-run",
            "--no-bentoml",
            "--no-save",
            *extra,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )


@pytest.mark.parametrize(
    ("regime", "expected_brand"),
    [
        ("scenario_a", "Kisqali"),
        ("scenario_a_balanced", "Kisqali"),
        ("scenario_b", "Fabhalta"),
        ("scenario_c", "Remibrutinib"),
    ],
)
def test_scenario_regime_auto_syncs_config_brand(regime: str, expected_brand: str) -> None:
    """``--regime scenario_*`` (no ``--brand``) sets ``CONFIG.brand`` to the regime brand."""
    result = _run_dry("--regime", regime)
    assert result.returncode == 0, (
        f"--dry-run with --regime {regime} should exit 0; "
        f"got {result.returncode}; stderr: {result.stderr[-500:]!r}"
    )
    assert f"Brand: {expected_brand}" in result.stdout, (
        f"--regime {regime} (no --brand) should auto-sync CONFIG.brand to "
        f"{expected_brand!r}; got stdout (truncated):\n{result.stdout[-1000:]}"
    )


def test_explicit_brand_overrides_regime_auto_sync() -> None:
    """``--brand competitor --regime scenario_b`` keeps the user's explicit brand."""
    result = _run_dry("--regime", "scenario_b", "--brand", "competitor")
    assert result.returncode == 0
    assert "Brand: competitor" in result.stdout, (
        "Explicit --brand must take precedence over the auto-sync; "
        f"got stdout (truncated):\n{result.stdout[-1000:]}"
    )


@pytest.mark.parametrize("regime", ["default", "adverse", "clean"])
def test_legacy_regime_does_not_auto_sync_brand(regime: str) -> None:
    """Legacy regimes (default/adverse/clean) preserve CONFIG.brand default.

    The auto-sync is gated on ``args.regime in _SCENARIO_REGIME_TO_NAME`` —
    legacy regimes are NOT in the scenario map, so they fall through and
    leave CONFIG.brand at its module-level default (``"Kisqali"``).
    """
    result = _run_dry("--regime", regime)
    assert result.returncode == 0
    assert "Brand: Kisqali" in result.stdout, (
        f"--regime {regime} should preserve CONFIG.brand default Kisqali; "
        f"got stdout (truncated):\n{result.stdout[-1000:]}"
    )


def test_explicit_brand_with_legacy_regime_unchanged() -> None:
    """``--brand X --regime default`` honors --brand for legacy regimes (regression pin)."""
    result = _run_dry("--regime", "default", "--brand", "ExplicitBrand")
    assert result.returncode == 0
    assert "Brand: ExplicitBrand" in result.stdout, (
        "Explicit --brand must work with legacy regimes (no regression); "
        f"got stdout (truncated):\n{result.stdout[-1000:]}"
    )
