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
    """Legacy regimes (default/adverse/clean) do NOT trigger the auto-sync.

    The auto-sync is gated on ``args.regime in _SCENARIO_REGIME_TO_BRAND`` —
    legacy regimes are NOT in the scenario map, so they fall through and
    leave CONFIG.brand at whatever its module default is. Asserting against
    the scenario-specific brands (Fabhalta/Remibrutinib) keeps this test
    robust to a future CONFIG.brand default rename — what we actually want
    to verify is that the auto-sync DID NOT fire (codex review LOW Q2).
    """
    result = _run_dry("--regime", regime)
    assert result.returncode == 0
    # Extract the brand line from stdout (deterministic format from line 4344).
    brand_lines = [
        line for line in result.stdout.splitlines() if line.strip().startswith("Brand:")
    ]
    assert len(brand_lines) == 1, (
        f"Expected exactly one 'Brand:' line in stdout for legacy --regime "
        f"{regime}; got {len(brand_lines)}. stdout (truncated):\n"
        f"{result.stdout[-1000:]}"
    )
    printed_brand = brand_lines[0].split(":", 1)[1].strip()
    # Auto-sync would have set the brand to a scenario-specific value;
    # legacy regimes must not produce those.
    assert printed_brand not in {"Fabhalta", "Remibrutinib"}, (
        f"--regime {regime} produced brand={printed_brand!r}; the auto-sync "
        f"must NOT fire for legacy regimes. stdout (truncated):\n"
        f"{result.stdout[-1000:]}"
    )


def test_explicit_brand_with_legacy_regime_unchanged() -> None:
    """``--brand X --regime default`` honors --brand for legacy regimes (regression pin)."""
    result = _run_dry("--regime", "default", "--brand", "ExplicitBrand")
    assert result.returncode == 0
    assert "Brand: ExplicitBrand" in result.stdout, (
        "Explicit --brand must work with legacy regimes (no regression); "
        f"got stdout (truncated):\n{result.stdout[-1000:]}"
    )


def test_scenario_regime_maps_have_identical_keysets() -> None:
    """``_SCENARIO_REGIME_TO_NAME`` and ``_SCENARIO_REGIME_TO_BRAND`` keysets match.

    Codex review MEDIUM (2026-05-09): the auto-sync now gates on
    ``_SCENARIO_REGIME_TO_BRAND`` membership and indexes the same dict, so
    the two are symmetrical at the override site. But other code (e.g.,
    ``_scenario_to_dataframe`` at line 1399) still gates on
    ``_SCENARIO_REGIME_TO_NAME`` and indexes ``_SCENARIO_REGIME_TO_BRAND``
    at line 1416. If a future scenario is added to one map but not the
    other, that callsite raises ``KeyError`` instead of fail-loud at
    config time.
    """
    import importlib

    runner = importlib.import_module("scripts.run_tier0_test")
    name_keys = set(runner._SCENARIO_REGIME_TO_NAME.keys())
    brand_keys = set(runner._SCENARIO_REGIME_TO_BRAND.keys())
    assert name_keys == brand_keys, (
        f"Scenario regime maps drifted out of sync. "
        f"Keys only in _SCENARIO_REGIME_TO_NAME: {sorted(name_keys - brand_keys)}; "
        f"keys only in _SCENARIO_REGIME_TO_BRAND: {sorted(brand_keys - name_keys)}. "
        f"Add the missing entry to whichever map is lacking."
    )
