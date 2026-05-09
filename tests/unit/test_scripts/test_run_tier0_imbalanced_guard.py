"""Unit tests for the `--imbalanced × scenario_*` CLI guard.

Pins backlog #21.7 (codex pass-2 follow-up): `generate_sample_data:1469`
short-circuits to `_scenario_to_dataframe` BEFORE the relabel block at
lines 1494-1506, so `--imbalanced RATIO` is silently dropped under any
`--regime scenario_*`. Discovered empirically during plan Phase 3.3
contrast (conditions A and C produced bit-identical metrics for seed=42).

The CLI guard at `scripts/run_tier0_test.py:6237-6262` errors out at the
argparse boundary so operators see a clear message + redirect to either
`--regime scenario_a_balanced` (if they wanted prevalence=0.50 with intact
signal) or a legacy regime (default/adverse/clean) for post-hoc relabel.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "run_tier0_test.py"


@pytest.mark.parametrize(
    "scenario_regime",
    ["scenario_a", "scenario_a_balanced", "scenario_b", "scenario_c"],
)
def test_imbalanced_with_scenario_regime_errors(scenario_regime: str) -> None:
    """`--imbalanced 0.50 --regime scenario_*` exits 2 with a clear message."""
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--regime",
            scenario_regime,
            "--imbalanced",
            "0.50",
            "--no-bentoml",
            "--no-save",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2, (
        f"Expected argparse exit 2 for --imbalanced 0.50 --regime "
        f"{scenario_regime}, got {result.returncode}; "
        f"stderr (truncated): {result.stderr[-500:]!r}"
    )
    err = result.stderr
    assert "--imbalanced" in err and "silently ignored" in err, (
        f"Error message lacks expected pointers for {scenario_regime}; got:\n{err}"
    )
    # Source-line reference must point at the actual short-circuit
    assert "generate_sample_data:1469" in err, (
        f"Error message must cite generate_sample_data:1469 for {scenario_regime}; got:\n{err}"
    )


def test_imbalanced_0_50_redirects_to_scenario_a_balanced() -> None:
    """At ratio=0.50, the guard recommends `scenario_a_balanced`."""
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--regime",
            "scenario_a",
            "--imbalanced",
            "0.50",
            "--no-bentoml",
            "--no-save",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    assert "scenario_a_balanced" in result.stderr, (
        f"At --imbalanced 0.50 the guard should redirect to "
        f"scenario_a_balanced; got:\n{result.stderr}"
    )


def test_imbalanced_non_half_redirects_to_legacy_regimes() -> None:
    """At ratio≠0.50, the guard redirects to legacy regimes (no balanced match)."""
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--regime",
            "scenario_a",
            "--imbalanced",
            "0.30",
            "--no-bentoml",
            "--no-save",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    err = result.stderr
    assert "default/adverse/clean" in err, (
        f"At --imbalanced 0.30 the guard should redirect to legacy regimes; got:\n{err}"
    )
    # Extract only the parser.error line (last non-empty line of stderr) —
    # argparse usage banner contains "scenario_a_balanced" as a choice
    # listing, but the error message itself should NOT recommend it.
    error_line = next(line for line in reversed(err.splitlines()) if line.strip())
    assert "scenario_a_balanced" not in error_line, (
        f"Guard error line should not recommend scenario_a_balanced for "
        f"non-0.50 ratio; got:\n{error_line}"
    )


def test_imbalanced_with_legacy_regime_passes_argparse() -> None:
    """`--imbalanced 0.50 --regime default` passes argparse (uses --dry-run).

    The guard must NOT fire for legacy regimes that DO honor the flag via
    the relabel block at lines 1494-1506 in `generate_sample_data`.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--regime",
            "default",
            "--imbalanced",
            "0.50",
            "--dry-run",
            "--no-bentoml",
            "--no-save",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"Expected exit 0 for --imbalanced 0.50 --regime default --dry-run, "
        f"got {result.returncode}; stderr (truncated): {result.stderr[-500:]!r}"
    )


def test_no_imbalanced_with_scenario_regime_passes() -> None:
    """`--regime scenario_a` (no `--imbalanced`) passes argparse cleanly."""
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--regime",
            "scenario_a",
            "--dry-run",
            "--no-bentoml",
            "--no-save",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"Expected exit 0 for --regime scenario_a --dry-run, "
        f"got {result.returncode}; stderr (truncated): {result.stderr[-500:]!r}"
    )
