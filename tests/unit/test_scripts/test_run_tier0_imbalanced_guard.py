"""Unit tests for the `--imbalanced × scenario_*` CLI guard + function guard.

Pins backlog #21.7 (codex pass-2 follow-up): `generate_sample_data:1579`
short-circuits to `_scenario_to_dataframe` BEFORE the relabel block at
lines 1609-1621, so `--imbalanced RATIO` is silently dropped under any
`--regime scenario_*`. Discovered empirically during plan Phase 3.3
contrast (conditions A and C produced bit-identical metrics for seed=42).

Two layers of defense covered here:

1. CLI guard at `scripts/run_tier0_test.py:7168-7192` errors out at the
   argparse boundary so operators see a clear message + redirect to either
   `--regime scenario_a_balanced` (if they wanted prevalence=0.50 with intact
   signal) or a legacy regime (default/adverse/clean) for post-hoc relabel.

2. Function-level guard in `_scenario_to_dataframe` (issue #195): mirrors
   the CLI redirect at the function boundary so a programmatic caller
   bypassing argparse (e.g. an interactive session importing
   `generate_sample_data`) ALSO cannot silently drop the ratio.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "run_tier0_test.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_tier0_test import (  # noqa: E402
    _scenario_to_dataframe,
    generate_sample_data,
)


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
    assert "generate_sample_data:1579" in err, (
        f"Error message must cite generate_sample_data:1579 for {scenario_regime}; got:\n{err}"
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
    the relabel block at lines 1609-1621 in `generate_sample_data`.
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


# ---------------------------------------------------------------------------
# Function-level defense-in-depth (issue #195 / backlog #21.7)
#
# A programmatic caller bypassing argparse (e.g. an interactive session that
# imports `generate_sample_data` directly) would have silently dropped the
# imbalance_ratio under a scenario regime, because `_scenario_to_dataframe`
# accepted no such kwarg and the dispatch in `generate_sample_data` returned
# BEFORE the relabel block. The fix threads `imbalance_ratio` through and
# raises ValueError immediately when it is non-None under a scenario regime.
# ---------------------------------------------------------------------------


class TestScenarioToDataframeImbalanceGuard:
    """Function-level mirror of the CLI guard. See module docstring layer #2."""

    @pytest.mark.parametrize(
        "scenario_regime",
        ["scenario_a", "scenario_a_balanced", "scenario_b", "scenario_c"],
    )
    def test_direct_call_with_ratio_raises(self, scenario_regime: str) -> None:
        """Direct call with imbalance_ratio set raises ValueError with redirect."""
        with pytest.raises(ValueError) as excinfo:
            _scenario_to_dataframe(scenario_regime, seed=42, imbalance_ratio=0.05)
        msg = str(excinfo.value)
        assert "scenario" in msg.lower(), f"Error must mention 'scenario'; got: {msg!r}"
        assert "#21.7" in msg, f"Error must cite backlog #21.7; got: {msg!r}"
        # ratio=0.05 is non-0.50 → legacy regime redirect
        assert ("scenario_a_balanced" in msg) or ("legacy" in msg.lower()), (
            f"Error must redirect to scenario_a_balanced or legacy regimes; got: {msg!r}"
        )

    def test_direct_call_at_half_redirects_to_balanced(self) -> None:
        """imbalance_ratio == 0.50 specifically recommends scenario_a_balanced."""
        with pytest.raises(ValueError) as excinfo:
            _scenario_to_dataframe("scenario_a", seed=42, imbalance_ratio=0.50)
        assert "scenario_a_balanced" in str(excinfo.value), (
            f"At ratio=0.50 the function guard must redirect to "
            f"scenario_a_balanced; got: {excinfo.value!s}"
        )

    def test_direct_call_without_ratio_succeeds(self) -> None:
        """Backward-compat: imbalance_ratio=None (and omitted) returns a DataFrame."""
        df_explicit = _scenario_to_dataframe("scenario_a", seed=42, imbalance_ratio=None)
        assert isinstance(df_explicit, pd.DataFrame) and len(df_explicit) > 0

        # Default-arg case (kwarg omitted entirely) must also succeed —
        # callers that pre-date issue #195 do not pass the new param.
        df_default = _scenario_to_dataframe("scenario_a", seed=42)
        assert isinstance(df_default, pd.DataFrame) and len(df_default) > 0

    def test_generate_sample_data_threads_ratio_to_scenario_path(self) -> None:
        """End-to-end: `generate_sample_data(_generator=..., imbalance_ratio=...)`
        must propagate the ratio so the in-function guard fires. This pins the
        threading from `generate_sample_data:1579-1585` and prevents a future
        refactor from dropping the kwarg silently again.
        """
        with pytest.raises(ValueError) as excinfo:
            generate_sample_data(_generator="scenario_a", imbalance_ratio=0.05, seed=42)
        assert "#21.7" in str(excinfo.value)

    # ----- codex pass-2 LOW-1: regime-aware redirect at ratio=0.50 -----

    @pytest.mark.parametrize("scenario_regime", ["scenario_b", "scenario_c"])
    def test_direct_call_at_half_scenario_bc_offers_dual_path(self, scenario_regime: str) -> None:
        """scenario_b/c have no balanced variant — at ratio=0.50 the redirect
        must offer BOTH scenario_a_balanced (different DGP, balanced) AND
        legacy regimes (post-hoc relabel). A naive "use scenario_a_balanced"
        would mislead users who picked scenario_b/c for their DGP.
        """
        with pytest.raises(ValueError) as excinfo:
            _scenario_to_dataframe(scenario_regime, seed=42, imbalance_ratio=0.50)
        msg = str(excinfo.value)
        assert "scenario_a_balanced" in msg, (
            f"{scenario_regime} + 0.50 must offer scenario_a_balanced as one option; got: {msg!r}"
        )
        assert "legacy" in msg.lower(), (
            f"{scenario_regime} + 0.50 must also offer legacy regimes as the "
            f"fidelity-preserving option; got: {msg!r}"
        )

    def test_direct_call_at_half_scenario_a_balanced_says_already_balanced(self) -> None:
        """scenario_a_balanced + 0.50 must NOT redirect to itself (the user is
        ALREADY using the balanced regime); the redirect must tell them to
        drop the imbalance_ratio flag instead.
        """
        with pytest.raises(ValueError) as excinfo:
            _scenario_to_dataframe("scenario_a_balanced", seed=42, imbalance_ratio=0.50)
        msg = str(excinfo.value)
        assert "already" in msg.lower(), (
            f"scenario_a_balanced + 0.50 must indicate the regime is already balanced; got: {msg!r}"
        )
        assert "drop" in msg.lower(), (
            f"scenario_a_balanced + 0.50 must tell user to drop imbalance_ratio; got: {msg!r}"
        )

    # ----- codex pass-2 LOW-2: edge-ratio + ordering coverage -----

    @pytest.mark.parametrize("ratio", [0.0, 1.0, -0.5, float("nan")])
    def test_direct_call_with_edge_ratio_raises(self, ratio: float) -> None:
        """Any non-None imbalance_ratio is rejected — including boundary (0.0,
        1.0), negative, and NaN values. The semantic check is `is not None`,
        not value validation: scenarios refuse ALL post-hoc relabel regardless
        of ratio shape.
        """
        with pytest.raises(ValueError) as excinfo:
            _scenario_to_dataframe("scenario_a", seed=42, imbalance_ratio=ratio)
        assert "#21.7" in str(excinfo.value), (
            f"Edge ratio {ratio!r} must still hit the function guard with the "
            f"backlog #21.7 citation; got: {excinfo.value!s}"
        )

    def test_unknown_regime_with_ratio_raises_regime_error_first(self) -> None:
        """When BOTH the regime is invalid AND imbalance_ratio is set, the
        regime-validation ValueError must fire FIRST (it is the first check
        in the function body). Pins the ordering so a future refactor that
        re-orders checks does not silently swallow invalid-regime errors.
        """
        with pytest.raises(ValueError) as excinfo:
            _scenario_to_dataframe("not_a_real_regime", seed=42, imbalance_ratio=0.05)
        assert "unknown synthetic_v2 regime" in str(excinfo.value), (
            f"Regime-validation error must fire before imbalance_ratio guard; "
            f"got: {excinfo.value!s}"
        )


def test_imbalanced_0_50_scenario_b_cli_offers_dual_path() -> None:
    """CLI mirror of codex pass-2 LOW-1: --regime scenario_b + --imbalanced 0.50
    must offer BOTH scenario_a_balanced AND legacy-regime paths in the parser
    error message.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--regime",
            "scenario_b",
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
    err = result.stderr
    # The argparse usage banner contains "scenario_a_balanced" as a regime
    # choice — strip it by looking only at the parser.error line (the last
    # non-empty stderr line).
    error_line = next(line for line in reversed(err.splitlines()) if line.strip())
    assert "scenario_a_balanced" in error_line, (
        f"scenario_b + 0.50 must offer scenario_a_balanced in the error line; got:\n{error_line}"
    )
    assert "legacy" in error_line.lower(), (
        f"scenario_b + 0.50 must also offer legacy regimes in the error line; got:\n{error_line}"
    )
