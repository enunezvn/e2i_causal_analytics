"""Integration test: --enable-evaluator populates verdict.evaluator_audit.

Marked live_lm — skipped in CI without ANTHROPIC_API_KEY. Exercises the
audit-eval wiring end-to-end on a small subset of the literature golden set.

The test verifies the env-var-before-import contract introduced by the
--enable-evaluator flag: ADAPTIVE_VALIDITY_EVALUATOR_ENABLED=1 must be set
before the classifier loader is imported so classify_feature attaches
evaluator_audit to each verdict, making the gated subset non-empty.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _live_lm_available() -> bool:
    """Match the project convention from test_miprov2_reproducibility.py:
    check key SHAPE, not just truthiness. CI may set ANTHROPIC_API_KEY to a
    placeholder value (any non-empty string) which would otherwise bypass a
    truthiness-only skip and try to hit Anthropic with a bogus key — that
    burns wall time, can crash the pytest worker, and surfaces as a failure
    rather than a skip. Project memory: feedback_live_lm_skip_must_check_key_shape.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return key.startswith("sk-ant-")


@pytest.mark.live_lm
@pytest.mark.skipif(
    not _live_lm_available(),
    reason=(
        "live_lm test requires a real sk-ant-* ANTHROPIC_API_KEY. Skipped "
        "in CI without one. Project convention: check key shape, not just "
        "truthiness, because CI may set the var to a placeholder value."
    ),
)
def test_enable_evaluator_flag_populates_evaluator_audit(tmp_path: Path) -> None:
    """--enable-evaluator + ANTHROPIC_API_KEY → gated subset non-empty."""

    project_root = Path(__file__).resolve().parents[2]
    full_path = project_root / "tests" / "fixtures" / "causal_role_golden_set.json"
    full = json.loads(full_path.read_text())

    # Build a 3-entry instrument subset (small enough for fast CI-like runs)
    entries = [e for e in full["entries"] if e.get("ground_truth_role") == "instrument"][:3]
    assert len(entries) == 3, (
        "literature golden set must contain at least 3 instrument entries; "
        f"found {len(entries)} with ground_truth_role='instrument'"
    )

    subset = {"schema_version": full.get("schema_version", 1), "entries": entries}
    subset_path = tmp_path / "golden_subset.json"
    subset_path.write_text(json.dumps(subset))

    report_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "measure_layer4_precision.py"),
            "--enable-evaluator",
            "--evaluator-gate",
            "both",
            "--golden-set",
            str(subset_path),
            "--report-path",
            str(report_path),
            "--threshold",
            "0.0",
        ],
        capture_output=True,
        text=True,
        timeout=240,
        cwd=str(project_root),
    )

    assert result.returncode == 0, (
        f"measure script exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert report_path.exists(), "report JSON was not written"
    report = json.loads(report_path.read_text())

    # Report structure: {"overall": {"gated": {..., "n_evaluated": N}, "ungated": {...}}, ...}
    overall = report.get("overall", {})
    gated = overall.get("gated", {})
    gated_n_evaluated = gated.get("n_evaluated", 0)

    assert gated_n_evaluated > 0, (
        "gated subset is empty even with --enable-evaluator — audit evaluator "
        "did not populate evaluator_audit. Check ANTHROPIC_API_KEY and "
        "ADAPTIVE_VALIDITY_EVALUATOR_ENABLED env var propagation.\n"
        f"Full report:\n{json.dumps(report, indent=2)}"
    )
