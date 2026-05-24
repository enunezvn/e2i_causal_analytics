"""Freshness guard for the data-sufficiency rollout plan doc.

Per CLAUDE.md REASON-BEFORE-RULES: this guard exists because the rollout doc
silently drifted from merged reality (issue #474). Each assertion ties a
stale-state string to the merged PR that made it stale, so the failure
message points reviewers at the truth.

The guard is intentionally narrow: substring checks on a single curated file.
We deliberately do NOT scan for arbitrary "stale-looking" patterns — that
would over-fit. Each rule below has a documented merge-state justification.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DOC_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "superpowers"
    / "plans"
    / "2026-05-22-data-sufficiency-diagnostics-rollout.md"
)


@pytest.fixture(scope="module")
def doc_text() -> str:
    assert DOC_PATH.exists(), f"rollout plan doc missing at {DOC_PATH}"
    return DOC_PATH.read_text(encoding="utf-8")


# --- Stale-state guards (must NOT appear) -----------------------------------


def test_phase1_not_marked_ci_blocked(doc_text: str) -> None:
    """PR #462 merged 2026-05-23; codex findings fixed in PR #472; CI fixed
    by raising integration-tests timeout 15→20 min in PR #472."""
    assert "CI BLOCKED" not in doc_text, (
        "Phase 1 (PR #462) is merged and its CI blocker was resolved in PR #472; "
        "remove 'CI BLOCKED' from the doc."
    )
    assert "IN PROGRESS" not in doc_text, "No phase is in progress: Phase 0/1/2/3 are all merged."


def test_phase2_not_marked_not_started(doc_text: str) -> None:
    """Phase 2 (learning curve) merged via PR #466 commit 33082359;
    file lives at src/agents/ml_foundation/model_trainer/nodes/learning_curve.py."""
    learning_curve_file = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "agents"
        / "ml_foundation"
        / "model_trainer"
        / "nodes"
        / "learning_curve.py"
    )
    assert learning_curve_file.exists(), (
        "Phase 2 learning_curve.py must exist on disk to justify 'merged' status"
    )
    # The doc must not still mark Phase 2 as not started.
    assert "Phase 2 (PR #463 — learning curve)" not in doc_text
    assert "Phase 2: Post-training learning curve in ModelTrainer (NOT STARTED)" not in doc_text


def test_phase3_not_marked_not_started(doc_text: str) -> None:
    """Phase 3 (synthetic preview) merged via PR #475;
    file lives at src/agents/ml_foundation/data_preparer/adapters/synthetic_preview.py.
    Note: 'PR #464' previously referenced here was reused for an unrelated chore."""
    preview_file = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "agents"
        / "ml_foundation"
        / "data_preparer"
        / "adapters"
        / "synthetic_preview.py"
    )
    assert preview_file.exists(), (
        "Phase 3 synthetic_preview.py must exist on disk to justify 'merged' status"
    )
    assert "PR #464 — Phase 3: Synthetic preview wiring (NOT STARTED)" not in doc_text


def test_no_stale_ci_blocker_followup(doc_text: str) -> None:
    """PR #462 CI failure is resolved (PR #472 raised timeout; PR #476 sharded).
    The 'current blocker' bullet under Open follow-ups must be gone."""
    assert "PR #462 CI failure" not in doc_text
    assert "current blocker" not in doc_text


# --- Positive-presence guards (must appear) ---------------------------------


def test_phase1_marked_merged_with_472(doc_text: str) -> None:
    """Doc must record Phase 1 merge + PR #472 hotfix lineage."""
    assert "PR #462" in doc_text
    assert "PR #472" in doc_text, "Phase 1 codex-findings hotfix lives in PR #472; cite it."


def test_phase2_marked_merged_with_466(doc_text: str) -> None:
    """Phase 2 learning curve actually merged as PR #466 (not #463)."""
    assert "PR #466" in doc_text, (
        "Phase 2 (learning curve) merged via PR #466; cite the real PR number."
    )


def test_phase3_marked_merged_with_475(doc_text: str) -> None:
    """Phase 3 synthetic preview actually merged as PR #475 (not #464)."""
    assert "PR #475" in doc_text, (
        "Phase 3 (synthetic preview) merged via PR #475; cite the real PR number."
    )


def test_ci_timeout_fix_recorded(doc_text: str) -> None:
    """The CI integration-tests timeout bump (15→20 min) is the diagnosis +
    fix for the prior 'CI BLOCKED' state and belongs in the doc as history."""
    # Either the explicit numeric bump or the workflow file name is acceptable.
    assert ("15→20" in doc_text) or ("integration-tests" in doc_text and "timeout" in doc_text), (
        "Record the integration-tests timeout diagnosis + fix from PR #472"
    )


def test_phase2_causal_branch_does_not_misclaim_synthetic_v2(doc_text: str) -> None:
    """Codex caught this: an earlier draft kept the spec sentence
    'Causal v2: uses synthetic_v2 with TRUE_ATE for bootstrap-style CI-width
    estimation' even after the Phase 2 section was flipped to MERGED. The
    merged learning_curve.py uses neither — see _bootstrap_ate_ci_width
    (difference-in-means on the train set). Asserting the false sentence is
    absent prevents future merged-section-with-stale-spec-detail drift."""
    forbidden = "uses `synthetic_v2` with `TRUE_ATE` for bootstrap-style CI-width estimation"
    assert forbidden not in doc_text, (
        "Phase 2 causal branch must describe the merged dim-bootstrap path, "
        "not the rejected synthetic_v2+TRUE_ATE spec."
    )
    # Cross-check against the real implementation: the merged file must
    # contain _bootstrap_ate_ci_width and NOT use TRUE_ATE in this module.
    learning_curve_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "agents"
        / "ml_foundation"
        / "model_trainer"
        / "nodes"
        / "learning_curve.py"
    )
    src = learning_curve_path.read_text(encoding="utf-8")
    assert "_bootstrap_ate_ci_width" in src, "merged Phase 2 must define the bootstrap helper"
    assert "TRUE_ATE" not in src, (
        "merged Phase 2 does not use TRUE_ATE; if it now does, update the doc"
    )
