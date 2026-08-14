"""Promotion metrics carry a classifier-baseline attribution (#1593).

``compute_run_metrics`` is documented as *"the signal any future active-mode
promotion would be judged against"*, and ``compute_threshold_proposals``
compiles the floor retune from ACCUMULATED labeled rows. Both read
``_is_engaged`` — the abstain rule — so both change meaning the moment the
classifier's abstain surface changes. #1593 does exactly that: teaching
DomainMapper the KPI-value-lookup SSOT stops the pipeline abstaining on 46 of
the 54 KPI rows in the #1337 gold set.

A series that silently spans that flip averages two different classifiers. So:

- every emitted metrics dict names the baseline it describes and counts how
  many rows in the window predate it;
- a floor recommendation is withheld unless the labeled set is PROVABLY all on
  the current baseline. Withholding is the fail-closed direction: the retune is
  a promotion-adjacent decision, and pooled-across-flip evidence cannot support
  it. The per-candidate evidence is still computed and returned.
"""

from __future__ import annotations

import json
from datetime import timedelta

from src.tasks.routing_metrics import (
    CLASSIFIER_BASELINE,
    CLASSIFIER_BASELINE_EPOCH,
    compute_run_metrics,
    compute_threshold_proposals,
)

_AFTER = (CLASSIFIER_BASELINE_EPOCH + timedelta(days=3)).isoformat()
_BEFORE = (CLASSIFIER_BASELINE_EPOCH - timedelta(days=3)).isoformat()


def _row(pattern, conf, was_correct, *, created_at=None, used_llm=False, source=None):
    row = {
        "routing_pattern": pattern,
        "confidence": conf,
        "used_llm_layer": used_llm,
        "was_correct": was_correct,
        "feedback_notes": json.dumps({"source": source}) if source else None,
    }
    if created_at is not None:
        row["created_at"] = created_at
    return row


# Same shape as the Phase-3 fixture: two rows in the (0.4, 0.5) band, both
# judged-correct, so lowering the floor to 0.40 is the profitable proposal.
def _prop_rows(created_at):
    return [
        _row("SINGLE_AGENT", 0.9, True, created_at=created_at),
        _row("SINGLE_AGENT", 0.7, True, created_at=created_at),
        _row("SINGLE_AGENT", 0.45, True, created_at=created_at),
        _row("TOOL_COMPOSER", 0.42, True, created_at=created_at),
        _row("SINGLE_AGENT", 0.8, False, created_at=created_at),
        _row("CLARIFICATION_NEEDED", 0.0, True, created_at=created_at),
    ]


class TestBaselineConstants:
    def test_epoch_is_timezone_aware(self):
        assert CLASSIFIER_BASELINE_EPOCH.tzinfo is not None

    def test_baseline_version_is_a_nonempty_slug(self):
        assert isinstance(CLASSIFIER_BASELINE, str) and CLASSIFIER_BASELINE.strip()

    def test_epoch_is_a_utc_day_boundary_after_the_deploy_day(self):
        """Rounded UP, so a same-day PRE-deploy row is never credited to the new
        classifier — that would fail OPEN on a promotion signal. Mis-attributing
        a same-day POST-deploy row as prior only withholds (codex iter-1)."""
        assert CLASSIFIER_BASELINE_EPOCH.utcoffset().total_seconds() == 0
        assert (
            CLASSIFIER_BASELINE_EPOCH.hour,
            CLASSIFIER_BASELINE_EPOCH.minute,
            CLASSIFIER_BASELINE_EPOCH.second,
        ) == (0, 0, 0)
        assert CLASSIFIER_BASELINE_EPOCH.date().isoformat() > CLASSIFIER_BASELINE[:10]


class TestRunMetricsAttribution:
    def test_metrics_name_the_baseline_they_describe(self):
        m = compute_run_metrics([_row("SINGLE_AGENT", 0.9, True, created_at=_AFTER)])
        assert m["classifier_baseline"]["version"] == CLASSIFIER_BASELINE
        assert m["classifier_baseline"]["epoch"] == CLASSIFIER_BASELINE_EPOCH.isoformat()

    def test_window_entirely_on_current_baseline(self):
        m = compute_run_metrics([_row("SINGLE_AGENT", 0.9, True, created_at=_AFTER)] * 3)
        cb = m["classifier_baseline"]
        assert (cb["rows_current"], cb["rows_prior"], cb["rows_undated"]) == (3, 0, 0)
        assert cb["mixed"] is False

    def test_window_spanning_the_epoch_is_flagged_mixed(self):
        m = compute_run_metrics(
            [
                _row("SINGLE_AGENT", 0.9, True, created_at=_BEFORE),
                _row("SINGLE_AGENT", 0.9, True, created_at=_BEFORE),
                _row("SINGLE_AGENT", 0.9, True, created_at=_AFTER),
            ]
        )
        cb = m["classifier_baseline"]
        assert (cb["rows_current"], cb["rows_prior"]) == (1, 2)
        assert cb["mixed"] is True

    def test_pre_epoch_only_window_is_not_mixed_but_is_prior(self):
        """A window fully BEFORE the flip is internally consistent — it just
        describes the old classifier, which the version stamp makes explicit."""
        m = compute_run_metrics([_row("SINGLE_AGENT", 0.9, True, created_at=_BEFORE)] * 2)
        cb = m["classifier_baseline"]
        assert (cb["rows_current"], cb["rows_prior"], cb["mixed"]) == (0, 2, False)

    def test_undated_rows_are_counted_not_guessed(self):
        m = compute_run_metrics([_row("SINGLE_AGENT", 0.9, True)] * 2)
        cb = m["classifier_baseline"]
        assert (cb["rows_current"], cb["rows_prior"], cb["rows_undated"]) == (0, 0, 2)

    def test_unparseable_timestamp_counts_as_undated(self):
        m = compute_run_metrics([_row("SINGLE_AGENT", 0.9, True, created_at="not-a-date")])
        assert m["classifier_baseline"]["rows_undated"] == 1

    def test_telemetry_still_emits_across_the_flip(self):
        """Attribution annotates; it must never suppress the safety telemetry."""
        m = compute_run_metrics(
            [
                _row("SINGLE_AGENT", 0.9, True, created_at=_BEFORE),
                _row("SINGLE_AGENT", 0.9, False, created_at=_AFTER),
            ]
        )
        assert m["total"] == 2
        assert m["overall_accuracy_pct"] == 50.0
        assert m["engagement_rate"] == 1.0

    def test_empty_window_is_safe(self):
        cb = compute_run_metrics([])["classifier_baseline"]
        assert (cb["rows_current"], cb["rows_prior"], cb["mixed"]) == (0, 0, False)


class TestThresholdProposalsBaselineGuard:
    def test_recommendation_stands_on_a_single_baseline(self):
        out = compute_threshold_proposals(
            _prop_rows(_AFTER), current_floor=0.5, candidates=[0.40], min_evidence=2
        )
        assert out["recommended_floor"] == 0.40
        assert out["classifier_baseline"]["mixed"] is False

    def test_rows_spanning_the_flip_withhold_the_recommendation(self):
        rows = _prop_rows(_AFTER) + [_row("SINGLE_AGENT", 0.44, True, created_at=_BEFORE)]
        out = compute_threshold_proposals(
            rows, current_floor=0.5, candidates=[0.40], min_evidence=2
        )
        assert out["classifier_baseline"]["mixed"] is True
        assert out["recommended_floor"] is None

    def test_pre_epoch_only_rows_withhold_the_recommendation(self):
        """Not 'mixed', but the evidence describes the OLD classifier — it
        cannot justify a floor for the one now running."""
        out = compute_threshold_proposals(
            _prop_rows(_BEFORE), current_floor=0.5, candidates=[0.40], min_evidence=2
        )
        assert out["classifier_baseline"]["mixed"] is False
        assert out["recommended_floor"] is None

    def test_undated_rows_withhold_the_recommendation(self):
        out = compute_threshold_proposals(
            _prop_rows(None), current_floor=0.5, candidates=[0.40], min_evidence=2
        )
        assert out["recommended_floor"] is None

    def test_evidence_is_still_computed_when_withheld(self):
        """Withholding must not blind the reviewer — the per-candidate maths is
        exactly what a human needs to judge the mixed window."""
        out = compute_threshold_proposals(
            _prop_rows(_BEFORE), current_floor=0.5, candidates=[0.40], min_evidence=2
        )
        cand = out["candidates"][0]
        assert cand["labeled_flips"] == 2
        assert cand["flips_judged_correct"] == 2
        assert cand["engaged_accuracy_pct"] == 80.0
        assert out["baseline_engaged_n"] == 3

    def test_withholding_reason_is_stated(self):
        out = compute_threshold_proposals(
            _prop_rows(_BEFORE), current_floor=0.5, candidates=[0.40], min_evidence=2
        )
        assert "baseline" in out["note"].lower()
