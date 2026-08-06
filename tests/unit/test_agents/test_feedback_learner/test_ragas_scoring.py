"""RAGAS bundle + combined-score blend (#1487).

The blend has exactly ONE source of truth in Python, and these tests pin it
against the SQL it must agree with: ``calculate_combined_score`` in
``database/ml/022_self_improvement_tables.sql``. The weights are PARSED out of
the migration rather than restated here, so editing either side without the
other turns this file red.

The partial-bundle tests exist because the SQL function ``COALESCE``s a missing
metric to 0 — the exact silent 40%-of-zero blend #1487 forbids. Python must
diverge there, deliberately and visibly.
"""

import copy
import json
import math
import pickle
import re
import warnings
from pathlib import Path
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from src.agents.feedback_learner.ragas_scoring import (
    HEURISTIC_EVALUATION_METHOD,
    RAGAS_BLEND_WEIGHT,
    RAGAS_METRIC_WEIGHTS,
    RUBRIC_BLEND_WEIGHT,
    RagasBundle,
    combined_score,
)

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent.parent.parent
    / "database"
    / "ml"
    / "022_self_improvement_tables.sql"
)


def _sql_function_body() -> str:
    sql = MIGRATION_PATH.read_text()
    start = sql.index("CREATE OR REPLACE FUNCTION calculate_combined_score")
    end = sql.index("$$ LANGUAGE plpgsql IMMUTABLE;", start)
    return sql[start:end]


class TestWeightsPinnedToMigrationSql:
    def test_metric_weights_match_sql_function(self):
        """Python's per-metric weights ARE the migration's, parsed from it."""
        body = _sql_function_body()
        sql_weights = {
            name: float(weight)
            for name, weight in re.findall(
                r"p_ragas_scores->>'(\w+)'\)::float,\s*0\)\s*\*\s*([\d.]+)", body
            )
        }
        assert sql_weights, "failed to parse metric weights out of the migration SQL"
        assert RAGAS_METRIC_WEIGHTS == sql_weights

    def test_blend_weights_match_sql_defaults(self):
        body = _sql_function_body()
        assert f"p_ragas_weight FLOAT DEFAULT {RAGAS_BLEND_WEIGHT}" in body
        assert f"p_rubric_weight FLOAT DEFAULT {RUBRIC_BLEND_WEIGHT}" in body

    def test_metric_weights_sum_to_one(self):
        """A complete bundle must not need renormalising to reach the SQL value."""
        assert math.isclose(sum(RAGAS_METRIC_WEIGHTS.values()), 1.0, abs_tol=1e-12)


class TestRagasWeighted:
    def test_complete_bundle_equals_sql_weighted_sum(self):
        scores = {
            "faithfulness": 0.80,
            "answer_relevancy": 0.40,
            "context_precision": 0.90,
            "context_recall": 0.70,
            "answer_correctness": 0.60,
        }
        expected = (
            0.80 * 0.25 + 0.40 * 0.20 + 0.90 * 0.20 + 0.70 * 0.20 + 0.60 * 0.15
        )  # the SQL expression, transcribed
        assert RagasBundle(scores=scores).weighted == pytest.approx(expected)

    def test_partial_bundle_renormalises_over_measured_metrics(self):
        """The #1485 real-pipeline shape: only faithfulness + answer_relevancy.

        COALESCE-to-zero would cap this bundle's weighted score at 0.45 forever;
        renormalising over the 0.45 of weight that was actually measured keeps it
        on the 0-1 scale the column's CHECK constraint documents.
        """
        bundle = RagasBundle(scores={"faithfulness": 0.524, "answer_relevancy": 0.179})
        expected = (0.524 * 0.25 + 0.179 * 0.20) / (0.25 + 0.20)
        assert bundle.weighted == pytest.approx(expected)
        coalesce_to_zero = 0.524 * 0.25 + 0.179 * 0.20
        assert bundle.weighted != pytest.approx(coalesce_to_zero)

    def test_all_unmeasured_bundle_has_no_aggregate(self):
        """An all-NaN judge run must not produce a plausible-looking number."""
        bundle = RagasBundle(
            scores={"faithfulness": None, "answer_relevancy": None},
            unmeasured_metrics=["faithfulness", "answer_relevancy"],
        )
        assert bundle.weighted is None
        assert bundle.measured == {}

    def test_empty_bundle_has_no_aggregate(self):
        assert RagasBundle(scores={}).weighted is None

    def test_weighted_stays_within_check_constraint_bounds(self):
        """All-1.0 must not float past 1.0 — the column CHECK rejects >1."""
        bundle = RagasBundle(scores=dict.fromkeys(RAGAS_METRIC_WEIGHTS, 1.0))
        assert bundle.weighted is not None
        assert 0.0 <= bundle.weighted <= 1.0


class TestBundleCoverage:
    def test_none_valued_metric_is_unmeasured_not_zero(self):
        bundle = RagasBundle(scores={"faithfulness": 0.8, "answer_relevancy": None})
        assert bundle.measured == {"faithfulness": 0.8}
        assert "answer_relevancy" in bundle.coverage["unmeasured"]

    def test_unmeasured_and_never_evaluated_are_distinguished(self):
        """#1488 NaN'd a metric; #1485 never asks for context_precision at all.

        Both persist as NULL, but only the first is a judge malfunction.
        """
        bundle = RagasBundle(
            scores={"faithfulness": 0.8, "answer_relevancy": None},
            unmeasured_metrics=["answer_relevancy"],
        )
        assert bundle.coverage["measured"] == ["faithfulness"]
        assert bundle.coverage["unmeasured"] == ["answer_relevancy"]
        assert bundle.coverage["not_evaluated"] == [
            "answer_correctness",
            "context_precision",
            "context_recall",
        ]

    def test_measured_weight_reports_the_renormalisation_denominator(self):
        bundle = RagasBundle(scores={"faithfulness": 0.5, "answer_relevancy": 0.5})
        assert bundle.coverage["measured_weight"] == pytest.approx(0.45)

    def test_signal_scores_payload_omits_unmeasured_keys(self):
        """learning_signals.ragas_scores: absence represents absence."""
        bundle = RagasBundle(scores={"faithfulness": 0.8, "answer_relevancy": None})
        assert bundle.as_signal_scores() == {"faithfulness": 0.8}


class TestBundleValidation:
    def test_unknown_metric_name_is_rejected(self):
        """A typo'd key would silently vanish from the weighted sum."""
        with pytest.raises(ValueError, match="unknown RAGAS metric"):
            RagasBundle(scores={"faithfullness": 0.8})

    def test_unknown_unmeasured_metric_name_is_rejected(self):
        with pytest.raises(ValueError, match="unknown RAGAS metric"):
            RagasBundle(scores={}, unmeasured_metrics=["answer_relevance"])

    def test_out_of_range_score_is_rejected(self):
        with pytest.raises(ValueError, match="outside"):
            RagasBundle(scores={"faithfulness": 1.4})

    def test_nan_score_is_rejected(self):
        """#1488 converts a NaN'd metric to None upstream; a NaN reaching here
        means that conversion was bypassed, and it would serialise to invalid
        JSON rather than to NULL."""
        with pytest.raises(ValueError, match="not a measurement"):
            RagasBundle(scores={"faithfulness": float("nan")})

    def test_metric_cannot_be_both_measured_and_unmeasured(self):
        with pytest.raises(ValueError, match="both measured and unmeasured"):
            RagasBundle(scores={"faithfulness": 0.8}, unmeasured_metrics=["faithfulness"])

    def test_heuristic_scored_bundle_is_refused(self):
        """Word-overlap heuristics are not RAGAS scores.

        ``evaluation_results`` has no column that could mark a row heuristic, so
        ``v_ragas_performance_trends`` would average them in as judged scores.
        """
        with pytest.raises(ValueError, match="heuristic"):
            RagasBundle(
                scores={"faithfulness": 0.125},
                evaluation_method=HEURISTIC_EVALUATION_METHOD,
            )

    def test_rubric_vocabulary_heuristic_is_also_refused(self):
        """codex iter-1 F1. There are TWO fallback vocabularies in this repo and
        they are near-anagrams: RAGAS stamps ``fallback_heuristic``, the rubric
        evaluator stamps ``heuristic_fallback``. Matching one exact string let
        the other through — exactly the cross-contamination a #1489 hook-up
        reading the wrong producer's key would commit.
        """
        with pytest.raises(ValueError, match="heuristic"):
            RagasBundle(
                scores={"faithfulness": 0.5},
                evaluation_method="heuristic_fallback",
            )

    def test_future_heuristic_vocabulary_drift_is_refused(self):
        """An exact-match list has to be updated every time a producer invents a
        label. No legitimate judged-path label contains the word."""
        with pytest.raises(ValueError, match="heuristic"):
            RagasBundle(
                scores={"faithfulness": 0.5},
                evaluation_method="ragas_heuristic_v2",
            )

    def test_heuristic_match_is_case_insensitive(self):
        with pytest.raises(ValueError, match="heuristic"):
            RagasBundle(
                scores={"faithfulness": 0.5},
                evaluation_method="Fallback_Heuristic",
            )

    def test_judged_path_labels_are_still_accepted(self):
        """The refusal must not over-reach: None is the judged path's own label
        in the #1485 judge script, and 'llm' is the rubric evaluator's."""
        for method in (None, "llm", "gpt-4o", "ragas"):
            assert RagasBundle(
                scores={"faithfulness": 0.5}, evaluation_method=method
            ).weighted == pytest.approx(0.5)


class TestBundleImmutability:
    """codex iter-1 F3. ``frozen=True`` blocks attribute assignment but NOT item
    mutation of the containers the model stores.

    The consequence is worse than a rejected insert: ``weighted`` clamps a
    poisoned 2.0 to 1.0 (passing the ``ragas_weighted`` CHECK) while
    ``as_signal_scores`` carries the raw 2.0 into the ``ragas_scores`` JSONB,
    which has no CHECK at all — so ``learning_signals`` would silently accept a
    fake-perfect row.
    """

    def test_scores_mapping_cannot_be_mutated_in_place(self):
        bundle = RagasBundle(scores={"faithfulness": 0.5})
        with pytest.raises(TypeError):
            bundle.scores["faithfulness"] = 2.0  # type: ignore[index]

    def test_a_poisoned_score_cannot_reach_the_jsonb_payload(self):
        """The column-level consequence, stated as its own assertion."""
        bundle = RagasBundle(scores={"faithfulness": 0.5})
        with pytest.raises(TypeError):
            bundle.scores["faithfulness"] = 2.0  # type: ignore[index]
        assert bundle.as_signal_scores() == {"faithfulness": 0.5}
        assert bundle.weighted == pytest.approx(0.5)

    def test_unmeasured_metrics_is_stored_as_a_tuple(self):
        bundle = RagasBundle(scores={"faithfulness": None}, unmeasured_metrics=["faithfulness"])
        assert isinstance(bundle.unmeasured_metrics, tuple)
        with pytest.raises(AttributeError):
            bundle.unmeasured_metrics.append("answer_relevancy")  # type: ignore[attr-defined]

    def test_serialising_the_bundle_stays_warning_free(self):
        """Making the mapping read-only must not cost a serializer warning on
        every dump — pydantic builds the serializer for a plain dict and warns
        that a mappingproxy 'may not be as expected'. A caller running under
        ``-W error`` or a filterwarnings=error suite would then break on a
        model_dump this fix had nothing to do with.
        """
        bundle = RagasBundle(scores={"faithfulness": 0.8, "answer_relevancy": None})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            dumped = bundle.model_dump()
            bundle.model_dump_json()
        assert dumped["scores"] == {"faithfulness": 0.8, "answer_relevancy": None}
        assert type(dumped["scores"]) is dict

    def test_a_dumped_bundle_round_trips(self):
        bundle = RagasBundle(
            scores={"faithfulness": 0.8, "answer_relevancy": None},
            unmeasured_metrics=["answer_relevancy"],
        )
        assert RagasBundle.model_validate(bundle.model_dump()).weighted == bundle.weighted


class TestBundleCopySemantics:
    """codex iter-2. Making ``scores`` read-only cost the bundle its copyability.

    ``MappingProxyType`` is not picklable, and pydantic's own ``__deepcopy__``
    copies ``__dict__`` directly rather than going through serialization — so
    all three of these worked before the immutability fix and stopped after it.
    Nothing copies a bundle today, but #1489 hands it to an offline producer,
    and the failure names ``mappingproxy`` with ``RagasBundle`` nowhere in the
    message.
    """

    def _bundle(self) -> RagasBundle:
        return RagasBundle(
            scores={"faithfulness": 0.8, "answer_relevancy": None},
            unmeasured_metrics=["answer_relevancy"],
            evaluation_model="gpt-4o",
            evaluation_duration_ms=2650,
        )

    def test_bundle_survives_deepcopy(self):
        bundle = self._bundle()
        assert dict(copy.deepcopy(bundle).scores) == dict(bundle.scores)

    def test_bundle_survives_a_pickle_round_trip(self):
        bundle = self._bundle()
        assert dict(pickle.loads(pickle.dumps(bundle)).scores) == dict(bundle.scores)

    def test_bundle_survives_model_copy_deep(self):
        bundle = self._bundle()
        assert dict(bundle.model_copy(deep=True).scores) == dict(bundle.scores)

    # --- guards below are green-after, NOT red-first: they pin properties the
    # --- fix must not lose, rather than behaviour it introduces.

    def test_immutability_survives_both_round_trips(self):
        """Guard (green-after). A copy that came back mutable would silently
        undo the F3 fix for every consumer downstream of a copy."""
        for restored in (
            copy.deepcopy(self._bundle()),
            pickle.loads(pickle.dumps(self._bundle())),
            self._bundle().model_copy(deep=True),
        ):
            assert type(restored.scores) is MappingProxyType
            assert isinstance(restored.unmeasured_metrics, tuple)
            with pytest.raises(TypeError):
                restored.scores["faithfulness"] = 2.0  # type: ignore[index]

    def test_equality_holds_across_both_round_trips(self):
        """Guard (green-after)."""
        bundle = self._bundle()
        assert copy.deepcopy(bundle) == bundle
        assert pickle.loads(pickle.dumps(bundle)) == bundle
        assert bundle.model_copy(deep=True) == bundle

    def test_a_tampered_payload_is_refused_on_rebuild(self):
        """Guard (green-after), and the reason this fix routes through
        ``model_validate`` rather than restoring ``__dict__``: an unpickle
        cannot smuggle in a bundle the constructor would have refused."""
        data = self._bundle().model_dump()
        data["scores"]["faithfulness"] = 2.0
        with pytest.raises(ValidationError):
            RagasBundle.model_validate(data)

    def test_the_sanctioned_serialisation_surface_is_json_safe(self):
        """Guard (green-after). ``json.dumps(bundle.scores)`` is NOT supported —
        a mappingproxy is not JSON-serialisable. Consumers serialise via
        ``model_dump``, ``as_signal_scores`` or ``measured``, all of which hand
        back a plain dict; those are the surfaces the writers actually use.
        """
        bundle = self._bundle()
        assert json.loads(json.dumps(bundle.as_signal_scores())) == {"faithfulness": 0.8}
        assert json.loads(json.dumps(bundle.measured)) == {"faithfulness": 0.8}
        assert json.loads(json.dumps(bundle.model_dump()))["scores"] == {
            "faithfulness": 0.8,
            "answer_relevancy": None,
        }


class TestCombinedScore:
    def test_matches_sql_blend_for_a_complete_bundle(self):
        scores = {
            "faithfulness": 0.80,
            "answer_relevancy": 0.40,
            "context_precision": 0.90,
            "context_recall": 0.70,
            "answer_correctness": 0.60,
        }
        rubric_total = 4.0
        ragas_weighted = 0.80 * 0.25 + 0.40 * 0.20 + 0.90 * 0.20 + 0.70 * 0.20 + 0.60 * 0.15
        expected = round(ragas_weighted * 0.4 + ((rubric_total - 1) / 4.0) * 0.6, 4)
        assert combined_score(RagasBundle(scores=scores).weighted, rubric_total) == expected

    def test_rubric_only_row_has_no_combined_score(self):
        """Never a silent 40%-of-zero blend: the column documents a two-half
        score, so a rubric-only row leaves it NULL."""
        assert combined_score(None, 4.0) is None

    def test_ragas_only_row_has_no_combined_score(self):
        assert combined_score(0.8, None) is None

    def test_rounds_half_up_like_postgres_numeric(self):
        """Postgres rounds the numeric CAST of the float (half-away-from-zero);
        Python's round() rounds the binary value and disagrees on genuine ties.

        This input is not synthetic — a 400k-sample sweep of the blend's own
        (ragas_weighted, rubric_total) domain hit this class of disagreement
        within a few hundred draws. Here the exact blend is 0.98435: Postgres
        emits 0.9844, ``round()`` emits 0.9843.
        """
        assert combined_score(0.99413, 4.91132) == 0.9844

    def test_rubric_total_outside_the_one_to_five_scale_is_rejected(self):
        """(total - 1) / 4 is only a normalisation on the rubric's own scale;
        a 0 would produce a negative score the column CHECK rejects."""
        with pytest.raises(ValueError, match="rubric_total"):
            combined_score(0.8, 0.0)

    def test_stays_within_check_constraint_bounds(self):
        value = combined_score(1.0, 5.0)
        assert value == 1.0
