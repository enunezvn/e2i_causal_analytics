"""Unit tests for `src.data.kg.adversarial_probe`.

Covers prefix-stable vs prefix-drifting derivations, edge cases (empty
inputs, NaN handling, non-numeric features, tolerance), error capture, and
argument-shape validation.

The probe's contract: a derivation that is anchor-aware and well-windowed
returns the same per-patient values whether it sees the full event stream
or each patient's prefix-censored slice. A derivation that pulls
post-anchor data drifts under censoring. The probe surfaces this drift
without needing a separate "observed" feature column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.kg.adversarial_probe import (
    AdversarialProbe,
    AdversarialProbeError,
)
from src.data.kg.types import AdversarialProbeResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def events_with_post_anchor() -> pd.DataFrame:
    """Events for 3 patients, some falling after each patient's anchor.

    Anchor dates (see ``anchors`` fixture):
        patient_1: 2024-01-15
        patient_2: 2024-02-20
        patient_3: 2024-03-10
    """
    return pd.DataFrame(
        {
            "patient_id": [
                "patient_1",
                "patient_1",
                "patient_1",
                "patient_2",
                "patient_2",
                "patient_3",
                "patient_3",
            ],
            "event_date": pd.to_datetime(
                [
                    "2024-01-01",  # patient_1 pre-anchor
                    "2024-01-10",  # patient_1 pre-anchor
                    "2024-02-01",  # patient_1 POST-anchor
                    "2024-02-15",  # patient_2 pre-anchor
                    "2024-03-01",  # patient_2 POST-anchor
                    "2024-03-05",  # patient_3 pre-anchor
                    "2024-03-15",  # patient_3 POST-anchor
                ]
            ),
            "value": [1, 2, 3, 4, 5, 6, 7],
        }
    )


@pytest.fixture
def anchors() -> pd.Series:
    return pd.Series(
        pd.to_datetime(["2024-01-15", "2024-02-20", "2024-03-10"]),
        index=["patient_1", "patient_2", "patient_3"],
        name="anchor_date",
    )


def _windowed_count(window_days: int):
    """Return a derivation that counts events in [anchor-window, anchor].

    The closure captures ``window_days``. Anchor-aware: reads ``anchors``
    and emits only window-bounded counts. Prefix-stable by construction:
    on full events, counts are bounded by the right edge of the window
    (≤ anchor); on prefix-censored events, the same bound applies and
    yields the same counts.
    """

    def _derivation(events: pd.DataFrame, anchors: pd.Series) -> pd.Series:
        window = pd.Timedelta(days=window_days)
        joined = events.merge(
            anchors.rename("_anchor"),
            left_on="patient_id",
            right_index=True,
            how="inner",
        )
        in_window = (joined["event_date"] <= joined["_anchor"]) & (
            joined["event_date"] >= joined["_anchor"] - window
        )
        windowed = joined.loc[in_window]
        counts = windowed.groupby("patient_id").size()
        return counts.reindex(anchors.index, fill_value=0)

    return _derivation


def _unwindowed_count(events: pd.DataFrame, anchors: pd.Series) -> pd.Series:
    """Anchor-IGNORANT derivation that counts all events per patient.

    This is the canonical leak: it pulls post-anchor events because it
    never bounds against the anchor. On full events, it sees more rows
    than the prefix-censored slice; the probe must catch the drift.
    """
    return events.groupby("patient_id").size().reindex(anchors.index, fill_value=0)


# ---------------------------------------------------------------------------
# Happy-path: well-windowed derivation
# ---------------------------------------------------------------------------


class TestPrefixStableDerivation:
    def test_180day_window_unchanged(
        self, events_with_post_anchor: pd.DataFrame, anchors: pd.Series
    ) -> None:
        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="med_count_180d",
            derivation=_windowed_count(180),
            events=events_with_post_anchor,
            anchors=anchors,
        )
        assert isinstance(result, AdversarialProbeResult)
        assert result.outcome == "unchanged"
        assert result.n_rows_compared == 3
        assert result.n_rows_changed == 0
        assert result.fraction_changed == 0.0
        assert result.error is None

    def test_30day_window_unchanged(
        self, events_with_post_anchor: pd.DataFrame, anchors: pd.Series
    ) -> None:
        # Tighter window — still prefix-stable because the window's right
        # edge is the anchor; censoring at the anchor doesn't change which
        # in-window events are visible.
        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="med_count_30d",
            derivation=_windowed_count(30),
            events=events_with_post_anchor,
            anchors=anchors,
        )
        assert result.outcome == "unchanged"


# ---------------------------------------------------------------------------
# Leak detection: anchor-ignorant aggregation
# ---------------------------------------------------------------------------


class TestPrefixDriftingDerivation:
    def test_unwindowed_count_flags_changed(
        self, events_with_post_anchor: pd.DataFrame, anchors: pd.Series
    ) -> None:
        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="all_med_fills",
            derivation=_unwindowed_count,
            events=events_with_post_anchor,
            anchors=anchors,
        )
        assert result.outcome == "changed"
        # All three patients have post-anchor events in the fixture.
        assert result.n_rows_compared == 3
        assert result.n_rows_changed == 3
        assert result.fraction_changed == 1.0
        # Patient 1 drops from 3 to 2 (delta=1); patient 2 drops 2 → 1 (1);
        # patient 3 drops 2 → 1 (1). Max delta = 1.
        assert result.max_abs_change == pytest.approx(1.0)

    def test_partial_leak_partial_unchanged(self) -> None:
        # Patient_1 has post-anchor events; patient_2 doesn't.
        events = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p2", "p2"],
                "event_date": pd.to_datetime(
                    ["2024-01-01", "2024-02-01", "2024-01-05", "2024-01-10"]
                ),
            }
        )
        anchors = pd.Series(
            pd.to_datetime(["2024-01-15", "2024-02-15"]),
            index=["p1", "p2"],
        )
        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="all_visits",
            derivation=_unwindowed_count,
            events=events,
            anchors=anchors,
        )
        assert result.outcome == "changed"
        assert result.n_rows_compared == 2
        assert result.n_rows_changed == 1
        assert result.fraction_changed == 0.5
        assert result.max_abs_change == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Edge cases: empty / NaN inputs
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_anchors_returns_inapplicable(self) -> None:
        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="x",
            derivation=_unwindowed_count,
            events=pd.DataFrame(
                {"patient_id": ["p1"], "event_date": pd.to_datetime(["2024-01-01"])}
            ),
            anchors=pd.Series([], index=pd.Index([], name="patient_id"), dtype="datetime64[ns]"),
        )
        assert result.outcome == "inapplicable"
        assert result.n_rows_compared == 0

    def test_no_events_for_anchored_patients_inapplicable(self) -> None:
        events = pd.DataFrame(
            {
                "patient_id": ["other_patient"],
                "event_date": pd.to_datetime(["2024-01-01"]),
            }
        )
        anchors = pd.Series(pd.to_datetime(["2024-01-15"]), index=["p1"])
        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="x",
            derivation=_unwindowed_count,
            events=events,
            anchors=anchors,
        )
        assert result.outcome == "inapplicable"
        assert any("no events" in n for n in result.notes)

    def test_patient_id_dtype_mismatch_surfaced_in_notes(self) -> None:
        # Codex L5: when events.patient_id has dtype int but anchors.index
        # has dtype str (or vice versa), pandas does NOT coerce, so .isin()
        # silently drops every row. The operator should see the dtype info
        # in the notes so they can spot the cross-type alias case without
        # having to inspect inputs.
        events = pd.DataFrame(
            {
                "patient_id": [1, 2],  # int dtype
                "event_date": pd.to_datetime(["2024-01-01", "2024-01-05"]),
            }
        )
        anchors = pd.Series(
            pd.to_datetime(["2024-01-15", "2024-01-20"]),
            index=["1", "2"],  # str dtype — different from events.patient_id
        )
        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="x",
            derivation=_unwindowed_count,
            events=events,
            anchors=anchors,
        )
        assert result.outcome == "inapplicable"
        # Both dtype labels surfaced for diagnosis.
        joined_notes = " ".join(result.notes)
        assert "patient_id" in joined_notes
        assert "anchors.index.dtype" in joined_notes
        assert "coerce" in joined_notes

    def test_all_nan_anchors_inapplicable(self) -> None:
        anchors = pd.Series([pd.NaT, pd.NaT], index=["p1", "p2"], dtype="datetime64[ns]")
        events = pd.DataFrame(
            {
                "patient_id": ["p1"],
                "event_date": pd.to_datetime(["2024-01-01"]),
            }
        )
        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="x",
            derivation=_unwindowed_count,
            events=events,
            anchors=anchors,
        )
        assert result.outcome == "inapplicable"
        # Both NaN anchors logged before falling through to "no valid anchors".
        assert any("dropped 2" in n for n in result.notes)

    def test_partial_nan_anchors_runs_on_rest(self) -> None:
        anchors = pd.Series(pd.to_datetime(["2024-01-15", pd.NaT]), index=["p1", "p2"])
        events = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p2"],
                "event_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-05"]),
            }
        )
        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="x",
            derivation=_unwindowed_count,
            events=events,
            anchors=anchors,
        )
        # Only p1 was probed (p2 had NaN anchor).
        assert result.n_rows_compared == 1
        assert any("dropped 1" in n for n in result.notes)

    def test_partial_nan_event_dates_filtered(self) -> None:
        events = pd.DataFrame(
            {
                "patient_id": ["p1", "p1", "p1"],
                "event_date": [
                    pd.Timestamp("2024-01-01"),
                    pd.NaT,
                    pd.Timestamp("2024-02-01"),
                ],
            }
        )
        anchors = pd.Series(pd.to_datetime(["2024-01-15"]), index=["p1"])
        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="x",
            derivation=_unwindowed_count,
            events=events,
            anchors=anchors,
        )
        # NaN-date row dropped → patient has 2 valid events, 1 pre, 1 post.
        # Unwindowed count: full=2, prefix=1 → changed.
        assert result.outcome == "changed"
        assert any("NaN" in n and "event_date" in n for n in result.notes)


# ---------------------------------------------------------------------------
# Derivation errors
# ---------------------------------------------------------------------------


class TestDerivationErrors:
    def test_derivation_raises_on_full_captures_error(
        self, events_with_post_anchor: pd.DataFrame, anchors: pd.Series
    ) -> None:
        def _bad(_e: pd.DataFrame, _a: pd.Series) -> pd.Series:
            raise RuntimeError("derivation failed")

        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="x",
            derivation=_bad,
            events=events_with_post_anchor,
            anchors=anchors,
        )
        assert result.outcome == "error"
        assert "derivation failed" in (result.error or "")

    def test_derivation_raises_on_prefix_captures_error(
        self, events_with_post_anchor: pd.DataFrame, anchors: pd.Series
    ) -> None:
        # Raise only when the input is smaller than the full event count.
        full_count = len(events_with_post_anchor)

        def _flaky(events: pd.DataFrame, anchors: pd.Series) -> pd.Series:
            if len(events) < full_count:
                raise RuntimeError("flaky on prefix")
            return events.groupby("patient_id").size().reindex(anchors.index, fill_value=0)

        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="x",
            derivation=_flaky,
            events=events_with_post_anchor,
            anchors=anchors,
        )
        assert result.outcome == "error"
        assert "prefix" in (result.error or "").lower()

    def test_derivation_returns_dict_captures_error(
        self, events_with_post_anchor: pd.DataFrame, anchors: pd.Series
    ) -> None:
        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="x",
            derivation=lambda _e, _a: {"p1": 1, "p2": 2, "p3": 3},
            events=events_with_post_anchor,
            anchors=anchors,
        )
        assert result.outcome == "error"
        assert "Series" in (result.error or "")

    def test_derivation_returns_series_with_unrecognized_index(
        self, events_with_post_anchor: pd.DataFrame, anchors: pd.Series
    ) -> None:
        # Derivation returns a Series whose index is NOT drawn from
        # anchors.index — e.g., a default RangeIndex from a missing
        # ``.reindex(anchors.index)``. Must produce outcome=error so a
        # positional-alignment silent-misprobe can't slip through.
        def _bad_index(events: pd.DataFrame, _a: pd.Series) -> pd.Series:
            return events.groupby("patient_id").size().reset_index(drop=True)

        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="x",
            derivation=_bad_index,
            events=events_with_post_anchor,
            anchors=anchors,
        )
        assert result.outcome == "error"
        assert "anchors.index" in (result.error or "")
        assert "RangeIndex" in (result.error or "")

    def test_derivation_returns_series_with_extra_patients(
        self, events_with_post_anchor: pd.DataFrame, anchors: pd.Series
    ) -> None:
        def _too_many(_events: pd.DataFrame, _a: pd.Series) -> pd.Series:
            return pd.Series(
                [1, 2, 3, 4],
                index=["patient_1", "patient_2", "patient_3", "patient_99"],
            )

        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="x",
            derivation=_too_many,
            events=events_with_post_anchor,
            anchors=anchors,
        )
        assert result.outcome == "error"
        assert "patient_99" in (result.error or "")


# ---------------------------------------------------------------------------
# Argument-shape validation
# ---------------------------------------------------------------------------


class TestArgumentValidation:
    def test_non_callable_derivation_raises(self) -> None:
        probe = AdversarialProbe()
        with pytest.raises(AdversarialProbeError, match="callable"):
            probe.probe(
                feature_name="x",
                derivation="not a function",  # type: ignore[arg-type]
                events=pd.DataFrame(),
                anchors=pd.Series(dtype="datetime64[ns]"),
            )

    def test_non_dataframe_events_raises(self) -> None:
        probe = AdversarialProbe()
        with pytest.raises(AdversarialProbeError, match="DataFrame"):
            probe.probe(
                feature_name="x",
                derivation=_unwindowed_count,
                events={"not": "a frame"},  # type: ignore[arg-type]
                anchors=pd.Series(dtype="datetime64[ns]"),
            )

    def test_non_series_anchors_raises(self) -> None:
        probe = AdversarialProbe()
        with pytest.raises(AdversarialProbeError, match="Series"):
            probe.probe(
                feature_name="x",
                derivation=_unwindowed_count,
                events=pd.DataFrame({"patient_id": [], "event_date": []}),
                anchors=[1, 2, 3],  # type: ignore[arg-type]
            )

    def test_missing_event_date_col_raises(self) -> None:
        probe = AdversarialProbe()
        with pytest.raises(AdversarialProbeError, match="event_date"):
            probe.probe(
                feature_name="x",
                derivation=_unwindowed_count,
                events=pd.DataFrame({"patient_id": ["p1"], "other": [1]}),
                anchors=pd.Series([pd.Timestamp("2024-01-01")], index=["p1"]),
            )

    def test_missing_patient_col_raises(self) -> None:
        probe = AdversarialProbe()
        with pytest.raises(AdversarialProbeError, match="patient_id"):
            probe.probe(
                feature_name="x",
                derivation=_unwindowed_count,
                events=pd.DataFrame({"event_date": pd.to_datetime(["2024-01-01"]), "value": [1]}),
                anchors=pd.Series([pd.Timestamp("2024-01-01")], index=["p1"]),
            )

    def test_duplicate_anchor_index_raises(self) -> None:
        probe = AdversarialProbe()
        anchors = pd.Series(
            pd.to_datetime(["2024-01-01", "2024-02-01"]),
            index=["p1", "p1"],  # duplicate
        )
        with pytest.raises(AdversarialProbeError, match="unique"):
            probe.probe(
                feature_name="x",
                derivation=_unwindowed_count,
                events=pd.DataFrame(
                    {"patient_id": ["p1"], "event_date": pd.to_datetime(["2024-01-01"])}
                ),
                anchors=anchors,
            )

    def test_dtype_mismatch_event_date_vs_anchor_returns_error(self) -> None:
        # event_date is timestamp, anchors is integer → comparison fails.
        events = pd.DataFrame(
            {
                "patient_id": ["p1"],
                "event_date": pd.to_datetime(["2024-01-01"]),
            }
        )
        anchors = pd.Series([100], index=["p1"], dtype="int64")
        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="x",
            derivation=_unwindowed_count,
            events=events,
            anchors=anchors,
        )
        assert result.outcome == "error"
        assert "dtype" in (result.error or "").lower()

    def test_tz_aware_vs_naive_returns_error(self) -> None:
        # Codex L4: pandas across versions raises TypeError vs ValueError
        # for tz-aware vs tz-naive datetime comparisons. The probe must
        # surface either as outcome=error rather than letting the
        # exception propagate.
        events = pd.DataFrame(
            {
                "patient_id": ["p1"],
                "event_date": pd.to_datetime(["2024-01-01"], utc=True),  # tz-aware
            }
        )
        anchors = pd.Series(pd.to_datetime(["2024-01-15"]), index=["p1"])  # tz-naive
        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="x",
            derivation=_unwindowed_count,
            events=events,
            anchors=anchors,
        )
        assert result.outcome == "error"
        assert "compare" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# Compare entry point
# ---------------------------------------------------------------------------


class TestCompare:
    def test_compare_identical_numeric_unchanged(self) -> None:
        probe = AdversarialProbe()
        baseline = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=baseline.copy(),
        )
        assert result.outcome == "unchanged"
        assert result.n_rows_changed == 0
        assert result.max_abs_change is None

    def test_compare_numeric_max_abs_change(self) -> None:
        probe = AdversarialProbe()
        baseline = pd.Series([10.0, 20.0, 30.0], index=["a", "b", "c"])
        prefix = pd.Series([10.0, 17.0, 5.0], index=["a", "b", "c"])
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=prefix,
        )
        assert result.outcome == "changed"
        assert result.n_rows_compared == 3
        assert result.n_rows_changed == 2
        # Max diff: |30-5|=25
        assert result.max_abs_change == pytest.approx(25.0)

    def test_compare_within_tolerance_unchanged(self) -> None:
        probe = AdversarialProbe()
        baseline = pd.Series([1.0, 2.0], index=["a", "b"])
        prefix = pd.Series([1.0 + 1e-10, 2.0 + 5e-10], index=["a", "b"])
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=prefix,
            rtol=1e-6,
            atol=1e-9,
        )
        assert result.outcome == "unchanged"

    def test_compare_outside_tolerance_changed(self) -> None:
        probe = AdversarialProbe()
        baseline = pd.Series([1.0, 2.0], index=["a", "b"])
        prefix = pd.Series([1.0 + 1e-3, 2.0], index=["a", "b"])
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=prefix,
            rtol=1e-6,
            atol=1e-9,
        )
        assert result.outcome == "changed"
        assert result.n_rows_changed == 1

    def test_compare_nan_equals_nan(self) -> None:
        probe = AdversarialProbe()
        baseline = pd.Series([1.0, np.nan, 3.0], index=["a", "b", "c"])
        prefix = pd.Series([1.0, np.nan, 3.0], index=["a", "b", "c"])
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=prefix,
        )
        assert result.outcome == "unchanged"

    def test_compare_nan_vs_value_changed(self) -> None:
        probe = AdversarialProbe()
        baseline = pd.Series([1.0, np.nan], index=["a", "b"])
        prefix = pd.Series([1.0, 5.0], index=["a", "b"])
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=prefix,
        )
        assert result.outcome == "changed"
        assert result.n_rows_changed == 1
        # NaN→value yields non-finite diff; max_abs_change must stay None
        # rather than reporting NaN. A NaN-ridden field would otherwise
        # silently mask real numeric drift in audit consumers.
        assert result.max_abs_change is None

    def test_compare_string_categorical_no_max_abs(self) -> None:
        probe = AdversarialProbe()
        baseline = pd.Series(["alpha", "beta", "gamma"], index=["a", "b", "c"])
        prefix = pd.Series(["alpha", "DIFFERENT", "gamma"], index=["a", "b", "c"])
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=prefix,
        )
        assert result.outcome == "changed"
        assert result.n_rows_changed == 1
        assert result.max_abs_change is None  # not numeric

    def test_compare_partial_index_overlap_notes(self) -> None:
        probe = AdversarialProbe()
        baseline = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
        prefix = pd.Series([1.0, 2.0, 4.0], index=["a", "b", "d"])
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=prefix,
        )
        # Only 'a', 'b' are common (both 1.0 and 2.0 unchanged).
        assert result.outcome == "unchanged"
        assert result.n_rows_compared == 2
        assert any("only in baseline" in n for n in result.notes)
        assert any("only in prefix" in n for n in result.notes)

    def test_compare_disjoint_indices_inapplicable(self) -> None:
        probe = AdversarialProbe()
        baseline = pd.Series([1.0], index=["a"])
        prefix = pd.Series([2.0], index=["b"])
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=prefix,
        )
        assert result.outcome == "inapplicable"
        assert result.n_rows_compared == 0

    def test_compare_extra_notes_propagate(self) -> None:
        probe = AdversarialProbe()
        baseline = pd.Series([1.0], index=["a"])
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=baseline.copy(),
            extra_notes=("upstream filtered 12 patients",),
        )
        assert "upstream filtered 12 patients" in result.notes

    def test_compare_duplicate_baseline_index_returns_error(self) -> None:
        # Codex H2: non-unique baseline index would expand `.loc[common_index]`
        # lookups and corrupt n_rows_changed / fraction_changed. Reject up
        # front rather than silently produce nonsense audit fields.
        probe = AdversarialProbe()
        baseline = pd.Series([1.0, 2.0], index=["p1", "p1"])  # duplicate
        prefix = pd.Series([1.0], index=["p1"])
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=prefix,
        )
        assert result.outcome == "error"
        assert "non-unique" in (result.error or "").lower()
        assert "baseline" in (result.error or "")

    def test_compare_duplicate_prefix_index_returns_error(self) -> None:
        probe = AdversarialProbe()
        baseline = pd.Series([1.0], index=["p1"])
        prefix = pd.Series([1.0, 2.0], index=["p1", "p1"])  # duplicate
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=prefix,
        )
        assert result.outcome == "error"
        assert "prefix" in (result.error or "")

    def test_compare_inf_vs_finite_max_abs_change_is_inf(self) -> None:
        # Codex M3: a real unbounded drift (one side inf, the other finite)
        # is the most extreme leak the probe can detect. It must surface as
        # max_abs_change=inf, not silently filter to None.
        probe = AdversarialProbe()
        baseline = pd.Series([float("inf")], index=["p1"])
        prefix = pd.Series([0.0], index=["p1"])
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=prefix,
        )
        assert result.outcome == "changed"
        assert result.max_abs_change == float("inf")

    def test_compare_inf_vs_neg_inf_max_abs_change_is_inf(self) -> None:
        probe = AdversarialProbe()
        baseline = pd.Series([float("inf")], index=["p1"])
        prefix = pd.Series([float("-inf")], index=["p1"])
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=prefix,
        )
        assert result.outcome == "changed"
        assert result.max_abs_change == float("inf")

    def test_compare_same_sign_inf_unchanged(self) -> None:
        # np.isclose(equal_nan=True) treats inf==inf (same sign) as close.
        probe = AdversarialProbe()
        baseline = pd.Series([float("inf"), float("-inf")], index=["p1", "p2"])
        prefix = pd.Series([float("inf"), float("-inf")], index=["p1", "p2"])
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=prefix,
        )
        assert result.outcome == "unchanged"

    def test_compare_bool_dtype_no_max_abs_change(self) -> None:
        # Codex N6: boolean features pass `is_numeric_dtype` but
        # `max_abs_change=1.0` is not semantically meaningful for a binary
        # flag flip. Special-case bool so the equality path runs and
        # max_abs_change stays None while still flagging drift.
        probe = AdversarialProbe()
        baseline = pd.Series([True, False, True], index=["a", "b", "c"])
        prefix = pd.Series([True, True, True], index=["a", "b", "c"])
        result = probe.compare(
            feature_name="has_med_fill",
            baseline_values=baseline,
            prefix_values=prefix,
        )
        assert result.outcome == "changed"
        assert result.n_rows_changed == 1
        assert result.max_abs_change is None  # not 1.0

    def test_compare_int_dtype_works(self) -> None:
        probe = AdversarialProbe()
        baseline = pd.Series([1, 2, 3], index=["a", "b", "c"], dtype="int64")
        prefix = pd.Series([1, 2, 5], index=["a", "b", "c"], dtype="int64")
        result = probe.compare(
            feature_name="x",
            baseline_values=baseline,
            prefix_values=prefix,
        )
        assert result.outcome == "changed"
        assert result.n_rows_changed == 1
        assert result.max_abs_change == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Integration: realistic synthetic trajectory
# ---------------------------------------------------------------------------


class TestRealisticTrajectory:
    def test_legitimate_180day_window_passes(self) -> None:
        # 50 patients, 5 events each scattered ±400 days from each anchor.
        rng = np.random.default_rng(seed=42)
        patient_ids = [f"p{i}" for i in range(50)]
        anchor_dates = pd.to_datetime(["2024-06-01"] * 50) + pd.to_timedelta(
            rng.integers(0, 365, size=50), unit="D"
        )
        anchors = pd.Series(anchor_dates, index=patient_ids)

        rows = []
        for pid, anchor in anchors.items():
            offsets = rng.integers(-400, 400, size=5)
            for off in offsets:
                rows.append(
                    {
                        "patient_id": pid,
                        "event_date": anchor + pd.Timedelta(days=int(off)),
                    }
                )
        events = pd.DataFrame(rows)

        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="med_count_180d",
            derivation=_windowed_count(180),
            events=events,
            anchors=anchors,
        )
        assert result.outcome == "unchanged"
        assert result.n_rows_compared == 50

    def test_unwindowed_count_flags_majority(self) -> None:
        rng = np.random.default_rng(seed=42)
        patient_ids = [f"p{i}" for i in range(50)]
        anchor_dates = pd.to_datetime(["2024-06-01"] * 50) + pd.to_timedelta(
            rng.integers(0, 365, size=50), unit="D"
        )
        anchors = pd.Series(anchor_dates, index=patient_ids)

        rows = []
        for pid, anchor in anchors.items():
            offsets = rng.integers(-400, 400, size=5)
            for off in offsets:
                rows.append(
                    {
                        "patient_id": pid,
                        "event_date": anchor + pd.Timedelta(days=int(off)),
                    }
                )
        events = pd.DataFrame(rows)

        probe = AdversarialProbe()
        result = probe.probe(
            feature_name="all_med_fills",
            derivation=_unwindowed_count,
            events=events,
            anchors=anchors,
        )
        assert result.outcome == "changed"
        # Most patients should have at least one post-anchor event in this
        # ±400-day spread; expect a high fraction_changed.
        assert result.fraction_changed > 0.5
