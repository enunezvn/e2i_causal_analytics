"""Phase 2.4 — AdversarialProbe.

Re-derives a feature using only prefix data (events with ``event_date`` at or
before each patient's anchor) and compares to the value the same derivation
produces on the full event stream. When the values differ, the derivation
depends on post-prefix data — a temporal leak that Layer 1 metadata audits
and Layer 2 KG-grounded LLMs cannot detect on their own because both reason
on declarative metadata, not on the raw event stream.

Semantics:
    The probe assumes the user-supplied ``derivation`` is the reference
    re-implementation. A *prefix-stable* derivation produces the same per-
    patient values whether it sees the full event stream or the per-patient
    censored stream — well-windowed aggregations (``[anchor-180, anchor]``)
    are prefix-stable; aggregations that read ``MAX(event_date)`` or
    otherwise pull post-anchor rows are not. The probe does not need a
    separate "observed" feature column to make this comparison: any
    self-disagreement of the derivation under censoring is itself the leak
    signal.

    This is **distinct from Layer 3** (``compute_adversarial_leakage_score``
    in ``src/data/leakage_detector/permutation.py``), which trains a
    discriminator and tests against a permutation null. Layer 3 catches
    *statistical* leaks (residual signal in feature values); Layer 2.4
    catches *temporal* leaks (the derivation reads post-anchor rows).

Usage:
    >>> from src.data.kg.adversarial_probe import AdversarialProbe
    >>> probe = AdversarialProbe()
    >>> result = probe.probe(
    ...     feature_name="med_fill_count_180d",
    ...     derivation=lambda evs, anchors: (
    ...         evs.groupby("patient_id").size().reindex(anchors.index, fill_value=0)
    ...     ),
    ...     events=med_events,
    ...     anchors=patient_index_dates,
    ... )
    >>> result.outcome
    'unchanged'   # this derivation is anchor-ignorant; the lambda is only
                  # prefix-stable when the events themselves are bounded.

Reference: ``.claude/plans/adaptive_temporal_validity_redesign.md`` Phase 2.4.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.data.kg.types import AdversarialProbeResult

logger = logging.getLogger(__name__)


_PROBE_ANCHOR_COL = "_probe_anchor"


class AdversarialProbeError(Exception):
    """Raised on argument-shape errors that the caller can fix.

    Wrapping ValueError/TypeError so callers can pinpoint probe-layer
    rejections separately from derivation-layer exceptions (which are
    captured into ``AdversarialProbeResult.error``).
    """


class AdversarialProbe:
    """Phase 2.4 — re-run feature derivation with prefix-only data.

    Stateless: a single instance can probe many features. Reuse for free.
    """

    def probe(
        self,
        *,
        feature_name: str,
        derivation: Callable[[pd.DataFrame, pd.Series], pd.Series],
        events: pd.DataFrame,
        anchors: pd.Series,
        event_date_col: str = "event_date",
        event_patient_col: str = "patient_id",
        rtol: float = 1e-6,
        atol: float = 1e-9,
    ) -> AdversarialProbeResult:
        """Re-derive ``feature_name`` on full and prefix-censored events.

        Args:
            feature_name: Name of the feature being probed; surfaces in
                logs and the result.
            derivation: Pure function ``(events, anchors) -> Series`` that
                computes the feature given an event-level DataFrame and a
                patient_id → anchor timestamp Series. The returned Series
                is indexed by patient ID. Anchor-aware derivations consult
                ``anchors``; anchor-ignorant ones simply ignore it.
            events: Event-level DataFrame containing at least
                ``event_date_col`` and ``event_patient_col``. Other columns
                are passed through to ``derivation`` untouched.
            anchors: Series indexed by patient ID, valued by anchor
                timestamp. Patients without an entry are excluded from the
                probe.
            event_date_col: Column in ``events`` carrying the event
                timestamp. Rows with NaN in this column are excluded from
                both the full and the prefix-censored slices (they cannot
                be classified as pre- or post-anchor).
            event_patient_col: Column in ``events`` carrying the patient
                identifier; matched against ``anchors.index``.
            rtol: Relative tolerance forwarded to ``np.isclose`` for
                numeric comparisons.
            atol: Absolute tolerance forwarded to ``np.isclose``.

        Returns:
            ``AdversarialProbeResult`` whose ``outcome`` is:
                - ``"unchanged"``: the derivation is prefix-stable on this
                  data — well-windowed aggregations or anchor-only features.
                - ``"changed"``: at least one patient's value drifted under
                  censoring — the derivation pulls post-anchor data.
                - ``"error"``: ``derivation`` raised on either invocation.
                - ``"inapplicable"``: insufficient inputs (no anchors, no
                  matching events, etc.). ``notes`` carries the reason.

        Raises:
            AdversarialProbeError: argument-shape errors the caller must
                fix (missing columns, wrong types). Derivation-layer
                exceptions are captured into the result, not raised.
        """

        notes: list[str] = []

        if not callable(derivation):
            raise AdversarialProbeError(
                f"derivation must be callable; got {type(derivation).__name__}"
            )
        if not isinstance(events, pd.DataFrame):
            raise AdversarialProbeError(
                f"events must be a pandas DataFrame; got {type(events).__name__}"
            )
        if not isinstance(anchors, pd.Series):
            raise AdversarialProbeError(
                f"anchors must be a pandas Series indexed by patient ID; "
                f"got {type(anchors).__name__}"
            )
        if event_date_col not in events.columns:
            raise AdversarialProbeError(
                f"events missing column {event_date_col!r}; available: {list(events.columns)}"
            )
        if event_patient_col not in events.columns:
            raise AdversarialProbeError(
                f"events missing column {event_patient_col!r}; available: {list(events.columns)}"
            )
        if not anchors.index.is_unique:
            # Duplicate patient anchors would silently produce ambiguous
            # ``map(anchors)`` lookups. Reject up front.
            raise AdversarialProbeError(
                "anchors.index must be unique (one anchor per patient); "
                f"saw {int((~anchors.index.duplicated()).sum())} unique vs "
                f"{len(anchors)} total"
            )

        # Drop NaN anchors. These patients can't be censored.
        anchor_nan = anchors.isna()
        if anchor_nan.any():
            n_dropped = int(anchor_nan.sum())
            notes.append(f"dropped {n_dropped} anchors with NaN timestamps")
            anchors = anchors[~anchor_nan]

        if anchors.empty:
            notes.append("no valid anchors remain after NaN filtering")
            return AdversarialProbeResult(
                feature_name=feature_name,
                outcome="inapplicable",
                notes=tuple(notes),
            )

        # Restrict events to patients with anchors so baseline and prefix
        # share a denominator. This is the right comparison: "for the
        # patients we have anchors for, does the derivation drift under
        # censoring?". Patients without anchors are unobservable to this
        # probe regardless.
        anchored_mask = events[event_patient_col].isin(anchors.index)
        events_anchored = events.loc[anchored_mask].copy()
        if events_anchored.empty:
            notes.append("no events for anchored patients")
            return AdversarialProbeResult(
                feature_name=feature_name,
                outcome="inapplicable",
                notes=tuple(notes),
            )

        # Drop rows with NaN event_date — they can't be classified as
        # pre- or post-anchor, so they pollute both baseline and prefix.
        date_nan = events_anchored[event_date_col].isna()
        if date_nan.any():
            n_drop = int(date_nan.sum())
            notes.append(f"dropped {n_drop} events with NaN {event_date_col!r}")
            events_anchored = events_anchored.loc[~date_nan]

        if events_anchored.empty:
            notes.append("no events remain after dropping NaN dates")
            return AdversarialProbeResult(
                feature_name=feature_name,
                outcome="inapplicable",
                notes=tuple(notes),
            )

        # Attach each row's patient anchor; censor by row date <= anchor.
        # ``map`` is safe because anchors.index is verified unique above.
        events_anchored[_PROBE_ANCHOR_COL] = events_anchored[event_patient_col].map(anchors)

        try:
            prefix_mask = events_anchored[event_date_col] <= events_anchored[_PROBE_ANCHOR_COL]
        except TypeError as exc:
            # Mismatched event_date / anchor dtypes (e.g., one is timestamp,
            # the other is string). The caller must align dtypes; we surface
            # this as an error rather than as a silent miscomparison.
            return AdversarialProbeResult(
                feature_name=feature_name,
                outcome="error",
                error=(
                    f"could not compare {event_date_col!r} to anchor: {exc}. "
                    "Confirm both are the same comparable dtype."
                ),
                notes=tuple(notes),
            )

        prefix_events = events_anchored.loc[prefix_mask].drop(columns=[_PROBE_ANCHOR_COL])
        full_events = events_anchored.drop(columns=[_PROBE_ANCHOR_COL])

        try:
            baseline = derivation(full_events, anchors)
        except Exception as exc:  # noqa: BLE001 — surface ANY derivation error
            return AdversarialProbeResult(
                feature_name=feature_name,
                outcome="error",
                error=f"derivation on full events raised: {exc}",
                notes=tuple(notes),
            )

        try:
            recomputed = derivation(prefix_events, anchors)
        except Exception as exc:  # noqa: BLE001
            return AdversarialProbeResult(
                feature_name=feature_name,
                outcome="error",
                error=f"derivation on prefix events raised: {exc}",
                notes=tuple(notes),
            )

        if not isinstance(baseline, pd.Series):
            return AdversarialProbeResult(
                feature_name=feature_name,
                outcome="error",
                error=(
                    "derivation on full events returned "
                    f"{type(baseline).__name__}; expected pandas Series"
                ),
                notes=tuple(notes),
            )
        if not isinstance(recomputed, pd.Series):
            return AdversarialProbeResult(
                feature_name=feature_name,
                outcome="error",
                error=(
                    "derivation on prefix events returned "
                    f"{type(recomputed).__name__}; expected pandas Series"
                ),
                notes=tuple(notes),
            )

        # The contract requires the returned Series be indexed by patient ID
        # (i.e., a subset of ``anchors.index``). A derivation that returns a
        # default ``RangeIndex(0..n)`` would otherwise positionally align in
        # ``compare()`` — silently comparing patient-A's full value to
        # patient-B's prefix value, which is a wrong-by-construction probe
        # result that can falsely PASS or FAIL the leak check depending on
        # group ordering. Verify both Series before calling ``compare()``.
        for label, returned in (("full", baseline), ("prefix", recomputed)):
            extra = returned.index.difference(anchors.index)
            if len(extra) > 0:
                return AdversarialProbeResult(
                    feature_name=feature_name,
                    outcome="error",
                    error=(
                        f"derivation on {label} events returned a Series whose "
                        f"index has {len(extra)} value(s) not present in anchors.index "
                        f"(first 3: {list(extra[:3])}). The probe contract requires "
                        "the returned Series to be indexed by patient ID drawn from "
                        "anchors.index — a default RangeIndex would silently align "
                        "positionally. Reindex via "
                        "`.reindex(anchors.index, fill_value=...)`."
                    ),
                    notes=tuple(notes),
                )

        return self.compare(
            feature_name=feature_name,
            baseline_values=baseline,
            prefix_values=recomputed,
            rtol=rtol,
            atol=atol,
            extra_notes=tuple(notes),
        )

    def compare(
        self,
        *,
        feature_name: str,
        baseline_values: pd.Series,
        prefix_values: pd.Series,
        rtol: float = 1e-6,
        atol: float = 1e-9,
        extra_notes: tuple[str, ...] = (),
    ) -> AdversarialProbeResult:
        """Compare two derivation outputs and produce a probe result.

        Lower-level entry point for callers that have already computed
        baseline and prefix-censored values — useful for vectorised
        pipelines that batch-derive many features at once.

        Both Series should be indexed by patient ID. Indices that appear in
        only one Series are dropped with a note (they cannot be compared).
        Numeric dtypes use ``np.isclose(equal_nan=True)``; non-numeric use
        per-element equality with NaN-equals-NaN semantics.
        """

        notes: list[str] = list(extra_notes)

        # Reject duplicate-index inputs up front: ``.loc[common_index]`` on
        # a non-unique index expands rows on duplicate-label match and can
        # make ``n_rows_changed > n_rows_compared`` and ``fraction_changed
        # > 1.0`` — a violation of the ``AdversarialProbeResult`` audit
        # invariants and a silent path to bogus drift counts.
        for label, values in (("baseline", baseline_values), ("prefix", prefix_values)):
            if not values.index.is_unique:
                return AdversarialProbeResult(
                    feature_name=feature_name,
                    outcome="error",
                    error=(
                        f"{label}_values has a non-unique index — duplicate "
                        "labels would corrupt per-patient comparison via "
                        "label-expanding `.loc` lookups."
                    ),
                    notes=tuple(notes),
                )

        common_index = baseline_values.index.intersection(prefix_values.index)
        only_baseline = baseline_values.index.difference(prefix_values.index)
        only_prefix = prefix_values.index.difference(baseline_values.index)
        if len(only_baseline):
            notes.append(f"{len(only_baseline)} patients only in baseline (not compared)")
        if len(only_prefix):
            notes.append(f"{len(only_prefix)} patients only in prefix (not compared)")

        if len(common_index) == 0:
            notes.append("no common patient indices between baseline and prefix")
            return AdversarialProbeResult(
                feature_name=feature_name,
                outcome="inapplicable",
                notes=tuple(notes),
            )

        baseline_aligned = baseline_values.loc[common_index]
        prefix_aligned = prefix_values.loc[common_index]

        is_numeric = pd.api.types.is_numeric_dtype(
            baseline_aligned
        ) and pd.api.types.is_numeric_dtype(prefix_aligned)

        max_abs_change: Optional[float] = None
        if is_numeric:
            baseline_arr = baseline_aligned.to_numpy(dtype=float, na_value=np.nan)
            prefix_arr = prefix_aligned.to_numpy(dtype=float, na_value=np.nan)
            close = np.isclose(
                baseline_arr,
                prefix_arr,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
            if (~close).any():
                diffs = np.abs(baseline_arr - prefix_arr)
                changed_diffs = diffs[~close]
                # Drop NaN diffs (NaN-vs-value pairs produce them) so they
                # can't pollute the reported max. KEEP inf diffs: a real
                # unbounded drift (baseline=inf vs prefix=0, or +inf vs
                # -inf) should surface as max_abs_change=inf rather than
                # be silently filtered to None — that would hide the most
                # extreme leak the probe can detect.
                non_nan_diffs = changed_diffs[~np.isnan(changed_diffs)]
                if non_nan_diffs.size > 0:
                    max_abs_change = float(non_nan_diffs.max())
        else:
            both_nan = (baseline_aligned.isna() & prefix_aligned.isna()).to_numpy()
            equal_or_na = baseline_aligned.eq(prefix_aligned).fillna(False).to_numpy()
            close = both_nan | equal_or_na.astype(bool)

        n_compared = int(len(common_index))
        n_changed = int((~close).sum())
        fraction_changed = n_changed / n_compared if n_compared > 0 else 0.0
        outcome: Any = "changed" if n_changed > 0 else "unchanged"

        return AdversarialProbeResult(
            feature_name=feature_name,
            outcome=outcome,
            n_rows_compared=n_compared,
            n_rows_changed=n_changed,
            fraction_changed=fraction_changed,
            max_abs_change=max_abs_change,
            notes=tuple(notes),
        )
