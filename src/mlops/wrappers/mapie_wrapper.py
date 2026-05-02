"""MAPIE cross-conformal wrapper.

Phase 1 W2 day-3: takes any sklearn-compatible binary classifier and produces
calibrated `predict_proba` via cross-conformal calibration (Vovk 2005;
Lei et al. 2018). The prediction-set output is logged for future-work but
NOT consumed by the existing v3 contract.

Reference: shard 19 §B of `.claude/plans/adaptive_criteria_v3_followup/`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class MapieConformalBinaryClassifier:
    """Cross-conformal calibration wrapper for binary classifiers.

    The wrapped estimator must implement sklearn-compatible fit / predict_proba.
    `method='lac'` (Least Ambiguous set-valued Classifier; Sadinle 2019) is the
    MAPIE 0.8+ default and produces calibrated softmax predict_proba. `cv` controls
    the cross-conformal splitting (5-fold default).

    Phase 1 alpha is a single value (0.10) — 90% marginal coverage. For
    multi-alpha reporting (80/90/95% bands), iterate alpha at evaluation time;
    this wrapper fits once and exposes the configured alpha via `predict_sets`.

    Amendments vs shard 19 §B.2 verbatim:
      1. `method="cv"` → `method="lac"`. MAPIE pre-0.7 accepted `"cv"`; MAPIE
         0.8.6 (pinned) requires one of `{naive, score, lac, cumulated_score,
         aps, top_k, raps}`. `lac` is MAPIE's own default and the most direct
         replacement.
      2. `MapieClassifier` in MAPIE 0.8.6 does NOT expose `predict_proba` —
         only `predict(X, alpha=...) → (y_pred, prediction_sets)`. The
         wrapper's `predict_proba` therefore delegates to the BASE estimator
         (the original probabilities). The Phase-1 conformal contribution is
         the prediction-set artifact (`predict_sets`), which is logged for
         future work but not consumed by the v3 contract. Re-evaluate at W4
         multi-disease results: if the non-calibration-native base entries
         (LightGBM_Conformal, LogisticRegression_Conformal) underperform on
         calibration metrics, revisit the `skip_post_hoc_calibration=True`
         registry default for those entries.
    """

    def __init__(
        self,
        base_estimator: Any,
        method: str = "lac",
        cv: int = 5,
        alpha: float = 0.10,
        random_state: int = 42,
    ) -> None:
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        self.alpha = alpha
        self.random_state = random_state
        self._mapie: Optional[Any] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: Any, y: Any) -> "MapieConformalBinaryClassifier":
        """Fit base on full data, then calibrate prediction sets via MAPIE prefit.

        Coupling base+MAPIE this way lets `predict_proba` return the base
        estimator's full-data probabilities directly (amendment 2, see class
        docstring) while still surfacing MAPIE's coverage-guaranteed prediction
        sets via `predict_sets`. The `cv` argument controls MAPIE's set-
        calibration internal split; the base estimator itself is fit ONCE on
        full (X, y) — not K times.
        """
        from mapie.classification import MapieClassifier

        self.base_estimator.fit(X, y)
        self._mapie = MapieClassifier(
            estimator=self.base_estimator,
            method=self.method,
            cv="prefit",
            random_state=self.random_state,
        )
        self._mapie.fit(X, y)
        self.classes_ = np.array(getattr(self.base_estimator, "classes_", np.unique(y)))
        return self

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return base-estimator probabilities (amendment 2 — see class docstring).

        MAPIE 0.8.6 doesn't expose `predict_proba` on `MapieClassifier`, so the
        wrapper delegates to the wrapped base estimator. The conformal-set
        artifact lives on `predict_sets`.
        """
        if self._mapie is None:
            raise RuntimeError("MapieConformalBinaryClassifier called before fit()")
        raw = np.asarray(self.base_estimator.predict_proba(X))
        if raw.ndim == 1:
            return np.column_stack([1.0 - raw, raw])
        if raw.ndim == 2 and raw.shape[1] == 1:
            p1 = raw[:, 0]
            return np.column_stack([1.0 - p1, p1])
        return raw

    def predict_sets(self, X: Any) -> Any:
        """Return prediction sets at the configured alpha.

        Phase 1: not consumed by the v3 contract; logged as future-work artifact.
        """
        if self._mapie is None:
            raise RuntimeError("MapieConformalBinaryClassifier called before predict_sets()")
        _, sets = self._mapie.predict(X, alpha=self.alpha)
        return sets
