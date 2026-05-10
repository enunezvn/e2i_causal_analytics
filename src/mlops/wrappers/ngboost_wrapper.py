"""sklearn-compatible NGBoost wrapper.

NGBoost is calibration-native (predicts a full distribution, not a point).
Phase 1 W2: used as a first-class classifier candidate in adaptive criteria
v3 Phase 1 evaluation. The wrapper guarantees a 2D predict_proba so the
existing evaluator path (evaluator.py:438-471) flows through unchanged.

Reference: shard 19 §A.1 of `.claude/plans/adaptive_criteria_v3_followup/`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class NGBoostBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Thin wrapper around ngboost.NGBClassifier for binary classification.

    Why wrap:
      - Some ngboost releases return (N,) predict_proba for binary; we force (N, 2).
      - Surface a stable `pred_dist_` accessor for downstream interval reporting.
      - Centralize default Bernoulli distribution choice (binary classification).
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.01,
        minibatch_frac: float = 1.0,
        col_sample: float = 1.0,
        verbose: bool = False,
        random_state: int = 42,
        # Base learner controls (passed to NGBClassifier as Base= kwargs):
        base_max_depth: int = 3,
        base_min_samples_leaf: int = 5,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.verbose = verbose
        self.random_state = random_state
        self.base_max_depth = base_max_depth
        self.base_min_samples_leaf = base_min_samples_leaf
        self._model: Optional[Any] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(
        self,
        X: Any,
        y: Any,
        X_val: Any = None,
        Y_val: Any = None,
    ) -> "NGBoostBinaryClassifier":
        from ngboost import NGBClassifier
        from ngboost.distns import Bernoulli
        from sklearn.tree import DecisionTreeRegressor

        base = DecisionTreeRegressor(
            criterion="friedman_mse",
            max_depth=self.base_max_depth,
            min_samples_leaf=self.base_min_samples_leaf,
            random_state=self.random_state,
        )
        self._model = NGBClassifier(
            Dist=Bernoulli,
            Base=base,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            col_sample=self.col_sample,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        fit_kwargs: dict = {}
        if X_val is not None and Y_val is not None:
            fit_kwargs["X_val"] = X_val
            fit_kwargs["Y_val"] = Y_val
        self._model.fit(X, y, **fit_kwargs)
        self.classes_ = np.array(getattr(self._model, "classes_", np.unique(y)))
        return self

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("NGBoostBinaryClassifier called before fit()")
        raw = self._model.predict_proba(X)
        raw = np.asarray(raw)
        if raw.ndim == 1:
            return np.column_stack([1.0 - raw, raw])
        if raw.ndim == 2 and raw.shape[1] == 1:
            p1 = raw[:, 0]
            return np.column_stack([1.0 - p1, p1])
        return raw  # already (N, 2)

    def pred_dist(self, X: Any) -> Any:
        if self._model is None:
            raise RuntimeError("NGBoostBinaryClassifier called before pred_dist()")
        return self._model.pred_dist(X)
