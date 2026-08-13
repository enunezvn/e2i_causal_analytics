"""#1548 iteration 2 — the ``rank_drivers`` SHAP compute must be BOUNDED, not
just off-loaded.

PR #1590 moved ``_compute_shap_from_frame`` onto the bounded heavy-compute pool
(``test_causal_discovery_offloop_1548.py`` pins that layering, which this file
does NOT relax). Post-deploy probes still died. Measured why (2026-08-13,
faithful prod shapes):

1. ``shap._cext.dense_tree_shap`` holds the GIL for its ENTIRE monolithic C
   call. Heartbeat-in-main-thread with the real explainer on a worker thread
   (n=6000x12, 100 trees) recorded a SINGLE 1240.544s heartbeat gap — the whole
   call. Off-loading a GIL-holding C call cannot protect the event loop, so
   uvicorn's ``callback_notify`` still never runs and gunicorn's arbiter still
   murders the worker at last-notify+120s.
2. The compute was intractable at prod scale: a 50-tree forest of UNBOUNDED
   depth explaining ALL rows (prod frames reached 37,515x12). Tree SHAP cost
   scales with rows x trees x leaves x depth^2.

Fix-candidate benchmark at 37,515x12 (50 trees, planted linear weights):

===============================================  =====  =======  ========
candidate                                          fit     shap     total
===============================================  =====  =======  ========
depth 8, explain all rows                        27.3s   114.9s    142.2s
depth 8, explain 2,000-row seeded sample         26.4s     5.8s     32.1s
unbounded depth, 2,000-row seeded sample         46.1s  1144.7s   1190.9s
===============================================  =====  =======  ========

Both bounds are load-bearing: unbounded depth alone is fatal, and 114.9s of
contiguous GIL hold still lands on the 120s arbiter. The chosen combination
leaves a ~5.8s GIL-held segment, and its rankings match the full-explain run at
Spearman 1.0000 (true-weight recovery 0.9983 for both).

Test-design reasoning (stated explicitly, per the #1548 brief):

* Every test drives the FULLY REAL ``_compute_shap_from_frame`` — real frames,
  real ``RandomForestRegressor`` fit, real ``TreeExplainer``. No mocked SHAP.
  ``test_forest_depth_bounded``'s only spy records the fitted estimator on its
  way into the explainer and then delegates to the real ``TreeExplainer``, so
  it observes the production object rather than substituting behaviour.
* Assertions are on shape and ordering, never on wall-clock, so nothing here is
  box-load dependent. The bound's *timing* payoff is measured in the issue; what
  a unit test can pin deterministically is that the bound is APPLIED.
* The expected bounds are written as literals here and cross-checked against the
  production constants. Pinning the values (not merely "whatever the module
  says") is deliberate: the value IS the contract — a constant re-tuned to
  50,000 rows would restore the outage while a self-referential assertion stayed
  green.
* Row subsampling is semantics-preserving because per-row SHAP never reaches a
  consumer: every path collapses the matrix to mean-|SHAP| per feature
  (``_predictive_only_ranking``, causal_discovery.py:979, and
  ``DriverRanker._compute_predictive_importance``, driver_ranker.py:367).
  ``test_ranking_stable_under_cap`` guards that claim end-to-end.

Falsifiability: removing ``max_depth`` or the explain cap from
``_compute_shap_from_frame`` fails ``test_explain_rows_capped`` and
``test_forest_depth_bounded`` on their intended assertions.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

import src.tool_registry.tools.causal_discovery as causal_discovery

# The bounds under test (see module docstring for why these are literals).
_EXPECTED_TREE_MAX_DEPTH = 8
_EXPECTED_MAX_EXPLAIN_ROWS = 2000

# The regression path (unbounded depth) is slow BY CONSTRUCTION: measured 38.0s
# for the n=2,500 frame below on the dev box, and 29.8s at n=600, against ~4.4s
# and ~1s once bounded. The suite-wide 30s timeout would abort the regression
# run before its assertion could report, so these tests get headroom — the
# failure signal is the assertion, not the clock.
_REGRESSION_HEADROOM_SECONDS = 180


def _linear_frame(
    n: int,
    weights: tuple[float, ...] = (3.0, 1.5, 0.5),
    seed: int = 7,
) -> pd.DataFrame:
    """A REAL frame whose target is a planted linear combination of features.

    ``weights`` are the true coefficients in descending magnitude, so the
    mean-|SHAP| ordering recovered from the frame has a known ground truth.
    """
    rng = np.random.default_rng(seed)
    features = {chr(ord("a") + i): rng.normal(size=n) for i in range(len(weights))}
    y = sum(w * features[name] for w, name in zip(weights, features, strict=True))
    return pd.DataFrame({**features, "y": y + rng.normal(size=n)})


@pytest.mark.timeout(_REGRESSION_HEADROOM_SECONDS)
def test_explain_rows_capped() -> None:
    """Frames larger than the cap are explained on a bounded row sample.

    Regression path: explaining all rows puts the full frame through the
    GIL-holding ``dense_tree_shap`` call, which at prod scale never finishes
    inside gunicorn's 120s arbiter window.
    """
    n = _EXPECTED_MAX_EXPLAIN_ROWS + 500
    shap_list, feats = causal_discovery._compute_shap_from_frame(_linear_frame(n), "y")

    assert len(shap_list) == _EXPECTED_MAX_EXPLAIN_ROWS, (
        f"#1548 regression: explained {len(shap_list)} rows of a {n}-row frame; "
        f"the explain set must be capped at {_EXPECTED_MAX_EXPLAIN_ROWS}."
    )
    assert causal_discovery._SHAP_MAX_EXPLAIN_ROWS == _EXPECTED_MAX_EXPLAIN_ROWS
    # The cap touches rows only — every feature still gets a column.
    assert all(len(row) == len(feats) for row in shap_list)


@pytest.mark.timeout(_REGRESSION_HEADROOM_SECONDS)
def test_forest_depth_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """The surrogate forest is fitted with an explicit depth bound.

    Depth is the binding term in tree-SHAP cost (measured: unbounded depth on a
    2,000-row explain sample took 1144.7s against 5.8s at depth 8). The fit
    itself stays on ALL rows — only depth is bounded — so model quality is
    unchanged in the row dimension.

    The spy sits on ``shap.TreeExplainer`` rather than on
    ``RandomForestRegressor``: shap's ``safe_isinstance`` recognises models by
    re-resolving ``sklearn.ensemble.RandomForestRegressor`` out of
    ``sys.modules`` and calling ``isinstance`` against it (_general.py:258), so
    substituting that attribute at all breaks model detection. Capturing the
    estimator on its way into the explainer is also the stronger assertion —
    it inspects the REAL fitted forest, not merely the kwargs requested.
    """
    import shap

    real_explainer_cls = shap.TreeExplainer
    captured: Dict[str, Any] = {}

    def recording_explainer(model: Any, *args: Any, **kwargs: Any) -> Any:
        """Record the fitted estimator, then delegate to the REAL explainer."""
        captured["model"] = model
        return real_explainer_cls(model, *args, **kwargs)

    monkeypatch.setattr(shap, "TreeExplainer", recording_explainer)

    shap_list, _ = causal_discovery._compute_shap_from_frame(_linear_frame(n=600), "y")

    model = captured.get("model")
    assert model is not None, "TreeExplainer was never constructed — test harness defect"
    assert model.max_depth == _EXPECTED_TREE_MAX_DEPTH, (
        "#1548 regression: the surrogate forest was fitted with max_depth="
        f"{model.max_depth!r}; unbounded-depth trees make TreeExplainer "
        "intractable at prod scale."
    )
    assert causal_discovery._SHAP_TREE_MAX_DEPTH == _EXPECTED_TREE_MAX_DEPTH
    # The bound reached the actual trees, not just the constructor.
    assert max(est.get_depth() for est in model.estimators_) <= _EXPECTED_TREE_MAX_DEPTH
    # The real forest really was fitted and really was explained.
    assert len(shap_list) == 600


def test_small_frames_unsampled_and_deterministic() -> None:
    """Frames within the cap are explained whole, and the result is reproducible.

    The cap must not perturb the ordinary case, and the seeded model plus seeded
    sampler must make repeat calls byte-identical — a chat turn re-run on the
    same frame cannot return a different driver ranking.
    """
    frame = _linear_frame(n=150)

    first, first_feats = causal_discovery._compute_shap_from_frame(frame, "y")
    second, second_feats = causal_discovery._compute_shap_from_frame(frame, "y")

    assert len(first) == 150, "frames within the cap must be explained in full"
    assert first_feats == second_feats
    assert first == second, "repeat SHAP derivation on one frame must be deterministic"


def _mean_abs_ranking(shap_list: list[list[float]], feats: list[str]) -> list[str]:
    """Reduce a SHAP matrix the way every consumer does: mean-|SHAP| per feature,
    descending. Mirrors ``_predictive_only_ranking`` and
    ``DriverRanker._compute_predictive_importance``."""
    mean_abs = np.abs(np.asarray(shap_list, dtype=float)).mean(axis=0)
    return [feats[i] for i in np.argsort(mean_abs)[::-1]]


@pytest.mark.timeout(_REGRESSION_HEADROOM_SECONDS)
def test_ranking_stable_under_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The capped explain set yields the SAME ranking as explaining every row.

    This is the semantics argument the row cap rests on, and it is checked
    against a real full-explain baseline rather than inferred: the cap is lifted
    for one run (so the identical production code explains all n rows) and left
    in place for the other, and the two mean-|SHAP| orderings must agree. Both
    arms must also recover the planted coefficient order, which pins the
    comparison to ground truth rather than to two matching-but-wrong runs.

    At prod scale (37,515x12) the same comparison measured Spearman 1.0000
    between capped and full explain — see the module docstring.
    """
    weights = (3.0, 1.5, 0.5)
    n = _EXPECTED_MAX_EXPLAIN_ROWS + 800
    frame = _linear_frame(n, weights=weights)
    planted = [chr(ord("a") + i) for i in range(len(weights))]

    # Baseline: same code path, cap raised so every row is explained.
    monkeypatch.setattr(causal_discovery, "_SHAP_MAX_EXPLAIN_ROWS", n)
    full_list, full_feats = causal_discovery._compute_shap_from_frame(frame, "y")
    monkeypatch.undo()

    capped_list, capped_feats = causal_discovery._compute_shap_from_frame(frame, "y")

    assert len(full_list) == n, "baseline arm did not explain every row"
    assert len(capped_list) == _EXPECTED_MAX_EXPLAIN_ROWS, "capped arm was not capped"
    assert full_feats == capped_feats

    full_ranking = _mean_abs_ranking(full_list, full_feats)
    capped_ranking = _mean_abs_ranking(capped_list, capped_feats)

    assert capped_ranking == full_ranking, (
        f"capped explain set reordered the drivers: {capped_ranking} against "
        f"{full_ranking} from explaining all {n} rows."
    )
    assert full_ranking == planted, (
        f"full-explain baseline itself missed the planted order: got "
        f"{full_ranking}, expected {planted} from weights {weights}."
    )
