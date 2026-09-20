"""KPI forecasting for the canonical monthly volume series (#2115, Lane B).

The platform had no forecaster before this package: ``e2i_data_query_tool``'s
``predictions`` type is a memory search and ``prediction_synthesizer`` ensembles
entity-level classifiers, so demo 6.5 ("Forecast Kisqali TRx volume for the next two
quarters and tell me the biggest risk to that forecast") refused on every run from
2026-07-29.

Four modules, each with one job:

* ``models``   — the swappable model interface and its members (Holt-Winters in-process,
                 TimesFM 2.5 on the forecast worker). A model is a name, a minimum
                 history and ``predict(y, horizon)``; it owns no notion of its own
                 uncertainty and does no scoring.
* ``backtest`` — rolling origins, the score, the champion, and the prediction band. All
                 models are graded on the SAME origins with the SAME metric, which is
                 what makes a state-space model and a pretrained transformer comparable.
* ``timesfm``  — the Celery dispatch for the worker-backed model, and the batching that
                 makes a 24-origin backtest one round trip instead of twenty-four.
* ``service``  — Lane A's canonical series in, champion forecast plus band out.

Two properties hold throughout. The band is always the champion's own MEASURED miss
distribution, never a model-internal interval. And every model here is UNIVARIATE, so
the forecast cannot see a competitor entry, a payer change or a regional step inside its
window — the payload says so, and the risk half of a question belongs to the gap and
causal tools.
"""
