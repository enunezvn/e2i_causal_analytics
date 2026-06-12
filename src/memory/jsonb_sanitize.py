"""Strict-JSON sanitization for JSONB-bound memory payloads (#891).

Why this exists
---------------
Postgres ``jsonb`` rejects the bare ``NaN``/``Infinity``/``-Infinity`` tokens
Python's default ``json.dumps`` emits, and supabase-py's JSON encoder is
strict (``allow_nan=False`` semantics): a single non-finite float anywhere in
a JSONB-bound payload raises ``ValueError: Out of range float values are not
JSON compliant`` BEFORE the request is sent. The agent memory hooks swallow
that exception, so the whole episodic write is SILENTLY DROPPED
(probe-verified live 2026-06-12, issue #891).

The historical variant of the same bug class — the pre-#888 writer
``json.dumps``-ing payloads itself — let the bare tokens reach the column as
JSON string scalars, producing the 137 NaN-bearing rows migration 073 had to
skip and ``scripts/maintenance/converge_episodic_nan_rows_891.py`` repairs.

model_trainer emits non-finite metrics for real (stacking fold means,
advanced_validation ``brier_*`` defaults, learning_curve ``score_mean``/
``score_std``), so this is live data loss, not a hypothetical.

Semantics
---------
Non-finite floats map to ``None`` (JSON null). Readers already treat
absent/None metrics as missing (``test_metrics.get("auc_roc")``-style
fallbacks), so null preserves the "missing value" meaning at rest. String
values — including ones that merely CONTAIN the text ``NaN`` (the codex-R2
corruption canary ``"threshold: NaN means missing, Infinity capped"``) — are
never touched: this operates on parsed values, never on serialized text, so
it is quote-aware by construction.
"""

import math
from typing import Any

__all__ = ["sanitize_jsonb_payload"]


def sanitize_jsonb_payload(obj: Any) -> Any:
    """Recursively map non-finite floats (nan/inf/-inf) to ``None``.

    - dicts/lists are rebuilt with sanitized values (keys untouched — JSON
      keys are strings and cannot carry floats);
    - tuples normalize to lists (what JSON encoding does anyway);
    - ``bool`` is untouched (subclasses ``int``, not ``float``); numpy float
      subclasses (``np.float64``) are handled by the ``float`` isinstance;
    - every other value passes through unchanged.

    The result is guaranteed to survive ``json.dumps(..., allow_nan=False)``
    as far as float-finiteness is concerned, and the function is idempotent.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_jsonb_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_jsonb_payload(v) for v in obj]
    return obj
