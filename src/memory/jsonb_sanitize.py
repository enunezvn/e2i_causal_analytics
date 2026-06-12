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


def _normalize_numpy_scalar(obj: Any) -> Any:
    """Collapse numpy scalars (np.float32/np.int32/np.bool_ ...) to Python.

    codex iter-1 MEDIUM: ``np.float32`` is NOT a ``float`` subclass (unlike
    ``np.float64``), and stdlib json rejects even FINITE numpy-only scalars
    with ``TypeError`` — recreating the silent-drop class through a different
    type. ``.item()`` is the numpy-blessed scalar conversion; the duck-typed
    check keeps this module free of a numpy import (numpy may legitimately be
    absent in slim environments).
    """
    if type(obj).__module__ == "numpy" and hasattr(obj, "item"):
        try:
            return obj.item()
        except (AttributeError, ValueError):  # 0-d-only guard; arrays pass through
            return obj
    return obj


def _sanitize_key(key: Any) -> Any:
    """Map a non-finite float dict KEY to ``None`` (json renders it "null").

    codex iter-1 MEDIUM: ``json.dumps({float("nan"): 1}, allow_nan=False)``
    raises just like a NaN value does — the key position was a residual
    silent-drop path. Finite float keys are left for json's normal key
    coercion ("0.5"). Distinct non-finite keys (nan + inf) collapse onto the
    single None key (last one wins) — JSON cannot represent them distinctly
    anyway, and a lossy-but-stored payload beats a silently dropped one.
    """
    key = _normalize_numpy_scalar(key)
    if isinstance(key, float) and not math.isfinite(key):
        return None
    return key


def sanitize_jsonb_payload(obj: Any) -> Any:
    """Recursively map non-finite floats (nan/inf/-inf) to ``None``.

    - dicts/lists are rebuilt with sanitized values; non-finite float KEYS
      map to ``None`` (rendered as the "null" key) — see :func:`_sanitize_key`;
    - tuples normalize to lists (what JSON encoding does anyway);
    - numpy scalars normalize to Python scalars first (``np.float32`` is not
      a ``float`` subclass and even finite ones crash stdlib json);
    - ``bool`` is untouched (subclasses ``int``, not ``float``);
    - every other value passes through unchanged.

    The result is guaranteed to survive ``json.dumps(..., allow_nan=False)``
    as far as float-finiteness is concerned, and the function is idempotent.
    """
    obj = _normalize_numpy_scalar(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {_sanitize_key(k): sanitize_jsonb_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_jsonb_payload(v) for v in obj]
    return obj
