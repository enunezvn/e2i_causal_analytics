"""Shared JSON normalisation for repository writers.

Algorithm wrappers hand back numpy scalars/arrays, diagnostics carry tuples and
enums, and refutation evidence can carry non-finite floats; the Supabase
transport (httpx) must see plain JSON types and encodes with ``allow_nan=False``.
Moved out of :mod:`src.repositories.discovered_dag` (lane 1, owner decision
2026-09-09) so :mod:`src.repositories.causal_validation` writes its evidence
rows in the same shape.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any
from uuid import UUID

import numpy as np


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        # Enum members (GateDecision / EdgeType / DiscoveryAlgorithmType).
        return value.value
    return str(value)


def to_plain_json(value: Any) -> Any:
    """Round-trip through JSON so every leaf is a plain JSON type.

    Non-finite floats (NaN / Infinity / -Infinity, including numpy ones)
    become ``None``: JSON has no such values and the transport encodes with
    ``allow_nan=False`` (httpx ``_content.py``), so one stray NaN in a score
    or a wrapper's metadata would otherwise fail the WHOLE write. ``null`` is
    the honest JSON reading of "no finite value" (codex iter-2 MED).
    """
    return json.loads(json.dumps(value, default=json_default), parse_constant=lambda _: None)
