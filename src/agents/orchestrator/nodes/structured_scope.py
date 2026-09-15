"""Reading a STRUCTURED scope out of a dispatch payload (#2114).

"Structured" means a caller DECIDED: the NLP layer's typed ``parsed_query``
entities, or the ``brand``/``region`` a chat caller stashed in ``user_context``.
The ask TEXT is evidence, never a decision, and is resolved elsewhere.

Three review rounds each found one more shape leaking through a "is this not
blank" test — two brands, then whitespace, then zero-width characters — so the
question is asked positively here instead: is this a scope the substrate
recognises? Validation precedes source selection, and recognition NORMALISES
without ever DELETING, because "unrecognised" and "absent" are different facts.
A blank names nobody; ``'Xolair'`` names somebody this substrate cannot serve,
and erasing it reads downstream as "no scope given" and silently widens the
query (r4 HIGH-2).

Lives beside ``dispatcher`` rather than inside it because that module is
ratchet-pinned (tests/unit/test_tests_meta/test_module_size_ratchet.py).
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Optional, cast


def _entity_value(payload: Dict[str, Any], entity_type: str) -> Optional[str]:
    """Return the first ``parsed_query.entities`` value of ``entity_type``.

    Mirrors the ``parsed_query.entities`` derivation used for ``drift_monitor``'s
    ``features_to_monitor`` default (KPI/feature mentions): walk the structured
    NLP entities the orchestrator already carries and return the first non-empty
    string ``value`` whose ``type`` matches. Returns ``None`` when no such entity
    exists (the caller then falls back to ``user_context`` or proceeds without).
    """
    parsed_query = payload.get("parsed_query") or {}
    entities = (parsed_query.get("entities") if isinstance(parsed_query, dict) else None) or []
    for ent in entities:
        if (
            isinstance(ent, dict)
            and ent.get("type") == entity_type
            and isinstance(ent.get("value"), str)
            and ent["value"].strip()
        ):
            return cast(str, ent["value"])
    return None


def _structured_brand(payload: Dict[str, Any]) -> Optional[str]:
    """The brand a STRUCTURED source decided on — typed NLP entities, else the
    ``user_context`` a chat caller stashed. Recognised values are NORMALISED to
    their ``brand_type`` label; an unrecognised one is passed on UNCHANGED so the
    fail-closed downstream still sees it; only a value naming nobody is ``None``."""
    from src.services.enum_labels import resolve_brand_label

    return _structured_scope(payload, "brand", resolve_brand_label)


def _structured_scope(payload: Dict[str, Any], key: str, resolve: Any, **kw: Any) -> Optional[str]:
    """Entities then ``user_context``, through ONE vocabulary-aware chooser so
    the clarify gate and the cohort extractor can never disagree (#2114)."""
    from src.services.enum_labels import first_named_scope

    ctx = payload.get("user_context") or {}
    stashed = ctx.get(key) if isinstance(ctx, dict) else None
    return first_named_scope((_entity_value(payload, key), stashed), resolve, **kw)


def _structured_region(payload: Dict[str, Any]) -> Optional[str]:
    """The region a STRUCTURED source decided on. Returned RAW: an unresolvable
    candidate no longer masks a servable one and a value naming nobody is
    ``None``, but the spelling is untouched — ``cohort_resolution`` folds region
    casing itself, so normalising here would change behaviour for no consumer."""
    from src.services.enum_labels import resolve_region_label

    resolve = functools.partial(resolve_region_label, allow_synonyms=True)
    return _structured_scope(payload, "region", resolve, normalise=False)
