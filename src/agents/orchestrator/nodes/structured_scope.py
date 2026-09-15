"""Reading a STRUCTURED scope out of a dispatch payload (#2114).

"Structured" means a caller DECIDED: the NLP layer's typed ``parsed_query``
entities, or the ``brand``/``region`` a chat caller stashed in ``user_context``.
The ask TEXT is evidence, never a decision, and is resolved elsewhere.

FIVE review rounds each found one more shape leaking through a "is this not
blank" test — two brands, brand-beside-another-brand's-indication, whitespace,
zero-width characters, and finally entity ORDERING — so the question is asked
positively here instead: is this a scope the substrate recognises? Validation
precedes source selection at BOTH levels, across entities and then across
sources, and recognition NORMALISES without ever DELETING, because
"unrecognised" and "absent" are different facts.
A blank names nobody; ``'Xolair'`` names somebody this substrate cannot serve,
and erasing it reads downstream as "no scope given" and silently widens the
query (r4 HIGH-2).

Lives beside ``dispatcher`` rather than inside it because that module is
ratchet-pinned (tests/unit/test_tests_meta/test_module_size_ratchet.py).
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional, cast


def _entity_values(payload: Dict[str, Any], entity_type: str) -> List[str]:
    """EVERY ``parsed_query.entities`` value of ``entity_type``, in order.

    Returns all of them rather than the first, because choosing here would
    choose BEFORE validating. The old single-value form returned the first
    entity whose value was ``.strip()``-truthy and stopped -- and U+200B is not
    whitespace, so a zero-width entity satisfied that test and the valid entity
    BEHIND it was never offered to the vocabulary (r5 HIGH-1). Measured:
    ``[U+200B, ' Kisqali ']`` bound nothing and ``['Xolair', ' Kisqali ']``
    bound ``'Xolair'``.

    That was the fifth shape to leak through a "is this not blank" test at this
    seam (two brands, brand+indication, whitespace, zero-width, ordering), so
    the fix is not another shape: hand every candidate to
    :func:`~src.services.enum_labels.first_named_scope` and let RECOGNITION
    decide, never position.

    A non-string or blank ``value`` is skipped here because it names nobody; a
    string that names something the substrate cannot serve is NOT skipped --
    that distinction belongs to the chooser, which normalises without deleting.

    Note the container guard: ``entities`` may be missing, ``None``, or -- from
    a malformed caller -- a TRUTHY NON-ITERABLE. ``... or []`` left the last one
    in place and ``for ent in 1`` raised ``TypeError``, killing the dispatch
    even when a perfectly valid ``user_context`` brand was present (r5 MEDIUM).
    A container that is not a list/tuple means "no structured entities", so the
    context fallback survives it. A ``str`` container is deliberately allowed
    through as an ordinary iterable: each character fails the ``dict`` check, so
    it already degraded to "absent" correctly.
    """
    parsed_query = payload.get("parsed_query")
    raw = parsed_query.get("entities") if isinstance(parsed_query, dict) else None
    entities = raw if isinstance(raw, (list, tuple, str)) else []
    out: List[str] = []
    for ent in entities:
        if (
            isinstance(ent, dict)
            and ent.get("type") == entity_type
            and isinstance(ent.get("value"), str)
        ):
            out.append(cast(str, ent["value"]))
    return out


def _structured_brand(payload: Dict[str, Any]) -> Optional[str]:
    """The brand a STRUCTURED source decided on — typed NLP entities, else the
    ``user_context`` a chat caller stashed. Recognised values are NORMALISED to
    their ``brand_type`` label; an unrecognised one is passed on UNCHANGED so the
    fail-closed downstream still sees it; only a value naming nobody is ``None``."""
    from src.services.enum_labels import resolve_brand_label

    return _structured_scope(payload, "brand", resolve_brand_label)


def _structured_scope(payload: Dict[str, Any], key: str, resolve: Any) -> Optional[str]:
    """Entities then ``user_context``, through ONE vocabulary-aware chooser so
    the clarify gate and the cohort extractor can never disagree (#2114)."""
    from src.services.enum_labels import first_named_scope

    ctx = payload.get("user_context") or {}
    stashed = ctx.get(key) if isinstance(ctx, dict) else None
    return first_named_scope((*_entity_values(payload, key), stashed), resolve)


def _structured_region(payload: Dict[str, Any]) -> Optional[str]:
    """The region a STRUCTURED source decided on, NORMALISED to its census label.

    Measured, not assumed: every live region predicate is
    ``LOWER(region::text) = LOWER($N)``, which folds CASE but never strips
    WHITESPACE — so a padded ``' West '`` from ``user_context`` matched no row
    and the KPI failed closed on a scope the caller had given correctly.
    ``cohort_resolution`` normalises for itself, but the Branch A KPI path passes
    this value straight into that predicate, so the normalisation belongs here
    where every consumer benefits (#2114 r4).
    """
    from src.services.enum_labels import resolve_region_label

    resolve = functools.partial(resolve_region_label, allow_synonyms=True)
    return _structured_scope(payload, "region", resolve)
