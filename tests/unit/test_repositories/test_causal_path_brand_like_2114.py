"""#2114: the ``causal_paths`` brand filter is an ILIKE, so it must escape.

``search_paths_for_outcome`` (async) and ``search_paths_for_outcome_sync`` both
filter ``brand`` with ``.ilike``, which is a PATTERN match: an unescaped ``%``
or ``_`` in a caller-supplied brand broadens it. Measured read-only against the
LIVE ``causal_paths`` table (109 rows, brands Fabhalta/Kisqali/Remibrutinib)
before these tests were written:

    .ilike("brand","%")        -> 109 rows, ALL THREE brands
    .ilike("brand","Kis%")     ->  37 rows, Kisqali    (an ask that named neither)
    .ilike("brand","_isqali")  ->  37 rows, Kisqali
    .ilike("brand","Xolair")   ->   0 rows             (no metachars -> already closed)
    escaped, same inputs       ->   0 rows
    escaped "Kisqali"/"kisqali"->  37 rows, Kisqali    (honest case preserved)

So the defect is specifically about METACHARACTERS, not about preserving an
unrecognised brand: a plain unknown brand already fails closed.

Both twins are covered because both carry the defect. The sync twin is reached
from the orchestrator (``_causal_path_evidence``); the async twin from three
chat-path callers (``chatbot_tools`` x2, ``insights.causal_context``) that never
pass through the dispatcher at all.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import pytest

ROWS: List[Dict[str, Any]] = [
    {
        "start_node": "HCP Visits",
        "end_node": "Total Prescriptions (TRx)",
        "brand": brand,
        "confidence_level": 0.9,
        "path_id": f"p-{brand}",
        # Both twins append the default-exclude provenance predicate
        # (``.eq('is_synthetic', False)``), so a row without the column is
        # filtered out and every assertion below would pass vacuously.
        "is_synthetic": False,
    }
    for brand in ("Kisqali", "Fabhalta", "Remibrutinib")
]


def _ilike_to_regex(pattern: str) -> re.Pattern[str]:
    """Model the ``PostgREST -> PostgreSQL`` boundary, NOT PostgreSQL alone.

    THE LAYER IS THE POINT. An earlier version of this helper modelled real
    PostgreSQL ILIKE, accurately, and its docstring said so -- which is exactly
    what hid a live defect for a round (#2114 codex r6). The stack under test is
    PostgREST in FRONT of PostgreSQL, and PostgREST translates ``*`` to ``%``
    server-side before the SQL is built. A PostgreSQL-only model sends ``*``
    through ``re.escape`` as a literal and returns 0 rows where the real stack
    returns every row in the table, so the tests passed for the wrong reason.

    A fake that is faithful to the wrong layer is not a careful fake; it is an
    unfalsifiable one.

    WHICH characters translate was ENUMERATED against the live server rather
    than reasoned about -- 31 candidates probed with a positive and a negative
    control, using a real value containing an underscore (``acceptance_status``)
    as the discriminator:

        wildcards : ``*`` (multi-char), ``%`` (multi-char), ``_`` (single char)
        literal   : , . : ( ) " ' ? # & = + space | [ ] { } ^ $ ! ~ / @ < > ; - `

    ``\\`` is not a wildcard but IS special: a dangling one is an API ERROR
    rather than a miss, which is why it is escaped first and never emitted bare.

    Escaping with ``\\`` tames all three wildcards, verified POSITIVELY and not
    merely by absence -- ``acceptance\\_status`` still matches the literal value
    (3 rows), so escaping preserves honest values instead of breaking them.

    Not distinguishable on this data, and not relevant to the property under
    test: whether PostgREST rewrites ``\\*`` to ``\\%`` (a literal percent) or
    to a literal ``*``. No row anywhere in the table contains ``%`` or ``*``, so
    both predict 0, and both mean "does not wildcard". Modelled as the simpler
    rule, "``\\`` makes the next character literal".
    """
    out: List[str] = []
    i = 0
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\" and i + 1 < len(pattern):
            out.append(re.escape(pattern[i + 1]))
            i += 2
            continue
        if ch in ("%", "*"):  # PostgREST translates * to % before SQL sees it
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
        i += 1
    return re.compile("^" + "".join(out) + "$", re.IGNORECASE)


class _Query:
    """Applies the filters it is given, and RECORDS the brand pattern verbatim.

    Recording matters: a fake that ignored its arguments could not witness
    whether the value reaching ``.ilike`` was escaped.
    """

    def __init__(self, rec: Dict[str, Any], rows: List[Dict[str, Any]]):
        self._rec = rec
        self._rows = rows

    def _next(self, rows: List[Dict[str, Any]]) -> "_Query":
        # ``type(self)`` so the async subclass keeps its awaitable execute()
        # through the whole filter chain.
        return type(self)(self._rec, rows)

    def select(self, *_a: Any, **_k: Any) -> "_Query":
        return self

    def or_(self, _expr: str) -> "_Query":
        return self

    def ilike(self, col: str, value: str) -> "_Query":
        if col == "brand":
            self._rec.setdefault("brand_patterns", []).append(value)
        rx = _ilike_to_regex(str(value))
        return self._next([r for r in self._rows if rx.match(str(r.get(col, "")))])

    def eq(self, col: str, value: Any) -> "_Query":
        return self._next([r for r in self._rows if r.get(col) == value])

    def gte(self, col: str, value: Any) -> "_Query":
        return self._next([r for r in self._rows if (r.get(col) or 0) >= value])

    def order(self, *_a: Any, **_k: Any) -> "_Query":
        return self

    def range(self, *_a: Any, **_k: Any) -> "_Query":
        return self

    def limit(self, *_a: Any, **_k: Any) -> "_Query":
        return self

    def _result(self) -> Any:
        rows = self._rows

        class _R:
            data = rows

        return _R()

    def execute(self) -> Any:
        return self._result()


class _AsyncQuery(_Query):
    """The async twin awaits ``.execute()``; everything else is identical."""

    async def execute(self) -> Any:  # type: ignore[override]
        return self._result()


class _Client:
    def __init__(self, rec: Dict[str, Any]):
        self._rec = rec

    def table(self, _name: str) -> _Query:
        return _Query(self._rec, list(ROWS))


class _AsyncClient:
    def __init__(self, rec: Dict[str, Any]):
        self._rec = rec

    def table(self, _name: str) -> _AsyncQuery:
        return _AsyncQuery(self._rec, list(ROWS))


def _brands(rows: List[Dict[str, Any]]) -> List[str]:
    return sorted({str(r["brand"]) for r in rows})


def _sync(brand: str, rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    from src.repositories.causal_path import search_paths_for_outcome_sync

    return search_paths_for_outcome_sync(
        "Total Prescriptions (TRx)", client=_Client(rec), brand=brand, limit=15
    )


class TestSyncTwinEscapesTheBrandPattern:
    def test_an_honest_brand_still_matches_its_own_rows(self) -> None:
        """POSITIVE CONTROL. Every 'returns nothing' assertion below is
        worthless unless a real brand still returns its own rows here."""
        rec: Dict[str, Any] = {}
        rows = _sync("Kisqali", rec)
        assert _brands(rows) == ["Kisqali"], rows
        assert rec["brand_patterns"] == ["Kisqali"]

    def test_case_insensitivity_survives_the_escaping(self) -> None:
        assert _brands(_sync("kisqali", {})) == ["Kisqali"]

    @pytest.mark.parametrize(
        "attack", ["%", "Kis%", "_isqali", "%%", "K_sqali", "*", "Kis*", "K*sqali", "**"]
    )
    def test_a_wildcard_brand_matches_nothing_instead_of_broadening(self, attack: str) -> None:
        rows = _sync(attack, {})
        assert rows == [], f"{attack!r} broadened the filter to {_brands(rows)}"

    def test_the_pattern_reaching_ilike_is_escaped(self) -> None:
        rec: Dict[str, Any] = {}
        _sync("Kis%", rec)
        assert rec["brand_patterns"] == ["Kis\\%"]

    def test_the_star_pattern_reaching_ilike_is_escaped(self) -> None:
        """#2114 r6: `*` is PostgREST's own wildcard, translated to `%` server
        side. The r5 helper escaped `\\`, `%` and `_` and left `*` live, so
        `.ilike("brand","*")` still returned every brand on the live table."""
        rec: Dict[str, Any] = {}
        _sync("Kis*", rec)
        assert rec["brand_patterns"] == ["Kis\\*"]

    def test_an_unrecognised_brand_without_metacharacters_already_failed_closed(self) -> None:
        """Pins the boundary of this fix: preserving 'Xolair' is NOT the defect
        -- it matches no row either way. Only metacharacters broaden."""
        assert _sync("Xolair", {}) == []


class TestAsyncTwinEscapesTheBrandPattern:
    """The async twin has three chat-path callers that never touch the
    dispatcher, so its copy of the defect is reachable independently."""

    @pytest.mark.asyncio
    async def test_an_honest_brand_still_matches_its_own_rows(self) -> None:
        from src.repositories.causal_path import CausalPathRepository

        rec: Dict[str, Any] = {}
        repo = CausalPathRepository(_AsyncClient(rec))
        rows = await repo.search_paths_for_outcome(
            "Total Prescriptions (TRx)", brand="Kisqali", limit=15
        )
        assert _brands(rows) == ["Kisqali"], rows

    @pytest.mark.asyncio
    @pytest.mark.parametrize("attack", ["%", "Kis%", "_isqali", "*", "Kis*", "K*sqali"])
    async def test_a_wildcard_brand_matches_nothing_instead_of_broadening(
        self, attack: str
    ) -> None:
        from src.repositories.causal_path import CausalPathRepository

        rec: Dict[str, Any] = {}
        repo = CausalPathRepository(_AsyncClient(rec))
        rows = await repo.search_paths_for_outcome(
            "Total Prescriptions (TRx)", brand=attack, limit=15
        )
        assert rows == [], f"{attack!r} broadened the filter to {_brands(rows)}"

    @pytest.mark.asyncio
    async def test_the_pattern_reaching_ilike_is_escaped(self) -> None:
        from src.repositories.causal_path import CausalPathRepository

        rec: Dict[str, Any] = {}
        repo = CausalPathRepository(_AsyncClient(rec))
        await repo.search_paths_for_outcome("Total Prescriptions (TRx)", brand="Kis%", limit=15)
        assert rec["brand_patterns"] == ["Kis\\%"]
