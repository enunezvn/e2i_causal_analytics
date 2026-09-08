"""#1972 -- every reader derives "active approval" the SAME way as the schema.

PR #1982 took the issue's option (a): ``expired`` is never written, expiry is
derived at read time from ``valid_until``. That only holds if every reader
derives it identically. Measured 2026-09-08 on the live droplet, they did not:

* SQL ``is_dag_approved()`` (010) tests
  ``approved AND (valid_until IS NULL OR valid_until >= CURRENT_DATE)``; the
  ``v_active_expert_approvals`` view filters on ``approval_status`` alone
  (expired rows included) and classifies each row's ``validity_status``:
  NULL -> ``'permanent'``, ``>= CURRENT_DATE`` -> ``'active'``, else ``'expired'``.
* Python ``is_dag_approved`` / ``get_dag_approval`` / ``get_expiring_reviews``
  used ``.gte("valid_until", today)`` -- and under SQL three-valued logic
  ``NULL >= date`` is not true, so a NULL row was EXCLUDED
  (``psql: select (null::date >= current_date) is true`` -> ``f``;
  PostgREST ``valid_until=gte.<today>`` -> 0 of 39 live rows,
  ``or=(valid_until.gte.<today>,valid_until.is.null)`` -> 39 of 39).
* ``get_reviews_for_dag`` used the OR form -> NULL INCLUDED.
* ``get_review_summary`` counted NULL as ``approved`` (never ``expired``).

Three definitions of one fact. The fix is ONE definition, owned by the schema
(NULL ``valid_until`` = permanent), implemented once in
``src/repositories/expert_review.py`` and routed through by every reader.

This file pins that structurally (AST: no raw ``valid_until`` filter outside
the shared helpers) and behaviourally (a PostgREST-faithful in-memory client:
comparisons against NULL are false, ``is.null`` matches None, ``or`` is any).
"""

from __future__ import annotations

import ast
import logging
import re
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytest

from src.repositories import expert_review as er
from src.repositories.expert_review import ExpertReviewRepository

REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_FILE = REPO_ROOT / "src" / "repositories" / "expert_review.py"
SQL_FILE = REPO_ROOT / "database" / "ml" / "010_causal_validation_tables.sql"

# The ONLY places a PostgREST filter on valid_until may live.
QUERY_HELPERS = frozenset({"_apply_active_validity", "_apply_expiring_window"})
# The ONLY places that may CLASSIFY a row's validity.
CLASSIFIERS = frozenset({"approval_validity", "is_active_approval"})
# Parses valid_until without classifying it; a caller that uses it must still classify.
PARSER = "_valid_until_date"
PURE_HELPERS = CLASSIFIERS | {PARSER}

TODAY = date.today()


def _iso(days: int) -> str:
    return (TODAY + timedelta(days=days)).isoformat()


# --------------------------------------------------------------------------
# Structural guard (AST) -- and its positive control
# --------------------------------------------------------------------------


def _is_valid_until_filter(call: ast.Call) -> bool:
    """A PostgREST builder call that filters on valid_until."""
    func = call.func
    if not isinstance(func, ast.Attribute) or not call.args:
        return False
    first = call.args[0]
    if func.attr in {"gte", "gt", "lte", "lt", "eq", "neq", "is_", "filter", "not_"}:
        return isinstance(first, ast.Constant) and first.value == "valid_until"
    if func.attr == "or_":
        return "valid_until" in ast.unparse(first)
    return False


def _filters_approved(fn: ast.AST) -> bool:
    """Contains ``.eq("approval_status", "approved")``."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "eq":
                consts = [a.value for a in node.args if isinstance(a, ast.Constant)]
                if consts[:2] == ["approval_status", "approved"]:
                    return True
    return False


def _reads_valid_until(fn: ast.AST) -> bool:
    """Reads a row's valid_until: ``row.get("valid_until")``, ``row["valid_until"]``
    or the parser ``_valid_until_date(row)`` (codex iter-1 MED: a reader that
    parses via the helper and then compares by hand is the same drift)."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == PARSER:
                return True
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
                a = node.args[0]
                if isinstance(a, ast.Constant) and a.value == "valid_until":
                    return True
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
            s = node.slice
            if isinstance(s, ast.Constant) and s.value == "valid_until":
                return True
    return False


def _declares_expired_included(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """A reader may legitimately want stored-'approved' rows regardless of
    validity (an audit/history view). It must SAY so: an ``include_expired``
    parameter, or a docstring stating that expired rows are included. Silence
    is what R2 forbids (codex iter-1 MED: R2 must not conflate stored status
    with active approval, but it must not let the choice be implicit either)."""
    if any(a.arg == "include_expired" for a in fn.args.args + fn.args.kwonlyargs):
        return True
    doc = (ast.get_docstring(fn) or "").lower()
    # codex iter-2/iter-3 MED: "expired rows are NOT included" / "never
    # including expired rows" must not count as a declaration. Accept only an
    # affirmative phrase, and reject the docstring outright if any negation
    # word ("not", "never", "no", "without", "neither", "nor"), "exclud" or an
    # "active ... only" restriction appears in the same sentence as "includ"
    # or "expired". Conservative on purpose: an `include_expired` parameter is
    # the unambiguous, structural way to declare a historical reader.
    affirmative = re.search(
        r"\b(including|includes)\s+expired\b|\bexpired\b(\s+\w+){0,2}\s+included\b", doc
    )
    # codex iter-4 MED: the negation may come AFTER the phrase ("Including
    # expired rows is never permitted"), so test both orders.
    neg = r"\b(not|never|no|without|neither|nor)\b"
    key = r"\b(includ|expired)"
    negated = re.search(
        rf"{neg}[^.;]*{key}|{key}[^.;]*{neg}"
        r"|\bexclud"
        r"|\bactive\b[^.;]*\bonly\b|\bonly\b[^.;]*\bactive\b",
        doc,
    )
    return bool(affirmative) and not negated


def _called_names(fn: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def _functions(tree: ast.AST) -> List[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def find_validity_violations(source: str) -> List[str]:
    """Return every way ``source`` lets a reader drift from the one definition.

    R1 a raw valid_until filter lives outside QUERY_HELPERS
    R2 a function filters approval_status=='approved' without a query helper,
       and does not declare that expired rows are deliberately included
    R3 a function reads (or parses) a row's valid_until without classifying it
       through approval_validity()/is_active_approval()
    R4 a helper is missing altogether
    """
    tree = ast.parse(source)
    fns = _functions(tree)
    defined = {fn.name for fn in fns}
    out: List[str] = []
    for helper in sorted(QUERY_HELPERS | PURE_HELPERS):
        if helper not in defined:
            out.append(f"R4 missing shared helper {helper}")
    for fn in fns:
        if fn.name in QUERY_HELPERS:
            continue
        raw = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and _is_valid_until_filter(n)]
        if raw:
            out.append(
                f"R1 {fn.name}: raw valid_until filter outside the shared helpers: "
                f"{ast.unparse(raw[0])}"
            )
        called = _called_names(fn)
        if (
            _filters_approved(fn)
            and not (called & QUERY_HELPERS)
            and not _declares_expired_included(fn)
        ):
            out.append(
                f"R2 {fn.name}: filters approval_status=='approved' without routing "
                "validity through a shared query helper (or declaring that expired "
                "rows are included)"
            )
        if fn.name not in PURE_HELPERS and _reads_valid_until(fn) and not (called & CLASSIFIERS):
            out.append(
                f"R3 {fn.name}: interprets a row's valid_until by hand instead of via "
                "approval_validity()/is_active_approval()"
            )
    return out


COMPLIANT_MODULE = """
from datetime import date

def _valid_until_date(row):
    v = row.get("valid_until")
    return None if v is None else date.fromisoformat(v)

def approval_validity(row, today=None):
    d = _valid_until_date(row)
    if d is None:
        return "permanent"
    return "active" if d >= (today or date.today()) else "expired"

def is_active_approval(row, today=None):
    return row.get("approval_status") == "approved" and approval_validity(row, today) != "expired"

def _apply_active_validity(query, today=None):
    return query.or_(f"valid_until.gte.{(today or date.today()).isoformat()},valid_until.is.null")

def _apply_expiring_window(query, days, today=None):
    t = today or date.today()
    return query.gte("valid_until", t.isoformat()).lte("valid_until", t.isoformat())

class Repo:
    async def is_dag_approved(self, h):
        q = self.client.table("t").select("*").eq("approval_status", "approved")
        q = _apply_active_validity(q)
        return bool((await q.execute()).data)

    async def get_expiring(self):
        q = self.client.table("t").select("*").eq("approval_status", "approved")
        return (await _apply_expiring_window(q, 14).execute()).data

    async def summary(self):
        rows = (await self.client.table("t").select("*").execute()).data
        return sum(1 for r in rows if approval_validity(r) == "expired")
"""

BYPASSING_READER = COMPLIANT_MODULE.replace(
    "        q = _apply_active_validity(q)\n",
    '        q = q.gte("valid_until", date.today().isoformat())\n',
)
assert BYPASSING_READER != COMPLIANT_MODULE

HAND_ROLLED_SUMMARY = COMPLIANT_MODULE.replace(
    'return sum(1 for r in rows if approval_validity(r) == "expired")',
    'return sum(1 for r in rows if r.get("valid_until") and r["valid_until"] < "2000")',
)
assert HAND_ROLLED_SUMMARY != COMPLIANT_MODULE

# codex iter-1 MED: parses via the helper, then re-derives validity by hand --
# and gets it wrong (a permanent approval is excluded).
PARSE_THEN_COMPARE = COMPLIANT_MODULE.replace(
    'return sum(1 for r in rows if approval_validity(r) == "expired")',
    "d = [_valid_until_date(r) for r in rows]\n"
    "        return sum(1 for x in d if x is not None and x >= date.today())",
)
assert PARSE_THEN_COMPARE != COMPLIANT_MODULE

HISTORICAL_READER_SILENT = (
    COMPLIANT_MODULE
    + """
    async def all_approvals_ever(self):
        q = self.client.table("t").select("*").eq("approval_status", "approved")
        return (await q.execute()).data
"""
)

HISTORICAL_READER_DECLARED = (
    COMPLIANT_MODULE
    + '''
    async def all_approvals_ever(self):
        """Every stored approval, expired ones included (audit view)."""
        q = self.client.table("t").select("*").eq("approval_status", "approved")
        return (await q.execute()).data
'''
)

# codex iter-2 MED: a docstring that mentions both words while EXCLUDING
# expired rows is a documented active reader that lost its validity filter.
HISTORICAL_READER_NEGATED = [
    COMPLIANT_MODULE
    + f'''
    async def active_only(self):
        """{doc}"""
        q = self.client.table("t").select("*").eq("approval_status", "approved")
        return (await q.execute()).data
'''
    for doc in (
        "Active approvals. Expired rows are not included.",
        "Approvals, excluding expired ones.",
        "Approvals without expired rows; nothing else is included.",
        # codex iter-3 MED: the affirmative regex matched "including expired"
        # and the negation list had no "never".
        "Active approvals only, never including expired rows.",
        "Current approvals; no expired rows are included.",
        # codex iter-4 MED: negation AFTER the affirmative phrase.
        "Including expired rows is never permitted.",
    )
]


class TestStructuralGuard:
    """No reader may re-derive validity on its own."""

    def test_positive_control_compliant_module_has_no_violations(self):
        assert find_validity_violations(COMPLIANT_MODULE) == []

    def test_positive_control_bypassing_reader_is_caught(self):
        found = find_validity_violations(BYPASSING_READER)
        assert any(v.startswith("R1 is_dag_approved") for v in found), found
        assert any(v.startswith("R2 is_dag_approved") for v in found), found

    def test_positive_control_hand_rolled_summary_is_caught(self):
        found = find_validity_violations(HAND_ROLLED_SUMMARY)
        assert found == [
            "R3 summary: interprets a row's valid_until by hand instead of via "
            "approval_validity()/is_active_approval()"
        ]

    def test_positive_control_parse_then_compare_bypass_is_caught(self):
        found = find_validity_violations(PARSE_THEN_COMPARE)
        assert found == [
            "R3 summary: interprets a row's valid_until by hand instead of via "
            "approval_validity()/is_active_approval()"
        ]

    def test_positive_control_silent_historical_reader_is_caught(self):
        found = find_validity_violations(HISTORICAL_READER_SILENT)
        assert any(v.startswith("R2 all_approvals_ever") for v in found), found

    def test_positive_control_declared_historical_reader_is_allowed(self):
        assert find_validity_violations(HISTORICAL_READER_DECLARED) == []

    @pytest.mark.parametrize(
        "module",
        HISTORICAL_READER_NEGATED,
        ids=[
            "not-included",
            "excluding",
            "without",
            "never-including",
            "no-expired",
            "never-after",
        ],
    )
    def test_positive_control_negated_declaration_is_still_caught(self, module):
        found = find_validity_violations(module)
        assert any(v.startswith("R2 active_only") for v in found), found

    def test_positive_control_missing_helper_is_caught(self):
        stripped = COMPLIANT_MODULE.replace("def _apply_expiring_window", "def _renamed")
        found = find_validity_violations(stripped)
        assert "R4 missing shared helper _apply_expiring_window" in found

    def test_the_real_repository_routes_every_reader_through_the_helpers(self):
        found = find_validity_violations(REPO_FILE.read_text(encoding="utf-8"))
        assert found == [], "\n".join(found)

    def test_the_python_definition_mirrors_the_schema_definition(self):
        """If 010's predicate changes, the Python helpers must change with it."""
        sql = SQL_FILE.read_text(encoding="utf-8")
        fn = re.search(r"FUNCTION is_dag_approved\(.*?\$\$ LANGUAGE plpgsql", sql, re.S)
        assert fn, "SQL is_dag_approved not found"
        assert "(valid_until IS NULL OR valid_until >= CURRENT_DATE)" in fn.group(0)
        assert "WHEN valid_until IS NULL THEN 'permanent'" in sql
        doc = (
            ast.get_docstring(
                next(
                    f
                    for f in _functions(ast.parse(REPO_FILE.read_text()))
                    if f.name == "approval_validity"
                )
            )
            or ""
        )
        assert "v_active_expert_approvals" in doc, (
            "approval_validity must cite the schema view it mirrors so the two are changed together"
        )


# --------------------------------------------------------------------------
# A PostgREST-faithful in-memory client (records every filter call)
# --------------------------------------------------------------------------

_Pred = Callable[[Dict[str, Any]], bool]


def _cmp(op: str, col: str, val: str) -> _Pred:
    """Three-valued: any comparison against NULL is not true (psql probe)."""

    def pred(row: Dict[str, Any]) -> bool:
        v = row.get(col)
        if v is None:
            return False
        return {
            "gte": v >= val,
            "gt": v > val,
            "lte": v <= val,
            "lt": v < val,
            "eq": v == val,
        }[op]

    return pred


def _parse_or(filters: str) -> _Pred:
    parts: List[_Pred] = []
    for term in filters.split(","):
        col, op, val = term.split(".", 2)
        if op == "is":
            assert val == "null", term
            parts.append(lambda r, c=col: r.get(c) is None)
        else:
            parts.append(_cmp(op, col, val))
    return lambda r: any(p(r) for p in parts)


class FakeQuery:
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        log: List[Tuple[str, Any]],
        fail_with: Optional[BaseException] = None,
    ):
        self._rows = rows
        self._log = log
        self._fail_with = fail_with
        self._preds: List[_Pred] = []
        self._order: Optional[Tuple[str, bool]] = None
        self._limit: Optional[int] = None
        self._insert: Optional[Dict[str, Any]] = None
        self._update: Optional[Dict[str, Any]] = None

    def select(self, cols: str = "*") -> "FakeQuery":
        self._log.append(("select", cols))
        return self

    def eq(self, col: str, val: Any) -> "FakeQuery":
        self._log.append(("eq", col, val))
        self._preds.append(_cmp("eq", col, val))
        return self

    def gte(self, col: str, val: Any) -> "FakeQuery":
        self._log.append(("gte", col, val))
        self._preds.append(_cmp("gte", col, val))
        return self

    def lte(self, col: str, val: Any) -> "FakeQuery":
        self._log.append(("lte", col, val))
        self._preds.append(_cmp("lte", col, val))
        return self

    def is_(self, col: str, val: Any) -> "FakeQuery":
        self._log.append(("is_", col, val))
        assert val == "null"
        self._preds.append(lambda r: r.get(col) is None)
        return self

    def or_(self, filters: str) -> "FakeQuery":
        self._log.append(("or_", filters))
        self._preds.append(_parse_or(filters))
        return self

    def order(self, col: str, desc: bool = False) -> "FakeQuery":
        self._log.append(("order", col, desc))
        self._order = (col, desc)
        return self

    def limit(self, n: int) -> "FakeQuery":
        self._log.append(("limit", n))
        self._limit = n
        return self

    def insert(self, row: Dict[str, Any]) -> "FakeQuery":
        self._log.append(("insert", dict(row)))
        self._insert = dict(row)
        return self

    def update(self, data: Dict[str, Any]) -> "FakeQuery":
        self._log.append(("update", dict(data)))
        self._update = dict(data)
        return self

    async def execute(self) -> SimpleNamespace:
        if self._fail_with is not None:
            raise self._fail_with
        if self._insert is not None:
            new = {"review_id": f"rev-{len(self._rows) + 1}", **self._insert}
            self._rows.append(new)
            return SimpleNamespace(data=[new])
        if self._update is not None:
            # PostgREST UPDATE ... WHERE <all filters>; returns the touched rows.
            touched = [r for r in self._rows if all(p(r) for p in self._preds)]
            for r in touched:
                r.update(self._update)
            return SimpleNamespace(data=[dict(r) for r in touched])
        out = [r for r in self._rows if all(p(r) for p in self._preds)]
        if self._order:
            col, desc = self._order
            out.sort(key=lambda r: (r.get(col) is None, r.get(col) or ""), reverse=desc)
        if self._limit is not None:
            out = out[: self._limit]
        return SimpleNamespace(data=out)


class FakeClient:
    def __init__(self, rows: List[Dict[str, Any]], fail_with: Optional[BaseException] = None):
        self.rows = [dict(r) for r in rows]
        self.log: List[Tuple[str, Any]] = []
        self.fail_with = fail_with

    def table(self, name: str) -> FakeQuery:
        assert name == "expert_reviews", name
        return FakeQuery(self.rows, self.log, self.fail_with)


def _row(status: str, valid_until: Optional[str], **extra: Any) -> Dict[str, Any]:
    base = {
        "review_id": f"rev-{status}-{valid_until}",
        "dag_version_hash": "dag-1",
        "approval_status": status,
        "valid_until": valid_until,
        "approved_at": None if status != "approved" else f"{_iso(-90)}T00:00:00+00:00",
    }
    base.update(extra)
    return base


PERMANENT = _row("approved", None)
EXPIRED = _row("approved", _iso(-1))
EXPIRES_TODAY = _row("approved", _iso(0))
EXPIRING_SOON = _row("approved", _iso(7))
FAR_FUTURE = _row("approved", _iso(60))
PENDING = _row("pending", None)
REJECTED = _row("rejected", None)


def _repo(rows: List[Dict[str, Any]]) -> Tuple[ExpertReviewRepository, FakeClient]:
    client = FakeClient(rows)
    repo = ExpertReviewRepository(supabase_client=client)
    return repo, client


class TestFakeIsFaithful:
    """The double's semantics are pinned to what psql/PostgREST returned live."""

    def test_comparison_against_null_is_not_true(self):
        assert _cmp("gte", "valid_until", _iso(0))({"valid_until": None}) is False

    def test_or_with_is_null_admits_a_null_row(self):
        pred = _parse_or(f"valid_until.gte.{_iso(0)},valid_until.is.null")
        assert pred({"valid_until": None}) is True
        assert pred({"valid_until": _iso(-1)}) is False
        assert pred({"valid_until": _iso(0)}) is True


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


class TestPureHelpers:
    def test_null_valid_until_is_permanent(self):
        assert er.approval_validity(PERMANENT, TODAY) == "permanent"

    def test_today_is_still_active_boundary_matches_sql_gte(self):
        assert er.approval_validity(EXPIRES_TODAY, TODAY) == "active"

    def test_yesterday_is_expired(self):
        assert er.approval_validity(EXPIRED, TODAY) == "expired"

    def test_accepts_a_date_object_as_well_as_the_postgrest_iso_string(self):
        assert er.approval_validity({"valid_until": TODAY - timedelta(days=1)}, TODAY) == "expired"

    @pytest.mark.parametrize("bad", ["", "2026-09-08garbage", "not-a-date"])
    def test_malformed_valid_until_is_never_silently_classified(self, bad):
        """codex iter-1 LOW: a lenient parse turned garbage into a date. A DATE
        column never yields these; if one ever appears, raising is the honest
        outcome (the summary's outer except logs it and returns zeros)."""
        with pytest.raises(ValueError):
            er.approval_validity({"valid_until": bad}, TODAY)

    def test_is_active_requires_approved_status(self):
        assert er.is_active_approval(PERMANENT, TODAY) is True
        assert er.is_active_approval(EXPIRES_TODAY, TODAY) is True
        assert er.is_active_approval(EXPIRED, TODAY) is False
        assert er.is_active_approval(PENDING, TODAY) is False
        assert er.is_active_approval(REJECTED, TODAY) is False


# --------------------------------------------------------------------------
# Readers, through the faithful fake
# --------------------------------------------------------------------------


class TestActiveApprovalReaders:
    async def test_is_dag_approved_treats_null_valid_until_as_active(self):
        repo, client = _repo([PERMANENT])
        assert await repo.is_dag_approved("dag-1") is True

    async def test_is_dag_approved_uses_the_or_is_null_form_not_a_bare_gte(self):
        repo, client = _repo([PERMANENT])
        await repo.is_dag_approved("dag-1")
        ors = [f for (m, *a) in client.log if m == "or_" for f in a]
        assert any("valid_until.is.null" in f for f in ors), client.log
        assert not any(m == "gte" and a[0] == "valid_until" for (m, *a) in client.log), client.log

    async def test_is_dag_approved_rejects_an_expired_row(self):
        repo, _ = _repo([EXPIRED])
        assert await repo.is_dag_approved("dag-1") is False

    async def test_is_dag_approved_is_fail_closed_without_a_client(self, caplog):
        repo = ExpertReviewRepository(supabase_client=None)
        with caplog.at_level(logging.WARNING):
            assert await repo.is_dag_approved("dag-1") is False
        assert any("No Supabase client" in r.getMessage() for r in caplog.records)

    async def test_get_dag_approval_returns_the_permanent_row(self):
        repo, _ = _repo([EXPIRED, PERMANENT])
        got = await repo.get_dag_approval("dag-1")
        assert got is not None and got["valid_until"] is None

    async def test_get_dag_approval_skips_expired(self):
        repo, _ = _repo([EXPIRED, PENDING])
        assert await repo.get_dag_approval("dag-1") is None

    async def test_get_reviews_for_dag_default_keeps_permanent_drops_expired(self):
        repo, _ = _repo([PERMANENT, EXPIRED, PENDING, FAR_FUTURE])
        ids = {r["review_id"] for r in await repo.get_reviews_for_dag("dag-1")}
        assert PERMANENT["review_id"] in ids
        assert PENDING["review_id"] in ids
        assert FAR_FUTURE["review_id"] in ids
        assert EXPIRED["review_id"] not in ids

    @pytest.mark.parametrize(
        "row",
        [PERMANENT, EXPIRED, EXPIRES_TODAY, EXPIRING_SOON, FAR_FUTURE, PENDING, REJECTED],
        ids=lambda r: r["review_id"],
    )
    async def test_query_side_and_pure_side_agree_row_by_row(self, row):
        """The drift this issue is about: the gate's query must say exactly what
        the pure predicate says, for every kind of row."""
        repo, _ = _repo([row])
        assert await repo.is_dag_approved("dag-1") is er.is_active_approval(row, TODAY)


class TestExpiringReviews:
    async def test_permanent_approval_is_never_expiring_soon(self):
        repo, client = _repo([PERMANENT, EXPIRING_SOON, FAR_FUTURE, EXPIRED])
        ids = [r["review_id"] for r in await repo.get_expiring_reviews(14)]
        assert ids == [EXPIRING_SOON["review_id"]]
        assert not any(m == "or_" and "is.null" in a[0] for (m, *a) in client.log), client.log

    async def test_window_includes_today_and_the_horizon(self):
        edge = _row("approved", _iso(14))
        repo, _ = _repo([EXPIRES_TODAY, edge, _row("approved", _iso(15))])
        ids = {r["review_id"] for r in await repo.get_expiring_reviews(14)}
        assert ids == {EXPIRES_TODAY["review_id"], edge["review_id"]}

    def test_docstring_says_permanent_is_never_expiring(self):
        doc = ExpertReviewRepository.get_expiring_reviews.__doc__ or ""
        assert "permanent" in doc.lower() and "never" in doc.lower()


class TestSummary:
    async def test_permanent_counts_as_approved_only(self):
        repo, _ = _repo([PERMANENT])
        got = await repo.get_review_summary()
        assert got == {"pending": 0, "approved": 1, "rejected": 0, "expired": 0, "expiring_soon": 0}

    async def test_expired_counts_as_expired_only(self):
        repo, _ = _repo([EXPIRED])
        got = await repo.get_review_summary()
        assert got == {"pending": 0, "approved": 0, "rejected": 0, "expired": 1, "expiring_soon": 0}

    async def test_expiring_soon_is_a_subset_of_approved(self):
        repo, _ = _repo([EXPIRING_SOON, FAR_FUTURE, PENDING, REJECTED])
        got = await repo.get_review_summary()
        assert got == {"pending": 1, "approved": 2, "rejected": 1, "expired": 0, "expiring_soon": 1}

    async def test_summary_agrees_with_the_pure_predicate(self):
        rows = [PERMANENT, EXPIRED, EXPIRES_TODAY, EXPIRING_SOON, FAR_FUTURE, PENDING, REJECTED]
        repo, _ = _repo(rows)
        got = await repo.get_review_summary()
        assert got["approved"] == sum(er.is_active_approval(r, TODAY) for r in rows)
        assert got["expired"] == sum(
            r["approval_status"] == "approved" and er.approval_validity(r, TODAY) == "expired"
            for r in rows
        )


class TestRenewalRow:
    async def test_renewal_is_a_pending_row_with_no_validity_of_its_own(self):
        """Validity is assigned only when the renewal is APPROVED (submit_review
        writes valid_until = today + validity_days); the renewal row itself
        must not carry one, or the summary would see a pending row with a date."""
        repo, client = _repo([PERMANENT])
        new_id = await repo.renew_review(PERMANENT["review_id"], reviewer_id="u2")
        assert new_id is not None
        inserted = next(a[0] for (m, *a) in client.log if m == "insert")
        assert inserted["approval_status"] == "pending"
        assert inserted["supersedes_review_id"] == PERMANENT["review_id"]
        for k in ("valid_from", "valid_until", "approved_at"):
            assert k not in inserted, inserted

    async def test_approving_a_renewal_never_revokes_a_permanent_original(self):
        """codex iter-1 HIGH: nothing filters on supersedes_review_id, so the
        original stays eligible. While the renewal is active it is the record
        reported (newest approved_at); once it expires the permanent original
        is reported again -- the DAG is approved throughout."""
        original = dict(PERMANENT, review_id="rev-orig", approved_at=f"{_iso(-400)}T00:00:00+00:00")
        active_renewal = _row(
            "approved",
            _iso(30),
            review_id="rev-renew",
            supersedes_review_id="rev-orig",
            approved_at=f"{_iso(-60)}T00:00:00+00:00",
        )
        expired_renewal = dict(active_renewal, valid_until=_iso(-1))

        repo, _ = _repo([original, active_renewal])
        assert (await repo.get_dag_approval("dag-1"))["review_id"] == "rev-renew"
        assert await repo.is_dag_approved("dag-1") is True

        repo, _ = _repo([original, expired_renewal])
        assert (await repo.get_dag_approval("dag-1"))["review_id"] == "rev-orig"
        assert await repo.is_dag_approved("dag-1") is True

    async def test_a_time_limited_original_is_reported_again_only_while_still_active(self):
        """codex iter-2 HIGH: "the original again once the renewal expires"
        holds only while the original itself is active. Both expired -> None."""
        still_valid = _row(
            "approved", _iso(20), review_id="rev-orig", approved_at=f"{_iso(-70)}T00:00:00+00:00"
        )
        lapsed = dict(still_valid, valid_until=_iso(-2))
        expired_renewal = _row(
            "approved",
            _iso(-1),
            review_id="rev-renew",
            supersedes_review_id="rev-orig",
            approved_at=f"{_iso(-60)}T00:00:00+00:00",
        )
        repo, _ = _repo([still_valid, expired_renewal])
        assert (await repo.get_dag_approval("dag-1"))["review_id"] == "rev-orig"

        repo, _ = _repo([lapsed, expired_renewal])
        assert await repo.get_dag_approval("dag-1") is None
        assert await repo.is_dag_approved("dag-1") is False

    def test_docstring_states_what_renewing_a_permanent_approval_does(self):
        doc = ExpertReviewRepository.renew_review.__doc__ or ""
        assert "permanent" in doc.lower()
        assert "supersedes_review_id" in doc
        assert "never revoke" in doc.lower(), (
            "the docstring must not claim a renewal makes a permanent approval time-limited"
        )
        assert "still active" in doc.lower(), (
            "the original is reported again after the renewal expires ONLY while it is "
            "itself still active -- the docstring must say so (codex iter-2)"
        )


# --------------------------------------------------------------------------
# R1 (lane-1971 audit): a store error must not read as "nothing on file"
# --------------------------------------------------------------------------


class StoreDown(RuntimeError):
    """Stands in for any transient client/transport failure."""


class TestReadErrorsAreNotSwallowed:
    """The gate's rejection probe reads an empty result as "structure clear".
    Before this change both readers turned a query error into []/None, so an
    outage looked like a clean slate. They must raise (after logging) so the
    gate can report ``unavailable`` and the node ``unknown``."""

    async def test_get_dag_approval_raises_on_store_error(self, caplog):
        repo = ExpertReviewRepository(supabase_client=FakeClient([PERMANENT], StoreDown("down")))
        with caplog.at_level(logging.ERROR):
            with pytest.raises(StoreDown):
                await repo.get_dag_approval("dag-1")
        assert any("Failed to get DAG approval" in r.getMessage() for r in caplog.records)

    async def test_get_reviews_for_dag_raises_on_store_error(self, caplog):
        repo = ExpertReviewRepository(supabase_client=FakeClient([PENDING], StoreDown("down")))
        with caplog.at_level(logging.ERROR):
            with pytest.raises(StoreDown):
                await repo.get_reviews_for_dag("dag-1")
        assert any("Failed to get reviews for DAG" in r.getMessage() for r in caplog.records)

    async def test_no_client_early_returns_are_unchanged(self):
        repo = ExpertReviewRepository(supabase_client=None)
        assert await repo.get_dag_approval("dag-1") is None
        assert await repo.get_reviews_for_dag("dag-1") == []

    def test_docstrings_say_they_raise(self):
        for fn in (
            ExpertReviewRepository.get_dag_approval,
            ExpertReviewRepository.get_reviews_for_dag,
        ):
            assert "Raises" in (fn.__doc__ or ""), fn.__name__


# --------------------------------------------------------------------------
# R2 (lane-1971 audit): only a PENDING row can be resolved
# --------------------------------------------------------------------------


class TestSubmitReviewResolvesOnlyPending:
    async def test_pending_row_is_resolved(self):
        repo, client = _repo([PENDING])
        ok = await repo.submit_review(PENDING["review_id"], "approved", {"c": True})
        assert ok is True
        assert client.rows[0]["approval_status"] == "approved"
        assert ("eq", "approval_status", "pending") in client.log, client.log

    async def test_already_approved_older_row_cannot_be_re_resolved(self):
        """An older approval re-resolved to 'rejected' while a NEWER approval
        exists would silently flip history. The UPDATE must carry the pending
        filter so it matches zero rows -> False (route -> 404)."""
        older = dict(FAR_FUTURE, review_id="rev-older")
        newer = dict(FAR_FUTURE, review_id="rev-newer")
        repo, client = _repo([older, newer])
        ok = await repo.submit_review("rev-older", "rejected", {"c": False})
        assert ok is False
        assert client.rows[0]["approval_status"] == "approved", "the row must be untouched"
        assert ("eq", "approval_status", "pending") in client.log, client.log

    async def test_rejected_row_cannot_be_flipped_to_approved(self):
        repo, client = _repo([REJECTED])
        assert await repo.submit_review(REJECTED["review_id"], "approved", {"c": True}) is False
        assert client.rows[0]["approval_status"] == "rejected"

    def test_docstring_states_pending_only_is_enforced(self):
        doc = ExpertReviewRepository.submit_review.__doc__ or ""
        assert "pending" in doc.lower() and "approval_status" in doc, (
            "the docstring must say the UPDATE itself carries approval_status = 'pending'"
        )
