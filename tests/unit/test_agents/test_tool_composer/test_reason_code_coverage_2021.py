"""#2021: every refusal raise site names a code. An AST check, so it cannot drift.

A grep would miss a multi-line call and match a comment. Walking the AST of the
module source finds every ``raise ToolRefusalError(...)`` / ``raise ToolInputError(...)``
however it is formatted.
"""

import ast
from pathlib import Path

import pytest

from src.agents.tool_composer.reason_codes import ReasonCode

_TARGETS = {"ToolRefusalError", "ToolInputError"}
_SOURCES = [
    Path("src/agents/tool_composer/tool_registrations.py"),
]


def _raise_sites(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        func = node.exc.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in _TARGETS:
            yield node.lineno, name, node.exc


@pytest.mark.parametrize("path", _SOURCES, ids=lambda p: p.name)
def test_every_raise_site_names_a_reason_code(path: Path):
    uncoded = []
    for lineno, name, call in _raise_sites(path):
        keywords = {kw.arg for kw in call.keywords}
        if "reason_code" not in keywords:
            uncoded.append(f"{path}:{lineno} {name}")
    assert uncoded == [], "raise sites without a reason_code (#2021):\n  " + "\n  ".join(uncoded)


def _literal_member(value: ast.expr, members: set) -> bool:
    """True when ``value`` is a literal ``ReasonCode.MEMBER``."""
    return (
        isinstance(value, ast.Attribute)
        and isinstance(value.value, ast.Name)
        and value.value.id == "ReasonCode"
        and value.attr in members
    )


def _threaded_params(tree: ast.Module) -> dict:
    """Functions that take ``reason_code: ReasonCode`` and pass it to their raise.

    One guard can serve two roles — ``_refuse_unless_binary_01`` checks the
    ``treatment`` of two tools and the ``outcome`` of a third — so the only place
    the code can be chosen correctly is the call site, where the role is known.
    Such a function is allowed to raise with its parameter INSTEAD of a literal,
    but then every call to it must supply a literal, which
    ``test_a_threaded_reason_code_is_literal_at_every_call_site`` enforces. The
    drift guard follows one hop; it is not relaxed.
    """
    threaded = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for arg in node.args.kwonlyargs + node.args.args:
            annotation = getattr(arg.annotation, "id", None)
            if arg.arg == "reason_code" and annotation == "ReasonCode":
                threaded[node.name] = node
    return threaded


@pytest.mark.parametrize("path", _SOURCES, ids=lambda p: p.name)
def test_every_reason_code_is_a_member_of_the_closed_set(path: Path):
    """A literal string or an unknown ReasonCode attribute is a drift vector."""
    members = {c.name for c in ReasonCode}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    threaded = _threaded_params(tree)
    # the line span of each function that legitimately raises with its parameter
    spans = [(f.lineno, f.end_lineno) for f in threaded.values()]
    bad = []
    for lineno, name, call in _raise_sites(path):
        for kw in call.keywords:
            if kw.arg != "reason_code":
                continue
            value = kw.value
            if _literal_member(value, members):
                continue
            inside_threaded = any(start <= lineno <= end for start, end in spans)
            if inside_threaded and isinstance(value, ast.Name) and value.id == "reason_code":
                continue
            bad.append(f"{path}:{lineno} {ast.unparse(value)} (not ReasonCode.MEMBER)")
    assert bad == [], "reason_code values outside the closed set:\n  " + "\n  ".join(bad)


def _threaded_call_violations(tree: ast.Module, label: str) -> tuple:
    """Calls to a threaded guard that do not supply a literal ``ReasonCode.MEMBER``.

    Returns ``(bad, seen)``: the offending call sites, and how many calls each
    threaded guard received.
    """
    members = {c.name for c in ReasonCode}
    threaded = _threaded_params(tree)
    bad = []
    seen = dict.fromkeys(threaded, 0)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # ``_guard(...)`` and ``module._guard(...)`` reach the same guard.
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        if name not in threaded:
            continue
        seen[name] += 1
        supplied = {kw.arg: kw.value for kw in node.keywords}
        value = supplied.get("reason_code")
        if value is None:
            bad.append(f"{label}:{node.lineno} {name}(...) supplies no reason_code")
        elif not _literal_member(value, members):
            bad.append(f"{label}:{node.lineno} {name}(reason_code={ast.unparse(value)})")
    return bad, seen


@pytest.mark.parametrize("path", _SOURCES, ids=lambda p: p.name)
def test_a_threaded_reason_code_is_literal_at_every_call_site(path: Path):
    """The one hop the closed-set check allows must itself end in a literal.

    Without this, threading the code through a parameter would be an unchecked
    hole: a caller could pass a string, a variable, or nothing at all.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert _threaded_params(tree), "no threaded guard found — delete this test if that is intended"
    bad, seen = _threaded_call_violations(tree, str(path))
    assert bad == [], "threaded reason_code not a literal member:\n  " + "\n  ".join(bad)
    uncalled = [name for name, count in seen.items() if count == 0]
    assert uncalled == [], f"threaded guard is never called, so nothing pins it: {uncalled}"


_SYNTHETIC_GUARD = """
def _guard(value, *, reason_code: ReasonCode):
    raise ToolRefusalError("x", reason_code=reason_code)
"""


@pytest.mark.parametrize(
    "call",
    [
        pytest.param('_guard(1, reason_code="oops")', id="name-string"),
        pytest.param('tr._guard(1, reason_code="oops")', id="attribute-string"),
        pytest.param("tr._guard(1)", id="attribute-no-code"),
        pytest.param("tr._guard(1, reason_code=ReasonCode.NOT_A_MEMBER)", id="attribute-unknown"),
        pytest.param("a.b._guard(1, reason_code=code)", id="nested-attribute-variable"),
    ],
)
def test_the_call_site_guard_flags_a_bad_call_however_the_guard_is_reached(call):
    """Proof of teeth: a green run on the real file proves nothing about a call shape it
    does not contain, such as ``tr._refuse_unless_binary_01(..., reason_code="oops")``."""
    tree = ast.parse(_SYNTHETIC_GUARD + f"\ndef caller():\n    {call}\n")
    bad, seen = _threaded_call_violations(tree, "synthetic")
    assert len(bad) == 1, bad
    assert seen == {"_guard": 1}


@pytest.mark.parametrize(
    "call",
    [
        pytest.param("_guard(1, reason_code=ReasonCode.NON_BINARY_TREATMENT)", id="name"),
        pytest.param("tr._guard(1, reason_code=ReasonCode.NON_BINARY_OUTCOME)", id="attribute"),
    ],
)
def test_the_call_site_guard_accepts_a_literal_member_however_the_guard_is_reached(call):
    tree = ast.parse(_SYNTHETIC_GUARD + f"\ndef caller():\n    {call}\n")
    bad, seen = _threaded_call_violations(tree, "synthetic")
    assert bad == []
    assert seen == {"_guard": 1}


def test_the_site_count_is_what_the_lane_measured():
    """A floor, not a ceiling: new sites are fine, a silent drop to zero is not.

    87, measured by AST on this lane's base ``0a16e9c18``: 66 ``ToolRefusalError``
    plus 21 ``ToolInputError``. The plan's 94 was wrong for TWO independent reasons,
    and neither alone explains the gap.

    * **Method.** ``grep -c "ToolRefusalError\\|ToolInputError"`` counts LINES that
      mention either name, and collapses several matches on one line to one. Many of
      those lines are prose — the module and tool docstrings, the comments explaining
      the #1600 split, the ``from .errors import``. On the lane base that is 103
      matching lines against 87 real ``raise`` statements.
    * **Base drift.** The 94 was measured on ``6c6a6a0ae``, which predates PR #2059
      (``8bb85a772``, ``772733dc2`` — the #2022 sensitivity work). That commit added
      five ``ToolRefusalError`` sites. On ``6c6a6a0ae`` grep says 94 and the AST says
      82; on the lane base grep says 103 and the AST says 87.

    The AST on the branch you are actually on is the measure; a line count against a
    stale base is not.
    """
    total = sum(1 for path in _SOURCES for _ in _raise_sites(path))
    assert total >= 87, f"expected at least the 87 sites measured for #2021, found {total}"
