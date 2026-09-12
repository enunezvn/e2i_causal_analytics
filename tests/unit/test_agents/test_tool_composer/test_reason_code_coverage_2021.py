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


@pytest.mark.parametrize("path", _SOURCES, ids=lambda p: p.name)
def test_a_threaded_reason_code_is_literal_at_every_call_site(path: Path):
    """The one hop the closed-set check allows must itself end in a literal.

    Without this, threading the code through a parameter would be an unchecked
    hole: a caller could pass a string, a variable, or nothing at all.
    """
    members = {c.name for c in ReasonCode}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    threaded = _threaded_params(tree)
    assert threaded, "no threaded guard found — delete this test if that is intended"

    bad = []
    seen = dict.fromkeys(threaded, 0)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in threaded:
            continue
        seen[node.func.id] += 1
        supplied = {kw.arg: kw.value for kw in node.keywords}
        value = supplied.get("reason_code")
        if value is None:
            bad.append(f"{path}:{node.lineno} {node.func.id}(...) supplies no reason_code")
        elif not _literal_member(value, members):
            bad.append(f"{path}:{node.lineno} {node.func.id}(reason_code={ast.unparse(value)})")
    assert bad == [], "threaded reason_code not a literal member:\n  " + "\n  ".join(bad)
    uncalled = [name for name, count in seen.items() if count == 0]
    assert uncalled == [], f"threaded guard is never called, so nothing pins it: {uncalled}"


def test_the_site_count_is_what_the_lane_measured():
    """A floor, not a ceiling: new sites are fine, a silent drop to zero is not.

    87, not the 94 the lane's plan quoted. That 94 came from
    ``grep -c "ToolRefusalError\\|ToolInputError"``, which counts LINES that mention
    either name — and 16 of those are the module and tool docstrings, the comments
    that explain the #1600 split, and the ``from .errors import`` line. 103 matching
    lines minus those 16 is 87, which is what walking the AST for actual ``raise``
    statements finds. The AST is the measure; a grep over prose is not.
    """
    total = sum(1 for path in _SOURCES for _ in _raise_sites(path))
    assert total >= 87, f"expected at least the 87 sites measured for #2021, found {total}"
