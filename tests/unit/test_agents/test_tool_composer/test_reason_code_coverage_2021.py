"""#2021: every refusal raise site names a code. An AST check, so it cannot drift.

A grep would miss a multi-line call and match a comment. Walking the AST of the
module source finds every ``raise ToolRefusalError(...)`` / ``raise ToolInputError(...)``
however it is formatted.
"""

import ast
from pathlib import Path

import pytest

from src.agents.tool_composer.reason_codes import EXECUTOR_ASSIGNED, ReasonCode

REPO_ROOT = Path(__file__).resolve().parents[4]

_TARGETS = {"ToolRefusalError", "ToolInputError"}
_SOURCES = [
    REPO_ROOT / "src/agents/tool_composer/tool_registrations.py",
]

# Codes a tool may raise with: every member except those only the executor assigns.
_TOOL_MEMBERS = {c.name for c in ReasonCode} - {c.name for c in EXECUTOR_ASSIGNED}


def _raise_sites_in(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        func = node.exc.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name in _TARGETS:
            yield node.lineno, name, node.exc


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", _SOURCES, ids=lambda p: p.name)
def test_every_raise_site_names_a_reason_code(path: Path):
    uncoded = []
    for lineno, name, call in _raise_sites_in(_parse(path)):
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


_EFFECT_MAPPER = "effect_reason_code"
_EFFECT_TABLE = "EFFECT_CAUSE_CODES"
# The table lives with the vocabulary; tool_registrations imports only the mapper.
_EFFECT_TABLE_SOURCE = REPO_ROOT / "src/agents/tool_composer/reason_codes.py"


def _effect_mapper_call(value: ast.expr) -> bool:
    """True for exactly ``effect_reason_code(<cause>, fallback=ReasonCode.MEMBER)`` (#2021 9b).

    The mapper returns its fallback or a value of ``EFFECT_CAUSE_CODES``: the fallback must be
    a literal tool member here, and ``test_the_effect_cause_table_holds_only_tool_members``
    checks the table. Like the threaded guards, this follows one hop and ends in literals.
    """
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == _EFFECT_MAPPER
        and len(value.args) == 1
        and len(value.keywords) == 1
        and value.keywords[0].arg == "fallback"
        and _literal_member(value.keywords[0].value, _TOOL_MEMBERS)
    )


def _effect_table_violations(tree: ast.Module, label: str) -> list:
    """The module-level ``EFFECT_CAUSE_CODES`` is a dict literal of string keys to tool members."""
    tables = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == _EFFECT_TABLE for t in node.targets)
        )
        or (isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == _EFFECT_TABLE)
    ]
    if len(tables) != 1:
        return [f"{label}: expected one module-level {_EFFECT_TABLE}, found {len(tables)}"]
    (table,) = tables
    if not isinstance(table.value, ast.Dict):
        return [f"{label}:{table.lineno} {_EFFECT_TABLE} is not a dict literal"]
    bad = []
    for key, value in zip(table.value.keys, table.value.values, strict=True):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            shown = "**" if key is None else ast.unparse(key)
            bad.append(f"{label}:{value.lineno} key {shown} is not a string literal")
        if not _literal_member(value, _TOOL_MEMBERS):
            bad.append(
                f"{label}:{value.lineno} {ast.unparse(value)} (not a tool ReasonCode.MEMBER)"
            )
    return bad


def _raise_site_violations(tree: ast.Module, label: str) -> list:
    """Raise sites whose ``reason_code`` is not a literal tool-legal member."""
    threaded = _threaded_params(tree)
    # the line span of each function that legitimately raises with its parameter
    spans = [(f.lineno, f.end_lineno) for f in threaded.values()]
    bad = []
    for lineno, _name, call in _raise_sites_in(tree):
        for kw in call.keywords:
            if kw.arg != "reason_code":
                continue
            value = kw.value
            if _literal_member(value, _TOOL_MEMBERS) or _effect_mapper_call(value):
                continue
            inside_threaded = any(start <= lineno <= end for start, end in spans)
            if inside_threaded and isinstance(value, ast.Name) and value.id == "reason_code":
                continue
            bad.append(f"{label}:{lineno} {ast.unparse(value)} (not a tool ReasonCode.MEMBER)")
    return bad


@pytest.mark.parametrize("path", _SOURCES, ids=lambda p: p.name)
def test_every_reason_code_is_a_member_of_the_closed_set(path: Path):
    """A literal string or an unknown ReasonCode attribute is a drift vector."""
    bad = _raise_site_violations(_parse(path), str(path))
    assert bad == [], "reason_code values outside the closed set:\n  " + "\n  ".join(bad)


def _threaded_call_violations(tree: ast.Module, label: str) -> tuple:
    """Calls to a threaded guard that do not supply a literal tool-legal member.

    Returns ``(bad, seen)``: the offending call sites, and how many calls each
    threaded guard received.
    """
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
        elif not _literal_member(value, _TOOL_MEMBERS):
            bad.append(f"{label}:{node.lineno} {name}(reason_code={ast.unparse(value)})")
    return bad, seen


@pytest.mark.parametrize("path", _SOURCES, ids=lambda p: p.name)
def test_a_threaded_reason_code_is_literal_at_every_call_site(path: Path):
    """The one hop the closed-set check allows must itself end in a literal.

    Without this, threading the code through a parameter would be an unchecked
    hole: a caller could pass a string, a variable, or nothing at all.
    """
    tree = _parse(path)
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
        pytest.param("_guard(1, reason_code=ReasonCode.TOOL_ERROR)", id="name-executor-code"),
        pytest.param("tr._guard(1, reason_code=ReasonCode.CIRCUIT_OPEN)", id="attr-executor-code"),
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


@pytest.mark.parametrize(
    ("code", "flagged"),
    [
        pytest.param("TOOL_ERROR", True, id="tool-error"),
        pytest.param("TOOL_TIMEOUT", True, id="tool-timeout"),
        pytest.param("PLAN_DEFECT", True, id="plan-defect"),
        pytest.param("REFERENCE_UNRESOLVABLE", True, id="reference-unresolvable"),
        pytest.param("DEPENDENCY_UNMET", True, id="dependency-unmet"),
        pytest.param("CIRCUIT_OPEN", True, id="circuit-open"),
        pytest.param("TOOL_NOT_REGISTERED", True, id="tool-not-registered"),
        pytest.param("NO_USABLE_ROWS", False, id="tool-code"),
    ],
)
def test_a_tool_raise_site_may_not_use_an_executor_assigned_code(code, flagged):
    """M6: those codes describe what the EXECUTOR observed; a tool claiming one would be
    counted as a timeout, an open circuit or a plan defect that never happened."""
    source = f'def tool():\n    raise ToolRefusalError("x", reason_code=ReasonCode.{code})\n'
    bad = _raise_site_violations(ast.parse(source), "synthetic")
    assert bool(bad) is flagged, bad


def test_the_site_count_is_what_the_lane_measured():
    """A floor, not a ceiling: new sites are fine, a silent drop is not.

    Measured by AST, not grep (a line count also matches docstrings and imports).
    """
    total = sum(1 for path in _SOURCES for _ in _raise_sites_in(_parse(path)))
    assert total >= 89, f"expected at least the 89 sites measured for #2021, found {total}"


# ---------------------------------------------------------------------------
# #2021 9b: counterfactual_simulator's per-cause code goes through one mapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        pytest.param("effect_reason_code(c)", id="no-fallback"),
        pytest.param('effect_reason_code(c, fallback="x")', id="string-fallback"),
        pytest.param("effect_reason_code(c, fallback=ReasonCode.TOOL_ERROR)", id="executor"),
        pytest.param("effect_reason_code(c, fallback=ReasonCode.NOT_A_MEMBER)", id="unknown"),
        pytest.param("effect_reason_code(c, fallback=code)", id="variable-fallback"),
        pytest.param("effect_reason_code(c, ReasonCode.NO_USABLE_ROWS)", id="positional"),
        pytest.param("other_mapper(c, fallback=ReasonCode.NO_USABLE_ROWS)", id="other-function"),
    ],
)
def test_a_mapped_reason_code_is_flagged_unless_its_fallback_is_a_tool_member(code):
    source = f'def tool():\n    raise ToolRefusalError("x", reason_code={code})\n'
    assert len(_raise_site_violations(ast.parse(source), "synthetic")) == 1


def test_a_mapped_reason_code_with_a_tool_member_fallback_is_accepted():
    source = (
        "def tool():\n"
        '    raise ToolRefusalError("x", reason_code=effect_reason_code('
        "cause, fallback=ReasonCode.EFFECT_NOT_ESTIMABLE))\n"
    )
    assert _raise_site_violations(ast.parse(source), "synthetic") == []


def test_the_effect_cause_table_holds_only_tool_members():
    path = _EFFECT_TABLE_SOURCE
    assert _effect_table_violations(_parse(path), str(path)) == []


@pytest.mark.parametrize(
    "table",
    [
        pytest.param('{"empty_cohort": ReasonCode.TOOL_ERROR}', id="executor-code"),
        pytest.param('{"empty_cohort": "no_usable_rows"}', id="plain-string"),
        pytest.param('{"empty_cohort": ReasonCode.NOT_A_MEMBER}', id="unknown-member"),
        pytest.param('{"empty_cohort": code}', id="variable"),
        pytest.param("{cause: ReasonCode.NO_USABLE_ROWS}", id="non-literal-key"),
        pytest.param("{**OTHER}", id="spread"),
        pytest.param("dict(empty_cohort=ReasonCode.NO_USABLE_ROWS)", id="not-a-dict-literal"),
    ],
)
def test_the_effect_cause_table_check_flags_a_value_that_is_not_a_tool_member(table):
    source = f"EFFECT_CAUSE_CODES: Dict[str, ReasonCode] = {table}\n"
    assert _effect_table_violations(ast.parse(source), "synthetic") != []


def test_the_effect_cause_table_check_flags_a_missing_table_and_accepts_a_literal_one():
    assert _effect_table_violations(ast.parse("OTHER = {}\n"), "synthetic") != []
    source = 'EFFECT_CAUSE_CODES = {"empty_cohort": ReasonCode.NO_USABLE_ROWS}\n'
    assert _effect_table_violations(ast.parse(source), "synthetic") == []
