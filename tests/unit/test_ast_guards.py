"""Self-tests for :mod:`tests.unit.ast_guards` (#2020).

The real guards live with the errors they keep authored: the phase-error guard in
``test_phase_error_text_2020.py`` and the coded-refusal guard in
``test_refusal_text_is_authored_2020.py``. These tests pin what the shared helper flags and what
it allows, on synthetic sources, for each form it follows.
"""

from __future__ import annotations

import ast
import textwrap
from typing import List, Tuple

import pytest

from tests.unit.ast_guards import CaughtInterpolation, caught_exception_interpolations

TARGETS = frozenset({"ToolRefusalError", "EstimationError", "PlanningError", "ExecutionError"})


def _in_handler(body: str) -> str:
    return "try:\n    pass\nexcept Exception as e:\n" + textwrap.indent(
        textwrap.dedent(body), "    "
    )


def _flagged(body: str) -> List[Tuple[int, str, str]]:
    found = caught_exception_interpolations(ast.parse(_in_handler(body)), TARGETS)
    return [(site.line, site.raised, site.name) for site in found]


# ---------------------------------------------------------------------------
# Direct references in the raised call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raise_line",
    [
        'raise PlanningError(f"failed: {e}") from e',
        "raise PlanningError(str(e))",
        "raise PlanningError(repr(e))",
        'raise PlanningError("failed: %s" % e)',
        'raise PlanningError("failed: {}".format(e))',
        "raise PlanningError(message=e.args[0])",
    ],
    ids=["fstring", "str", "repr", "percent", "format", "attribute-kwarg"],
)
def test_guard_flags_every_reference_form(raise_line):
    source = f"try:\n    pass\nexcept Exception as e:\n    {raise_line}\n"
    assert caught_exception_interpolations(ast.parse(source), TARGETS) == [
        CaughtInterpolation(4, None, ("Exception",), "PlanningError", "e")
    ]


def test_guard_follows_a_local_built_from_the_exception():
    source = (
        "try:\n    pass\nexcept Exception as e:\n"
        '    detail = f"{e}"\n    message = "failed: " + detail\n'
        "    raise executor.ExecutionError(message)\n"
    )
    assert caught_exception_interpolations(ast.parse(source), TARGETS) == [
        CaughtInterpolation(6, None, ("Exception",), "ExecutionError", "e")
    ]


@pytest.mark.parametrize(
    "handler",
    [
        'except Exception as e:\n    raise PlanningError("fixed sentence") from e',
        "except PlanningError:\n    raise",
        'except Exception as e:\n    raise ValueError(f"not a phase error: {e}")',
        'except Exception as e:\n    logger.warning("x: %s", e)\n    raise PlanningError("fixed")',
    ],
    ids=["from-cause", "bare-reraise", "other-class", "logged-only"],
)
def test_guard_allows_what_does_not_render_the_exception(handler):
    source = f"try:\n    pass\n{handler}\n"
    assert caught_exception_interpolations(ast.parse(source), TARGETS) == []


# ---------------------------------------------------------------------------
# Derived forms, flagged and allowed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "line"),
    [
        ('msg = "failed: "\nmsg += str(e)\nraise ToolRefusalError(msg)\n', 6),
        ("print((d := str(e)))\nraise ToolRefusalError(d)\n", 5),
        ("if (d := str(e)):\n    raise ToolRefusalError(d)\n", 5),
        ("for arg in e.args:\n    raise ToolRefusalError(arg)\n", 5),
        ('err = ToolRefusalError(f"{e}")\nraise err\n', 5),
        ('err = ToolRefusalError(f"{e}")\nalias = err\nraise alias\n', 6),
        ('def build():\n    return f"{e}"\nraise ToolRefusalError(build())\n', 6),
        ('def build():\n    raise ToolRefusalError(f"{e}")\n', 5),
        ("with open(str(e)) as handle:\n    raise ToolRefusalError(handle)\n", 5),
        ('msg = str(e)\nif cond:\n    msg = "fixed"\nraise ToolRefusalError(msg)\n', 7),
        ("while cond:\n    raise ToolRefusalError(msg)\n    msg = str(e)\n", 5),
    ],
    ids=[
        "augassign",
        "walrus",
        "walrus-in-test",
        "for-over-args",
        "prebuilt-instance",
        "prebuilt-alias",
        "nested-def-called",
        "nested-def-raises",
        "with-as",
        "rebound-on-one-branch-only",
        "loop-carries-taint-back",
    ],
)
def test_helper_flags_derived_forms(body, line):
    raised = "ToolRefusalError"
    assert _flagged(body) == [(line, raised, "e")]


@pytest.mark.parametrize(
    "body",
    [
        'msg = "failed"\nmsg += "!"\nraise ToolRefusalError(msg) from e\n',
        'if (d := "fixed"):\n    raise ToolRefusalError(d)\n',
        'for arg in ("a", "b"):\n    raise ToolRefusalError(arg)\n',
        'err = ToolRefusalError("fixed")\nraise err from e\n',
        'err = ValueError(f"{e}")\nraise err\n',
        'def build():\n    return "fixed"\nraise ToolRefusalError(build())\n',
        'def build(e):\n    return str(e)\nraise ToolRefusalError(build("fixed"))\n',
        "raise ToolRefusalError(type(e).__name__)\n",
        "raise ToolRefusalError(e.__class__.__name__)\n",
        'raise ToolRefusalError(f"{type(e)}")\n',
        'raise ToolRefusalError("bad value" if isinstance(e, ValueError) else "failed")\n',
        'e = "fixed"\nraise ToolRefusalError(f"{e}")\n',
        'msg = str(e)\nmsg = "fixed"\nraise ToolRefusalError(msg)\n',
        'msg = {}\nmsg["k"] = 1\nraise ToolRefusalError(msg)\n',
    ],
    ids=[
        "augassign-untainted",
        "walrus-untainted",
        "for-over-literal",
        "prebuilt-authored",
        "prebuilt-other-class",
        "nested-def-untainted",
        "nested-def-shadowing-param",
        "type-name",
        "class-name",
        "type-repr",
        "isinstance",
        "handler-name-rebound",
        "derived-name-rebound",
        "subscript-store-untainted",
    ],
)
def test_helper_allows_what_does_not_render_the_exception(body):
    assert _flagged(body) == []


# ---------------------------------------------------------------------------
# Nested handlers and what each site reports
# ---------------------------------------------------------------------------


def test_helper_reports_a_nested_handler_once_under_its_own_clause():
    source = textwrap.dedent(
        """\
        def tool():
            try:
                pass
            except Exception as e:
                try:
                    pass
                except (ValueError, errors.EffectDataUnavailable) as e:
                    raise ToolRefusalError(f"{e}")
        """
    )
    assert caught_exception_interpolations(ast.parse(source), TARGETS) == [
        CaughtInterpolation(
            8, "tool", ("ValueError", "EffectDataUnavailable"), "ToolRefusalError", "e"
        )
    ]


def test_helper_carries_a_nested_handlers_bindings_to_the_outer_scan():
    body = "try:\n    pass\nexcept ValueError:\n    msg = str(e)\nraise ToolRefusalError(msg)\n"
    assert _flagged(body) == [(8, "ToolRefusalError", "e")]


def test_helper_names_the_enclosing_function_and_caught_types():
    source = textwrap.dedent(
        """\
        try:
            pass
        except RuntimeError as e:
            raise ToolRefusalError(str(e))

        async def outer():
            try:
                pass
            except (KeyError, module.LookupFailure) as exc:
                raise EstimationError(repr(exc))
        """
    )
    assert caught_exception_interpolations(ast.parse(source), TARGETS) == [
        CaughtInterpolation(4, None, ("RuntimeError",), "ToolRefusalError", "e"),
        CaughtInterpolation(10, "outer", ("KeyError", "LookupFailure"), "EstimationError", "exc"),
    ]
