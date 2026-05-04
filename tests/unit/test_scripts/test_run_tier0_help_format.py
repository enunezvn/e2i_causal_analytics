"""Pin the `--help` CLI surface against argparse %-expansion errors.

Discovered 2026-05-04 during the `feat/phase5p2-initiation-revalidation`
shard: `python scripts/run_tier0_test.py --help` raised
``ValueError: unsupported format character 'p' (0x70) at index 117``
because argparse's ``HelpFormatter._expand_help`` treats the help string
as a printf-style template (``help_str % params``), and an unescaped
``%`` in the ``--regime`` help string produced a ``%p`` (or ``% p``)
sequence that Python interprets as an invalid format spec.

This test pins the contract that:

1. Every help string in ``_build_parser()`` survives the ``% params``
   expansion that argparse performs when formatting help text.
2. ``parser.format_help()`` returns successfully without raising.

Both invariants are violated TODAY by the unescaped ``13-18%`` in the
``--regime`` help string at ``scripts/run_tier0_test.py:5641``. The
fix is to escape the literal percent as ``13-18%%``.

Per `feedback_pr_merge_workflow.md` §7, the test includes a
discriminating-coverage assertion (parser has ≥10 actions) so it
cannot pass vacuously if `_build_parser` ever degenerates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from run_tier0_test import _build_parser  # noqa: E402,I001


def _expand_params_for_action(action: argparse.Action) -> dict[str, object]:
    """Mirror the dict argparse uses in HelpFormatter._expand_help.

    See cpython argparse.py: `_expand_help` populates a dict of
    {'prog', 'default', 'choices', 'metavar', ...} based on the action
    attributes, then formats the help string as `help_str % params`.
    """
    params: dict[str, object] = {"prog": "run_tier0_test"}
    if action.default is not None and action.default is not argparse.SUPPRESS:
        params["default"] = action.default
    if action.metavar:
        params["metavar"] = action.metavar
    if action.choices:
        params["choices"] = list(action.choices)
    return params


# --------------------------------------------------------------------------- #
# Discriminating coverage (vacuous-pass guard per feedback §7)                 #
# --------------------------------------------------------------------------- #


def test_build_parser_yields_at_least_10_actions() -> None:
    """If _build_parser ever degenerates to a stub, this catches it."""
    parser = _build_parser()
    # Strip the auto-injected --help action; assert the user-defined
    # surface is non-trivial.
    user_actions = [a for a in parser._actions if not isinstance(a, argparse._HelpAction)]
    assert len(user_actions) >= 10, f"expected ≥10 user-defined arguments; got {len(user_actions)}"


# --------------------------------------------------------------------------- #
# Per-action help-string % expansion                                           #
# --------------------------------------------------------------------------- #


def test_every_help_string_survives_percent_expansion() -> None:
    """Each action's help string must format cleanly under `% params`.

    This is the exact code path argparse takes inside
    `HelpFormatter._expand_help`. A bare `%` in a help string that is
    not escaped as `%%` will raise here.

    On TODAY's main (`fix/cli-help-format-string-escape` parent), this
    test FAILS on `--regime` with `ValueError: unsupported format
    character 'p' (0x70) at index 117`. After the fix (escape `13-18%`
    to `13-18%%`), the test PASSES.
    """
    parser = _build_parser()
    failures: list[tuple[str, str]] = []
    for action in parser._actions:
        if not action.help:
            continue
        params = _expand_params_for_action(action)
        try:
            _ = action.help % params
        except Exception as e:
            failures.append((str(action.option_strings), f"{type(e).__name__}: {e}"))
    assert not failures, (
        "help string %-expansion failed for one or more actions:\n"
        + "\n".join(f"  {opt} -> {err}" for opt, err in failures)
        + "\n\nFix: escape literal % as %% in the help text."
    )


# --------------------------------------------------------------------------- #
# End-to-end: parser.format_help() must succeed                               #
# --------------------------------------------------------------------------- #


def test_parser_format_help_does_not_raise() -> None:
    """The full help-formatting code path must produce a string.

    `parser.format_help()` is what `--help` invokes internally. It calls
    `HelpFormatter._expand_help` for every action, which is where the
    %-expansion bug surfaces. This test pins the user-visible behaviour:
    `--help` must not crash.
    """
    parser = _build_parser()
    help_text = parser.format_help()
    assert isinstance(help_text, str)
    assert len(help_text) > 100, f"help text suspiciously short: {len(help_text)} chars"


def test_parser_help_action_does_not_raise_via_parse_args() -> None:
    """End-to-end: `parser.parse_args(['--help'])` must SystemExit cleanly.

    argparse calls `parser.print_help()` from the `--help` action, which
    invokes `format_help()`. If the format-string substitution fails,
    `print_help()` raises `ValueError` BEFORE `SystemExit` fires —
    catching the user-visible failure mode.
    """
    parser = _build_parser()
    # parse_args(['--help']) must SystemExit, NOT ValueError.
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    # Help action exits with code 0, not an error.
    assert excinfo.value.code == 0
