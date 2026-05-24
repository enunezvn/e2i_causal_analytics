"""Guard test: every DSPy-touching CLI script in ``scripts/`` MUST call
``load_dotenv()`` at module level.

Rationale (GitHub issue #470)
-----------------------------
CLI scripts that depend on a configured DSPy LM (``dspy.configure(lm=...)``
or ``ensure_dspy_lm_configured()`` from
``src/data/causal_role_classifier_loader.py``) silently no-op when invoked
directly from the shell if no module bridges ``.env`` → process env. The
chain:

1. ``.env`` has ``ANTHROPIC_API_KEY=...``
2. ``python scripts/<x>.py`` — process inherits shell env, which lacks
   the key unless exported.
3. DSPy LM config reads ``os.environ.get("ANTHROPIC_API_KEY")`` → empty
   → no LM configured.
4. ``classify_feature(...)`` returns ``None`` instead of raising.
5. Script exits 0 with a misleading "No instrument predictions made"
   log line.

``tests/conftest.py`` calls ``load_dotenv(override=True)`` at collection
time, so pytest paths work; CLI paths do not. The remediation is to add
``from dotenv import load_dotenv`` + ``load_dotenv()`` at the top of each
DSPy-touching CLI script, BEFORE any import that may read provider env
vars at import time.

This test parses each candidate script with ``ast`` and asserts:

- It imports ``load_dotenv`` from ``dotenv`` somewhere at module level.
- It calls ``load_dotenv(...)`` somewhere at module level.

The candidate set is computed dynamically from the union of patterns
documented in #470: ``dspy.configure``, ``dspy.LM``,
``ensure_dspy_lm_configured``, ``classify_feature``,
``load_compiled_classifier``. Library modules without an ``if __name__
== "__main__"`` block are excluded.

Allow-list
----------
``EXCLUDED_SCRIPTS`` — scripts that legitimately do NOT need
``load_dotenv()``. Each entry MUST carry a one-line rationale that
explains why this script is safe to skip.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Substrings whose presence in a script's source flags it as
# DSPy-touching (per issue #470 audit pattern).
_DSPY_TRIGGER_PATTERNS: tuple[str, ...] = (
    "dspy.configure",
    "dspy.LM",
    "ensure_dspy_lm_configured",
    "classify_feature",
    "load_compiled_classifier",
)

# Scripts intentionally excluded from the load_dotenv() requirement.
# Each entry MUST carry a one-line rationale documenting WHY the script
# is safe to skip — see #470 reasoning discipline.
EXCLUDED_SCRIPTS: dict[str, str] = {
    # (currently empty — extend with `"name.py": "rationale ..."` if needed)
}


def _candidate_scripts() -> list[Path]:
    """Return CLI scripts in ``scripts/`` that touch DSPy LM config.

    A script qualifies if:
    - It is a ``.py`` file directly under ``scripts/``.
    - Its source contains at least one of ``_DSPY_TRIGGER_PATTERNS``.
    - It exposes a ``if __name__ == "__main__"`` block (CLI entrypoint),
      i.e. it is NOT a pure library module.
    """
    out: list[Path] = []
    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not any(pat in src for pat in _DSPY_TRIGGER_PATTERNS):
            continue
        if '__name__ == "__main__"' not in src and "__name__ == '__main__'" not in src:
            continue
        out.append(path)
    return out


def _imports_load_dotenv_at_module_level(tree: ast.Module) -> bool:
    """Return True iff the module has ``from dotenv import load_dotenv``
    (or ``import dotenv``) at module top level."""
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "dotenv":
            for alias in node.names:
                if alias.name == "load_dotenv":
                    return True
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "dotenv":
                    return True
    return False


def _calls_load_dotenv_at_module_level(tree: ast.Module) -> bool:
    """Return True iff the module calls ``load_dotenv(...)`` at module
    top level (NOT inside a function/class)."""
    for node in tree.body:
        if not isinstance(node, ast.Expr):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        # `load_dotenv(...)` — direct name
        if isinstance(func, ast.Name) and func.id == "load_dotenv":
            return True
        # `dotenv.load_dotenv(...)` — attribute access
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "load_dotenv"
            and isinstance(func.value, ast.Name)
            and func.value.id == "dotenv"
        ):
            return True
    return False


_CANDIDATES = _candidate_scripts()


def test_at_least_one_candidate_discovered() -> None:
    """Sanity check: discovery returned a non-empty list. Otherwise the
    guard test would silently pass even if every CLI lost its ``__main__``
    block by accident."""
    assert _CANDIDATES, (
        "No DSPy-touching CLI scripts discovered in scripts/. Either every "
        "candidate was removed (verify deliberately) or the discovery "
        "logic regressed. Update _DSPY_TRIGGER_PATTERNS if patterns "
        "evolved."
    )


@pytest.mark.parametrize("script_path", _CANDIDATES, ids=lambda p: p.name)
def test_dspy_touching_script_has_load_dotenv(script_path: Path) -> None:
    """Every DSPy-touching CLI script MUST call ``load_dotenv()`` at
    module top, BEFORE any import that may read provider env vars at
    import time (issue #470)."""
    if script_path.name in EXCLUDED_SCRIPTS:
        pytest.skip(f"Excluded by allow-list: {EXCLUDED_SCRIPTS[script_path.name]}")

    src = script_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(script_path))

    assert _imports_load_dotenv_at_module_level(tree), (
        f"{script_path.name} touches DSPy LM config but does not import "
        "`load_dotenv` from `dotenv` at module level. Add "
        "`from dotenv import load_dotenv` near the top of the file. "
        "See issue #470 for the failure mode this prevents."
    )
    assert _calls_load_dotenv_at_module_level(tree), (
        f"{script_path.name} imports `load_dotenv` but does not call it "
        "at module level. Add `load_dotenv()` immediately after the "
        "import, BEFORE any DSPy / causal_role_classifier_loader import. "
        "See issue #470 for the failure mode this prevents."
    )
