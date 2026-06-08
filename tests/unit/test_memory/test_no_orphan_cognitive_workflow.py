"""Anti-resurrection guard for the orphan M12 module.

`src/memory/004_cognitive_workflow.py` imported from `src.memory.memory_backends`
and `src.memory.agent_registry` -- neither of which exists as a real module in
`src/memory/`. The file name also starts with a digit, so it can never be
imported via normal `import`/`from` syntax. It was therefore UNRUNNABLE in
production: its only "passing" test (`test_cognitive_workflow_v2.py`) worked
exclusively by injecting fabricated `MockMemoryBackends`/`MockAgentRegistry`
into `sys.modules` before an `importlib.import_module` call.

The live cognitive-workflow factories used in production are the unrelated
`create_dspy_cognitive_workflow` / `create_production_cognitive_workflow`
functions defined in `src/rag/cognitive_rag_dspy.py` -- NOT this 004 module.
The file was deleted in PR (Refs #694, M12).

This guard asserts the file stays deleted.
"""

from __future__ import annotations

from pathlib import Path

# tests/unit/test_memory/<this file> -> repo root is parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
_ORPHAN = _SRC / "memory" / "004_cognitive_workflow.py"


def test_orphan_cognitive_workflow_file_is_absent() -> None:
    """The orphan 004 cognitive-workflow module file must not exist."""
    assert not _ORPHAN.exists(), (
        f"{_ORPHAN} was deleted as an orphan, unrunnable module "
        "(Refs #694, M12); do not resurrect it. The production cognitive "
        "workflow lives in src/rag/cognitive_rag_dspy.py "
        "(create_dspy_cognitive_workflow / create_production_cognitive_workflow)."
    )


def test_no_src_module_loads_orphan_cognitive_workflow() -> None:
    """No source module may dynamically load the deleted 004 module file."""
    needle = "004_cognitive_workflow"
    offenders: list[str] = []
    for py in _SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        if needle in text:
            offenders.append(str(py.relative_to(_REPO_ROOT)))
    assert not offenders, (
        "Found references to the deleted orphan 004 module in: "
        f"{offenders}. It was removed (Refs #694, M12)."
    )
