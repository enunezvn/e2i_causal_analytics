"""Anti-resurrection guard for the vestigial M11 module.

`src/memory/006_memory_backends_v1_3.py` was a ~2074-LOC monolithic predecessor
of the per-concern production memory modules. Every symbol it defined
(`search_episodic_memory`, `insert_episodic_memory`, `get_working_memory`,
`RedisWorkingMemory`, `FalkorDBSemanticMemory`, `get_semantic_memory`, ...) now
lives in the real, importable modules `src/memory/episodic_memory.py`,
`src/memory/working_memory.py`, and `src/memory/semantic_memory.py`.

The file name starts with a digit, so it can NEVER be imported via normal
`import`/`from` syntax (module names cannot start with a digit). No production
code references it (verified by grep across src/, frontend/, scripts/, *.yaml,
*.yml, *.sh). It was deleted in PR (Refs #694, M11).

This guard asserts the file stays deleted and that nothing in src/ resurrects it
via a dynamic load.
"""

from __future__ import annotations

from pathlib import Path

# tests/unit/test_memory/<this file> -> repo root is parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
_VESTIGIAL = _SRC / "memory" / "006_memory_backends_v1_3.py"


def test_vestigial_backends_module_file_is_absent() -> None:
    """The vestigial 006 module file must not exist."""
    assert not _VESTIGIAL.exists(), (
        f"{_VESTIGIAL} was deleted as a vestigial duplicate (Refs #694, M11); "
        "do not resurrect it. Use src/memory/episodic_memory.py, "
        "working_memory.py, and semantic_memory.py instead."
    )


def test_no_src_module_references_vestigial_backends() -> None:
    """No source module may reference the deleted 006 module (string or load)."""
    needles = ("006_memory_backends", "memory_backends_v1_3")
    offenders: list[str] = []
    for py in _SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8", errors="ignore")
        if any(needle in text for needle in needles):
            offenders.append(str(py.relative_to(_REPO_ROOT)))
    assert not offenders, (
        "Found references to the deleted vestigial 006 module in: "
        f"{offenders}. It was removed (Refs #694, M11); update these callers "
        "to the real per-concern memory modules."
    )
