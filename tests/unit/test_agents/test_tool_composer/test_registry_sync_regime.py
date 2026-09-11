"""The migration-per-schema-change regime is retired; the runtime sync replaced it (spec §4).

#2003 / PR #2012 kept the DB ``tool_registry`` rows in step with the code by generating a
migration whenever a tool's declared inputs or output model changed. ml/040's
``sync_tool_registry``, called at API startup by ``registry_sync``, now makes the rows equal to
the running code, so the generator, the drift test's DB section and the never-wired G3 database
registration (``register_from_database`` / ``sync_to_database``, placeholder callables that
always raise, a ``ToolCategory`` enum whose values match no DB enum, a category filter that
returns ``[]``) are gone. ``database/ml/037`` stays: it is applied history.

Filesystem and import checks only, so this runs in CI.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[4]
THIS = Path(__file__).resolve()

RETIRED_NAMES = re.compile(
    r"generate_tool_registry_sync_migration|register_from_database|sync_to_database"
    r"|NOT_SEEDED_IN_DB|_create_placeholder_callable"
)
# get_tools_by_category is also a real, unrelated GEPA helper.
GEPA_OWNERS = ("src/optimization/gepa/", "scripts/gepa_integration_test.py")


def _python_files() -> Iterator[Path]:
    for top in ("src", "scripts", "tests"):
        for path in sorted((REPO_ROOT / top).rglob("*.py")):
            if "__pycache__" in path.parts or path.resolve() == THIS:
                continue
            yield path


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def test_generator_script_is_deleted():
    assert not (REPO_ROOT / "scripts" / "generate_tool_registry_sync_migration.py").exists()


def test_no_code_references_the_retired_regime():
    hits = [
        f"{_rel(path)}:{number}"
        for path in _python_files()
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1)
        if RETIRED_NAMES.search(line)
    ]
    assert hits == []


def test_generic_registry_has_no_database_registration_or_category_surface():
    import src.tool_registry.registry as generic

    assert not hasattr(generic, "ToolCategory")
    assert "ToolCategory" not in generic.__all__
    for name in ("register_from_database", "sync_to_database", "get_tools_by_category"):
        assert not hasattr(generic.ToolRegistry, name), name


def test_nothing_imports_the_retired_category_names():
    hits = []
    for path in _python_files():
        rel = _rel(path)
        text = path.read_text(errors="replace")
        if (
            "get_tools_by_category" in text
            and not rel.startswith(GEPA_OWNERS)
            and "src.optimization.gepa" not in text  # a consumer of the GEPA helper
        ):
            hits.append(f"{rel}: get_tools_by_category")
        if "ToolCategory" not in text:
            continue
        for node in ast.walk(ast.parse(text, filename=rel)):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "src.tool_registry.registry"
                and any(alias.name == "ToolCategory" for alias in node.names)
            ):
                hits.append(f"{rel}:{node.lineno}: ToolCategory from src.tool_registry.registry")
    assert hits == []


def test_composer_registry_comment_names_the_runtime_sync():
    text = (REPO_ROOT / "src" / "agents" / "tool_composer" / "tool_registry.py").read_text()
    assert "registry_sync.sync_tool_registry_once()" in text and "(ml/040)" in text


def test_registry_doc_describes_runtime_sync():
    doc = (REPO_ROOT / "docs" / "data" / "03-ML-PIPELINE-SCHEMA.md").read_text()
    start = doc.index("### 4.1 `tool_registry`")
    section = doc[start : doc.index("### 4.2", start)]
    assert "never registered at runtime" not in section
    assert "sync_tool_registry" in section
    assert "success_rate" not in section  # dropped by ml/040
