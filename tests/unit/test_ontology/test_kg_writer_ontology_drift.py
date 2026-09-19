"""#2178 follow-up: the ontology must declare every label and edge type the code writes.

``config/ontology/node_types.yaml`` / ``edge_types.yaml`` have no runtime
loader, so nothing kept them in step with the graph writers. On 2026-09-18 the
live ``e2i_causal`` graph held 21 node labels and 24 relationship types; the
ontology declared 7 and 8 of them. This test gives the ontology a role: it is
the declared vocabulary of the graph, and a writer that adds a label or edge
type must declare it.

How the writers are found, without a database:

* ``add_e2i_entity(entity_type=...)`` — a string literal, or an
  ``E2IEntityType`` member resolved through ``E2I_TO_LABEL``.
* ``add_relationship(relationship_type=...)`` and
  ``add_e2i_relationship(rel_type=...)`` — string literals.
* Cypher string literals / f-strings containing MERGE or CREATE —
  ``(x:Label`` and ``[r:TYPE`` tokens.
* Edge-spec dicts in ``scripts/`` (``{"rel_type": "PRESCRIBES", ...}``), the
  shape ``seed_falkordb.py`` uses.

This extraction recovered every label and relationship type present in the live
graph on 2026-09-18 (a faithful check, not an assumption). Writers whose type
comes from runtime data (an LLM triplet's predicate, a caller-supplied list)
cannot be checked statically; they are pinned in ``DYNAMIC_WRITER_SITES`` so a
new one is a reviewed decision, not a silent gap.

Only this direction is enforced. An ontology entry that no static writer
produces (e.g. ``GENERATED``) may still arrive through a dynamic site, so its
absence from the static scan proves nothing.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
NODE_TYPES_PATH = REPO_ROOT / "config" / "ontology" / "node_types.yaml"
EDGE_TYPES_PATH = REPO_ROOT / "config" / "ontology" / "edge_types.yaml"

# Files that contain graph-write code but never write e2i_causal.
EXCLUDED_FILES = {
    # The retired seeder of the separate e2i_semantic graph (#2178).
    "scripts/seed_semantic_graph.py",
}

# Call sites whose label / edge type is computed at runtime. Adding one means
# the ontology can no longer vouch for everything the graph may contain —
# review it, then add it here with the reason.
DYNAMIC_WRITER_SITES = {
    # add_e2i_relationship creates both endpoints from caller-supplied enum types.
    ("src/memory/semantic_memory.py", "add_e2i_relationship", "label"),
    # Relationships listed in the causal_impact output.
    ("src/agents/causal_impact/agent.py", "update_semantic_memory", "rel"),
    # Triplet predicates (sync_to_semantic_graph).
    ("src/memory/semantic_memory.py", "sync_to_semantic_graph", "rel"),
    # cognitive RAG relationship_type.upper().
    ("src/rag/cognitive_backends.py", "store_relationship", "rel"),
    # Test-data script: relationship spec list.
    ("scripts/e2e_cognitive_rag_test.py", "test_populate_semantic_memory", "rel"),
}

_NODE_TOKEN = re.compile(r"\(\s*\w*\s*:\s*([A-Z][A-Za-z_]*)")
_REL_TOKEN = re.compile(r"\[\s*\w*\s*:\s*([A-Z][A-Z_]+)")
_WRITE_VERB = re.compile(r"\b(MERGE|CREATE)\b")

Found = Dict[str, Set[str]]


def _enum_labels() -> Dict[str, str]:
    from src.memory.semantic_memory import E2I_TO_LABEL

    return {member.name: label for member, label in E2I_TO_LABEL.items()}


def _enclosing_functions(tree: ast.AST) -> Dict[int, str]:
    owner: Dict[int, str] = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if hasattr(node, "lineno"):
                    # ast.walk is breadth-first, so the innermost function wins.
                    owner[node.lineno] = fn.name
    return owner


def _scan() -> Tuple[Found, Found, Set[Tuple[str, str, str]]]:
    enum_labels = _enum_labels()
    labels: Found = defaultdict(set)
    rels: Found = defaultdict(set)
    dynamic: Set[Tuple[str, str, str]] = set()
    files = sorted((REPO_ROOT / "src").rglob("*.py")) + sorted(
        (REPO_ROOT / "scripts").rglob("*.py")
    )
    for path in files:
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        if rel_path in EXCLUDED_FILES:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        owner = _enclosing_functions(tree)
        # Bare string statements (docstrings) are prose, never executed as Cypher.
        prose = {
            id(stmt.value)
            for stmt in ast.walk(tree)
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Constant, ast.JoinedStr))
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
                kwargs = {k.arg: k.value for k in node.keywords}
                if name == "add_e2i_entity" and (node.args or "entity_type" in kwargs):
                    arg = node.args[0] if node.args else kwargs["entity_type"]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        labels[arg.value].add(rel_path)
                    elif isinstance(arg, ast.Attribute) and arg.attr in enum_labels:
                        labels[enum_labels[arg.attr]].add(rel_path)
                    else:
                        dynamic.add((rel_path, owner.get(node.lineno, "<module>"), "label"))
                elif name in ("add_relationship", "add_e2i_relationship"):
                    key = "relationship_type" if name == "add_relationship" else "rel_type"
                    positional = 2 if name == "add_relationship" else 4
                    value = kwargs.get(key)
                    if value is None and len(node.args) > positional:
                        value = node.args[positional]
                    if value is None:
                        continue
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        rels[value.value].add(rel_path)
                    else:
                        dynamic.add((rel_path, owner.get(node.lineno, "<module>"), "rel"))
            elif isinstance(node, ast.Dict) and rel_path.startswith("scripts/"):
                for key, value in zip(node.keys, node.values, strict=True):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "rel_type"
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                    ):
                        rels[value.value].add(rel_path)
            elif id(node) not in prose:
                text = None
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    text = node.value
                elif isinstance(node, ast.JoinedStr):
                    text = "".join(
                        v.value
                        for v in node.values
                        if isinstance(v, ast.Constant) and isinstance(v.value, str)
                    )
                if text and _WRITE_VERB.search(text):
                    for label in _NODE_TOKEN.findall(text):
                        labels[label].add(rel_path)
                    for rel in _REL_TOKEN.findall(text):
                        rels[rel].add(rel_path)
    return labels, rels, dynamic


@pytest.fixture(scope="module")
def scan() -> Tuple[Found, Found, Set[Tuple[str, str, str]]]:
    return _scan()


def _declared(path: Path, key: str) -> Set[str]:
    return set(yaml.safe_load(path.read_text(encoding="utf-8"))[key])


def test_the_scan_sees_known_writers(scan) -> None:
    """Teeth: an extractor that silently finds nothing would pass every check below."""
    labels, rels, _ = scan
    assert "src/agents/ml_foundation/scope_definer/memory_hooks.py" in labels["ScopeSpec"]
    assert "scripts/sync_causal_paths_to_falkordb.py" in labels["Variable"]
    assert "scripts/seed_falkordb.py" in labels["HCP"]
    assert "src/agents/ml_foundation/scope_definer/memory_hooks.py" in rels["HAS_TYPE"]
    assert "scripts/seed_falkordb.py" in rels["PRESCRIBES"]
    assert "scripts/sync_causal_paths_to_falkordb.py" in rels["CAUSES"]
    assert "src/agents/drift_monitor/memory_hooks.py" in rels["HAS_DRIFT"]


def test_every_written_label_is_declared(scan) -> None:
    labels, _, _ = scan
    declared = _declared(NODE_TYPES_PATH, "node_types")
    missing = {label: sorted(files) for label, files in labels.items() if label not in declared}
    assert not missing, (
        "Graph writers create node labels that config/ontology/node_types.yaml does not "
        f"declare. Declare each one (description + writer) or rename the write: {missing}"
    )


def test_every_written_edge_type_is_declared(scan) -> None:
    _, rels, _ = scan
    declared = _declared(EDGE_TYPES_PATH, "edge_types")
    missing = {rel: sorted(files) for rel, files in rels.items() if rel not in declared}
    assert not missing, (
        "Graph writers create relationship types that config/ontology/edge_types.yaml does "
        f"not declare. Declare each one (description + writer) or rename the write: {missing}"
    )


def test_dynamic_writer_sites_are_the_reviewed_set(scan) -> None:
    _, _, dynamic = scan
    assert dynamic == DYNAMIC_WRITER_SITES, (
        "The set of graph writes whose label / edge type is computed at runtime changed. "
        f"New: {sorted(dynamic - DYNAMIC_WRITER_SITES)}; gone: {sorted(DYNAMIC_WRITER_SITES - dynamic)}. "
        "A new site can write types the ontology never declares — review it and update "
        "DYNAMIC_WRITER_SITES with the reason."
    )
