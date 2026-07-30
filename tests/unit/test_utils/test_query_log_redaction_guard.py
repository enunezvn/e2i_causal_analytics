"""Regression guard for #1367: query-bearing log lines must route through redact_query.

The scattered ``query[:N]`` idiom (and one untruncated ``{query}``) was replaced
by the :func:`src.utils.redaction.redact_query` SSOT. This guard parses the AST
of the governed files and inspects **only** ``logger.<level>(...)`` calls: any
f-string that interpolates a user-query variable (``query``, ``chat_request.query``,
``original_query``, ...) must wrap it in ``redact_query(...)`` — a bare
``{query}`` or a raw ``{query[:50]}`` slice fails.

Why AST and not grep: these same files legitimately carry non-logger query
slices — ``query[:500]`` Opik/MLflow trace payloads, ``request.query[:50]``
session-title derivation, ``query[:200]`` memory descriptions — that are out of
scope for #1367. A text grep would false-positive on all of them; keying on
``logger.*`` call nodes and the exact query variable names does not.

Limitation: the guard keys on a fixed set of query variable names in logger
calls within a fixed file list. It does not police query logging in files
outside that list, nor variables under other names. It is a regression fence
for the reviewed surface, not a proof of repo-wide coverage.
"""

import ast
from pathlib import Path

import pytest

# The query-logging surface #1367 governs (repo-relative).
GOVERNED_FILES = [
    "src/agents/causal_impact/agent.py",
    "src/api/routes/chatbot_tools.py",
    "src/api/routes/copilotkit.py",
    "src/api/routes/chatbot_graph.py",
    "src/api/routes/chatbot_dspy.py",
    "src/rag/retriever.py",
    "src/rag/insight_enricher.py",
    "src/rag/query_optimizer.py",
    "src/nlp/typo_handler.py",
    "src/agents/tool_composer/composer.py",
    "src/agents/tool_composer/decomposer.py",
    "src/agents/tool_composer/synthesizer.py",
    "src/agents/orchestrator/nodes/intent_classifier.py",
    "src/api/routes/causal.py",
    "src/rag/evaluation.py",
]

# Exact variable names that carry raw user query text.
QUERY_NAMES = frozenset(
    {
        "query",
        "question",
        "user_query",
        "original_query",
        "query_text",
        "rewritten_query",
        "corrected_query",
    }
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _is_logger_call(node: ast.Call) -> bool:
    """True for ``logger.<level>(...)`` calls."""
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "logger"
    )


def _query_name(value: ast.expr) -> str | None:
    """Return the query variable name a node references, else None.

    Handles ``query`` (Name) and ``chat_request.query`` (Attribute); unwraps a
    ``query[:N]`` Subscript and the ``(request.query or "")[:N]`` guard idiom
    (BoolOp) to the underlying query name.
    """
    if isinstance(value, ast.Subscript):
        value = value.value
    if isinstance(value, ast.BoolOp):
        for operand in value.values:
            name = _query_name(operand)
            if name is not None:
                return name
        return None
    if isinstance(value, ast.Name) and value.id in QUERY_NAMES:
        return value.id
    if isinstance(value, ast.Attribute) and value.attr in QUERY_NAMES:
        return value.attr
    return None


def _wrapped_in_redact(value: ast.expr) -> bool:
    """True if ``value`` is a ``redact_query(...)`` call."""
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "redact_query"
    )


def _raw_query_logs(tree: ast.AST) -> list[tuple[int, str]]:
    """Find logger calls interpolating a raw (unredacted) query variable."""
    findings: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _is_logger_call(node)):
            continue
        for fstring in ast.walk(node):
            if not isinstance(fstring, ast.FormattedValue):
                continue
            if _wrapped_in_redact(fstring.value):
                continue
            name = _query_name(fstring.value)
            if name is not None:
                findings.append((fstring.lineno, name))
    return findings


@pytest.mark.parametrize("rel_path", GOVERNED_FILES)
def test_logger_query_interpolation_is_redacted(rel_path):
    path = _REPO_ROOT / rel_path
    assert path.exists(), f"governed file moved or renamed: {rel_path}"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    findings = _raw_query_logs(tree)
    assert not findings, (
        f"{rel_path}: logger call(s) interpolate a raw user-query variable "
        f"without redact_query() — route through src.utils.redaction.redact_query "
        f"(#1367). Offending (line, name): {findings}"
    )
