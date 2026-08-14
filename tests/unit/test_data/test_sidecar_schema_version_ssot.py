"""The sidecar `schema_version` has ONE source of truth (#1620).

The producer (`graph.py::write_adaptive_verdicts_sidecar`) and the reader
(`audit_sidecar_reader.py`) must agree on `schema_version` exactly — the reader
WARNs otherwise, and `test_role_propagation_contract.py` asserts no such WARN is
emitted. So "they always match" is a property the system *requires*.

It used to be maintained by hand: two independent string literals plus two
near-identical copies of the full bump changelog (1.1 → 1.8). Nothing made them
agree; only a test noticed, after the fact, in CI.

That cost a real CI cycle. PR #1618 bumped 1.7 → 1.8, updated the three pins under
`tests/unit/`, and missed the two under `tests/integration/` — two red shards, fixed
in `f16e2311`. A unit-scoped local run could not have caught it.

These tests pin the *structural* property (one literal, one changelog). The five
version pins elsewhere in the tree stay hardcoded on purpose: they are tripwires
whose whole function is to trip on a bump and force an explicit, reviewed
confirmation that the change is additive and MAJOR-preserving. Deriving them from
the constant would turn them into `assert X == X`.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WRITER = _REPO_ROOT / "src" / "agents" / "ml_foundation" / "data_preparer" / "graph.py"
_READER = _REPO_ROOT / "src" / "data" / "audit_sidecar_reader.py"

_VERSION_LITERAL = re.compile(r"""["']\d+\.\d+["']""")


def test_writer_does_not_hardcode_its_own_schema_version() -> None:
    """The producer must derive the version, not restate it.

    A second literal is what allows writer and reader to drift silently.
    """
    source = _WRITER.read_text()
    hardcoded = re.findall(r"""["']schema_version["']\s*:\s*(["']\d+\.\d+["'])""", source)
    assert not hardcoded, (
        f"graph.py hardcodes schema_version={hardcoded} instead of importing the "
        "shared constant. That is the second source of truth #1620 removed — writer "
        "and reader can then drift with nothing but a downstream test to notice."
    )


def test_writer_imports_the_shared_constant() -> None:
    source = _WRITER.read_text()
    assert "SIDECAR_SCHEMA_VERSION" in source, (
        "graph.py should import SIDECAR_SCHEMA_VERSION from "
        "src.data.audit_sidecar_reader and use it for the payload's schema_version."
    )


def test_writer_and_reader_agree_at_runtime() -> None:
    """The property the reader's WARN path depends on, asserted directly."""
    import importlib

    reader = importlib.import_module("src.data.audit_sidecar_reader")
    graph = importlib.import_module("src.agents.ml_foundation.data_preparer.graph")

    version = reader.SIDECAR_SCHEMA_VERSION
    assert re.fullmatch(r"\d+\.\d+", version), f"malformed schema version {version!r}"
    assert getattr(graph, "SIDECAR_SCHEMA_VERSION", None) == version, (
        "graph.py does not expose the same SIDECAR_SCHEMA_VERSION object as the "
        "reader — the import is not actually wired through."
    )


#: Marker the reader puts above its changelog block. Keying the test to an explicit
#: sentinel rather than to comment formatting means the entries can be reworded or
#: restyled freely without a false CI failure (codex round 1, LOW).
_CHANGELOG_MARKER = "SIDECAR SCHEMA CHANGELOG"


def test_the_bump_changelog_lives_in_exactly_one_place() -> None:
    """One changelog, not two hand-synced copies.

    Both files previously carried the full 1.1 → 1.8 history in near-identical
    prose, so a bump meant editing two comment blocks as well as two literals.
    """
    reader_source = _READER.read_text()
    writer_source = _WRITER.read_text()

    assert _CHANGELOG_MARKER in reader_source, (
        f"audit_sidecar_reader.py should carry the {_CHANGELOG_MARKER!r} marker above "
        "the bump history — it is what makes the single-source claim checkable."
    )
    assert _CHANGELOG_MARKER not in writer_source, (
        "graph.py carries the changelog marker; the history belongs in exactly one "
        "file, beside the constant."
    )

    # A second, format-independent check on the writer: a *copy* of the history would
    # show up as several version-labelled comment lines. A pointer mentions none.
    writer_entries = re.findall(r"^\s*#\s*v?1\.\d+\b", writer_source, flags=re.MULTILINE)
    assert len(writer_entries) <= 1, (
        f"graph.py still carries {len(writer_entries)} version-labelled comment lines, "
        "which looks like a re-duplicated changelog. Keep the history beside the "
        "constant so a future bump edits ONE block."
    )
