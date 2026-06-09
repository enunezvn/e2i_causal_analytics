"""Guard: memory-subsystem schema docs reflect the current table reality.

Audit finding **F6** (`docs/reports/memory-system-audit-20260605.md`) first asked
the docs to stop presenting the dropped trio as live. Migrations ``031``/``032``
dropped the ``016`` RPCs + ``cognitive_cycles``/``investigation_hops``.

**Reversal (owner decision 2026-06-09):** that drop was wrong — ``cognitive_cycles``
is the parent ledger of the live 4-phase cognitive cycle (its ``cycle_id`` is
already threaded onto ``episodic_memories`` / ``learning_signals``), so migration
``042`` RESTORED both tables and the 4-phase workflow now writes the parent row.
The docs must therefore present these tables as LIVE (restored, producer wired),
NOT retired. ``semantic_memory_cache`` remains genuinely seed-only (F2/F4).

This is an anti-staleness guard (real assertions against the real docs, no mocks):
it FAILS if the docs still present the restored tables as retired/dropped, drop
the restore (042) provenance, omit the producer, or stop labelling
``semantic_memory_cache`` as seed-only.
"""

from __future__ import annotations

import re
from pathlib import Path

# parents[3] == repo root, resolved from the worktree top regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DOC = REPO_ROOT / "docs" / "data" / "07-SUPPORTING-SCHEMAS.md"
ONBOARDING = REPO_ROOT / "docs" / "ONBOARDING.md"

# Tables restored by migration 042 (reverses the 032 drop).
RESTORED_TABLES = ("cognitive_cycles", "investigation_hops")


def _section(text: str, heading: str) -> str:
    """Return the body of a ``### <heading>`` section, up to the next heading."""
    pattern = re.compile(
        r"^###\s+" + re.escape(heading) + r"\s*$(.*?)(?=^#{2,3}\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    assert match is not None, f"section '### {heading}' not found in {SCHEMA_DOC}"
    return match.group(1)


def test_restored_tables_marked_restored_with_migration_provenance() -> None:
    """Each restored table's section states it is LIVE/RESTORED and cites mig 042,
    and must NOT carry a retired/dropped banner."""
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    for table in RESTORED_TABLES:
        body = _section(text, table)
        assert re.search(r"restored|live", body, re.IGNORECASE), (
            f"### {table} section must state it was RESTORED / is live (migration 042)."
        )
        assert "042" in body, (
            f"### {table} section must cite migration 042 (the restore) for provenance."
        )
        assert "🗑️ RETIRED" not in body, (
            f"### {table} must not still carry a 'RETIRED' banner — it was restored (mig 042)."
        )


def test_cognitive_cycles_documents_its_producer() -> None:
    """The cognitive_cycles section names the live producer so a reader knows the
    parent ledger is now written (not an orphan)."""
    body = _section(SCHEMA_DOC.read_text(encoding="utf-8"), "cognitive_cycles")
    assert "cognitive_integration" in body or "CognitiveService" in body, (
        "### cognitive_cycles must name its producer "
        "(src/memory/cognitive_integration.py::CognitiveService) — the 4-phase "
        "workflow persists the parent cycle row (audit-F1 reversal, mig 042)."
    )


def test_semantic_memory_cache_marked_seed_only() -> None:
    """`semantic_memory_cache` is documented as deploy-seed-only (F2/F4)."""
    body = _section(SCHEMA_DOC.read_text(encoding="utf-8"), "semantic_memory_cache")
    assert re.search(
        r"seed-only|no live producer|no live reader|dormant|audit 2026-06-05|F2",
        body,
        re.IGNORECASE,
    ), (
        "### semantic_memory_cache must note it is deploy-seed-only with no live "
        "producer/reader (audit F2/F4); the dead sync wrappers were retired in "
        "commit 9cb0dc19."
    )


def test_onboarding_lists_cognitive_cycles_as_live() -> None:
    """The onboarding memory-table list advertises cognitive_cycles again now that
    it is restored + wired (mig 042)."""
    text = ONBOARDING.read_text(encoding="utf-8")
    assert "cognitive_cycles" in text, (
        "docs/ONBOARDING.md must list cognitive_cycles as a live memory table again "
        "(restored mig 042; producer wired in cognitive_integration.py)."
    )
