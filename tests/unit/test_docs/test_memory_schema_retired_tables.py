"""Guard: memory-subsystem schema docs reflect the 2026-06-05 audit remediation.

Audit finding **F6** (`docs/reports/memory-system-audit-20260605.md`): the
schema docs presented `cognitive_cycles` / `investigation_hops` as live tables
and labelled `episodic_memories.cycle_id` / `learning_signals.cycle_id` as live
``FK -> cognitive_cycles``. Those tables were subsequently **dropped** by the
remediation (migrations ``031``/``032``; verified absent in the droplet DB), and
the CASCADE drop removed the FK constraints — so the columns are now plain
orphaned UUIDs, not foreign keys. Separately, the dormant `semantic_memory_cache`
sync wrappers were retired (commit ``9cb0dc19``, F2/F4): the table + its
populating RPC remain as a deploy-seed substrate, but there is no live
producer/reader.

This is an anti-resurrection guard (real assertions against the real docs, no
mocks): it FAILS if the docs re-present the dropped tables as live, or drop the
RETIRED provenance, or stop labelling `semantic_memory_cache` as seed-only.
"""

from __future__ import annotations

import re
from pathlib import Path

# parents[3] == repo root, resolved from the worktree top regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DOC = REPO_ROOT / "docs" / "data" / "07-SUPPORTING-SCHEMAS.md"
ONBOARDING = REPO_ROOT / "docs" / "ONBOARDING.md"

# Tables dropped by migration 032 (FK child investigation_hops, then parent
# cognitive_cycles). The drop migration that retired the 016 RPCs is 031.
RETIRED_TABLES = ("cognitive_cycles", "investigation_hops")


def _section(text: str, heading: str) -> str:
    """Return the body of a ``### <heading>`` section, up to the next heading."""
    pattern = re.compile(
        r"^###\s+" + re.escape(heading) + r"\s*$(.*?)(?=^#{2,3}\s|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    assert match is not None, f"section '### {heading}' not found in {SCHEMA_DOC}"
    return match.group(1)


def test_retired_tables_marked_retired_with_migration_provenance() -> None:
    """Each dropped table's section states it was RETIRED and cites mig 032."""
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    for table in RETIRED_TABLES:
        body = _section(text, table)
        assert re.search(r"retired|removed|dropped", body, re.IGNORECASE), (
            f"### {table} section must state it was RETIRED — it is dropped in "
            f"the live DB (migrations 031/032); docs must not present it as live."
        )
        assert "032" in body, (
            f"### {table} section must cite migration 032 (the drop) for provenance."
        )


def test_no_live_fk_label_to_dropped_cognitive_cycles() -> None:
    """`cycle_id` columns must not be labelled as a live FK to a dropped table."""
    text = SCHEMA_DOC.read_text(encoding="utf-8")
    offenders = re.findall(r"FK\s*(?:->|→)\s*`?cognitive_cycles`?", text)
    assert not offenders, (
        "docs still label cycle_id as a live 'FK -> cognitive_cycles', but "
        "cognitive_cycles was dropped (mig 032; CASCADE removed the FK). "
        "Annotate the column as legacy/retired instead."
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


def test_onboarding_does_not_list_cognitive_cycles_as_live() -> None:
    """The onboarding memory-table list must not advertise a dropped table."""
    text = ONBOARDING.read_text(encoding="utf-8")
    assert "cognitive_cycles" not in text, (
        "docs/ONBOARDING.md still lists cognitive_cycles as a live memory table; "
        "it was dropped (mig 032)."
    )
