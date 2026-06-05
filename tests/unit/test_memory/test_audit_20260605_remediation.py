"""Anti-resurrection + characterization guards for the memory-system audit
remediation (audit 2026-06-05: F5 / F1 / F3).

All checks are real static assertions against the actual source tree and
migration files — no mocks, no DB. They lock in the audit decisions and fail
loudly if a retired artifact is resurrected or a live one is accidentally
dropped.
"""

from __future__ import annotations

from pathlib import Path

# tests/unit/test_memory/<this> -> repo root is parents[3]
_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
_DB = _ROOT / "database"


# ---------------------------------------------------------------------------
# F5 — invalidation cascade scope is a deliberate, documented exclusion.
# ---------------------------------------------------------------------------
def test_f5_invalidatable_tables_excludes_procedural_and_cache() -> None:
    src = (_SRC / "memory" / "lifecycle" / "invalidator.py").read_text(encoding="utf-8")
    start = src.index("INVALIDATABLE_TABLES")
    block = src[start : src.index("}", start)]
    assert "procedural_memories" not in block, (
        "procedural_memories must NOT be in INVALIDATABLE_TABLES (audit F5, by design)"
    )
    assert "semantic_memory_cache" not in block, (
        "semantic_memory_cache must NOT be in INVALIDATABLE_TABLES (audit F5, by design)"
    )
    assert "BY-DESIGN EXCLUSION (audit 2026-06-05, F5" in src, (
        "the by-design rationale comment must be present so the exclusion is not "
        "'fixed' by a future reader"
    )


# ---------------------------------------------------------------------------
# F1 — the cognitive_cycles trio is retired.
# ---------------------------------------------------------------------------
def test_f1_conversation_repository_is_retired() -> None:
    assert not (_SRC / "repositories" / "conversation.py").exists(), (
        "src/repositories/conversation.py was retired (audit F1); do not resurrect "
        "it. The live RAG-over-history path is episodic_memories + HybridRetriever; "
        "conversation history lives in chatbot_conversations."
    )
    init_text = (_SRC / "repositories" / "__init__.py").read_text(encoding="utf-8")
    # 'ConversationRepository' must not appear as its own token (ChatbotConversationRepository is fine).
    import re

    bare = re.findall(r"(?<!Chatbot)\bConversationRepository\b", init_text)
    assert not bare, f"repositories/__init__.py still references ConversationRepository: {bare}"


def test_f1_no_src_writer_resurrects_cognitive_cycles() -> None:
    import re

    pat = re.compile(r"""table\(\s*["']cognitive_cycles["']\s*\)\s*\.\s*(insert|upsert|update)""")
    offenders = [
        str(p.relative_to(_ROOT))
        for p in _SRC.rglob("*.py")
        if pat.search(p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not offenders, f"cognitive_cycles must have no writer (audit F1); found: {offenders}"


def test_f1_retire_and_drop_migrations_present() -> None:
    rpc = (_DB / "memory" / "031_retire_conversation_similarity_rpcs.sql").read_text()
    assert "search_similar_conversations" in rpc and "get_conversations_with_feedback" in rpc
    drop = (_DB / "memory" / "032_drop_cognitive_cycles_trio.sql").read_text()
    assert "DROP TABLE IF EXISTS investigation_hops" in drop
    assert "DROP TABLE IF EXISTS cognitive_cycles" in drop
    # FK child must be dropped before the parent.
    assert drop.index("investigation_hops") < drop.index("cognitive_cycles CASCADE")


# ---------------------------------------------------------------------------
# F3 — orphan dspy tables dropped; live table + newer GEPA tables preserved.
# ---------------------------------------------------------------------------
def test_f3_drop_migration_targets_only_the_orphans() -> None:
    drop = (_DB / "memory" / "033_drop_orphan_dspy_tables.sql").read_text()
    for orphan in (
        "dspy_optimization_runs",
        "dspy_prompt_versions",
        "dspy_cognitive_context_history",
    ):
        assert f"DROP TABLE IF EXISTS {orphan}" in drop, f"033 must drop {orphan}"
    # The live training-signals table must NOT be dropped.
    assert "DROP TABLE IF EXISTS dspy_agent_training_signals" not in drop


def test_f3_preserves_live_training_signals_table() -> None:
    adapters = (_SRC / "rag" / "memory_adapters.py").read_text(encoding="utf-8")
    assert 'table("dspy_agent_training_signals")' in adapters, (
        "dspy_agent_training_signals is LIVE (writer in memory_adapters.py); it must "
        "not be dropped — only the 3 orphan dspy tables are retired (audit F3)."
    )


def test_f3_preserves_newer_gepa_tables() -> None:
    gepa = _DB / "ml" / "023_gepa_optimization_tables.sql"
    assert gepa.exists(), "the newer GEPA tables (023) are the current stake — keep them"
    text = gepa.read_text(encoding="utf-8")
    assert "prompt_optimization_runs" in text and "optimized_instructions" in text


# ---------------------------------------------------------------------------
# F2 / F4 (S3) — dormant semantic_memory_cache sync wrappers + inert TTL retired.
# ---------------------------------------------------------------------------
def test_s3_dead_cache_sync_wrappers_removed() -> None:
    epi = (_SRC / "memory" / "episodic_memory.py").read_text(encoding="utf-8")
    sem = (_SRC / "memory" / "semantic_memory.py").read_text(encoding="utf-8")
    assert "def sync_treatment_relationships_to_cache" not in epi, (
        "dead sync wrapper must stay removed (audit F2/S3 — zero non-test callers; "
        "the relationship-lookup capability is live via FalkorDB graph traversal)"
    )
    assert "def sync_data_layer_to_semantic_cache" not in sem, (
        "dead sync wrapper must stay removed (audit F2/S3)"
    )


def test_s3_inert_cache_ttl_control_removed() -> None:
    cfg = (_SRC / "memory" / "graphiti_config.py").read_text(encoding="utf-8")
    assert "cache_ttl_minutes: int" not in cfg, (
        "inert cache_ttl_minutes field must stay removed (audit F4/S3 — never enforced)"
    )
    assert "cache_ttl_minutes=raw_config" not in cfg, (
        "inert cache_ttl_minutes loader must stay removed (audit F4/S3)"
    )
