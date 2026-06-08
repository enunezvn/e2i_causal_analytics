from pathlib import Path


def test_migration_059_creates_gaps_feedback_tables():
    sql = Path("database/migrations/059_gaps_feedback_persistence.sql").read_text()
    for table in (
        "gap_analyses",
        "feedback_learning_batches",
        "feedback_patterns",
        "feedback_knowledge_updates",
        "feedback_items",
    ):
        assert f"CREATE TABLE IF NOT EXISTS public.{table}" in sql, table
    # JSONB payload column on each row-store table + indexed scalar filters.
    assert sql.count("payload JSONB NOT NULL") >= 5
    assert "CREATE INDEX IF NOT EXISTS idx_gap_analyses_brand_status" in sql
    assert "CREATE INDEX IF NOT EXISTS idx_feedback_updates_status" in sql
