#!/usr/bin/env python3
"""
RAG Database Migration Runner

Applies RAG schema migrations to Supabase in order.
Run this after setting up the base memory schema.

Usage:
    python scripts/run_rag_migrations.py

Environment variables required:
    SUPABASE_URL: Your Supabase project URL
    SUPABASE_SERVICE_ROLE_KEY: Your Supabase service role key

Part of Phase 1, Checkpoint 1.4.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Check environment
def check_environment() -> bool:
    """Check that required environment variables are set."""
    required = ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"]
    missing = [var for var in required if not os.getenv(var)]

    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("\nPlease set these in your .env file or environment:")
        for var in missing:
            print(f"  export {var}=<your-value>")
        return False

    return True


def get_migration_files() -> list[Path]:
    """Get all RAG migration files in order."""
    rag_dir = project_root / "database" / "rag"
    memory_dir = project_root / "database" / "memory"

    migrations = []

    # First, check if memory migrations exist
    memory_migrations = [
        memory_dir / "001_agentic_memory_schema_v1.3.sql",
        memory_dir / "011_hybrid_search_functions_fixed.sql",
    ]

    for m in memory_migrations:
        if m.exists():
            migrations.append(m)
        else:
            print(f"⚠️  Memory migration not found: {m.name}")

    # Then RAG-specific migrations
    if rag_dir.exists():
        rag_migrations = sorted(rag_dir.glob("*.sql"))
        # Filter out validation queries
        rag_migrations = [m for m in rag_migrations if "validation" not in m.name.lower()]
        migrations.extend(rag_migrations)
    else:
        print(f"⚠️  RAG migration directory not found: {rag_dir}")

    return migrations


def run_migration(supabase_client, migration_path: Path) -> bool:
    """
    Run a single migration file.

    Args:
        supabase_client: Supabase client instance
        migration_path: Path to the SQL file

    Returns:
        True if successful, False otherwise
    """
    print(f"\n📄 Running: {migration_path.name}")

    try:
        sql_content = migration_path.read_text()

        # Skip empty files
        if not sql_content.strip():
            print(f"   ⏭️  Skipping empty file")
            return True

        # Execute the SQL
        # Note: Supabase's Python client doesn't have a direct execute_raw
        # We need to use the REST API or psycopg2 for raw SQL
        # For now, we'll use the RPC approach with a helper function

        # Split by statement for better error handling
        # This is a simple split - complex SQL may need better parsing
        statements = []
        current_statement = []

        for line in sql_content.split('\n'):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith('--') or not stripped:
                continue

            current_statement.append(line)

            # Check for statement end
            if stripped.endswith(';') and not stripped.endswith('$$;'):
                statements.append('\n'.join(current_statement))
                current_statement = []

            # Handle $$ blocks for functions
            if '$$;' in stripped:
                statements.append('\n'.join(current_statement))
                current_statement = []

        # Execute each statement
        executed = 0
        for stmt in statements:
            if stmt.strip():
                try:
                    # Use raw SQL execution
                    supabase_client.postgrest.rpc('exec_sql', {'sql': stmt}).execute()
                    executed += 1
                except Exception as e:
                    if "function exec_sql" in str(e).lower():
                        # exec_sql doesn't exist, need to use direct connection
                        print(f"   ⚠️  exec_sql function not available")
                        print(f"   💡 Run this migration directly in Supabase SQL Editor")
                        return False
                    else:
                        # Check for common "already exists" errors
                        error_str = str(e).lower()
                        if any(x in error_str for x in ['already exists', 'duplicate']):
                            print(f"   ℹ️  Object already exists, continuing...")
                            continue
                        raise

        print(f"   ✅ Executed {executed} statements")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def run_migration_via_direct_sql(migration_path: Path) -> bool:
    """
    Run migration using direct PostgreSQL connection.

    This is the preferred method for running migrations.
    """
    import psycopg2
    from urllib.parse import urlparse

    db_url = os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")

    if not db_url:
        # Construct from Supabase URL
        supabase_url = os.getenv("SUPABASE_URL", "")
        if "supabase.co" in supabase_url:
            # Extract project ref
            parsed = urlparse(supabase_url)
            project_ref = parsed.netloc.split('.')[0]
            db_host = f"db.{project_ref}.supabase.co"
            db_pass = os.getenv("SUPABASE_DB_PASSWORD", "")
            if db_pass:
                db_url = f"postgresql://postgres:{db_pass}@{db_host}:5432/postgres"

    if not db_url:
        return False

    print(f"\n📄 Running: {migration_path.name}")

    try:
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        cursor = conn.cursor()

        sql_content = migration_path.read_text()
        cursor.execute(sql_content)

        cursor.close()
        conn.close()

        print(f"   ✅ Migration applied successfully")
        return True

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def main():
    """Main entry point."""
    print("=" * 60)
    print("E2I RAG Database Migration Runner")
    print("=" * 60)

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Get migrations
    migrations = get_migration_files()

    if not migrations:
        print("\n❌ No migration files found")
        sys.exit(1)

    print(f"\n📋 Found {len(migrations)} migration file(s):")
    for m in migrations:
        print(f"   - {m.name}")

    # Try to import Supabase client
    try:
        from supabase import create_client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        supabase = create_client(supabase_url, supabase_key)
        print("\n✅ Connected to Supabase")
    except ImportError:
        print("\n⚠️  supabase package not installed")
        supabase = None

    # Run migrations
    print("\n" + "=" * 60)
    print("Running Migrations")
    print("=" * 60)

    success_count = 0
    fail_count = 0

    for migration in migrations:
        # Try direct SQL first (preferred)
        if run_migration_via_direct_sql(migration):
            success_count += 1
        elif supabase and run_migration(supabase, migration):
            success_count += 1
        else:
            fail_count += 1
            print(f"\n💡 To apply {migration.name} manually:")
            print(f"   1. Open Supabase SQL Editor")
            print(f"   2. Copy contents of: {migration}")
            print(f"   3. Execute the SQL")

    # Summary
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {fail_count}")

    if fail_count > 0:
        print(f"\n⚠️  Some migrations failed. Please run them manually in Supabase SQL Editor.")
        print(f"\nMigration files are located in:")
        print(f"  - {project_root / 'database' / 'memory'}")
        print(f"  - {project_root / 'database' / 'rag'}")
        sys.exit(1)
    else:
        print("\n🎉 All migrations applied successfully!")

        # Run validation
        print("\n" + "=" * 60)
        print("Validation")
        print("=" * 60)
        print("Run the validation queries to verify installation:")
        print(f"  - {project_root / 'database' / 'rag' / '002_validation_queries.sql'}")


if __name__ == "__main__":
    main()
