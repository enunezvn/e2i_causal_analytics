#!/usr/bin/env python3
"""
Database Migration Runner

Runs SQL migration files against Supabase PostgreSQL database.

Usage:
    python scripts/run_migration.py <migration_file.sql>
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client


def run_migration(migration_file: str) -> None:
    """
    Run a SQL migration file against Supabase.

    Args:
        migration_file: Path to SQL migration file
    """
    # Load environment variables
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

    if not supabase_url or not supabase_key:
        print("❌ Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) must be set")
        sys.exit(1)

    # Read migration file
    migration_path = Path(migration_file)
    if not migration_path.exists():
        print(f"❌ Error: Migration file not found: {migration_file}")
        sys.exit(1)

    print(f"📄 Reading migration: {migration_file}")
    with open(migration_path, "r") as f:
        sql = f.read()

    # Connect to Supabase
    print(f"🔌 Connecting to Supabase: {supabase_url}")
    supabase: Client = create_client(supabase_url, supabase_key)

    # Execute migration
    print("⚙️  Executing migration...")
    try:
        # Supabase Python client doesn't have direct SQL execution
        # We need to use the REST API's rpc method with a custom function
        # OR use psycopg2 directly with connection string

        # For now, let's try using rpc to execute SQL
        # Note: This requires a helper function in Supabase
        # Alternative: Use psycopg2 with DATABASE_URL

        database_url = os.getenv("DATABASE_URL")
        if database_url:
            # Use psycopg2 for direct SQL execution
            import psycopg2

            print("🔌 Connecting via psycopg2...")
            conn = psycopg2.connect(database_url)
            conn.autocommit = True

            with conn.cursor() as cursor:
                cursor.execute(sql)

            conn.close()
            print("✅ Migration completed successfully!")
        else:
            print("⚠️  DATABASE_URL not set. Cannot execute raw SQL via Supabase client.")
            print("Please set DATABASE_URL to your PostgreSQL connection string.")
            print("Example: postgresql://postgres:password@db.xxx.supabase.co:5432/postgres")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_migration.py <migration_file.sql>")
        sys.exit(1)

    migration_file = sys.argv[1]
    run_migration(migration_file)
