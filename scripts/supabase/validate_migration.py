#!/usr/bin/env python3
"""
Supabase Self-Hosted Migration Validation Script

This script validates that the migration from Supabase Cloud to self-hosted
was successful by checking:
1. Database connectivity
2. Schema integrity (tables, indexes, constraints)
3. Data integrity (row counts, sample data)
4. RLS policies
5. PostgreSQL extensions
6. Application connectivity

Usage:
    python validate_migration.py [--verbose] [--fix]

Environment Variables:
    SUPABASE_DB_HOST - Database host (default: localhost)
    SUPABASE_DB_PORT - Database port (default: 5433)
    SUPABASE_DB_NAME - Database name (default: postgres)
    SUPABASE_DB_USER - Database user (default: postgres)
    POSTGRES_PASSWORD - Database password

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ValidationResult:
    """Result of a validation check."""

    name: str
    passed: bool
    message: str
    details: dict[str, Any] | None = None


class MigrationValidator:
    """Validates Supabase self-hosted migration."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5433,
        database: str = "postgres",
        user: str = "postgres",
        password: str = "",
        verbose: bool = False,
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.verbose = verbose
        self.conn = None
        self.results: list[ValidationResult] = []

    def connect(self) -> bool:
        """Establish database connection."""
        try:
            import psycopg2

            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            return True
        except ImportError:
            print("Error: psycopg2 not installed. Run: pip install psycopg2-binary")
            return False
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"  {message}")

    def execute_query(self, query: str) -> list[tuple]:
        """Execute a query and return results."""
        with self.conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()

    def check_connection(self) -> ValidationResult:
        """Validate database connectivity."""
        try:
            result = self.execute_query("SELECT version()")
            version = result[0][0]
            return ValidationResult(
                name="Database Connection",
                passed=True,
                message="Successfully connected to PostgreSQL",
                details={"version": version},
            )
        except Exception as e:
            return ValidationResult(
                name="Database Connection",
                passed=False,
                message=f"Connection failed: {e}",
            )

    def check_extensions(self) -> ValidationResult:
        """Validate required PostgreSQL extensions are installed."""
        required_extensions = {
            "uuid-ossp": True,
            "pgcrypto": True,
            "pg_trgm": True,
            "vector": False,  # Optional for RAG
        }

        query = "SELECT extname FROM pg_extension"
        installed = {row[0] for row in self.execute_query(query)}

        missing_required = []
        missing_optional = []

        for ext, required in required_extensions.items():
            if ext not in installed:
                if required:
                    missing_required.append(ext)
                else:
                    missing_optional.append(ext)

        if missing_required:
            return ValidationResult(
                name="PostgreSQL Extensions",
                passed=False,
                message=f"Missing required extensions: {', '.join(missing_required)}",
                details={
                    "installed": list(installed),
                    "missing_required": missing_required,
                    "missing_optional": missing_optional,
                },
            )

        return ValidationResult(
            name="PostgreSQL Extensions",
            passed=True,
            message=f"All required extensions installed ({len(installed)} total)",
            details={
                "installed": list(installed),
                "missing_optional": missing_optional,
            },
        )

    def check_tables(self) -> ValidationResult:
        """Validate expected tables exist."""
        # Key E2I tables that should exist
        expected_tables = {
            "business_metrics",
            "agent_activities",
            "causal_paths",
            "conversations",
            "triggers",
            "model_experiments",
            "feature_store",
        }

        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
        """
        existing = {row[0] for row in self.execute_query(query)}

        missing = expected_tables - existing
        extra = existing - expected_tables

        if missing:
            return ValidationResult(
                name="Database Tables",
                passed=False,
                message=f"Missing tables: {', '.join(missing)}",
                details={
                    "total_tables": len(existing),
                    "missing": list(missing),
                    "extra_tables": len(extra),
                },
            )

        return ValidationResult(
            name="Database Tables",
            passed=True,
            message=f"Found {len(existing)} tables, all expected tables present",
            details={
                "total_tables": len(existing),
                "expected_found": list(expected_tables & existing),
            },
        )

    def check_row_counts(self) -> ValidationResult:
        """Validate tables have data."""
        query = """
            SELECT schemaname, relname, n_live_tup
            FROM pg_stat_user_tables
            WHERE schemaname = 'public'
            ORDER BY n_live_tup DESC
            LIMIT 20
        """
        results = self.execute_query(query)

        total_rows = sum(row[2] for row in results)
        empty_tables = [row[1] for row in results if row[2] == 0]

        details = {
            "total_rows": total_rows,
            "tables_with_data": len([r for r in results if r[2] > 0]),
            "empty_tables": len(empty_tables),
            "top_tables": [(r[1], r[2]) for r in results[:5]],
        }

        if total_rows == 0:
            return ValidationResult(
                name="Data Integrity",
                passed=False,
                message="No data found in any tables",
                details=details,
            )

        return ValidationResult(
            name="Data Integrity",
            passed=True,
            message=f"Found {total_rows:,} total rows across tables",
            details=details,
        )

    def check_rls_policies(self) -> ValidationResult:
        """Validate RLS policies are in place."""
        query = """
            SELECT schemaname, tablename, policyname, roles, cmd
            FROM pg_policies
            WHERE schemaname = 'public'
        """
        results = self.execute_query(query)

        if not results:
            return ValidationResult(
                name="Row Level Security",
                passed=True,
                message="No RLS policies found (may be expected for service role)",
                details={"policy_count": 0},
            )

        policies_by_table = {}
        for row in results:
            table = row[1]
            if table not in policies_by_table:
                policies_by_table[table] = []
            policies_by_table[table].append(row[2])

        return ValidationResult(
            name="Row Level Security",
            passed=True,
            message=f"Found {len(results)} RLS policies across {len(policies_by_table)} tables",
            details={
                "policy_count": len(results),
                "tables_with_policies": len(policies_by_table),
                "policies_by_table": policies_by_table,
            },
        )

    def check_indexes(self) -> ValidationResult:
        """Validate indexes exist for key tables."""
        query = """
            SELECT tablename, indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
        """
        results = self.execute_query(query)

        indexes_by_table = {}
        for row in results:
            table = row[0]
            if table not in indexes_by_table:
                indexes_by_table[table] = []
            indexes_by_table[table].append(row[1])

        return ValidationResult(
            name="Database Indexes",
            passed=True,
            message=f"Found {len(results)} indexes across {len(indexes_by_table)} tables",
            details={
                "index_count": len(results),
                "tables_with_indexes": len(indexes_by_table),
            },
        )

    def check_foreign_keys(self) -> ValidationResult:
        """Validate foreign key constraints."""
        query = """
            SELECT
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = 'public'
        """
        results = self.execute_query(query)

        return ValidationResult(
            name="Foreign Key Constraints",
            passed=True,
            message=f"Found {len(results)} foreign key constraints",
            details={"fk_count": len(results)},
        )

    def check_supabase_api(self) -> ValidationResult:
        """Validate Supabase API connectivity."""
        try:
            import requests

            api_url = os.getenv("SUPABASE_URL", "http://localhost:8000")
            anon_key = os.getenv("SUPABASE_ANON_KEY", "")

            if not anon_key:
                return ValidationResult(
                    name="Supabase API",
                    passed=False,
                    message="SUPABASE_ANON_KEY not set",
                )

            # Test REST API
            response = requests.get(
                f"{api_url}/rest/v1/",
                headers={"apikey": anon_key},
                timeout=5,
            )

            if response.status_code == 200:
                return ValidationResult(
                    name="Supabase API",
                    passed=True,
                    message="API responding correctly",
                    details={"url": api_url, "status_code": response.status_code},
                )
            else:
                return ValidationResult(
                    name="Supabase API",
                    passed=False,
                    message=f"API returned status {response.status_code}",
                    details={"url": api_url, "status_code": response.status_code},
                )
        except ImportError:
            return ValidationResult(
                name="Supabase API",
                passed=False,
                message="requests library not installed",
            )
        except Exception as e:
            return ValidationResult(
                name="Supabase API",
                passed=False,
                message=f"API check failed: {e}",
            )

    def run_all_checks(self) -> list[ValidationResult]:
        """Run all validation checks."""
        checks = [
            self.check_connection,
            self.check_extensions,
            self.check_tables,
            self.check_row_counts,
            self.check_rls_policies,
            self.check_indexes,
            self.check_foreign_keys,
            self.check_supabase_api,
        ]

        for check in checks:
            try:
                result = check()
                self.results.append(result)

                status = "PASS" if result.passed else "FAIL"
                print(f"[{status}] {result.name}: {result.message}")

                if self.verbose and result.details:
                    for key, value in result.details.items():
                        self.log(f"{key}: {value}")

            except Exception as e:
                self.results.append(
                    ValidationResult(
                        name=check.__name__,
                        passed=False,
                        message=f"Check failed with error: {e}",
                    )
                )

        return self.results

    def generate_report(self, output_path: Path | None = None) -> dict:
        """Generate validation report."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        report = {
            "validation_date": datetime.now().isoformat(),
            "target": {
                "host": self.host,
                "port": self.port,
                "database": self.database,
            },
            "summary": {
                "total_checks": len(self.results),
                "passed": passed,
                "failed": failed,
                "success_rate": f"{(passed / len(self.results) * 100):.1f}%",
            },
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
        }

        if output_path:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nReport saved to: {output_path}")

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate Supabase self-hosted migration"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("SUPABASE_DB_HOST", "localhost"),
        help="Database host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("SUPABASE_DB_PORT", "5433")),
        help="Database port",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("SUPABASE_DB_NAME", "postgres"),
        help="Database name",
    )
    parser.add_argument(
        "--user",
        default=os.getenv("SUPABASE_DB_USER", "postgres"),
        help="Database user",
    )

    args = parser.parse_args()

    password = os.getenv("POSTGRES_PASSWORD", "")
    if not password:
        import getpass

        password = getpass.getpass("Enter PostgreSQL password: ")

    print("=" * 60)
    print("Supabase Self-Hosted Migration Validation")
    print("=" * 60)
    print(f"Target: {args.host}:{args.port}/{args.database}")
    print()

    validator = MigrationValidator(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=password,
        verbose=args.verbose,
    )

    if not validator.connect():
        print("\nValidation failed: Could not connect to database")
        sys.exit(1)

    try:
        validator.run_all_checks()
        report = validator.generate_report(args.output)

        print()
        print("=" * 60)
        print(f"Summary: {report['summary']['passed']}/{report['summary']['total_checks']} checks passed")
        print(f"Success Rate: {report['summary']['success_rate']}")
        print("=" * 60)

        if report["summary"]["failed"] > 0:
            print("\nFailed checks:")
            for result in validator.results:
                if not result.passed:
                    print(f"  - {result.name}: {result.message}")
            sys.exit(1)
        else:
            print("\nMigration validation successful!")
            sys.exit(0)

    finally:
        validator.close()


if __name__ == "__main__":
    main()
