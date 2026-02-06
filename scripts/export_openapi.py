#!/usr/bin/env python
"""
Export OpenAPI specification from the FastAPI app without running a server.

Usage:
    python -m scripts.export_openapi --output openapi.json
    python -m scripts.export_openapi  # prints to stdout

Environment:
    Automatically sets E2I_TESTING_MODE=true and mock API keys so the app
    can be imported without real service connections.
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Set testing/mock environment BEFORE any app imports
# ---------------------------------------------------------------------------
os.environ.setdefault("E2I_TESTING_MODE", "true")
os.environ.setdefault("DISABLE_RATE_LIMITING", "true")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-export-openapi")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-export-openapi")
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "test-key")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-key")
os.environ.setdefault("REDIS_PASSWORD", "test-password")
os.environ.setdefault("FALKORDB_PASSWORD", "test-password")

# ---------------------------------------------------------------------------
# Import app and extract spec
# ---------------------------------------------------------------------------

from src.api.main import app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OpenAPI spec as JSON")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: stdout)",
    )
    args = parser.parse_args()

    spec = app.openapi()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(spec, f, indent=2, default=str)
        print(f"OpenAPI spec written to {args.output}", file=sys.stderr)
    else:
        json.dump(spec, sys.stdout, indent=2, default=str)


if __name__ == "__main__":
    main()
