"""Shared plumbing for the three READ-ONLY live proofs of #2286 / #2287.

Each live script: tees its stdout to ``<script>.out`` beside itself, reads prod Supabase
through the app's own service-role clients with SELECTs only (no insert / update /
delete / rpc anywhere in these scripts or the code paths they call), and ends with ONE
``VERDICT: ...`` line. The clients are injectable so the same ``run()`` is exercised
against a throwaway Postgres + PostgREST before an owner runs it live.

The app reads its connection from the environment and nothing loads ``.env`` for it;
``load_env()`` loads the ``.env`` of this checkout, or of the main checkout when run
from a git worktree (worktrees carry no ``.env``).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MIGRATION_163 = REPO / "database" / "migrations" / "163_registry_cohort_contract_hcp_adoption.sql"
BRANDS = ("Remibrutinib", "Fabhalta", "Kisqali")
VIEW = "hcp_adoption_goldstd_v"


def model_name(brand: str) -> str:
    return f"hcp_adoption_{brand.lower()}_goldstd_lr_v1"


class Tee:
    """Duplicate writes to the terminal and ``<script>.out``."""

    def __init__(self, stream: TextIO, path: Path):
        self._stream = stream
        self._file = open(path, "w")

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()


def tee_to_out(script_file: str) -> Path:
    out = Path(script_file).with_suffix(".out")
    sys.stdout = Tee(sys.stdout, out)  # type: ignore[assignment]
    return out


def load_env() -> Optional[Path]:
    from dotenv import load_dotenv

    candidates: List[Path] = [REPO / ".env"]
    try:
        common = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        if common:
            candidates.append(Path(common).parent / ".env")
    except (OSError, subprocess.SubprocessError):
        pass
    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)
            return path
    return None


def live_sync_client() -> Any:
    from src.memory.services.factories import get_supabase_client

    return get_supabase_client()


async def live_async_client() -> Any:
    from src.memory.services.factories import get_async_supabase_client

    return await get_async_supabase_client()


def migration_163_rows() -> Dict[str, Dict[str, Any]]:
    """{model_name: {"set": {column: literal}, "where": text}} parsed from migration 163."""
    text = "\n".join(
        line for line in MIGRATION_163.read_text().splitlines() if not line.strip().startswith("--")
    )
    out: Dict[str, Dict[str, Any]] = {}
    for m in re.finditer(
        r"UPDATE ml_model_registry\s+SET (?P<set>.*?)\s+WHERE model_name = '(?P<model>[^']+)'"
        r"(?P<where>.*?);",
        text,
        re.S,
    ):
        sets = dict(re.findall(r"(cohort_\w+) = '([^']*)'", m["set"]))
        out[m["model"]] = {"set": sets, "where": m["where"]}
    return out


def header(title: str) -> None:
    import src

    print(f"{title}")
    print(f"code under test: {os.path.dirname(src.__file__)}")
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = "?"
    print(f"git HEAD: {sha}")
