"""Run migration 143's GENERATED statements against planted business_metrics rows.

Canonical TRx lane, codex r2. Each query runs in its own psql session inside
BEGIN ... ROLLBACK. A TEMP table named business_metrics (pg_temp is searched
first) shadows public.business_metrics, so a scenario sees ONLY its planted rows
and nothing persists. CURRENT_DATE is pinned so the in-progress-month rule is
deterministic. The object quacks like the sync supabase client's
``rpc("kpi_query", ...)``, so the real calculator and series reader run on it.
Skips without docker.
"""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
from datetime import date
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import pytest

from scripts import gen_canonical_volume_registry as gen

STATEMENTS: Dict[str, str] = {qid: sql for qid, sql, _n, _note in gen.registry_rows()}
_COLUMNS = ("metric_id", "metric_date", "metric_type", "brand", "region", "value", "is_synthetic")
_IDS = itertools.count(1)


def _literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def bind(sql: str, params: Sequence[Any]) -> str:
    """Positional $N -> SQL literals, highest N first so $1 never rewrites $10."""
    for n in range(len(params), 0, -1):
        sql = sql.replace(f"${n}", _literal(params[n - 1]))
    return sql


def bm_row(
    metric_type: str,
    month: str,
    brand: Optional[str],
    region: Optional[str],
    value: float,
    *,
    synthetic: bool = False,
) -> Dict[str, Any]:
    return {
        "metric_id": f"planted-{next(_IDS)}",
        "metric_date": month,
        "metric_type": metric_type,
        "brand": brand,
        "region": region,
        "value": value,
        "is_synthetic": synthetic,
    }


class PlantedBusinessMetrics:
    """A sync ``kpi_query`` client over planted rows (see the module docstring)."""

    def __init__(self, rows: Sequence[Dict[str, Any]], today: date):
        if shutil.which("docker") is None:
            pytest.skip("docker not available")
        if (
            subprocess.run(
                ["docker", "exec", "supabase-db", "true"], capture_output=True
            ).returncode
            != 0
        ):
            pytest.skip("supabase-db container not reachable")
        self.rows = list(rows)
        self.today = today
        self.calls: List[Dict[str, Any]] = []

    def query(self, query_id: str, params: Sequence[Any]) -> List[Dict[str, Any]]:
        sql = bind(STATEMENTS[query_id], params).replace(
            "CURRENT_DATE", f"DATE '{self.today.isoformat()}'"
        )
        inserts = "".join(
            f"INSERT INTO business_metrics ({', '.join(_COLUMNS)}) "
            f"VALUES ({', '.join(_literal(row[c]) for c in _COLUMNS)});\n"
            for row in self.rows
        )
        script = (
            "BEGIN;\n"
            "CREATE TEMP TABLE business_metrics (LIKE public.business_metrics INCLUDING DEFAULTS) "
            "ON COMMIT DROP;\n"
            + inserts
            + f"SELECT coalesce(json_agg(row_to_json(q)), '[]'::json) FROM ({sql}) q;\n"
            + "ROLLBACK;\n"
        )
        out = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                "supabase-db",
                "psql",
                "-U",
                "postgres",
                "-d",
                "postgres",
                "-v",
                "ON_ERROR_STOP=1",
                "-qtA",
            ],
            input=script,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert out.returncode == 0, out.stderr
        return json.loads([line for line in out.stdout.splitlines() if line.strip()][-1])

    def rpc(self, name: str, payload: Dict[str, Any]) -> Any:
        assert name == "kpi_query", name
        self.calls.append(dict(payload))
        rows = self.query(payload["query_id"], payload["params"])
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=rows))

    def eligible(self, include_synthetic: bool) -> List[Dict[str, Any]]:
        """What the history fetch (fetch_canonical_rows) returns under the synthetic policy."""
        current = self.today.replace(day=1).isoformat()
        return [
            row
            for row in self.rows
            if row["metric_date"] < current and (include_synthetic or not row["is_synthetic"])
        ]
