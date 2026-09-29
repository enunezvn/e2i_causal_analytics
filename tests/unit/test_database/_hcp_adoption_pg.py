"""Prod-free real-Postgres + real-PostgREST harness for migrations 162/163 (#2286/#2287).

Nothing here touches ``supabase-db``. A throwaway container of prod's OWN Postgres image
(``docker inspect supabase-db`` -> image tag; container metadata only) gets:

* the enums + the ``hcp_profiles`` columns the view reads, typed exactly as prod
  (``\\d hcp_profiles``, 2026-09-29), then migrations 076 + 082 VERBATIM for
  ``hcp_brand_adoption``;
* a minimal ``ml_model_registry`` (the columns migration 163 touches) and
  ``schema_migrations``;
* rows from the committed generators that reproduce the live cohort's SHAPE
  (``HCPGenerator(seed=42, scv)`` reproduces live ``hcp_profiles`` ids +
  ``influence_network_size``; adoption seed 427 reproduces the live arm and aggregate
  prevalence, not the per-row labels — ``scripts/load_hcp_brand_adoption.py``).

``PostgrestServer`` fronts it with PostgREST at prod's image tag
(``docker inspect supabase-rest``), connecting as ``authenticator`` with anon role
``anon`` exactly like prod, so a request's grants are the real ones: service_role via a
signed JWT, anon without one.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, List, Optional

REPO = Path(__file__).resolve().parents[3]
MIGRATIONS = REPO / "database" / "migrations"
M076 = MIGRATIONS / "076_hcp_brand_adoption.sql"
M082 = MIGRATIONS / "082_hcp_brand_adoption_treatment_arm.sql"
M162_PATH = MIGRATIONS / "162_hcp_adoption_goldstd_view.sql"
VIEW = "hcp_adoption_goldstd_v"
BRANDS = ("Remibrutinib", "Fabhalta", "Kisqali")
OPT_IN = "E2I_DB_INTEGRATION"

# Types and columns as prod carries them (\d hcp_profiles / enum DDL in
# database/core/e2i_ml_complete_v3_schema.sql). Only the columns 162's view reads.
BASE_DDL = """
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE TYPE data_split_type AS ENUM ('train', 'validation', 'test', 'holdout', 'unassigned');
CREATE TYPE brand_type AS ENUM ('Remibrutinib', 'Fabhalta', 'Kisqali', 'competitor', 'other');
CREATE TYPE region_type AS ENUM ('northeast', 'south', 'midwest', 'west');
CREATE TABLE hcp_profiles (
    hcp_id                 VARCHAR(20) PRIMARY KEY,
    specialty              VARCHAR(100),
    geographic_region      region_type,
    years_experience       INTEGER,
    influence_network_size INTEGER,
    peer_influence_score   NUMERIC(3,2),
    is_synthetic           BOOLEAN NOT NULL DEFAULT false
);
CREATE TABLE ml_model_registry (
    id                             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name                     VARCHAR(100) NOT NULL,
    stage                          VARCHAR(20),
    is_synthetic                   BOOLEAN NOT NULL DEFAULT false,
    cohort_data_source             TEXT,
    cohort_target_outcome          TEXT,
    cohort_feature_manifest_source TEXT
);
CREATE TABLE IF NOT EXISTS public.schema_migrations
    (filename TEXT PRIMARY KEY, applied_at TIMESTAMPTZ DEFAULT now());
"""


PROD_GRANTS = """
REVOKE ALL ON hcp_profiles, hcp_brand_adoption, ml_model_registry FROM PUBLIC, anon, authenticated;
GRANT ALL ON hcp_profiles, hcp_brand_adoption, ml_model_registry TO service_role;
"""


def prod_image(container: str) -> Optional[str]:
    """The image tag a prod container runs; ``docker inspect`` reads metadata only."""
    if shutil.which("docker") is None:
        return None
    proc = subprocess.run(
        ["docker", "inspect", container, "--format", "{{.Config.Image}}"],
        capture_output=True,
        timeout=60,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.decode().strip() or None


def generated_frames(n_hcps: int = 5000) -> tuple[Any, Any]:
    """(hcp_profiles frame, hcp_brand_adoption frame) from the committed generators."""
    from src.ml.synthetic.generators import GeneratorConfig, HCPGenerator
    from src.ml.synthetic.generators.hcp_brand_adoption_generator import (
        generate_hcp_brand_adoption_frame,
    )

    hcp = HCPGenerator(GeneratorConfig(id_prefix="scv", seed=42, n_records=n_hcps)).generate()
    adoption = generate_hcp_brand_adoption_frame(
        hcp, seed=427, end_date=date(2026, 6, 1), brands=BRANDS, n_months=37
    )
    return hcp, adoption


def build_base(conn: Any) -> None:
    """Enums + hcp_profiles + registry stub, then 076 and 082 verbatim, as ``postgres``."""
    conn.execute(BASE_DDL, user="postgres")
    for path in (M076, M082):
        conn.pg.rows(conn.db, "select 1")  # container liveness before each file
        proc = conn.pg.run_script(conn.db, path.read_bytes(), user="postgres")
        if proc.returncode != 0:
            raise RuntimeError(f"{path.name}: {proc.stderr.decode()}")
    # Grants as prod holds them (information_schema.role_table_grants, 2026-09-29):
    # postgres + service_role only; the image's default ACL would also grant anon.
    conn.execute(PROD_GRANTS, user="postgres")


def seed(conn: Any, n_hcps: int = 5000) -> dict:
    """Load the generated cohort; returns row counts."""
    hcp, adoption = generated_frames(n_hcps)
    with conn.connect() as c, c.cursor() as cur:
        with cur.copy(
            "COPY hcp_profiles (hcp_id, specialty, geographic_region, years_experience, "
            "influence_network_size, peer_influence_score, is_synthetic) FROM STDIN"
        ) as cp:
            for r in hcp.itertuples(index=False):
                cp.write_row(
                    (
                        r.hcp_id,
                        r.specialty,
                        r.geographic_region,
                        int(r.years_experience),
                        int(r.influence_network_size),
                        round(float(r.peer_influence_score), 2),
                        True,
                    )
                )
        with cur.copy(
            "COPY hcp_brand_adoption (hcp_id, brand, consideration_date, adopted, "
            "adoption_category, data_split, is_synthetic) FROM STDIN"
        ) as cp:
            for r in adoption.itertuples(index=False):
                cp.write_row(
                    (
                        r.hcp_id,
                        r.brand,
                        r.consideration_date,
                        int(r.adopted),
                        r.adoption_category,
                        r.data_split,
                        True,
                    )
                )
        c.commit()
    return {"hcp_profiles": len(hcp), "hcp_brand_adoption": len(adoption)}


def apply_file(conn: Any, path: Path, *, record: bool = True) -> None:
    """Apply one migration file the way run_migrations.sh does: as postgres, in ONE
    transaction, with its ledger row."""
    script = path.read_bytes()
    if record:
        script += (
            f"\nINSERT INTO public.schema_migrations(filename) VALUES ('{path.name}') "
            "ON CONFLICT DO NOTHING;\n"
        ).encode()
    proc = conn.pg.run_script(conn.db, script, single_transaction=True, user="postgres")
    if proc.returncode != 0:
        raise RuntimeError(f"{path.name}: {proc.stderr.decode()}")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def sign_jwt(secret: str, role: str) -> str:
    import jwt

    return jwt.encode({"role": role, "exp": int(time.time()) + 3600}, secret, algorithm="HS256")


@dataclass
class PostgrestServer:
    """PostgREST (prod's image) in front of one throwaway database, host-networked."""

    image: str
    pg: Any
    db: str
    name: str = field(default_factory=lambda: "e2i-learnloop-pgrst-" + secrets.token_hex(4))
    jwt_secret: str = field(default_factory=lambda: secrets.token_hex(32), repr=False)
    _auth_password: str = field(default_factory=lambda: secrets.token_hex(16), repr=False)
    port: int = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, ready_timeout_s: int = 60) -> None:
        # supabase/postgres ships ``authenticator`` (NOINHERIT, member of anon /
        # authenticated / service_role) — the role prod's PostgREST logs in as.
        self.pg.rows(
            "postgres",
            f"ALTER ROLE authenticator WITH LOGIN PASSWORD '{self._auth_password}'",
        )
        self.port = _free_port()
        env = {
            **os.environ,
            "PGRST_DB_URI": (
                f"postgres://authenticator:{self._auth_password}@127.0.0.1:"
                f"{self.pg.host_port}/{self.db}"
            ),
            "PGRST_DB_SCHEMAS": "public",
            "PGRST_DB_ANON_ROLE": "anon",
            "PGRST_JWT_SECRET": self.jwt_secret,
            "PGRST_SERVER_PORT": str(self.port),
            "PGRST_DB_USE_LEGACY_GUCS": "false",
        }
        names = [k for k in env if k.startswith("PGRST_")]
        cmd = ["docker", "run", "-d", "--name", self.name, "--network", "host"]
        cmd += ["--label", "e2i.learnloop.owner_pid=" + str(os.getpid()), "--memory", "256m"]
        for k in names:
            cmd += ["-e", k]  # values travel in the environment, never argv
        cmd.append(self.image)
        proc = subprocess.run(cmd, capture_output=True, env=env, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(f"postgrest docker run failed: {proc.stderr.decode()}")
        deadline = time.monotonic() + ready_timeout_s
        import httpx

        while time.monotonic() < deadline:
            try:
                if httpx.get(self.url + "/", timeout=2).status_code < 500:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
        logs = subprocess.run(["docker", "logs", self.name], capture_output=True).stderr
        self.stop()
        raise RuntimeError(f"postgrest not ready: {logs.decode()[-2000:]}")

    def reload_schema(self) -> None:
        subprocess.run(["docker", "kill", "-s", "SIGUSR1", self.name], capture_output=True)
        time.sleep(2)

    def stop(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=60)

    def client(self, role: Optional[str] = "service_role", *, is_async: bool = False) -> Any:
        """A postgrest-py client authenticated as ``role`` (``None`` = anonymous)."""
        from postgrest import AsyncPostgrestClient, SyncPostgrestClient

        headers = {}
        if role is not None:
            headers["Authorization"] = f"Bearer {sign_jwt(self.jwt_secret, role)}"
        cls = AsyncPostgrestClient if is_async else SyncPostgrestClient
        return cls(self.url, headers=headers)


class SupabaseShim:
    """What MLDataLoader / FeatureBuilder call on a supabase client: ``.table(name)``."""

    def __init__(self, postgrest_client: Any):
        self._pg = postgrest_client

    def table(self, name: str) -> Any:
        return self._pg.from_(name)


def safe_db_name(prefix: str, node_name: str) -> str:
    return prefix + re.sub(r"[^a-z0-9]", "_", node_name.lower())[:40]


def rows_of(conn: Any, sql: str) -> List[str]:
    return conn.rows(sql)
