"""Nightly mirror of adaptive-validity sidecars into ``adaptive_validity_verdicts`` (#238, #2273).

#238 made the sidecars on the ``audit_artifacts`` volume the canonical audit record
and specified "a nightly batch job that mirrors the sidecar payload" into Postgres
for cross-experiment queries. ``scripts/mirror_audit_sidecar_to_supabase.py`` is that
batch (idempotent upsert, ``max(imported_at) - 1h`` cursor), but it was never
scheduled; this task is the schedule. It runs on the ``analytics`` queue, whose
consumer (worker_medium) mounts the volume.

DARK by default
---------------
The task does nothing unless ``AUDIT_SIDECAR_MIRROR_ENABLED`` is truthy. #2260 changes
the mirror's ON CONFLICT key together with migration 157, and the mirror must not
write prod before 157 is applied. Arming it: set ``AUDIT_SIDECAR_MIRROR_ENABLED=true``
in the host ``.env`` (forwarded by x-common-env) and recreate worker_medium.

Why a subprocess
----------------
Same reasons as ``graph_reseed_tasks``: a top-level ``import scripts.*`` in a task
module crash-looped every worker on 2026-05-26, and a subprocess's RSS is returned
the moment it exits. The script reads ``DATABASE_URL``; the workers get
``SUPABASE_DB_URL`` from x-common-env (the ETL convention), so it is passed through
the child's environment, never argv, where ``ps`` would show the password.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Final

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

ENABLE_FLAG: Final = "AUDIT_SIDECAR_MIRROR_ENABLED"
MIRROR_TIMEOUT_SECONDS: Final = 900

# The src/data/audit_sidecar_reader.py warnings that mean a sidecar's verdicts did
# NOT reach the table: an unreadable file, an unparseable written_at (both skipped),
# and a non-list adaptive_verdicts (read as empty). Its other warnings (schema
# drift, unknown keys) are about rows that still land.
_DROPPED_SIDECAR_MARKERS: Final = (
    "SidecarReader: skipping malformed sidecar",
    "has unparseable written_at=",
    "has non-list adaptive_verdicts=",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MIRROR_SCRIPT = _REPO_ROOT / "scripts" / "mirror_audit_sidecar_to_supabase.py"


class AuditSidecarMirrorError(RuntimeError):
    """The armed mirror ran and failed: the DB mirror is behind the canonical sidecars."""


def _enabled() -> bool:
    return os.environ.get(ENABLE_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


@celery_app.task(
    name="src.tasks.mirror_audit_sidecars",
    # No retries: a FAILED run rolls its batch back and leaves max(imported_at) where
    # it was, so the next nightly tick re-reads the same sidecars (the upsert is
    # idempotent). A sidecar the reader DROPS is different: once later imports move
    # the cursor past its written_at, no nightly run reads it again, which is why a
    # drop is reported "degraded" at ERROR with the file named.
    max_retries=0,
)
def mirror_audit_sidecars() -> Dict[str, Any]:
    """Run the #238 mirror once. Returns a status dict; raises when an armed run fails."""
    if not _enabled():
        logger.info("audit-sidecar mirror is disabled (%s unset/false); skipping", ENABLE_FLAG)
        return {"status": "disabled"}

    artifacts_dir = os.environ.get("ADAPTIVE_VALIDITY_ARTIFACTS_DIR")
    db_url = os.environ.get("SUPABASE_DB_URL") or os.environ.get("DATABASE_URL")
    if not artifacts_dir or not db_url:
        raise AuditSidecarMirrorError(
            "audit-sidecar mirror is armed but ADAPTIVE_VALIDITY_ARTIFACTS_DIR or "
            "SUPABASE_DB_URL/DATABASE_URL is unset"
        )

    cmd = [sys.executable, str(_MIRROR_SCRIPT), "--artifacts-dir", artifacts_dir]
    env = {**os.environ, "DATABASE_URL": db_url}
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=MIRROR_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AuditSidecarMirrorError(
            f"audit-sidecar mirror timed out after {MIRROR_TIMEOUT_SECONDS}s"
        ) from exc

    # The script logs through logging's default stream (stderr) as
    # "<asctime> <LEVEL> <logger> <message>"; its last "done: read=…" line is the
    # run summary.
    stderr = proc.stderr or ""
    lines = stderr.splitlines()
    if proc.returncode != 0:
        logger.error(
            "audit-sidecar mirror FAILED rc=%s — adaptive_validity_verdicts is behind "
            "the canonical sidecars\nstderr tail:\n%s",
            proc.returncode,
            stderr[-2000:],
        )
        raise AuditSidecarMirrorError(
            f"audit-sidecar mirror exited {proc.returncode}: {stderr[-500:]}"
        )

    summary = next((line for line in reversed(lines) if " done: " in line or "nothing" in line), "")
    # The reader skips an unreadable sidecar with a WARNING and the script still
    # exits 0, so that canonical record never reaches the table. Say so at ERROR and
    # report the run degraded; other reader warnings (schema drift on a row that did
    # land) are relayed at WARNING.
    skipped = [line for line in lines if any(m in line for m in _DROPPED_SIDECAR_MARKERS)]
    warnings = [line for line in lines if " WARNING " in line and line not in skipped]
    for line in warnings:
        logger.warning("audit-sidecar mirror: %s", line)
    if skipped:
        logger.error(
            "audit-sidecar mirror SKIPPED %d sidecar(s); those canonical records are "
            "not in adaptive_validity_verdicts:\n%s",
            len(skipped),
            "\n".join(skipped),
        )
        return {"status": "degraded", "summary": summary, "skipped_sidecars": skipped}
    logger.info("audit-sidecar mirror OK: %s", summary)
    return {"status": "ok", "summary": summary, "warnings": len(warnings)}
