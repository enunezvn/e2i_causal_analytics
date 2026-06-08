"""M2 disproof lock: the API serves /api/gaps and /api/feedback from >1 worker
process, so the in-memory route stores cannot be read back reliably. This test
fails if someone reverts the compose file to a single worker (which would make
this whole shard OPTIONAL) — forcing a re-evaluation rather than silent rot.
"""

from pathlib import Path


def test_api_runs_multiple_gunicorn_workers():
    compose = Path("docker/docker-compose.yml").read_text()
    # The api service command pins gunicorn with --workers 2 (env WORKERS: 2).
    assert "--workers 2" in compose, "API no longer multi-worker; re-evaluate M2 (may be OPTIONAL)"
    assert "uvicorn.workers.UvicornWorker" in compose
    # --max-requests recycling also wipes per-worker dicts mid-life.
    assert "--max-requests 1000" in compose
