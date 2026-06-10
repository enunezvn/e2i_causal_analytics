"""#857: the deploy CLI must WRITE the manifest where the factory READS it, and
both must live under a writable on-box location.

In the prod api container ``/app/data`` is read-only; only named-volume subdirs
like ``data/ml_artifacts`` are writable. The old defaults (``data/...``) made the
bare runbook ``python -m src.mlops.prediction_synthesizer_deploy`` fail with
``OSError: Read-only file system`` and meant the live factory could never load a
manifest. These tests pin the two defaults together so they can't drift apart.
"""

from __future__ import annotations

from pathlib import Path

from src.agents.factory import DEFAULT_DEPLOYMENT_MANIFEST_PATH
from src.mlops.prediction_synthesizer_deploy import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MANIFEST_PATH,
)

_WRITABLE_VOLUME = Path("data/ml_artifacts")


def test_deploy_defaults_under_writable_volume():
    """Deploy CLI defaults must live under the writable ml_artifacts volume."""
    assert _WRITABLE_VOLUME in DEFAULT_MANIFEST_PATH.parents, DEFAULT_MANIFEST_PATH
    assert _WRITABLE_VOLUME in DEFAULT_ARTIFACT_DIR.parents, DEFAULT_ARTIFACT_DIR


def test_factory_reads_where_deploy_writes():
    """The factory's default manifest path must equal the deploy CLI's, or the
    live agent loads no ``model_clients`` even after a successful activation."""
    assert DEFAULT_DEPLOYMENT_MANIFEST_PATH == DEFAULT_MANIFEST_PATH
