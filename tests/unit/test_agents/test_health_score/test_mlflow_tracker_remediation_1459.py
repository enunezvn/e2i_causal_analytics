"""#1459 — the artifact-write-blocked remediation must never advise mounting at '/'.

Red-first regression tests for the operator-facing message defect found by a
post-merge review of #1452:

    ... or mount a writable volume at '/'.

For the REAL production artifact URI
(``/mlflow/artifacts/9/ebf7988490e845f09d351785eac6450a/artifacts``) the
``blocked_root`` computed by ``classify_artifact_destination`` is ``/`` — the
nearest EXISTING unwritable ancestor — so the #1452 message told operators to
mount a writable volume at the container ROOT, defeating the ``read_only: true``
hardening (docker/docker-compose.yml, ``e2i_api``) the same message cites.

Measured ground truth (issue #1459, verified in the e2i_api container):

* ``/mlflow`` does not exist in the container — the MISSING path is ``/mlflow``,
  not ``/``. ``/`` merely happens to be the nearest existing ancestor.
* The recreation remediation (rename/archive the pre-b0a30f11 experiment so it
  is recreated with ``artifact_location='mlflow-artifacts:/'``) is the correct
  primary fix and must survive.

Required behavior pinned here:

1. ``classify_artifact_destination`` reports the missing artifact root (the
   first nonexistent path component, e.g. ``/mlflow``) SEPARATELY from the
   unwritable ancestor (``blocked_root``).
2. The rendered warning names the real missing root and NEVER advises mounting
   a writable volume — not at ``/``, not anywhere (the mount clause is dropped
   unconditionally; experiment recreation is the single recommended path).
3. Proxied URIs (``mlflow-artifacts:/...``) stay non-blocked.

Nothing here mocks the units under test: every assertion runs the real
classification / reporting logic against real URIs and real filesystem state.
"""

from __future__ import annotations

import logging
import os

import pytest

from src.agents.health_score.mlflow_tracker import (
    classify_artifact_destination,
    report_artifact_write_blocked,
    reset_tracking_reports,
)

# The exact production strings from issue #1459.
PROD_LEGACY_ARTIFACT_URI = "/mlflow/artifacts/9/ebf7988490e845f09d351785eac6450a/artifacts"
APP_MLRUNS_ARTIFACT_URI = "/app/mlruns/1/abc/artifacts"
PROD_PROXIED_ARTIFACT_URI = "mlflow-artifacts:/9/abc/artifacts"

LOGGER = "src.agents.health_score.mlflow_tracker"

# Preconditions that make the literal production URIs classify on this box
# exactly as they do in the container: '/' unwritable, '/mlflow' and '/app'
# absent. Root would bypass permissions; a box that HAS these paths would
# change the classification, so skip honestly rather than fake the filesystem.
not_faithful_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="root bypasses filesystem permissions"
)


@pytest.fixture(autouse=True)
def _clean_report_state():
    """The once-per-process report ledger is module state; isolate each test."""
    reset_tracking_reports()
    yield
    reset_tracking_reports()


# =============================================================================
# CLASSIFICATION: missing root reported separately from the unwritable ancestor
# =============================================================================


class TestMissingRootClassification:
    @not_faithful_root
    @pytest.mark.skipif(os.path.exists("/mlflow"), reason="/mlflow exists on this box")
    def test_production_uri_names_slash_mlflow_as_the_missing_root(self):
        """The real gap is '/mlflow' (absent), not '/' (merely the unwritable ancestor)."""
        dest = classify_artifact_destination(PROD_LEGACY_ARTIFACT_URI)
        assert dest.is_blocked is True
        assert dest.blocked_root == "/"
        assert dest.missing_root == "/mlflow"

    @not_faithful_root
    @pytest.mark.skipif(os.path.exists("/app"), reason="/app exists on this box")
    def test_app_mlruns_uri_names_slash_app_as_the_missing_root(self):
        dest = classify_artifact_destination(APP_MLRUNS_ARTIFACT_URI)
        assert dest.is_blocked is True
        assert dest.blocked_root == "/"
        assert dest.missing_root == "/app"

    def test_proxied_uri_stays_non_blocked(self):
        dest = classify_artifact_destination(PROD_PROXIED_ARTIFACT_URI)
        assert dest.is_blocked is False
        assert dest.local_path is None
        assert dest.blocked_root is None
        assert dest.missing_root is None

    @not_faithful_root
    def test_readonly_ancestor_with_missing_children(self, tmp_path):
        """Deterministic shape of the container failure: existing read-only dir,
        target path entirely absent below it."""
        jail = tmp_path / "jail"
        jail.mkdir()
        os.chmod(jail, 0o555)
        try:
            dest = classify_artifact_destination(str(jail / "mlruns" / "1" / "abc" / "artifacts"))
            assert dest.is_blocked is True
            assert dest.blocked_root == str(jail)
            assert dest.missing_root == str(jail / "mlruns")
        finally:
            os.chmod(jail, 0o755)

    @not_faithful_root
    def test_fully_existing_unwritable_path_has_no_missing_root(self, tmp_path):
        """When the destination itself exists (just unwritable) nothing is missing."""
        jail = tmp_path / "jail"
        jail.mkdir()
        os.chmod(jail, 0o555)
        try:
            dest = classify_artifact_destination(str(jail))
            assert dest.is_blocked is True
            assert dest.blocked_root == str(jail)
            assert dest.missing_root is None
        finally:
            os.chmod(jail, 0o755)

    def test_writable_destination_is_not_blocked(self, tmp_path):
        dest = classify_artifact_destination(str(tmp_path / "exp" / "run" / "artifacts"))
        assert dest.is_blocked is False
        assert dest.blocked_root is None


# =============================================================================
# RENDERED MESSAGE PIN: never advise a mount; name the real missing root
# =============================================================================


def _sole_warning(caplog) -> str:
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one WARNING, got {len(warnings)}"
    return warnings[0].getMessage()


class TestRemediationMessage:
    @not_faithful_root
    @pytest.mark.skipif(os.path.exists("/mlflow"), reason="/mlflow exists on this box")
    def test_production_message_never_advises_mounting_at_root(self, caplog):
        """The #1459 pin, on the exact production URI.

        Pre-fix the message ends "...or mount a writable volume at '/'." —
        instructing an operator to defeat the e2i_api read_only hardening.
        """
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            assert (
                report_artifact_write_blocked(
                    PROD_LEGACY_ARTIFACT_URI, run_id="ebf7988490e845f09d351785eac6450a"
                )
                is True
            )
        message = _sole_warning(caplog)

        # Never advise mounting at '/' — nor anywhere else (clause dropped
        # unconditionally; recreation is the single recommended remediation).
        assert "mount a writable volume at '/'" not in message
        assert "mount" not in message.lower()

        # The message must name the REAL gap: '/mlflow' as its own token, not
        # merely as a prefix of the full local path.
        assert "'/mlflow'" in message

        # The correct primary remediation survives.
        assert "mlflow-artifacts:" in message
        assert "artifact_location" in message
        # And the run id is still traceable.
        assert "ebf7988490e845f09d351785eac6450a" in message

    @not_faithful_root
    def test_message_names_missing_root_deterministic_jail(self, tmp_path, caplog):
        """Same pin against a real read-only dir — no dependence on / or /mlflow."""
        jail = tmp_path / "jail"
        jail.mkdir()
        os.chmod(jail, 0o555)
        try:
            uri = str(jail / "artifacts" / "9" / "run" / "artifacts")
            with caplog.at_level(logging.WARNING, logger=LOGGER):
                assert report_artifact_write_blocked(uri, run_id="r-jail") is True
        finally:
            os.chmod(jail, 0o755)
        message = _sole_warning(caplog)

        assert "mount" not in message.lower()
        # The first nonexistent component is the named gap...
        assert repr(str(jail / "artifacts")) in message
        # ...and the unwritable ancestor is still identified for diagnosis.
        assert repr(str(jail)) in message
        assert "mlflow-artifacts:" in message

    @not_faithful_root
    def test_fully_existing_unwritable_destination_message_still_renders(self, tmp_path, caplog):
        """No missing component: the message must degrade gracefully (no 'None')."""
        jail = tmp_path / "jail"
        jail.mkdir()
        os.chmod(jail, 0o555)
        try:
            with caplog.at_level(logging.WARNING, logger=LOGGER):
                assert report_artifact_write_blocked(str(jail), run_id="r-exists") is True
        finally:
            os.chmod(jail, 0o755)
        message = _sole_warning(caplog)

        assert "mount" not in message.lower()
        assert "None" not in message
        assert repr(str(jail)) in message
        assert "mlflow-artifacts:" in message
