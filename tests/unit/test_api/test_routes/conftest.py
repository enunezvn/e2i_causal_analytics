"""Route-test fixtures (M2).

The gaps + feedback route handlers now persist to Supabase by default and only
fall back to their process-local dicts when Supabase is unconfigured OR the dev
flag ``E2I_GAPS_FEEDBACK_INMEMORY=1`` is set. On this box, pytest autoloads
``.env`` (``tests/conftest.py`` calls ``load_dotenv(override=True)`` and
``pyproject.toml`` sets ``env_files = [".env"]``), and ``.env`` provides a real
``SUPABASE_URL`` + service key — so without this fixture the legacy dict-based
route tests (``test_gaps.py`` / ``test_feedback*.py``) would silently hit real
Supabase and fail.

This autouse fixture pins those tests to the deterministic in-memory path. The
dedicated persistence tests (``test_gaps_persistence.py`` /
``test_feedback_persistence.py``) monkeypatch ``_use_inmemory_fallback``
directly, so they override this env flag and exercise the repo path regardless.
"""

import pytest


@pytest.fixture(autouse=True)
def _force_inmemory_gaps_feedback(monkeypatch):
    """Force the gaps/feedback route stores onto the in-memory dev path."""
    monkeypatch.setenv("E2I_GAPS_FEEDBACK_INMEMORY", "1")
