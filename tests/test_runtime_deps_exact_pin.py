"""Forcing-function test: runtime deps that previously carried range/unbounded
pins must be pinned exactly in requirements.txt.

FU3-C / #528 (item C): ``imbalanced-learn`` (unbounded ``>=0.12.0``), ``ngboost``
(``>=0.5.1,<0.6``), and ``mapie`` (``>=0.8.0,<0.9``) carried range pins, so a
Docker rebuild (``pip install -r requirements.txt``, no lock file) could resolve
a silently-different version than production was tested against — the same drift
class as the #491 ragas/langchain break. The exact versions are sourced from a
resolved, import-tested environment (the live api image ``pip freeze``), NOT range
maxima.

``mlflow`` / ``mlflow-skinny`` / ``mlflow-tracing`` are deliberately EXCLUDED from
this guard: their bounded ``>=3.11.0,<3.12.0`` range is CVE-anchored and lockstep-
managed across 4 files (#442), enforced by ``tests/test_mlflow_upgrade_pin.py``.
This test asserts the ``==`` *form* (not a hardcoded version), so a future, tested
version bump only edits requirements.txt — it does not need to touch this test.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_REQS = REPO_ROOT / "requirements.txt"

# Deps that must use an exact ``==`` pin (mlflow* intentionally excluded — see
# module docstring + test_mlflow_upgrade_pin.py).
EXACT_PIN_PACKAGES = ("imbalanced-learn", "ngboost", "mapie")


def _constraint_for(text: str, pkg: str) -> str:
    """Return the version specifier on ``pkg``'s requirement line (the line that
    STARTS with the package name, so comment lines mentioning it are ignored)."""
    pattern = re.compile(rf"^\s*{re.escape(pkg)}\s*([<>=!~][^#\s]*)", re.MULTILINE | re.IGNORECASE)
    match = pattern.search(text)
    assert match, f"no {pkg!r} requirement line found in {ROOT_REQS}"
    return match.group(1).strip()


def test_runtime_deps_use_exact_pins():
    """imbalanced-learn / ngboost / mapie must be pinned with ``==`` in requirements.txt."""
    text = ROOT_REQS.read_text()
    offenders = {
        pkg: c
        for pkg in EXACT_PIN_PACKAGES
        if not (c := _constraint_for(text, pkg)).startswith("==")
    }
    assert not offenders, (
        "these runtime deps must use exact == pins (range pins risk silent version "
        f"skew on a Docker rebuild with no lock file): {offenders}"
    )
