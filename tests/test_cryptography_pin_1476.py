"""Forcing-function tests for #1476: cryptography 49.0.0 + allowlist cleanup.

#1456 held four cryptography advisories because mlflow's ``cryptography<47``
cap (3.11.x line) made any 47+ pin unsatisfiable. #1476 bumps mlflow to the
3.15.x line (cap ``<50``), unlocking ``cryptography==49.0.0`` which clears
three of the four:

- GHSA-537c-gmf6-5ccf (HIGH, vulnerable bundled OpenSSL — fix 48.0.1)
- CVE-2026-69248 (MEDIUM, permittedSubtrees wildcard escape — fix 49.0.0)
- CVE-2026-69249 (HIGH, exponential path-building DoS — fix 49.0.0)

The fourth, CVE-2026-69247 (PKCS#7 ``EnvelopedData`` oracle — fix 50.0.0),
REMAINS held: no mlflow release admits ``cryptography>=50`` yet, and the repo
has zero pkcs7/EnvelopedData usage (never imports ``cryptography`` at all).

These tests are RED on the pre-#1476 tree (cryptography==46.0.7 + 4 held
entries) and GREEN once the bump + allowlist cleanup land together. Style
matches ``test_mlflow_upgrade_pin.py`` (repo-top-level pin tests).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_REQS = REPO_ROOT / "requirements.txt"
REQS_LOCK = REPO_ROOT / "requirements.lock"
SECURITY_YML = REPO_ROOT / ".github" / "workflows" / "security.yml"

# The exact cryptography pin required post-#1476. 49.0.0 is the only 49.x
# release (verified on PyPI 2026-08-04) and the highest version admitted by
# mlflow 3.15.1's ``cryptography<50,>=43.0.0`` constraint.
CRYPTOGRAPHY_REQUIRED_PIN = "49.0.0"

# Advisories CLEARED by 49.0.0 — their --ignore-vuln entries must be GONE
# from security.yml (every occurrence; the file carries two dated blocks).
CLEARED_CRYPTO_IDS = (
    "GHSA-537c-gmf6-5ccf",
    "CVE-2026-69248",
    "CVE-2026-69249",
)

# The single residual cryptography advisory that must REMAIN ignored: fix is
# 50.0.0, which no mlflow release admits (3.15.1 caps cryptography<50).
RESIDUAL_CRYPTO_ID = "CVE-2026-69247"

_CRYPTO_REQ_RE = re.compile(r"^cryptography==([^\s\\#]+)", re.MULTILINE)


def test_requirements_txt_pins_cryptography_49() -> None:
    """requirements.txt must pin cryptography==49.0.0 exactly."""
    matches = _CRYPTO_REQ_RE.findall(ROOT_REQS.read_text())
    assert matches, "no cryptography== pin found in requirements.txt"
    assert matches == [CRYPTOGRAPHY_REQUIRED_PIN], (
        f"requirements.txt pins cryptography=={matches!r}; expected exactly "
        f"[{CRYPTOGRAPHY_REQUIRED_PIN!r}] (#1476 — 49.0.0 clears "
        f"GHSA-537c-gmf6-5ccf + CVE-2026-69248 + CVE-2026-69249)."
    )


def test_requirements_lock_pins_cryptography_49() -> None:
    """requirements.lock must carry the same cryptography version (the lock is
    what docker/Dockerfile actually installs — a stale lock silently ships the
    vulnerable 46.0.7 wheels regardless of requirements.txt)."""
    matches = _CRYPTO_REQ_RE.findall(REQS_LOCK.read_text())
    assert matches == [CRYPTOGRAPHY_REQUIRED_PIN], (
        f"requirements.lock pins cryptography=={matches!r}; expected exactly "
        f"[{CRYPTOGRAPHY_REQUIRED_PIN!r}] — regenerate the lock (command in "
        f"its header) after bumping requirements.txt."
    )


def test_cleared_crypto_advisories_removed_from_security_yml() -> None:
    """The three advisories fixed at/below 49.0.0 must have NO --ignore-vuln
    entry left anywhere in security.yml (both dated allowlist blocks)."""
    text = SECURITY_YML.read_text()
    stale = [
        vuln_id
        for vuln_id in CLEARED_CRYPTO_IDS
        if re.search(rf"--ignore-vuln\s+{re.escape(vuln_id)}\b", text)
    ]
    assert not stale, (
        f"security.yml still ignores cryptography advisories cleared by the "
        f"49.0.0 bump: {stale}. Remove every occurrence (the file has two "
        f"dated blocks — check both)."
    )


def test_residual_crypto_advisory_still_held_with_current_rationale() -> None:
    """CVE-2026-69247 must remain ignored (fix 50.0.0 > mlflow 3.15.1's <50
    cap) and its rationale must be the REWRITTEN one — the pre-#1476 text
    blamed the mlflow<47 constraint, which no longer exists."""
    text = SECURITY_YML.read_text()
    entries = re.findall(rf"--ignore-vuln\s+{re.escape(RESIDUAL_CRYPTO_ID)}\b[^`]*`([^`]*)`", text)
    assert entries, (
        f"security.yml no longer ignores {RESIDUAL_CRYPTO_ID}; it is still "
        f"unfixable (fix 50.0.0; mlflow 3.15.1 caps cryptography<50) and "
        f"removing it turns the pip-audit gate red."
    )
    for rationale in entries:
        assert "50" in rationale and "49.0.0" in rationale, (
            f"{RESIDUAL_CRYPTO_ID} rationale looks stale (must state the "
            f"installed 49.0.0 and the 50.0.0 fix ceiling): {rationale!r}"
        )
        assert "46.0.7" not in rationale, (
            f"{RESIDUAL_CRYPTO_ID} rationale still references the pre-#1476 "
            f"cryptography 46.0.7: {rationale!r}"
        )
