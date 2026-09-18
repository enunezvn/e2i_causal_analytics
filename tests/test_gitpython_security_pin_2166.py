"""Regression guards for the GitPython 3.1.60 security bump (#2166)."""

from __future__ import annotations

import re
from pathlib import Path

from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_VERSION = "3.1.60"


def _requirements_pin(path: Path) -> list[str]:
    return re.findall(
        r"^gitpython==([^\s\\#]+)",
        path.read_text(),
        re.IGNORECASE | re.MULTILINE,
    )


def test_gitpython_security_pin_is_consistent_across_dependency_locks() -> None:
    """Every supported install path must resolve the patched GitPython release."""
    for filename in ("requirements.txt", "requirements-dev.txt", "requirements.lock"):
        pins = _requirements_pin(REPO_ROOT / filename)
        assert pins == [REQUIRED_VERSION], (
            f"{filename} pins GitPython at {pins!r}; expected [{REQUIRED_VERSION!r}] "
            "to clear PYSEC-2026-3982/3983/3984"
        )

    uv_versions = re.findall(
        r'name = "gitpython"\nversion = "([^"]+)"',
        (REPO_ROOT / "uv.lock").read_text(),
    )
    assert len(uv_versions) == 1 and Version(uv_versions[0]) >= Version(REQUIRED_VERSION), (
        f"uv.lock locks GitPython at {uv_versions!r}; expected one version at or above "
        f"{REQUIRED_VERSION!r}"
    )
