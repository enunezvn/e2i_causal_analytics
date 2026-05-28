"""Phase A wiring guard: the Optum operator runner must thread
``feature_manifest_source`` into ``tier0.run_pipeline`` so Layer-5's
manifest-driven Layer-1 verdicts (and PR #544's declared-safe FDR honor)
actually engage for Optum cohorts.

Pre-fix bug: ``scripts/run_optum_tier0_test.py`` called
``tier0.run_pipeline(data_dir=...)`` but omitted ``feature_manifest_source``
entirely, so every Optum run silently no-op'd the manifest pass — the post-index
leak catch and the declared-safe σ-inflation never fired. The CSU runner already
resolved it via ``_resolve_feature_manifest_source``; the Optum runner did not.

These tests pin that:
  - each Optum cohort dir (``data/rwd/optum/<cohort>``) auto-resolves to ``"optum"``;
  - the resolved value reaches ``run_pipeline`` as the keyword argument;
  - an explicit ``--feature-manifest-source`` override is honored;
  - a conflicting override (e.g. ``csu`` against Optum data) fails fast (M2).
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest


def _load_optum_runner() -> Any:
    return importlib.import_module("scripts.run_optum_tier0_test")


def _patch_capturing_run_pipeline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace ``tier0.run_pipeline`` with an async stub that records its
    keyword arguments and returns an empty state dict. Returns the capture box.
    """
    runner = _load_optum_runner()
    captured: dict[str, Any] = {}

    async def _stub(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {}

    monkeypatch.setattr(runner.tier0, "run_pipeline", _stub)
    return captured


@pytest.mark.parametrize("cohort", ["initiation", "discontinuation", "persistence"])
def test_optum_cohort_autoresolves_to_optum_manifest(
    cohort: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each Optum cohort dir auto-resolves to the ``optum`` manifest source and
    threads it into ``run_pipeline`` (dry-run skips the carve-out + dir check)."""
    runner = _load_optum_runner()
    captured = _patch_capturing_run_pipeline(monkeypatch)
    monkeypatch.setattr(
        runner.sys, "argv", ["run_optum_tier0_test.py", "--cohort", cohort, "--dry-run"]
    )

    rc = runner.main()

    assert rc == 0
    assert captured["kwargs"].get("feature_manifest_source") == "optum", (
        f"Optum {cohort} run did not thread feature_manifest_source='optum' "
        f"into run_pipeline; got {captured['kwargs'].get('feature_manifest_source')!r}"
    )


def test_explicit_override_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``--feature-manifest-source optum`` is passed through."""
    runner = _load_optum_runner()
    captured = _patch_capturing_run_pipeline(monkeypatch)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "run_optum_tier0_test.py",
            "--cohort",
            "initiation",
            "--dry-run",
            "--feature-manifest-source",
            "optum",
        ],
    )

    rc = runner.main()

    assert rc == 0
    assert captured["kwargs"].get("feature_manifest_source") == "optum"


def test_conflicting_override_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``--feature-manifest-source csu`` override against an Optum data_dir must
    raise (M2 conflict contract) rather than silently apply the wrong manifest."""
    runner = _load_optum_runner()
    _patch_capturing_run_pipeline(monkeypatch)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "run_optum_tier0_test.py",
            "--cohort",
            "initiation",
            "--dry-run",
            "--feature-manifest-source",
            "csu",
        ],
    )

    with pytest.raises(ValueError, match="conflicts with"):
        runner.main()
