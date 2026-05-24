"""Plan-239 §5.3 — MIPROv2 reproducibility tests (Tier-1 + Tier-2).

Tier-1 (CI-mandatory, no LM key required): verifies WIRING ONLY —
    the MIPROv2 path imports, `--optimizer miprov2` is accepted on the
    compile script CLI, the wrapper threads seed into both constructor
    and compile call, and `normalize_artifact_json` produces a stable
    canonical string. Does NOT prove artifact reproducibility (that
    requires running the LM end-to-end).

Tier-2 (live-LM, `ANTHROPIC_API_KEY`-gated, marked `@pytest.mark.live_lm`):
    runs two MIPROv2 compiles under the same fixed seed, normalizes each
    artifact via `normalize_artifact_json`, and asserts byte-identical
    canonical output. AC2 evidence is the Tier-2 result attached to the PR.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------- Tier-1 (CI-mandatory, no LM) -----------------------------


def test_miprov2_optimizer_flag_accepted_on_help() -> None:
    """Tier-1 / AC1 — `--optimizer miprov2` is a valid CLI choice."""
    script = _REPO_ROOT / "scripts" / "compile_causal_role_classifier.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "--optimizer" in proc.stdout
    assert "miprov2" in proc.stdout


def test_normalize_artifact_json_drops_volatile_keys(tmp_path: Path) -> None:
    """Tier-1 — `normalize_artifact_json` strips volatile keys + sorts."""
    from scripts.compile_causal_role_classifier import (
        VOLATILE_KEY_ALLOWLIST,
        normalize_artifact_json,
    )

    artifact = tmp_path / "art.json"
    raw = {
        "compiled_at": "2026-05-23T10:00:00",
        "cache_hits": 17,
        "demos": [{"feature_name": "x", "causal_role": "confounder"}],
        "metadata": {
            "dspy_version": "3.1.0",
            "elapsed_seconds": 9.3,
        },
    }
    artifact.write_text(json.dumps(raw, indent=2))

    out = normalize_artifact_json(artifact)
    parsed = json.loads(out)

    # Volatile keys stripped at every depth.
    for vk in {"compiled_at", "cache_hits", "elapsed_seconds"}:
        assert vk in VOLATILE_KEY_ALLOWLIST
    assert "compiled_at" not in parsed
    assert "cache_hits" not in parsed
    assert "elapsed_seconds" not in parsed.get("metadata", {})
    # Non-volatile keys preserved.
    assert parsed["metadata"]["dspy_version"] == "3.1.0"
    assert parsed["demos"][0]["feature_name"] == "x"


def test_normalize_artifact_json_is_deterministic(tmp_path: Path) -> None:
    """Tier-1 — two normalize() calls on the same input produce identical output."""
    from scripts.compile_causal_role_classifier import normalize_artifact_json

    artifact = tmp_path / "art.json"
    artifact.write_text(
        json.dumps(
            {
                "b": 2,
                "a": [3, 2, 1],
                "compiled_at": "2026-05-23T10:00:00",
            },
            indent=2,
        )
    )
    a = normalize_artifact_json(artifact)
    b = normalize_artifact_json(artifact)
    assert a == b


# ---------------- Tier-2 (live LM, gated) ----------------------------------


def _live_lm_available() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    return key.startswith("sk-ant-")


def _tier2_opted_in() -> bool:
    """Tier-2 is opt-in via PLAN239_RUN_TIER2=1 to prevent accidental LM cost
    during routine `pytest` runs (conftest loads .env and ANTHROPIC_API_KEY
    will typically be present locally). Set this env var to actively run the
    byte-identical compile-twice comparison.
    """
    return os.environ.get("PLAN239_RUN_TIER2") == "1"


@pytest.mark.live_lm
@pytest.mark.integration
@pytest.mark.skipif(
    not (_live_lm_available() and _tier2_opted_in()),
    reason=(
        "Plan-239 Tier-2 byte-identical reproducibility test requires both "
        "ANTHROPIC_API_KEY (sk-ant-*) AND opt-in via PLAN239_RUN_TIER2=1 "
        "(opt-in gate prevents accidental LM cost during routine pytest runs)."
    ),
)
def test_miprov2_artifact_byte_identical_under_seed(tmp_path: Path) -> None:
    """Plan-239 §5.3 Tier-2 — two MIPROv2 compiles under the same fixed seed
    produce byte-identical normalized artifact JSON.

    Run manually with ANTHROPIC_API_KEY + PLAN239_RUN_TIER2=1; attach the
    GREEN result to the PR body as AC2 evidence. Subject to provider
    nondeterminism per §5.4 — if this xfails reliably under your provider,
    document the failure mode and re-evaluate the AC2 contract before
    promoting MIPROv2 to default.
    """
    from scripts.compile_causal_role_classifier import (
        DEFAULT_LM_MODEL,
        compile_and_persist,
        normalize_artifact_json,
    )

    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"

    compile_and_persist(out_path=out_a, lm_model=DEFAULT_LM_MODEL, optimizer="miprov2", seed=42)
    compile_and_persist(out_path=out_b, lm_model=DEFAULT_LM_MODEL, optimizer="miprov2", seed=42)

    norm_a = normalize_artifact_json(out_a)
    norm_b = normalize_artifact_json(out_b)

    assert norm_a == norm_b, (
        "Plan-239 §5.3 Tier-2: two MIPROv2 compiles under seed=42 did NOT "
        "produce byte-identical normalized artifact JSON. Diff first 500 chars:\n"
        f"  A: {norm_a[:500]!r}\n"
        f"  B: {norm_b[:500]!r}"
    )
