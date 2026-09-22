"""Lane B (real-data causal estimation, 2026-09-22) — ``scripts/measure_structural_author.py``
with the fake LM (spec §6: "CLI with a fake LM"). The replay of the committed
CSU blind edges through the full author pipeline must reproduce the committed
validation record (28/31 exact, 0 missed leaks); the paid run must refuse
without ``--i-accept-cost`` after writing its cost estimate.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "scripts" / "measure_structural_author.py"


def _load():
    spec = importlib.util.spec_from_file_location("measure_structural_author", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.timeout(180)
def test_fake_replay_on_csu_reproduces_the_committed_record(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-be-blanked")
    mod = _load()
    out = tmp_path / "measure"
    rc = mod.main(
        [
            "--lm",
            "fake",
            "--fake-source",
            "replay",
            "--cohort",
            "CSU_remibrutinib",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    # Provider keys are blanked in-process on a fake run.
    assert "OPENAI_API_KEY" not in os.environ
    score = json.loads((out / "score.json").read_text())
    assert score["n"] == 31 and score["n_review"] == 0
    assert score["exact_role_agreement"] == 28
    assert score["missed_leaks"] == [] and score["gate_passed"] is True
    assert score["meta"]["lm"] == "fake" and score["meta"]["resolver"] == "offline"
    # codex r2 HIGH 4: every capture names the tree it ran on.
    assert len(score["meta"]["tree"]["commit"]) == 40
    assert isinstance(score["meta"]["tree"]["dirty_src_scripts_tests"], bool)
    authored = json.loads((out / "authored.json").read_text())
    assert len(authored["records"]) == 31
    rec = authored["records"][0]
    assert rec["model_id"] == "dummy" and rec["provenance"] == "machine"
    assert len(rec["guide_hash"]) == 64 and len(rec["prompt_hash"]) == 64
    # The offline resolver never verifies: every non-estimand edge is unsupported.
    grades = {p["grade"] for r in authored["records"] for p in r["edge_provenance"]}
    assert grades <= {"unsupported", "estimand"}
    summary = (out / "summary.md").read_text()
    assert "PASS: gate missed_leaks == 0" in summary
    cost = json.loads((out / "cost_estimate.json").read_text())
    assert cost["n_briefs"] == 31
    # The prompt carries the whole guide (~16.5k chars) plus the brief: measured, not guessed.
    assert cost["prompt_chars_per_brief_mean"] > 16_000
    assert cost["usd_estimate"] > 0


@pytest.mark.timeout(180)
def test_real_lm_refuses_without_cost_acceptance_but_writes_the_estimate(tmp_path, monkeypatch):
    mod = _load()
    out = tmp_path / "real"
    rc = mod.main(["--lm", "real", "--cohort", "CSU_remibrutinib", "--out", str(out)])
    assert rc == 3
    assert (out / "cost_estimate.json").exists()
    assert not (out / "authored.json").exists()
    assert not (out / "score.json").exists()


@pytest.mark.timeout(60)
def test_live_resolver_is_refused_on_a_fake_run(tmp_path):
    mod = _load()
    rc = mod.main(
        [
            "--lm",
            "fake",
            "--resolver",
            "live",
            "--cohort",
            "CSU_remibrutinib",
            "--out",
            str(tmp_path / "x"),
        ]
    )
    assert rc == 4
