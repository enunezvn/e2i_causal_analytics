"""Lane E item 4 — ``scripts/measure_feature_role_panel.py``.

Runs the panel over a parquet frame and writes ``panel.json`` + ``summary.md``.
Layer 4 defaults to a fake LM; ``--layer4 real`` is the paid run and is an owner
decision (the script prints the cost estimate and refuses without ``--i-accept-cost``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.test_causal_engine.test_feature_role_panel.conftest import (
    OUTCOME,
    TREATMENT,
    make_panel_frame,
)

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "measure_feature_role_panel.py"


def test_cli_help_runs() -> None:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert "--layer4" in proc.stdout


def test_run_on_a_parquet_frame_with_a_fake_layer_4(tmp_path: Path, monkeypatch) -> None:
    from scripts.measure_feature_role_panel import main

    frame = make_panel_frame()
    parquet = tmp_path / "frame.parquet"
    frame.to_parquet(parquet)
    out = tmp_path / "out"
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    rc = main(
        [
            "--parquet",
            str(parquet),
            "--manifest-source",
            "optum",
            "--treatment",
            TREATMENT,
            "--outcome",
            OUTCOME,
            "--out",
            str(out),
            "--layer4",
            "fake",
            "--seed",
            "7",
        ]
    )
    assert rc == 0
    payload = json.loads((out / "panel.json").read_text())
    assert payload["treatment"] == TREATMENT and payload["outcome"] == OUTCOME
    assert set(payload["records"]) == {c for c in frame.columns if c not in (TREATMENT, OUTCOME)}
    assert payload["layer_activity"]["layer_4"]["lm"] == "fake"
    summary = (out / "summary.md").read_text()
    assert "Layer 4" in summary and "abstain" in summary.lower()
    assert "treatment_initiated" in summary


def test_real_layer_4_refuses_without_cost_acceptance(tmp_path: Path) -> None:
    from scripts.measure_feature_role_panel import main

    frame = make_panel_frame()
    parquet = tmp_path / "frame.parquet"
    frame.to_parquet(parquet)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--parquet",
                str(parquet),
                "--manifest-source",
                "optum",
                "--treatment",
                TREATMENT,
                "--outcome",
                OUTCOME,
                "--out",
                str(tmp_path / "out"),
                "--layer4",
                "real",
            ]
        )
    assert exc.value.code != 0


def test_covariates_file_restricts_the_panel(tmp_path: Path, monkeypatch) -> None:
    from scripts.measure_feature_role_panel import main

    frame = make_panel_frame()
    parquet = tmp_path / "frame.parquet"
    frame.to_parquet(parquet)
    covs = tmp_path / "covs.txt"
    covs.write_text("age_at_index\nnoise_feature\n")
    out = tmp_path / "out"
    rc = main(
        [
            "--parquet",
            str(parquet),
            "--manifest-source",
            "optum",
            "--treatment",
            TREATMENT,
            "--outcome",
            OUTCOME,
            "--out",
            str(out),
            "--layer4",
            "off",
            "--covariates-file",
            str(covs),
        ]
    )
    assert rc == 0
    payload = json.loads((out / "panel.json").read_text())
    assert payload["features"] == ["age_at_index", "noise_feature"]
    assert payload["layer_activity"]["layer_4"]["fired"] == 0
