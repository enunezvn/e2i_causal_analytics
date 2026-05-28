"""measure_layer4_precision.py --decider structural (Plan v4 Layer B / Phase 2, Task 5).

The eval harness can score the deterministic ``extract_role`` over authored DAG
edges (``--decider structural``) instead of the LLM classifier — the path the
Track-2B non-circular precision test (Task 8) will use once the literature
golden set is edge-augmented. Runs LLM-free (no ANTHROPIC_API_KEY needed).
"""

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def test_decider_structural_scores_via_extract_role(tmp_path):
    out = tmp_path / "report.json"
    rc = subprocess.run(
        [
            sys.executable,
            "scripts/measure_layer4_precision.py",
            "--decider",
            "structural",
            "--golden-set",
            "tests/fixtures/causal_role_attestation_edges_sample.json",
            "--report-path",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
    )
    assert rc.returncode == 0, rc.stderr
    report = json.loads(out.read_text())
    # Both fixture entries are classified correctly by extract_role
    # (age→confounder, post_event→collider) → perfect macro precision.
    assert report["macro_precision"] == 1.0
    assert report.get("decider") == "structural"


def test_decider_structural_requires_edges(tmp_path):
    # An edge-less golden set (the current 91-entry literature fixture shape)
    # must fail loudly under --decider structural, NOT silently score nothing.
    edgeless = tmp_path / "edgeless.json"
    edgeless.write_text(
        json.dumps({"entries": [{"feature_name": "x", "ground_truth_role": "confounder"}]})
    )
    rc = subprocess.run(
        [
            sys.executable,
            "scripts/measure_layer4_precision.py",
            "--decider",
            "structural",
            "--golden-set",
            str(edgeless),
            "--report-path",
            str(tmp_path / "r.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
    )
    assert rc.returncode != 0
    assert "edges" in (rc.stderr + rc.stdout).lower()
