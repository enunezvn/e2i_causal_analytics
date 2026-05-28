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


# --- Track-2B-v2: structural decider internals (graceful unclassifiable + leak-decision metric) ---


def _load_measure_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "measure_layer4_precision", _REPO / "scripts" / "measure_layer4_precision.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: the module-level @dataclass looks itself up in
    # sys.modules during decoration (dataclasses._is_type); without this the
    # load fails with AttributeError on a NoneType module.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_structural_predict_unclassifiable_is_scored_not_crash():
    # An entry WITH edges but whose feature node has no relation to T or Y is
    # unclassifiable; extract_role raises ValueError. The harness must score it
    # as "unclassified" (a miss / review), NOT crash the whole eval.
    mod = _load_measure_module()
    entry = {
        "feature_name": "x",
        "feature_node": "x",
        "treatment_node": "T",
        "outcome_node": "Y",
        "edges": [["T", "Y"], ["x", "q"]],
    }
    assert mod._structural_predict(entry) == "unclassified"


def test_leak_decision_metrics_buckets_and_missed_leaks():
    # Layer-4's functional output is the leak-vs-accept bucket (LEAK_ROLES ->
    # high severity). The leak-decision metric scores that bucket, surfaces the
    # safety-critical missed-leak count, and routes "unclassified" to review.
    mod = _load_measure_module()
    cm = mod.CohortMetrics(cohort="C", gate="ungated")
    cm.confusion = {
        "mediator": {"collider": 1},  # LEAK->LEAK: correct decision (within-bucket role miss)
        "descendant": {"collider": 1},  # LEAK->LEAK: correct
        "confounder": {"ancestor": 1},  # ACCEPT->ACCEPT: correct
        "instrument": {"unclassified": 1},  # ACCEPT->REVIEW: reviewed (not decided)
        "collider": {"confounder": 1},  # LEAK->ACCEPT: MISSED LEAK
        "ancestor": {"mediator": 1},  # ACCEPT->LEAK: false alarm
    }
    m = mod._leak_decision_metrics({("C", "ungated"): cm})
    assert m["decided"] == 5
    assert m["reviewed"] == 1
    assert m["missed_leaks"] == 1
    assert m["false_alarms"] == 1
    assert abs(m["leak_decision_accuracy"] - 0.6) < 1e-9


def test_structural_gate_fails_on_missed_leak(tmp_path):
    # A feature labeled collider (LEAK) but authored as a confounder (feat->T,
    # feat->Y => ACCEPT) is a MISSED LEAK; the structural gate must exit 1.
    fx = tmp_path / "missed.json"
    fx.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "feature_name": "ml",
                        "ground_truth_role": "collider",
                        "feature_node": "ml",
                        "treatment_node": "T",
                        "outcome_node": "Y",
                        "edges": [["ml", "T"], ["ml", "Y"]],
                    }
                ]
            }
        )
    )
    rc = subprocess.run(
        [
            sys.executable,
            "scripts/measure_layer4_precision.py",
            "--decider",
            "structural",
            "--golden-set",
            str(fx),
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO),
    )
    assert rc.returncode == 1
    assert "missed leak" in (rc.stderr + rc.stdout).lower()
