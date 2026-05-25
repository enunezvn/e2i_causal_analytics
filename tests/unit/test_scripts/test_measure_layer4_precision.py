"""Tests for ``scripts/measure_layer4_precision.py`` flag surface.

Plan-239 §6.0 Phase-0 adds two flags to support the §6.8 A/B test:

- ``--classifier-artifact PATH`` — override the compiled-classifier path
  used by ``load_compiled_classifier()`` so the A/B test can score two
  artifacts (``causal_role_classifier.json`` vs
  ``causal_role_classifier_miprov2.json``) without mutating on-disk state.

- ``--disagreements-path PATH`` — optional JSON-list emit of disagreement
  records with exactly four keys per record:
  ``{cohort, gate, predicted_role, ground_truth_role}``. Feature names and
  derivation pseudocode are EXCLUDED BY CONSTRUCTION (plan-239 §6.4 HARD
  RULE) to keep the golden set from leaking into compile-set authoring.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SCRIPT = _REPO_ROOT / "scripts" / "measure_layer4_precision.py"
_PY = sys.executable


def test_measure_layer4_precision_accepts_classifier_artifact_flag() -> None:
    """Plan-239 §6.0 F1 / §6.2 R5 — argparse must accept --classifier-artifact.

    Subprocess invocation with --help: success exit + flag appears in usage.
    """
    proc = subprocess.run(
        [_PY, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"--help should exit 0; got {proc.returncode}. stderr={proc.stderr}"
    )
    assert "--classifier-artifact" in proc.stdout, (
        "--classifier-artifact flag missing from --help output. Plan-239 §6.0 F1."
    )


def test_measure_layer4_precision_accepts_disagreements_path_flag() -> None:
    """Plan-239 §6.0 F2 — argparse must accept --disagreements-path."""
    proc = subprocess.run(
        [_PY, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "--disagreements-path" in proc.stdout, (
        "--disagreements-path flag missing from --help output. Plan-239 §6.0 F2."
    )


def _golden_fixture(tmp_path: Path) -> Path:
    """Minimal golden fixture with one entry per role to exercise disagreement emit."""
    fixture = tmp_path / "mini_golden.json"
    fixture.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "cohort": "CSU_remibrutinib",
                        "feature_name": "f_iv",
                        "derivation_pseudocode": "select x from y",
                        "dataset_context": "ctx",
                        "ground_truth_role": "instrument",
                    },
                    {
                        "cohort": "CSU_remibrutinib",
                        "feature_name": "f_conf",
                        "derivation_pseudocode": "select x from y",
                        "dataset_context": "ctx",
                        "ground_truth_role": "confounder",
                    },
                ]
            }
        )
    )
    return fixture


class _FakeEvaluator:
    satisfied = True


class _FakeVerdict:
    def __init__(self, role: str) -> None:
        self.causal_role = role
        self.evaluator_audit = _FakeEvaluator()


def test_measure_layer4_precision_disagreements_path_emits_4key_records(
    tmp_path: Path,
) -> None:
    """Plan-239 §6.0 F2 / §6.2 R6 — --disagreements-path writes a JSON list whose
    records contain EXACTLY {cohort, gate, predicted_role, ground_truth_role}.
    NO feature_name. NO derivation_pseudocode. Plan-239 §6.4 HARD RULE.
    """
    import importlib

    # Reload the script as a module so we can patch its load_compiled_classifier.
    spec = importlib.util.spec_from_file_location("_mlp_module", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_mlp_module"] = module
    spec.loader.exec_module(module)

    golden = _golden_fixture(tmp_path)
    disagreements = tmp_path / "disagreements.json"

    # Patch load_compiled_classifier to return a sentinel and classify_feature
    # to return mismatching roles (so every entry disagrees with ground truth).
    fake_classifier = object()

    def _load() -> object:
        return fake_classifier

    def _classify(*, feature_name, derivation_pseudocode, dataset_context, classifier):
        # Return wrong role to force disagreement.
        if "iv" in feature_name:
            return _FakeVerdict("confounder")  # ground truth instrument
        return _FakeVerdict("instrument")  # ground truth confounder

    # Override the symbols on the loaded module.
    module.load_compiled_classifier = _load  # type: ignore[attr-defined]
    module.classify_feature = _classify  # type: ignore[attr-defined]

    # Drive main() via sys.argv manipulation.
    argv = [
        "measure_layer4_precision.py",
        "--golden-set",
        str(golden),
        "--cohort",
        "all",
        "--evaluator-gate",
        "true",
        "--threshold",
        "0.0",
        "--disagreements-path",
        str(disagreements),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        rc = module.main()
    finally:
        sys.argv = old_argv

    assert rc == 0, f"main() should exit 0 with threshold=0.0; got {rc}"
    assert disagreements.exists(), (
        f"--disagreements-path={disagreements} not written. Plan-239 §6.0 F2."
    )

    records: list[dict[str, Any]] = json.loads(disagreements.read_text())
    assert isinstance(records, list), "disagreements file must contain a JSON list"
    assert len(records) >= 1, "at least one disagreement should be recorded"

    expected_keys = {"cohort", "gate", "predicted_role", "ground_truth_role"}
    for rec in records:
        assert set(rec.keys()) == expected_keys, (
            f"disagreement record keys = {sorted(rec.keys())}; expected exactly "
            f"{sorted(expected_keys)}. Plan-239 §6.4 HARD RULE: NO feature_name, "
            f"NO derivation_pseudocode."
        )
        assert "feature_name" not in rec, "feature_name must NOT leak"
        assert "derivation_pseudocode" not in rec, "derivation_pseudocode must NOT leak"


def test_measure_layer4_precision_classifier_artifact_override(tmp_path: Path) -> None:
    """Plan-239 §6.0 F1 — --classifier-artifact must be threaded into
    load_compiled_classifier(). Run with a non-existent artifact and confirm the
    script surfaces a clear error or load_compiled_classifier receives the path.
    """
    import importlib

    spec = importlib.util.spec_from_file_location("_mlp_module2", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_mlp_module2"] = module
    spec.loader.exec_module(module)

    captured_paths: list[Any] = []

    def _spy_load(*args, **kwargs):
        # Capture either positional or kwarg path.
        if args:
            captured_paths.append(args[0])
        elif "artifact_path" in kwargs:
            captured_paths.append(kwargs["artifact_path"])
        elif "path" in kwargs:
            captured_paths.append(kwargs["path"])
        else:
            captured_paths.append(None)
        return None  # simulate no-LM path

    module.load_compiled_classifier = _spy_load  # type: ignore[attr-defined]

    fake_artifact = tmp_path / "fake_artifact.json"
    fake_artifact.write_text("{}")

    golden = _golden_fixture(tmp_path)

    argv = [
        "measure_layer4_precision.py",
        "--golden-set",
        str(golden),
        "--cohort",
        "all",
        "--evaluator-gate",
        "true",
        "--threshold",
        "0.0",
        "--classifier-artifact",
        str(fake_artifact),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        rc = module.main()
    finally:
        sys.argv = old_argv

    assert rc == 0
    assert captured_paths, "load_compiled_classifier was not called"
    # The path passed to load_compiled_classifier should equal the artifact path.
    passed = captured_paths[0]
    assert passed is not None and str(passed) == str(fake_artifact), (
        f"--classifier-artifact not threaded; load_compiled_classifier received "
        f"{passed!r}, expected {fake_artifact!r}. Plan-239 §6.0 F1."
    )


def test_measure_layer4_precision_ensemble_flag_routes_to_ensemble(tmp_path: Path) -> None:
    """#242 — ``--ensemble`` routes classification through
    ``classify_feature_ensemble`` (multi-model) instead of the single-model
    ``classify_feature``, so the offline A/B harness can score the ensemble.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_mlp_ens_module", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_mlp_ens_module"] = module
    spec.loader.exec_module(module)

    golden = _golden_fixture(tmp_path)
    ensemble_calls: list[str] = []
    single_calls: list[str] = []

    def _load(*a: Any, **k: Any) -> object:
        return object()

    def _ensemble(*, feature_name, derivation_pseudocode, dataset_context, classifier, **kw):
        ensemble_calls.append(feature_name)
        return _FakeVerdict("instrument")

    def _single(*, feature_name, derivation_pseudocode, dataset_context, classifier, **kw):
        single_calls.append(feature_name)
        return _FakeVerdict("instrument")

    module.load_compiled_classifier = _load  # type: ignore[attr-defined]
    module.classify_feature_ensemble = _ensemble  # type: ignore[attr-defined]
    module.classify_feature = _single  # type: ignore[attr-defined]
    module.ensure_dspy_lm_configured = lambda *a, **k: None  # type: ignore[attr-defined]

    argv = [
        "measure_layer4_precision.py",
        "--golden-set",
        str(golden),
        "--evaluator-gate",
        "false",
        "--threshold",
        "0.0",
        "--ensemble",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        rc = module.main()
    finally:
        sys.argv = old_argv

    assert rc == 0
    assert ensemble_calls, "--ensemble must route through classify_feature_ensemble"
    assert not single_calls, "single-model classify_feature must NOT run under --ensemble"
