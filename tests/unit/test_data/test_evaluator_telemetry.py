"""Issue #241 — telemetry (latency + cost + tokens) for the Layer-4
Haiku audit evaluator.

Plan: GitHub issue #241 (R6 follow-up from
``.claude/plans/archive/15_layer4_evaluator_audit_signal_DONE_710058e0-13570eb8.md``).

Tests pin:
  * Per-call wall-clock latency captured in ``LLMEvaluatorAudit.latency_ms``.
  * Input/output token counts captured from the underlying DSPy LM
    ``history`` (litellm/Anthropic ``usage.prompt_tokens`` /
    ``usage.completion_tokens``).
  * Cost computed from documented Haiku rates:
    ``HAIKU_INPUT_USD_PER_MTOK`` × input_tokens / 1e6 +
    ``HAIKU_OUTPUT_USD_PER_MTOK`` × output_tokens / 1e6.
  * When evaluator is disabled (``ADAPTIVE_VALIDITY_EVALUATOR_ENABLED`` unset
    or missing API key), the 4 new fields are ``None``-stamped (not
    missing) at all 4 verdict-composition sites.
  * On evaluator exceptions (rate-limit, transient errors), ``_run_evaluator``
    returns ``None`` to preserve the worker verdict — the audit object is
    therefore absent and the sidecar telemetry fields are ``None``. The
    measured latency is surfaced via the WARNING log line so operators can
    still observe slow / timed-out calls.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Constants — pin canonical Haiku pricing so a silent edit to the rate
# constants trips the test.
# ---------------------------------------------------------------------------


def test_haiku_pricing_constants_pinned_to_documented_rates():
    """Anthropic Haiku 4-5 published rates (per million tokens):
    USD 1.00 input / USD 5.00 output.

    Source: https://www.anthropic.com/pricing (Haiku 4-5; checked 2026-05-15).
    Pinning the constants here makes any rate drift surface in CI —
    operators must consciously bump them when Anthropic re-prices.
    """
    from src.data.causal_role_evaluator import (
        HAIKU_INPUT_USD_PER_MTOK,
        HAIKU_OUTPUT_USD_PER_MTOK,
    )

    assert HAIKU_INPUT_USD_PER_MTOK == 1.00
    assert HAIKU_OUTPUT_USD_PER_MTOK == 5.00


def test_compute_haiku_cost_uses_documented_rates():
    from src.data.causal_role_evaluator import compute_haiku_cost_usd

    # 1M input tokens * $1.00 + 1M output * $5.00 = $6.00
    assert compute_haiku_cost_usd(input_tokens=1_000_000, output_tokens=1_000_000) == 6.00
    # 100 input + 250 output: 100*1e-6 + 250*5e-6 = 1.35e-3
    assert (
        abs(
            compute_haiku_cost_usd(input_tokens=100, output_tokens=250)
            - (100 * 1.00 / 1_000_000 + 250 * 5.00 / 1_000_000)
        )
        < 1e-12
    )


def test_compute_haiku_cost_handles_none_as_zero():
    """When usage extraction returns None (missing tokens key), the
    helper coerces to 0.0 rather than raising — the audit must still
    write SOMETHING for cost rather than dropping the field."""
    from src.data.causal_role_evaluator import compute_haiku_cost_usd

    assert compute_haiku_cost_usd(input_tokens=None, output_tokens=None) == 0.0
    assert compute_haiku_cost_usd(input_tokens=10, output_tokens=None) > 0.0
    assert compute_haiku_cost_usd(input_tokens=None, output_tokens=10) > 0.0


# ---------------------------------------------------------------------------
# Telemetry on the audit dataclass itself.
# ---------------------------------------------------------------------------


def test_llm_evaluator_audit_carries_telemetry_fields():
    """The 4 new telemetry fields are part of the audit dataclass and
    default to None so old call-sites that don't pass them still
    construct."""
    from src.data.kg.types import LLMEvaluatorAudit

    audit = LLMEvaluatorAudit(
        satisfied=True,
        rationale_complete=True,
        missed_considerations=(),
        notes="ok",
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )
    # Defaults — None when telemetry was not attached.
    assert audit.latency_ms is None
    assert audit.input_tokens is None
    assert audit.output_tokens is None
    assert audit.cost_usd is None

    # Explicit construction with telemetry.
    audit2 = LLMEvaluatorAudit(
        satisfied=True,
        rationale_complete=True,
        missed_considerations=(),
        notes="ok",
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
        latency_ms=123.4,
        input_tokens=200,
        output_tokens=100,
        cost_usd=0.0007,
    )
    assert audit2.latency_ms == 123.4
    assert audit2.input_tokens == 200
    assert audit2.output_tokens == 100
    assert audit2.cost_usd == 0.0007


# ---------------------------------------------------------------------------
# _run_evaluator wrapper: latency timing + usage extraction.
# ---------------------------------------------------------------------------


def _make_stub_lm_with_usage(
    *, prompt_tokens: int = 200, completion_tokens: int = 50
) -> SimpleNamespace:
    """Mimic a DSPy LM with a populated ``.history`` after a call.

    The ``base_lm._process_lm_response`` path appends
    ``{"usage": dict(response.usage), ...}`` — for Anthropic via litellm
    that's the OpenAI-style ``{"prompt_tokens": ..., "completion_tokens": ...,
    "total_tokens": ...}``.
    """
    return SimpleNamespace(
        history=[
            {
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            }
        ]
    )


# ---------------------------------------------------------------------------
# Issue #270 — direct unit tests for ``_extract_lm_usage`` to lock in the
# Anthropic-native usage-shape branch that was implemented in PR #262 but
# left uncovered by the test suite. ``_make_stub_lm_with_usage`` above only
# exercises the OpenAI / litellm-normalised shape; the fallback at lines
# 459-462 of ``causal_role_classifier_loader.py`` (``input_tokens`` /
# ``output_tokens``) was provably falsifiable — deleting those lines did not
# trip any existing test. These three parameter rows lock the fallback in.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "usage_block,expected_in,expected_out",
    [
        # Row 1 — OpenAI / litellm-normalised shape (baseline; locks existing
        # branch so a future "drop the OpenAI path" regression also trips).
        (
            {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
            200,
            50,
        ),
        # Row 2 — Anthropic-native ONLY (no OpenAI keys present at all). This
        # is the falsifiable case the issue calls out: a future provider
        # change to direct Anthropic SDK (no litellm normalisation) would
        # ship only these keys, and the existing test suite would silently
        # report None.
        (
            {"input_tokens": 200, "output_tokens": 50},
            200,
            50,
        ),
        # Row 3 — Mixed: OpenAI keys present but ``None``, Anthropic keys
        # populated. Exercises the ``in_t is None`` / ``out_t is None``
        # fallback predicate at lines 459-462 specifically — a stricter
        # implementation that returned early on "OpenAI key present" would
        # trip this row even though Row 2 still passes.
        (
            {
                "prompt_tokens": None,
                "completion_tokens": None,
                "input_tokens": 200,
                "output_tokens": 50,
            },
            200,
            50,
        ),
    ],
    ids=["openai_shape", "anthropic_native_shape", "mixed_openai_none_anthropic_populated"],
)
def test_extract_lm_usage_accepts_both_shapes(usage_block, expected_in, expected_out):
    """Issue #270. ``_extract_lm_usage`` must extract tokens from EITHER the
    OpenAI / litellm-normalised usage block (``prompt_tokens`` /
    ``completion_tokens``) OR the Anthropic-native block (``input_tokens`` /
    ``output_tokens``). The Anthropic branch is fallback-only: a future
    provider-drift away from litellm normalisation would otherwise silently
    drop token / cost telemetry.

    Falsifiability (issue #270): deleting lines 459-462 of
    ``causal_role_classifier_loader.py`` (the ``if in_t is None: in_t = ...
    input_tokens`` fallback block) would not trip any pre-existing test.
    Rows 2 and 3 of this parameterisation trip in that scenario.
    """
    from src.data.causal_role_classifier_loader import _extract_lm_usage

    stub_lm = SimpleNamespace(history=[{"usage": usage_block}])

    in_t, out_t = _extract_lm_usage(stub_lm)

    assert in_t == expected_in
    assert out_t == expected_out


def test_extract_lm_usage_prefers_openai_shape_when_both_present():
    """Issue #270 corollary — when BOTH shapes are present with non-None
    values, the OpenAI / litellm-normalised keys win. This is the
    documented precedence at lines 454-462 of
    ``causal_role_classifier_loader.py`` ("Accept OpenAI shape first") and
    matters because litellm currently always normalises Anthropic responses
    to OpenAI keys — if the fallback ever shadowed the primary, the
    telemetry surface would silently report Anthropic-derived values from a
    duplicated/echoed block instead of the litellm-canonical totals.
    """
    from src.data.causal_role_classifier_loader import _extract_lm_usage

    stub_lm = SimpleNamespace(
        history=[
            {
                "usage": {
                    # Primary — OpenAI / litellm shape.
                    "prompt_tokens": 200,
                    "completion_tokens": 50,
                    "total_tokens": 250,
                    # Fallback — Anthropic-native, intentionally different
                    # numbers to disambiguate which branch fires.
                    "input_tokens": 9999,
                    "output_tokens": 9999,
                }
            }
        ]
    )

    in_t, out_t = _extract_lm_usage(stub_lm)

    # OpenAI shape wins; the 9999/9999 fallback is shadowed.
    assert in_t == 200
    assert out_t == 50


def test_run_evaluator_attaches_latency_and_token_telemetry(monkeypatch):
    """When the evaluator succeeds, the returned LLMEvaluatorAudit
    carries:
      * latency_ms > 0 (real wall-clock)
      * input_tokens / output_tokens from the LM ``.history``
      * cost_usd computed from the Haiku rates
    """
    import src.data.causal_role_classifier_loader as loader
    from src.data.kg.types import LLMEvaluatorAudit, LLMVerdict

    canned_audit = LLMEvaluatorAudit(
        satisfied=True,
        rationale_complete=True,
        missed_considerations=(),
        notes="ok",
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )

    stub_evaluator = MagicMock()
    stub_evaluator.evaluate.return_value = canned_audit

    stub_lm = _make_stub_lm_with_usage(prompt_tokens=200, completion_tokens=50)

    # Patch dspy.LM in the loader's namespace and the dspy.settings.context
    # so we don't actually call out to the network.
    monkeypatch.setattr(loader.dspy, "LM", lambda **_kw: stub_lm)

    class _CtxStub:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(loader.dspy.settings, "context", lambda **_kw: _CtxStub())

    audit = loader._run_evaluator(
        stub_evaluator,
        feature_name="f",
        derivation_pseudocode="d",
        dataset_context="c",
        worker_verdict=LLMVerdict(
            causal_role="confounder",
            mechanism="m",
            recommended_remediation="keep_with_caveat",
        ),
    )
    assert audit is not None
    assert isinstance(audit, LLMEvaluatorAudit)
    # Latency: positive real number.
    assert audit.latency_ms is not None
    assert audit.latency_ms >= 0.0
    assert isinstance(audit.latency_ms, float)
    # Token counts pulled from history.
    assert audit.input_tokens == 200
    assert audit.output_tokens == 50
    # Cost from the documented Haiku rates.
    from src.data.causal_role_evaluator import compute_haiku_cost_usd

    assert audit.cost_usd == compute_haiku_cost_usd(input_tokens=200, output_tokens=50)


def test_run_evaluator_telemetry_resilient_to_empty_history(monkeypatch):
    """When the LM ``history`` is empty (e.g. cache-hit path or stub LM
    that doesn't update history), telemetry still attaches latency_ms
    but the token / cost fields stay None — better to surface
    partial-telemetry than to drop the audit."""
    import src.data.causal_role_classifier_loader as loader
    from src.data.kg.types import LLMEvaluatorAudit, LLMVerdict

    canned_audit = LLMEvaluatorAudit(
        satisfied=True,
        rationale_complete=True,
        missed_considerations=(),
        notes="ok",
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
    )
    stub_evaluator = MagicMock()
    stub_evaluator.evaluate.return_value = canned_audit

    stub_lm = SimpleNamespace(history=[])
    monkeypatch.setattr(loader.dspy, "LM", lambda **_kw: stub_lm)

    class _CtxStub:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(loader.dspy.settings, "context", lambda **_kw: _CtxStub())

    audit = loader._run_evaluator(
        stub_evaluator,
        feature_name="f",
        derivation_pseudocode="d",
        dataset_context="c",
        worker_verdict=LLMVerdict(
            causal_role="confounder",
            mechanism="m",
            recommended_remediation="keep_with_caveat",
        ),
    )
    assert audit is not None
    # Latency still attaches.
    assert audit.latency_ms is not None
    # Token / cost fields are None when usage cannot be extracted.
    assert audit.input_tokens is None
    assert audit.output_tokens is None
    assert audit.cost_usd is None


def test_run_evaluator_returns_none_when_evaluator_raises(monkeypatch, caplog):
    """When the evaluator raises (rate-limit, etc.), _run_evaluator
    returns None to preserve the worker's verdict. This is the
    pre-existing contract — telemetry must not break it.

    Note: latency telemetry is dropped because the audit object itself
    is None. The INFO/WARNING log line carries the timing if operators
    need it.
    """
    import src.data.causal_role_classifier_loader as loader
    from src.data.kg.types import LLMVerdict

    stub_evaluator = MagicMock()
    stub_evaluator.evaluate.side_effect = RuntimeError("rate-limit")
    monkeypatch.setattr(loader.dspy, "LM", lambda **_kw: SimpleNamespace(history=[]))

    class _CtxStub:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    monkeypatch.setattr(loader.dspy.settings, "context", lambda **_kw: _CtxStub())

    with caplog.at_level("WARNING"):
        audit = loader._run_evaluator(
            stub_evaluator,
            feature_name="f",
            derivation_pseudocode="d",
            dataset_context="c",
            worker_verdict=LLMVerdict(
                causal_role="confounder",
                mechanism="m",
                recommended_remediation="keep_with_caveat",
            ),
        )
    assert audit is None
    # Operator-facing log must include the latency so a slow rate-limit
    # response is visible.
    assert any("latency_ms" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# All 4 stamping sites carry the new fields — symmetry check.
# ---------------------------------------------------------------------------


def test_ensemble_to_legacy_dict_threads_telemetry_fields():
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _ensemble_to_legacy_dict,
    )
    from src.data.kg.types import EnsembleVerdict, LLMEvaluatorAudit, LLMVerdict

    audit = LLMEvaluatorAudit(
        satisfied=True,
        rationale_complete=True,
        missed_considerations=(),
        notes="ok",
        evaluator_model="anthropic/claude-haiku-4-5-20251001",
        latency_ms=42.5,
        input_tokens=200,
        output_tokens=50,
        cost_usd=0.00045,
    )
    llm = LLMVerdict(
        causal_role="confounder",
        mechanism="m",
        recommended_remediation="keep_with_caveat",
        evaluator_audit=audit,
    )
    verdict = EnsembleVerdict(
        feature_name="f",
        severity="moderate",
        remediation="keep_with_caveat",
        decided_by="llm",
        confidence=0.8,
        final_role="confounder",
        evidence=("layer-4 llm",),
        disagreements=(),
        llm_input=llm,
    )
    out = _ensemble_to_legacy_dict(
        verdict,
        adversarial_input={
            "feature": "f",
            "severity_pre_joint_check": "moderate",
            "z_score": 4.2,
            "delta_auc": 0.12,
            "delta_auc_below_floor": False,
            "_hblp_classified": True,
        },
    )
    assert out["evaluator_latency_ms"] == 42.5
    assert out["evaluator_input_tokens"] == 200
    assert out["evaluator_output_tokens"] == 50
    assert out["evaluator_cost_usd"] == 0.00045


def test_ensemble_to_legacy_dict_telemetry_none_when_audit_absent():
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _ensemble_to_legacy_dict,
    )
    from src.data.kg.types import EnsembleVerdict, LLMVerdict

    llm = LLMVerdict(
        causal_role="confounder",
        mechanism="m",
        recommended_remediation="keep_with_caveat",
    )
    verdict = EnsembleVerdict(
        feature_name="f",
        severity="moderate",
        remediation="keep_with_caveat",
        decided_by="llm",
        confidence=0.8,
        final_role="confounder",
        evidence=("layer-4 llm",),
        disagreements=(),
        llm_input=llm,
    )
    out = _ensemble_to_legacy_dict(
        verdict,
        adversarial_input={
            "feature": "f",
            "severity_pre_joint_check": "moderate",
            "z_score": 4.2,
            "delta_auc": 0.12,
            "delta_auc_below_floor": False,
            "_hblp_classified": True,
        },
    )
    for key in (
        "evaluator_latency_ms",
        "evaluator_input_tokens",
        "evaluator_output_tokens",
        "evaluator_cost_usd",
    ):
        assert key in out, f"telemetry key {key} missing from _ensemble_to_legacy_dict"
        assert out[key] is None


def test_legacy_adversarial_alone_verdict_stamps_telemetry_fields_none():
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _legacy_adversarial_alone_verdict,
    )

    out = _legacy_adversarial_alone_verdict(
        "f",
        {
            "feature": "f",
            "severity": "info",
            "remediation": "keep",
            "evidence": "z=1.0",
            "severity_pre_joint_check": "info",
            "z_score": 1.0,
            "delta_auc": 0.0,
            "delta_auc_below_floor": True,
            "_hblp_classified": True,
        },
    )
    for key in (
        "evaluator_latency_ms",
        "evaluator_input_tokens",
        "evaluator_output_tokens",
        "evaluator_cost_usd",
    ):
        assert key in out, f"telemetry key {key} missing from _legacy_adversarial_alone_verdict"
        assert out[key] is None


def test_legacy_info_verdict_stamps_telemetry_fields_none():
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _legacy_info_verdict,
    )

    out = _legacy_info_verdict(
        "f",
        adversarial_input={
            "feature": "f",
            "severity": "info",
            "remediation": "keep",
            "evidence": "Adversarial score undefined",
            "_hblp_classified": True,
        },
        evidence="Adversarial score undefined",
    )
    for key in (
        "evaluator_latency_ms",
        "evaluator_input_tokens",
        "evaluator_output_tokens",
        "evaluator_cost_usd",
    ):
        assert key in out, f"telemetry key {key} missing from _legacy_info_verdict"
        assert out[key] is None


def test_legacy_short_circuit_verdict_stamps_telemetry_fields_none():
    from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
        _legacy_short_circuit_verdict,
    )

    out = _legacy_short_circuit_verdict("f", evidence="too few rows")
    for key in (
        "evaluator_latency_ms",
        "evaluator_input_tokens",
        "evaluator_output_tokens",
        "evaluator_cost_usd",
    ):
        assert key in out, f"telemetry key {key} missing from _legacy_short_circuit_verdict"
        assert out[key] is None


# ---------------------------------------------------------------------------
# Sidecar JSON round-trip.
# ---------------------------------------------------------------------------


def test_sidecar_serialises_telemetry_keys_when_present(tmp_path, monkeypatch):
    import json

    from src.agents.ml_foundation.data_preparer.graph import (
        write_adaptive_verdicts_sidecar,
    )

    monkeypatch.setenv("ADAPTIVE_VALIDITY_ARTIFACTS_DIR", str(tmp_path))
    verdict = {
        "feature": "f",
        "layer": "4",
        "severity": "moderate",
        "remediation": "keep_with_caveat",
        "evidence": "layer-4 llm",
        "decided_by": "llm",
        "disagreements": [],
        "kg_signal": "no_signal",
        "z_score": 4.2,
        "actual_auc": 0.66,
        "null_mean": 0.5,
        "null_std": 0.02,
        "p_value": 0.0001,
        "n_permutations": 200,
        "delta_auc": 0.12,
        "delta_auc_floor": 0.10,
        "delta_auc_below_floor": False,
        "severity_pre_joint_check": "moderate",
        "ablation_z_score": None,
        "ablation_delta_auc": None,
        "ablation_null_mean": None,
        "ablation_null_std": None,
        "ablation_severity": None,
        "contract_source": None,
        "contract_window_days": None,
        "llm_role": "confounder",
        "llm_remediation": "keep_with_caveat",
        "evaluator_satisfied": True,
        "evaluator_rationale_complete": True,
        "evaluator_missed_considerations": ("pearl_arrows",),
        "evaluator_notes": "ok",
        "evaluator_model": "anthropic/claude-haiku-4-5-20251001",
        # New telemetry keys.
        "evaluator_latency_ms": 42.5,
        "evaluator_input_tokens": 200,
        "evaluator_output_tokens": 50,
        "evaluator_cost_usd": 0.00045,
    }
    state = {
        "experiment_id": "test-experiment",
        "data_source": "synthetic",
        "leakage_severity": "none",
        "leaked_features": [],
        "adaptive_flagged_features": [],
        "adaptive_verdicts": [verdict],
    }
    path = write_adaptive_verdicts_sidecar(state)
    assert path is not None and path.exists()
    payload = json.loads(path.read_text())
    out = payload["adaptive_verdicts"][0]
    assert out["evaluator_latency_ms"] == 42.5
    assert out["evaluator_input_tokens"] == 200
    assert out["evaluator_output_tokens"] == 50
    assert out["evaluator_cost_usd"] == 0.00045


# ---------------------------------------------------------------------------
# SidecarReader carries the telemetry through to VerdictRecord.
# ---------------------------------------------------------------------------


def test_sidecar_reader_surfaces_telemetry_fields(tmp_path):
    import json

    from src.data.audit_sidecar_reader import SidecarReader

    exp_dir = tmp_path / "exp-1"
    exp_dir.mkdir()
    sidecar = exp_dir / "adaptive_verdicts_20260515T120000Z.json"
    sidecar.write_text(
        json.dumps(
            {
                "experiment_id": "exp-1",
                "data_source": "synthetic",
                "written_at": "20260515T120000Z",
                "adaptive_verdicts": [
                    {
                        "feature": "f",
                        "layer": "4",
                        "severity": "moderate",
                        "remediation": "keep_with_caveat",
                        "evidence": "e",
                        "evaluator_satisfied": False,
                        "evaluator_rationale_complete": False,
                        "evaluator_missed_considerations": ["temporal_filter"],
                        "evaluator_notes": "thin",
                        "evaluator_model": "anthropic/claude-haiku-4-5-20251001",
                        "evaluator_latency_ms": 42.5,
                        "evaluator_input_tokens": 200,
                        "evaluator_output_tokens": 50,
                        "evaluator_cost_usd": 0.00045,
                    }
                ],
            }
        )
    )
    reader = SidecarReader(artifacts_dir=tmp_path)
    records = list(reader.iter_verdict_records())
    assert len(records) == 1
    rec = records[0]
    assert rec.evaluator_latency_ms == 42.5
    assert rec.evaluator_input_tokens == 200
    assert rec.evaluator_output_tokens == 50
    assert rec.evaluator_cost_usd == 0.00045


def test_sidecar_reader_handles_missing_telemetry_keys(tmp_path):
    """Older sidecars (pre-#241) don't have the telemetry keys. The
    reader must surface them as None, not raise."""
    import json

    from src.data.audit_sidecar_reader import SidecarReader

    exp_dir = tmp_path / "exp-2"
    exp_dir.mkdir()
    sidecar = exp_dir / "adaptive_verdicts_20260514T120000Z.json"
    sidecar.write_text(
        json.dumps(
            {
                "experiment_id": "exp-2",
                "data_source": "synthetic",
                "written_at": "20260514T120000Z",
                "adaptive_verdicts": [
                    {
                        "feature": "f",
                        "layer": "4",
                        "severity": "moderate",
                        "remediation": "keep_with_caveat",
                        "evidence": "e",
                        "evaluator_satisfied": True,
                        "evaluator_rationale_complete": True,
                        "evaluator_missed_considerations": [],
                        "evaluator_notes": "ok",
                        "evaluator_model": "anthropic/claude-haiku-4-5-20251001",
                        # No telemetry keys (pre-#241 sidecar).
                    }
                ],
            }
        )
    )
    reader = SidecarReader(artifacts_dir=tmp_path)
    records = list(reader.iter_verdict_records())
    assert len(records) == 1
    rec = records[0]
    assert rec.evaluator_latency_ms is None
    assert rec.evaluator_input_tokens is None
    assert rec.evaluator_output_tokens is None
    assert rec.evaluator_cost_usd is None
