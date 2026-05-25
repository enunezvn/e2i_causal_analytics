"""Tests for scripts/measure_ensemble_ab.py helpers (Issue #242).

TDD red-first suite covering the anti-money-waste (Fix A) and de-confound
(Fix B / zeroshot prompt mode) changes.  ALL tests are fully offline —
no live API calls, no DSPy LM, no network.

Fix A helpers under test:
  _load_checkpoint   — load previously-persisted rows from a JSON file
  _remaining         — filter entries to those not yet measured (+ --force)
  _estimate_cost     — upfront cost estimate print
  _is_quota_error    — detect Anthropic credit-exhaustion / quota errors
  _order_entries     — order control (file / reverse / shuffle)
  _persist           — write/overwrite the checkpoint file after each entry

Fix B helpers under test:
  run_ensemble_classification prompt_mode="zeroshot" — no demos attached
  run_ensemble_classification prompt_mode="compiled" — demos present (default)
  --prompt-mode flag threads end-to-end through the script's argument parser
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap (needed when running from repo root without installing)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Import the helpers under test — these will FAIL (ImportError / AttributeError)
# until the implementation exists (TDD red phase).
# ---------------------------------------------------------------------------
import scripts.measure_ensemble_ab as ab  # noqa: E402

# ===========================================================================
# Fix A — Checkpoint / resume helpers
# ===========================================================================


class TestLoadCheckpoint:
    """_load_checkpoint: reads persisted rows from a JSON file."""

    def test_returns_empty_dict_when_file_absent(self, tmp_path):
        result = ab._load_checkpoint(tmp_path / "nonexistent.json")
        assert result == {}

    def test_loads_existing_rows_keyed_by_feature_name(self, tmp_path):
        data = {
            "rows": [
                {"feature_name": "feat_a", "gt": "confounder", "sonnet": "confounder"},
                {"feature_name": "feat_b", "gt": "descendant", "sonnet": "descendant"},
            ]
        }
        p = tmp_path / "checkpoint.json"
        p.write_text(json.dumps(data))
        result = ab._load_checkpoint(p)
        assert set(result.keys()) == {"feat_a", "feat_b"}
        assert result["feat_a"]["gt"] == "confounder"

    def test_returns_empty_dict_on_corrupt_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json")
        result = ab._load_checkpoint(p)
        assert result == {}


class TestRemaining:
    """_remaining: returns entries not yet measured, respecting --force and contaminated."""

    ENTRIES = [
        {"feature_name": "feat_a"},
        {"feature_name": "feat_b"},
        {"feature_name": "feat_c"},
    ]

    def test_all_entries_when_nothing_done(self):
        result = ab._remaining(self.ENTRIES, done={}, force=False)
        assert [e["feature_name"] for e in result] == ["feat_a", "feat_b", "feat_c"]

    def test_skips_already_done_entries(self):
        done = {"feat_a": {"feature_name": "feat_a", "contaminated": False}}
        result = ab._remaining(self.ENTRIES, done=done, force=False)
        names = [e["feature_name"] for e in result]
        assert "feat_a" not in names
        assert "feat_b" in names and "feat_c" in names

    def test_force_reruns_all_entries(self):
        done = {"feat_a": {"feature_name": "feat_a", "contaminated": False}}
        result = ab._remaining(self.ENTRIES, done=done, force=True)
        assert len(result) == 3

    def test_contaminated_row_is_retried(self):
        """A previously-contaminated (provider-outage) row is NOT considered done."""
        done = {"feat_a": {"feature_name": "feat_a", "contaminated": True}}
        result = ab._remaining(self.ENTRIES, done=done, force=False)
        names = [e["feature_name"] for e in result]
        # feat_a was contaminated → retry it
        assert "feat_a" in names

    def test_non_contaminated_done_row_is_skipped(self):
        done = {"feat_b": {"feature_name": "feat_b", "contaminated": False}}
        result = ab._remaining(self.ENTRIES, done=done, force=False)
        names = [e["feature_name"] for e in result]
        assert "feat_b" not in names


class TestEstimateCost:
    """_estimate_cost: upfront estimate in USD."""

    def test_returns_product_of_entries_times_per_call(self):
        est = ab._estimate_cost(n_entries=10, per_call_usd=0.06)
        assert est == pytest.approx(0.60)

    def test_zero_entries_returns_zero(self):
        assert ab._estimate_cost(0, 0.06) == pytest.approx(0.0)


class TestIsQuotaError:
    """_is_quota_error: detects Anthropic credit / quota exhaustion."""

    @pytest.mark.parametrize(
        "text",
        [
            "credit balance is too low",
            "Credit Balance Is Too Low",
            "quota exceeded",
            "insufficient_quota",
            "429 insufficient",
            "Your account has insufficient credits",
        ],
    )
    def test_recognises_quota_errors(self, text):
        assert ab._is_quota_error(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "rate limited — retry in 60s",
            "timeout connecting to API",
            "internal server error",
            "",
        ],
    )
    def test_non_quota_errors_return_false(self, text):
        assert ab._is_quota_error(text) is False

    def test_accepts_exception_object(self):
        exc = RuntimeError("credit balance is too low for this request")
        assert ab._is_quota_error(exc) is True

    def test_non_quota_exception_returns_false(self):
        exc = RuntimeError("rate limited")
        assert ab._is_quota_error(exc) is False


class TestOrderEntries:
    """_order_entries: file / reverse / shuffle ordering."""

    ENTRIES = [{"feature_name": f"feat_{i}"} for i in range(5)]

    def test_file_order_preserves_input(self):
        result = ab._order_entries(self.ENTRIES, order="file", seed=None)
        assert [e["feature_name"] for e in result] == [
            "feat_0",
            "feat_1",
            "feat_2",
            "feat_3",
            "feat_4",
        ]

    def test_reverse_order(self):
        result = ab._order_entries(self.ENTRIES, order="reverse", seed=None)
        assert [e["feature_name"] for e in result] == [
            "feat_4",
            "feat_3",
            "feat_2",
            "feat_1",
            "feat_0",
        ]

    def test_shuffle_is_deterministic_with_same_seed(self):
        r1 = ab._order_entries(self.ENTRIES, order="shuffle", seed=42)
        r2 = ab._order_entries(self.ENTRIES, order="shuffle", seed=42)
        assert r1 == r2

    def test_shuffle_with_different_seeds_differs(self):
        r1 = ab._order_entries(self.ENTRIES, order="shuffle", seed=1)
        r2 = ab._order_entries(self.ENTRIES, order="shuffle", seed=99)
        # Very unlikely to be equal for 5 items; if they happen to be equal
        # across two seeds that's a 1/120 chance — acceptable flake risk.
        assert [e["feature_name"] for e in r1] != [e["feature_name"] for e in r2]

    def test_shuffle_does_not_mutate_original(self):
        original = [e["feature_name"] for e in self.ENTRIES]
        ab._order_entries(self.ENTRIES, order="shuffle", seed=42)
        assert [e["feature_name"] for e in self.ENTRIES] == original

    def test_invalid_order_raises(self):
        with pytest.raises((ValueError, SystemExit)):
            ab._order_entries(self.ENTRIES, order="random_invalid_value", seed=None)


class TestPersist:
    """_persist: write (or overwrite) checkpoint file."""

    def test_creates_file_with_rows(self, tmp_path):
        rows = [{"feature_name": "feat_a", "gt": "confounder"}]
        p = tmp_path / "out.json"
        ab._persist(p, rows)
        data = json.loads(p.read_text())
        assert data["rows"] == rows

    def test_overwrites_existing_file(self, tmp_path):
        p = tmp_path / "out.json"
        ab._persist(p, [{"feature_name": "old"}])
        ab._persist(p, [{"feature_name": "new1"}, {"feature_name": "new2"}])
        data = json.loads(p.read_text())
        assert len(data["rows"]) == 2
        assert data["rows"][0]["feature_name"] == "new1"


# ===========================================================================
# Fix A — Budget guard integration (via _is_quota_error + accumulation)
# ===========================================================================


class TestBudgetGuard:
    """Budget cap: script stops before overspending --max-cost."""

    def _fake_classify(self, role="confounder", cost_per_call=0.02):
        """Return a fake EnsembleClassification-like object for monkeypatching."""

        class _FakeVote:
            def __init__(self, model, role):
                self.model = model
                self.causal_role = role
                self.mechanism = ""
                self.cost_usd = cost_per_call / 3

        class _FakeCLF:
            def __init__(self):
                self.fused_role = role
                self.agreement = "full"
                self.votes = [
                    _FakeVote("anthropic/claude-sonnet-4-6", role),
                    _FakeVote("anthropic/claude-opus-4-7", role),
                    _FakeVote("openai/gpt-5", role),
                ]
                self.total_cost_usd = cost_per_call

        return _FakeCLF()

    def test_stops_when_cost_exceeded(self, tmp_path, monkeypatch):
        """With max_cost=0.03 and per-entry cost=0.02, should stop after first entry
        (second would push total to 0.04 > cap)."""
        entries = [
            {
                "feature_name": f"feat_{i}",
                "ground_truth_role": "confounder",
                "derivation_pseudocode": "",
                "dataset_context": "",
            }
            for i in range(5)
        ]
        call_count = {"n": 0}
        fake_classify = self._fake_classify

        def _mock_run(*, feature_name, derivation_pseudocode, dataset_context, **kwargs):
            call_count["n"] += 1
            return fake_classify(cost_per_call=0.02)

        # Patch run_ensemble_classification inside the script module
        monkeypatch.setattr(ab, "_run_ensemble_for_entry", _mock_run, raising=False)

        rows, stopped = ab._run_measurement_loop(
            entries=entries,
            done={},
            models=("anthropic/claude-sonnet-4-6", "anthropic/claude-opus-4-7", "openai/gpt-5"),
            classifier=object(),
            max_cost=0.03,
            out=None,
            prompt_mode="compiled",
        )
        # Only 1 entry should have been measured before the cap triggered
        assert call_count["n"] == 1
        assert stopped == "budget"


class TestQuotaStop:
    """Quota exhaustion: script checkpoints and stops (no more CONTAMINATED rows)."""

    def test_quota_error_stops_and_checkpoints(self, tmp_path, monkeypatch):
        entries = [
            {
                "feature_name": f"feat_{i}",
                "ground_truth_role": "confounder",
                "derivation_pseudocode": "",
                "dataset_context": "",
            }
            for i in range(4)
        ]
        call_count = {"n": 0}

        def _mock_run(*, feature_name, derivation_pseudocode, dataset_context, **kwargs):
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise RuntimeError("credit balance is too low")

            class _FakeVote:
                def __init__(self, model, role):
                    self.model = model
                    self.causal_role = role
                    self.mechanism = ""
                    self.cost_usd = 0.01

            class _FakeCLF:
                fused_role = "confounder"
                agreement = "full"
                total_cost_usd = 0.03
                votes = [
                    _FakeVote("anthropic/claude-sonnet-4-6", "confounder"),
                    _FakeVote("anthropic/claude-opus-4-7", "confounder"),
                    _FakeVote("openai/gpt-5", "confounder"),
                ]

            return _FakeCLF()

        monkeypatch.setattr(ab, "_run_ensemble_for_entry", _mock_run, raising=False)

        rows, stopped = ab._run_measurement_loop(
            entries=entries,
            done={},
            models=("anthropic/claude-sonnet-4-6", "anthropic/claude-opus-4-7", "openai/gpt-5"),
            classifier=object(),
            max_cost=None,
            out=None,
            prompt_mode="compiled",
        )
        # Only first entry was measured cleanly
        assert call_count["n"] == 2  # called twice: first ok, second raises
        assert stopped == "quota"
        # Only 1 clean result — no contaminated entries added for remaining 3
        assert len(rows) == 1


# ===========================================================================
# Fix A — Argument parser: new flags are present with correct defaults
# ===========================================================================


class TestArgParser:
    """New CLI flags exist with correct types and defaults."""

    def _parse(self, args_str: str):
        """Parse a CLI arg string using the script's parser."""
        return ab._build_parser().parse_args(args_str.split())

    def test_max_cost_default_is_none(self):
        args = self._parse("")
        assert args.max_cost is None

    def test_max_cost_parses_float(self):
        args = self._parse("--max-cost 1.50")
        assert args.max_cost == pytest.approx(1.50)

    def test_order_default_is_file(self):
        args = self._parse("")
        assert args.order == "file"

    def test_order_accepts_valid_values(self):
        for val in ("file", "reverse", "shuffle"):
            args = self._parse(f"--order {val}")
            assert args.order == val

    def test_seed_default_is_none(self):
        args = self._parse("")
        assert args.seed is None

    def test_seed_parses_int(self):
        args = self._parse("--seed 42")
        assert args.seed == 42

    def test_force_default_is_false(self):
        args = self._parse("")
        assert args.force is False

    def test_force_flag_sets_true(self):
        args = self._parse("--force")
        assert args.force is True

    def test_prompt_mode_default_is_compiled(self):
        args = self._parse("")
        assert args.prompt_mode == "compiled"

    def test_prompt_mode_accepts_zeroshot(self):
        args = self._parse("--prompt-mode zeroshot")
        assert args.prompt_mode == "zeroshot"


# ===========================================================================
# Fix B — De-confound: zeroshot prompt mode
# ===========================================================================


class TestZeroshotPromptMode:
    """Fix B: zeroshot mode must NOT use compiled demos; compiled mode must."""

    def test_zeroshot_uses_uncompiled_classifier(self, monkeypatch):
        """In zeroshot mode, _predict_under_lm receives a fresh CausalRoleClassifier
        with no demos (empty predictors), not the compiled artifact."""
        import src.data.causal_role_classifier_ensemble as ens

        seen_classifiers = []

        def _capture_predict(classifier, lm, **kw):
            seen_classifiers.append(classifier)

            class _FakePred:
                causal_role = "confounder"
                mechanism = ""
                recommended_remediation = "keep_with_caveat"

            return _FakePred()

        monkeypatch.setattr(ens, "_make_lm", lambda model: MagicMock())
        monkeypatch.setattr(ens, "_predict_under_lm", _capture_predict)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")

        # A compiled classifier with demos attached (simulated)
        compiled_clf = MagicMock()
        compiled_clf.predictors.return_value = [MagicMock()]  # has predictors

        ens.run_ensemble_classification(
            feature_name="f",
            derivation_pseudocode="d",
            dataset_context="c",
            models=(
                "anthropic/claude-sonnet-4-6",
                "anthropic/claude-opus-4-7",
                "openai/gpt-5",
            ),
            classifier=compiled_clf,
            prompt_mode="zeroshot",
        )

        # Each model should have received a FRESH (uncompiled) classifier, not
        # the compiled_clf that was passed in.
        assert len(seen_classifiers) == 3
        for clf in seen_classifiers:
            # Must NOT be the compiled artifact passed in
            assert clf is not compiled_clf

    def test_compiled_mode_uses_passed_classifier(self, monkeypatch):
        """In compiled mode (default), the loaded compiled classifier is forwarded."""
        import src.data.causal_role_classifier_ensemble as ens

        seen_classifiers = []

        def _capture_predict(classifier, lm, **kw):
            seen_classifiers.append(classifier)

            class _FakePred:
                causal_role = "confounder"
                mechanism = ""
                recommended_remediation = "keep_with_caveat"

            return _FakePred()

        monkeypatch.setattr(ens, "_make_lm", lambda model: MagicMock())
        monkeypatch.setattr(ens, "_predict_under_lm", _capture_predict)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")

        compiled_clf = MagicMock(name="compiled_clf")

        ens.run_ensemble_classification(
            feature_name="f",
            derivation_pseudocode="d",
            dataset_context="c",
            models=(
                "anthropic/claude-sonnet-4-6",
                "anthropic/claude-opus-4-7",
                "openai/gpt-5",
            ),
            classifier=compiled_clf,
            prompt_mode="compiled",
        )

        # All three calls should have received the compiled_clf
        assert len(seen_classifiers) == 3
        for clf in seen_classifiers:
            assert clf is compiled_clf

    def test_prompt_mode_default_is_compiled(self, monkeypatch):
        """Default behaviour (no prompt_mode arg) must keep using compiled classifier."""
        import src.data.causal_role_classifier_ensemble as ens

        seen_classifiers = []

        def _capture_predict(classifier, lm, **kw):
            seen_classifiers.append(classifier)

            class _FakePred:
                causal_role = "descendant"
                mechanism = ""
                recommended_remediation = "drop"

            return _FakePred()

        monkeypatch.setattr(ens, "_make_lm", lambda model: MagicMock())
        monkeypatch.setattr(ens, "_predict_under_lm", _capture_predict)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")

        compiled_clf = MagicMock(name="compiled_clf_default")

        # Call WITHOUT prompt_mode kwarg (backward compat)
        ens.run_ensemble_classification(
            feature_name="f",
            derivation_pseudocode="d",
            dataset_context="c",
            models=(
                "anthropic/claude-sonnet-4-6",
                "anthropic/claude-opus-4-7",
                "openai/gpt-5",
            ),
            classifier=compiled_clf,
        )

        # All three calls used the compiled_clf
        assert all(clf is compiled_clf for clf in seen_classifiers)

    def test_prompt_mode_recorded_in_output_row(self, tmp_path, monkeypatch):
        """The prompt_mode value is recorded in each output row so runs are self-describing."""
        entries = [
            {
                "feature_name": "feat_a",
                "ground_truth_role": "confounder",
                "derivation_pseudocode": "",
                "dataset_context": "",
            }
        ]

        class _FakeCLF_obj:
            fused_role = "confounder"
            agreement = "full"
            total_cost_usd = 0.01

            class _V:
                model = "anthropic/claude-sonnet-4-6"
                causal_role = "confounder"
                mechanism = ""
                cost_usd = 0.003

            votes = [_V(), _V(), _V()]

        def _mock_run(*, feature_name, **kwargs):
            return _FakeCLF_obj()

        monkeypatch.setattr(ab, "_run_ensemble_for_entry", _mock_run, raising=False)

        rows, _ = ab._run_measurement_loop(
            entries=entries,
            done={},
            models=("anthropic/claude-sonnet-4-6", "anthropic/claude-opus-4-7", "openai/gpt-5"),
            classifier=object(),
            max_cost=None,
            out=None,
            prompt_mode="zeroshot",
        )
        assert len(rows) == 1
        assert rows[0].get("prompt_mode") == "zeroshot"


# ===========================================================================
# Fix B — classify_feature_ensemble: prompt_mode threads through correctly
# ===========================================================================


class TestPromptModeThreading:
    """prompt_mode must be accepted and propagated by classify_feature_ensemble."""

    def test_classify_feature_ensemble_accepts_prompt_mode_compiled(self, monkeypatch):
        import src.data.causal_role_classifier_ensemble as ens

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        models = ("anthropic/claude-sonnet-4-6", "anthropic/claude-opus-4-7", "openai/gpt-5")

        from src.data.kg.types import EnsembleModelVote

        def _fake(model, **kw):
            return EnsembleModelVote(model=model, causal_role="confounder")

        monkeypatch.setattr(ens, "_classify_one", _fake)

        verdict = ens.classify_feature_ensemble(
            feature_name="f",
            derivation_pseudocode="d",
            dataset_context="c",
            models=models,
            classifier=object(),
            prompt_mode="compiled",
        )
        assert verdict is not None
        assert verdict.causal_role == "confounder"

    def test_classify_feature_ensemble_accepts_prompt_mode_zeroshot(self, monkeypatch):
        import src.data.causal_role_classifier_ensemble as ens

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        models = ("anthropic/claude-sonnet-4-6", "anthropic/claude-opus-4-7", "openai/gpt-5")

        from src.data.kg.types import EnsembleModelVote

        def _fake(model, **kw):
            return EnsembleModelVote(model=model, causal_role="descendant")

        monkeypatch.setattr(ens, "_classify_one", _fake)

        verdict = ens.classify_feature_ensemble(
            feature_name="f",
            derivation_pseudocode="d",
            dataset_context="c",
            models=models,
            classifier=object(),
            prompt_mode="zeroshot",
        )
        assert verdict is not None
        assert verdict.causal_role == "descendant"
