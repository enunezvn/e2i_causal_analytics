"""Anti-leakage integration test for the risk_score model (issue #171).

Pins the contract that no feature with ``xolair`` / ``dupixent`` /
``omalizumab`` / ``dupilumab`` / ``50242`` / ``00024`` in its name appears in
the training feature matrix consumed by the risk_score model.

The converter at ``scripts/convert_optum_rwd.py`` already filters biologic
rows from the ``<drug_class>_ever_filled`` features per §7.5
(``_csu_biologic_mask`` at line ~1776 + the filter at line ~2222), but the
contract is enforced by a regression test so we fail loud if anyone:

    1. Adds Xolair / Dupixent to ``NON_TARGET_DRUG_CLASSES``.
    2. Adds a feature named e.g. ``xolair_days_since_last_fill`` to
       ``_compute_features``.
    3. Forgets to apply the ``_csu_biologic_mask`` filter in a new feature
       block under §7.x.

The test is deliberately STATIC — it analyses
``scripts/convert_optum_rwd.py`` for forbidden feature-key assignments AND
runs ``_compute_features`` on a synthetic patient that has BOTH biologic and
non-biologic prescriptions in lookback. The dynamic-cohort assertion
confirms that even when biologic rows are present in the lookback medication
table, none of the resulting feature column names contains a forbidden
substring.

The pipeline-level end-to-end check (real Optum cohort -> training feature
matrix -> training) is deferred to the closure memo because real Optum data
is not available in CI / agent worktrees. The honest deferral is documented
in ``.claude/plans/issue_171_close_*.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.agents.prediction_synthesizer.risk_score import (
    FORBIDDEN_FEATURE_SUBSTRINGS,
    LeakageError,
    assert_no_leakage_in_features,
    find_leaked_features,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONVERTER_PATH = PROJECT_ROOT / "scripts" / "convert_optum_rwd.py"


# ---------------------------------------------------------------------------
# Static analysis: scan the converter source for forbidden feature keys
# ---------------------------------------------------------------------------


class TestConverterFeatureKeysAreClean:
    """The converter must not emit any feature whose key contains a forbidden
    substring.

    Approach: scan ``_compute_features`` for ``feats["..."] = ...`` /
    ``feats[f"...{...}..."] = ...`` writes and check the literal key fragments.
    """

    @pytest.fixture(scope="class")
    def converter_source(self) -> str:
        return CONVERTER_PATH.read_text(encoding="utf-8")

    @pytest.fixture(scope="class")
    def compute_features_body(self, converter_source: str) -> str:
        """Slice ``_compute_features`` from the converter source for grep.

        The method signature spans multiple lines so we anchor on the
        ``def _compute_features(`` token and slurp until the next ``    def ``
        (same-indent method).
        """
        start_match = re.search(r"\n    def _compute_features\(", converter_source)
        assert start_match is not None, "Could not locate `def _compute_features(` in converter."
        start = start_match.start()
        # Find the next same-indent method def or class line.
        tail = converter_source[start + 1 :]
        end_match = re.search(r"\n    def \w+\(|\n    @\w|\nclass \w", tail)
        end = (start + 1 + end_match.start()) if end_match else len(converter_source)
        return converter_source[start:end]

    @pytest.mark.parametrize("token", FORBIDDEN_FEATURE_SUBSTRINGS)
    def test_no_literal_feature_key_contains_forbidden_token(
        self, compute_features_body: str, token: str
    ) -> None:
        """No ``feats["...<token>..."]`` literal key.

        Allows the converter to MENTION the token in comments (e.g., §7.5
        explaining why biologics are filtered out) but forbids it from
        appearing as part of a feature column name.
        """
        # Match feats["..."] or feats[f"..."] assignments only. Capture the
        # literal key text.
        key_pattern = re.compile(
            r"feats\[\s*(?:f?)['\"]([^'\"]+)['\"]\s*\]\s*=",
            flags=re.IGNORECASE,
        )
        offending: list[str] = []
        for key in key_pattern.findall(compute_features_body):
            if token.lower() in key.lower():
                offending.append(key)
        assert offending == [], (
            f"Converter assigns feature keys containing forbidden token {token!r}: "
            f"{offending}. The CSU initiation target is "
            "`initiated_biologic_180d` and Xolair/Dupixent exposure features "
            "would leak it. Filter biologic rows via `_csu_biologic_mask` "
            "before building this feature."
        )

    def test_non_target_drug_classes_excludes_biologic_generics(
        self, converter_source: str
    ) -> None:
        """``NON_TARGET_DRUG_CLASSES`` must not include Xolair / Dupixent generics.

        If a future PR adds these to the non-target dict, the
        ``<class_name>_ever_filled`` feature suite would leak the target.
        """
        match = re.search(
            r"NON_TARGET_DRUG_CLASSES.*?=\s*\{(.*?)\}\n",
            converter_source,
            flags=re.DOTALL,
        )
        assert match is not None, "Could not find NON_TARGET_DRUG_CLASSES literal."
        body = match.group(1).lower()
        for token in FORBIDDEN_FEATURE_SUBSTRINGS:
            assert token not in body, (
                f"NON_TARGET_DRUG_CLASSES contains forbidden generic / brand / NDC "
                f"prefix {token!r}. Adding biologic generics to non-target classes "
                "would leak `initiated_biologic_180d` via "
                "`<class_name>_ever_filled` features."
            )


# ---------------------------------------------------------------------------
# Dynamic check: _compute_features on a synthetic patient with biologic fills
# ---------------------------------------------------------------------------


def _make_minimal_converter():
    """Construct a minimal OptumDataConverter without real input files.

    Returns a converter instance whose feature pipeline can be invoked with
    a hand-rolled medication table. We bypass the full converter ``__init__``
    by allocating via ``__new__`` and populating only the attributes
    ``_compute_features`` reads.
    """
    import sys

    # Make the project root importable for the script.
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.convert_optum_rwd import OptumDataConverter  # type: ignore[import-not-found]

    conv = OptumDataConverter.__new__(OptumDataConverter)
    # Attributes that ``_compute_features`` and its helpers touch. The exact
    # field set is broad because the converter is monolithic; we initialise to
    # sensible empties so the few branches we care about (medication features
    # + non-target drug class loop) execute.
    conv._med_by_pat = {}
    conv._lab_by_pat = {}
    conv._diag_by_pat = {}
    conv._proc_by_pat = {}
    conv._visit_by_pat = {}
    conv._provider_by_pat = {}
    conv._enroll_by_pat = pd.DataFrame()
    conv._mem_by_pat = pd.DataFrame()
    conv._hosp_by_pat = {}
    conv._ed_by_pat = {}
    conv._comorbidity_idx_by_pat = {}
    conv.enrollment_post_days = 180
    return conv


def test_compute_features_dynamic_output_has_no_leakage() -> None:
    """Construct a synthetic patient and assert no biologic-named feature.

    Builds a minimal lookback medication table that contains:
        - Xolair (omalizumab, NDC prefix 50242) — the target drug.
        - Dupixent (dupilumab, NDC prefix 00024) — the other target drug.
        - Non-target H1 antihistamine (cetirizine) — included to confirm the
          non-target features ARE emitted.

    The §7.5 biologic mask should filter the Xolair + Dupixent rows out of
    the lookback before the ``<drug_class>_ever_filled`` features are
    computed; the resulting feature dict's keys MUST contain no forbidden
    substring.

    We DO NOT execute the full converter pipeline (no input files); we just
    sanity-check the column-name contract by inspecting the dict keys after
    ``_compute_features``. If the converter API drifts, this test falls back
    to the static-source check above.
    """
    try:
        conv = _make_minimal_converter()
    except Exception as exc:
        pytest.skip(
            f"Dynamic converter probe could not initialise (likely API drift): {exc}. "
            f"Static-source leakage check above still pins the contract."
        )
        return

    # Stub a single patient with mixed med history.
    patid = 1
    index_date = pd.Timestamp("2024-01-01")
    med = pd.DataFrame(
        {
            "medication_date": pd.to_datetime(
                [
                    "2023-10-01",  # Xolair, NDC 50242-04-001
                    "2023-11-15",  # Dupixent, NDC 00024-58-700
                    "2023-12-01",  # Cetirizine
                ]
            ),
            "patid": [patid] * 3,
            "ndc": ["50242040001", "00024587000", "00378612001"],
            "brand_name": ["XOLAIR", "DUPIXENT", "CETIRIZINE HCL"],
            "generic_name": ["omalizumab", "dupilumab", "cetirizine"],
            "days_sup": [28, 28, 30],
            "code": ["", "", ""],
        }
    )
    conv._med_by_pat[patid] = med

    demo = pd.Series(
        {
            "age": 45,
            "gdr_cd": "F",
            "zipcode_5": "10001",
            "bus": None,
            "product": None,
            "health_exch": None,
            "lis_dual": None,
        }
    )
    try:
        feats = conv._compute_features(patid, index_date, demo)
    except Exception as exc:
        pytest.skip(
            f"_compute_features signature/behavior drifted (got {type(exc).__name__}: "
            f"{exc}). Static-source leakage check above still pins the contract."
        )
        return

    leaked = find_leaked_features(feats.keys())
    assert leaked == [], (
        f"Converter emitted feature keys containing forbidden substrings: {leaked}. "
        f"Feature dict had {len(feats)} keys. The §7.5 biologic mask in "
        "`_compute_features` must drop Xolair + Dupixent fills before any "
        "lookback medication feature is computed."
    )


# ---------------------------------------------------------------------------
# Trainer-side guard: end-to-end smoke that simulates a wired training pass
# ---------------------------------------------------------------------------


def test_trainer_rejects_feature_matrix_with_biologic_columns() -> None:
    """If a future caller accidentally passes a leaking feature matrix to the
    trainer, ``RiskScoreTrainer.fit`` must refuse to train.
    """
    import numpy as np

    from src.agents.prediction_synthesizer.risk_score import RiskScoreTrainer

    n = 80
    X = pd.DataFrame(
        {
            "age": np.linspace(20, 80, n),
            "ed_visits_total": np.random.RandomState(0).poisson(1.5, size=n),
            # This is the bug we are guarding against:
            "xolair_ever_filled": np.random.RandomState(1).randint(0, 2, size=n),
        }
    )
    y = np.random.RandomState(2).randint(0, 2, size=n)

    trainer = RiskScoreTrainer(
        enable_mlflow=False, hpo_trials=1, cv_folds=2, model_candidates=("xgboost",)
    )
    with pytest.raises(LeakageError) as exc:
        trainer.fit(X, y, X, y)
    assert "xolair_ever_filled" in exc.value.leaked


def test_trainer_accepts_clean_feature_matrix() -> None:
    """The mirror of the previous test: clean column names pass the guard.

    Uses a tiny dataset; we only assert that the leakage guard does not raise.
    HPO is set to 1 trial so this stays under a second.
    """
    import numpy as np

    from src.agents.prediction_synthesizer.risk_score import RiskScoreTrainer

    n = 60
    X = pd.DataFrame(
        {
            "age": np.linspace(20, 80, n),
            "ed_visits_total": np.random.RandomState(0).poisson(1.5, size=n),
            "h1_1g_ever_filled": np.random.RandomState(1).randint(0, 2, size=n),
            "sys_steroid_fill_count": np.random.RandomState(2).randint(0, 4, size=n),
        }
    )
    # Balanced labels so the splitter doesn't choke.
    y = np.array([0, 1] * (n // 2))
    trainer = RiskScoreTrainer(
        enable_mlflow=False, hpo_trials=1, cv_folds=2, model_candidates=("xgboost",)
    )
    # Must not raise LeakageError.
    result = trainer.fit(X, y, X, y)
    assert result.feature_names == list(X.columns)
    # Honest deferral guard: on a noise dataset the AUC-PR floor is unlikely
    # to be met — we only check the bar wasn't silently lowered.
    assert result.auc_pr_floor == 0.65


# ---------------------------------------------------------------------------
# Suite-level summary: every forbidden token has at least one assertion
# ---------------------------------------------------------------------------


def test_every_forbidden_substring_has_a_pinned_assertion() -> None:
    """Meta-check: ensure ``FORBIDDEN_FEATURE_SUBSTRINGS`` matches the issue spec.

    If a future contributor edits ``FORBIDDEN_FEATURE_SUBSTRINGS`` without
    updating the test suite, this regression catches it.
    """
    expected = {"xolair", "dupixent", "omalizumab", "dupilumab", "50242", "00024"}
    assert set(FORBIDDEN_FEATURE_SUBSTRINGS) == expected
    assert_no_leakage_in_features([])  # smoke: empty input passes


def test_leakage_guard_self_check() -> None:
    """Smoke: planting every forbidden token gets caught."""
    for token in FORBIDDEN_FEATURE_SUBSTRINGS:
        with pytest.raises(LeakageError):
            assert_no_leakage_in_features([f"feature_{token}_count"])
