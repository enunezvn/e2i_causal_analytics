"""Fixtures for the Lane E feature-role panel tests.

The frame is synthetic but named after the ``optum`` manifest so the REAL Layer 1
(contracts) and the COMMITTED Layer 2 cache (``data/kg_cache/1cdaa038__96bfd2e0.json``,
target omalizumab) are exercised; Layer 3 is the real adversarial probe; Layer 4 is
a ``DummyLM`` (no paid call, ever — spec §6).
"""

from __future__ import annotations

from typing import Any

import dspy
import numpy as np
import pandas as pd
import pytest

TREATMENT = "treatment_arm"
OUTCOME = "initiated_biologic_180d"


def make_panel_frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """A causal-shaped frame: T, Y and five covariates of known character.

    * ``age_at_index`` — optum contract, enrollment (declared safe) AND attested
      (``causal_structure`` derives ``confounder``): under the causal profile the
      structural decider decides it and Layer 4 is never called for it, by the
      node's design. Its Layer-3 z lands in the pre-joint ``moderate`` band
      (0.26*y + 0.5*noise → z≈4.97 under the declared-safe HBLP multiplier).
    * ``moderate_probe`` — no contract, no attestation; 0.20*y + 0.5*noise →
      z≈4.09 with this exact draw order (drawn LAST so the other columns are
      unchanged): pre-joint ``moderate`` (Layer 4's trigger), |delta_AUC| 0.068 <
      the 0.10 floor so the final severity is joint-clamped to ``info``.
    * ``dx_total_csu`` — optum contract, index_date; the committed cache carries
      a ``leak_drug_treats_disease`` edge for it (omalizumab treats urticaria).
      Pure noise here so Layer 3 says info and Layer 2 is the only voice.
    * ``treatment_initiated`` — optum contract, POST-index → Layer 1 veto.
    * ``leak_probe`` — no contract; a near-copy of Y → Layer 3 high, no immunity.
    * ``noise_feature`` — no contract; pure noise → info.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            TREATMENT: t,
            OUTCOME: y,
            "age_at_index": 0.26 * y + 0.5 * rng.standard_normal(n),
            "dx_total_csu": rng.poisson(2.0, n).astype(float),
            "treatment_initiated": (y * 0.7 + rng.random(n) > 0.5).astype(int),
            "leak_probe": y + 0.01 * rng.standard_normal(n),
            "noise_feature": rng.standard_normal(n),
            "moderate_probe": 0.20 * y + 0.5 * rng.standard_normal(n),
        }
    )


@pytest.fixture
def stub_lm() -> Any:
    """A DummyLM answering ``confounder`` for every Layer-4 call; restored after."""
    from dspy.utils.dummies import DummyLM

    prior = getattr(dspy.settings, "lm", None)
    lm = DummyLM(
        [
            {
                "reasoning": "stub",
                "causal_role": "confounder",
                "mechanism": "stub mechanism: baseline severity drives both T and Y",
                "recommended_remediation": "keep_with_caveat",
            }
        ]
    )
    dspy.configure(lm=lm)
    yield lm
    try:
        if prior is None:
            dspy.settings.configure(lm=None)
        else:
            dspy.configure(lm=prior)
    except Exception:  # pragma: no cover
        pass


@pytest.fixture(autouse=True)
def _no_paid_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Belt and braces: no provider key can reach a real LM from these tests,
    and the voter stays audit-only (``ADAPTIVE_LAYER4_LLM_DECIDES`` unset)."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AZURE_API_KEY", "AZURE_OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("ADAPTIVE_LAYER4_LLM_DECIDES", raising=False)
