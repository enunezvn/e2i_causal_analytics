"""#2031: every planted ``true_ate_by_arm`` ATE is the POPULATION-WEIGHTED mean of its
per-segment CATE map.

Measured 2026-09-12 (``docs/demos/results/2026-09-12_estimator_calibration_2031/
disproof.md`` Result 3): ``copay_support -> adherent_180d``, ``copay_support ->
low_gap_180d`` and ``psp_enrolled -> adherent_180d`` stored the UNWEIGHTED mean over
the three severity segments while every neighbour weighted by the per-row segment.
With segment shares 0.54 / 0.31 / 0.16 the unweighted value overstated the population
ATE by ~0.02 (n = 5000: 0.120 vs 0.101, 0.115 vs 0.096, 0.099 vs 0.080), so a
coverage gate against it was unstateable. This pins the weighted invariant for EVERY
arm/outcome whose ``cate_by_segment`` is keyed by the severity segment.

One 600-row frame, ~seconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.dgp.treatment_arm import SEGMENT_HIGH, SEGMENT_LOW, SEGMENT_MEDIUM
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator

SEVERITY_SEGMENTS = {SEGMENT_HIGH, SEGMENT_MEDIUM, SEGMENT_LOW}


@pytest.fixture(scope="module")
def frame():
    cfg = GeneratorConfig(
        seed=21, n_records=600, brand=Brand.REMIBRUTINIB, dgp_type=DGPType.HETEROGENEOUS
    )
    return PatientGenerator(cfg).generate()


def _severity_keyed_pairs(frame):
    """(arm, outcome, ate, cate_by_segment) for every entry keyed by severity segment.

    Entries with no ``cate_by_segment`` (the brand clinical-axis arms store only an
    ``ate``) or keyed by something other than the severity segments are skipped
    explicitly -- the invariant below is only stateable against the severity map.
    """
    pairs, skipped = [], []
    for arm, outcomes in frame.attrs["true_ate_by_arm"].items():
        for outcome, entry in outcomes.items():
            cate = entry.get("cate_by_segment")
            if not cate:
                skipped.append((arm, outcome, "no cate_by_segment"))
                continue
            if set(cate) != SEVERITY_SEGMENTS:
                skipped.append((arm, outcome, f"keys {sorted(cate)} are not severity"))
                continue
            pairs.append((arm, outcome, float(entry["ate"]), cate))
    return pairs, skipped


def test_segment_assignment_holds_the_cate_keys(frame):
    assert set(frame["segment_assignment"].unique()) == SEVERITY_SEGMENTS
    shares = frame["segment_assignment"].value_counts(normalize=True)
    # Unequal shares are what makes weighted != unweighted; the pin must not be vacuous.
    assert shares.max() - shares.min() > 0.1, shares.to_dict()


def test_every_severity_keyed_ate_is_the_population_weighted_mean(frame):
    pairs, skipped = _severity_keyed_pairs(frame)
    segment = frame["segment_assignment"].to_numpy()
    assert len(pairs) >= 11, [(p[0], p[1]) for p in pairs]
    # The three axis arms (#1321) only carry an ``ate``; nothing else may be skipped.
    assert all(reason == "no cate_by_segment" for _, _, reason in skipped), skipped
    mismatches = {}
    for arm, outcome, ate, cate in pairs:
        weighted = float(np.mean([cate[str(s)] for s in segment]))
        unweighted = float(np.mean(list(cate.values())))
        if ate != pytest.approx(weighted, abs=1e-9):
            mismatches[(arm, outcome)] = {
                "stored": round(ate, 6),
                "weighted": round(weighted, 6),
                "unweighted": round(unweighted, 6),
            }
    assert not mismatches, f"ATE is not the population-weighted CATE mean: {mismatches}"
