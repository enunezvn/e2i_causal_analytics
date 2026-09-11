"""T4 assumption check: can DoWhy's rebuilt estimate give a CI for the NC outcome per production method?"""

import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import src

assert "lane-g-2007" in src.__file__, src.__file__
from dowhy import CausalModel

from src.agents.causal_impact.nodes.refutation import (
    _SELECTOR_TO_DOWHY_METHOD,
    _reconstruction_nuisance_init_params,
)
from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator

df = PatientGenerator(
    GeneratorConfig(
        seed=21, n_records=1500, brand=Brand.REMIBRUTINIB, dgp_type=DGPType.HETEROGENEOUS
    )
).generate()
pairs = [
    ("copay_support", "treatment_initiated", ["insurance_access_score", "disease_severity"]),
    ("rep_detailing_high", "persistent_180d", ["academic_hcp", "engagement_score"]),
]
methods = sorted(set(_SELECTOR_TO_DOWHY_METHOD.values()))
print("methods:", methods)
for method in methods:
    for t, o, cc in pairs:
        d = df[[t, o] + cc].dropna().copy()
        t0 = time.time()
        try:
            m = CausalModel(data=d, treatment=t, outcome=o, common_causes=cc, effect_modifiers=cc)
            est_id = m.identify_effect(proceed_when_unidentifiable=True)
            init = {"random_state": 42}
            if "DRLearner" not in method:
                init["discrete_treatment"] = True
            init.update(_reconstruction_nuisance_init_params(method, discrete_treatment=True))
            est = m.estimate_effect(
                est_id,
                method_name=method,
                method_params={"init_params": init, "fit_params": {}},
                test_significance=False,
            )
            t1 = time.time()
            ci = est.get_confidence_intervals()
            t2 = time.time()
            ci = np.asarray(ci, dtype=float).ravel()
            print(
                f"{method:38} {t:>18}->{o:<20} n={len(d)} value={float(est.value):+.4f} ci={ci.round(4).tolist()} fit={t1 - t0:.1f}s ci={t2 - t1:.1f}s"
            )
        except Exception as e:
            print(f"{method:38} {t:>18}->{o:<20} FAILED {type(e).__name__}: {str(e)[:160]}")
