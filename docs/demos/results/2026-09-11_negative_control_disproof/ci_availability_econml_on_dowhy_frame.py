import time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import src; assert "lane-g-2007" in src.__file__, src.__file__
from dowhy import CausalModel
from src.agents.causal_impact.nodes.refutation import _reconstruction_nuisance_init_params
from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator
df = PatientGenerator(GeneratorConfig(seed=21, n_records=1500, brand=Brand.REMIBRUTINIB, dgp_type=DGPType.HETEROGENEOUS)).generate()
print("dtypes:", {c: str(df[c].dtype) for c in ["academic_hcp", "engagement_score", "insurance_access_score", "disease_severity", "rep_detailing_high", "persistent_180d"]})
pairs = [("copay_support", "treatment_initiated", ["insurance_access_score", "disease_severity"]),
         ("rep_detailing_high", "persistent_180d", ["academic_hcp", "engagement_score"])]
for method in ["backdoor.econml.dml.LinearDML", "backdoor.econml.dr.DRLearner", "backdoor.econml.dml.CausalForestDML"]:
    for t, o, cc in pairs:
        d = df[[t, o] + cc].dropna().copy()
        m = CausalModel(data=d, treatment=t, outcome=o, common_causes=cc, effect_modifiers=cc)
        est_id = m.identify_effect(proceed_when_unidentifiable=True)
        init = {"random_state": 42}
        if "DRLearner" not in method:
            init["discrete_treatment"] = True
        init.update(_reconstruction_nuisance_init_params(method, discrete_treatment=True))
        est = m.estimate_effect(est_id, method_name=method, method_params={"init_params": init, "fit_params": {}}, test_significance=False)
        inner = est.estimator; econ = inner.estimator
        attrs = [a for a in dir(inner) if "modif" in a.lower()]
        Xd = getattr(inner, "_effect_modifiers", None)
        info = f"type={type(Xd).__name__}" + (f" shape={getattr(Xd,'shape',None)} cols={list(Xd.columns) if hasattr(Xd,'columns') else None} dtypes={[str(x) for x in Xd.dtypes] if hasattr(Xd,'dtypes') else None}" if Xd is not None else "")
        t0 = time.time()
        try:
            inf = econ.ate_inference(np.asarray(Xd, dtype=float) if Xd is not None else None)
            lo, hi = (float(v) for v in inf.conf_int_mean())
            print(f"{method.split('.')[-1]:15} {t}->{o}: dowhy={float(est.value):+.4f} econml_on_dowhy_X={float(inf.mean_point):+.4f} ci=[{lo:+.4f},{hi:+.4f}] {time.time()-t0:.2f}s | {info} | attrs={attrs}")
        except Exception as e:
            print(f"{method.split('.')[-1]:15} {t}->{o}: FAILED {type(e).__name__}: {str(e)[:120]} | {info} | attrs={attrs}")
