import sys, time, json, logging
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, "docs/demos/results/2026-09-22_optum_biologic_persistence_cert")
logging.basicConfig(level=logging.WARNING)
import src

assert ".worktrees/real-data-causal" in src.__file__
from preflight_agent import TREATMENT, build_frame

t0 = time.perf_counter()
frame, cov = build_frame("persistent_at_180d_g28")
print("frame", frame.shape, "build_s", round(time.perf_counter() - t0, 1), flush=True)
from dowhy import CausalModel

model = CausalModel(
    data=frame,
    treatment=TREATMENT,
    outcome="persistent_at_180d_g28",
    common_causes=list(cov),
    effect_modifiers=list(cov),
)
print(
    "graph nodes",
    len(model._graph._graph.nodes),
    "has U:",
    [n for n in model._graph._graph.nodes if "nobserved" in n],
    flush=True,
)
t0 = time.perf_counter()
est = model.identify_effect(proceed_when_unidentifiable=True, optimize_backdoor=True)
dt = time.perf_counter() - t0
bd = {k: len(v) for k, v in (est.backdoor_variables or {}).items()}
ga = {k: len(v) for k, v in (est.general_adjustment_variables or {}).items()}
print(
    json.dumps(
        {
            "identify_optimized_s": round(dt, 1),
            "backdoor_sets": bd,
            "general_adjustment_sets": ga,
            "default_backdoor_id": est.default_backdoor_id,
            "default_adjustment_set_id": getattr(est, "default_adjustment_set_id", None),
            "estimand_type": str(est.estimand_type),
            "identifier_method": getattr(est, "identifier_method", None),
        },
        indent=1,
    ),
    flush=True,
)
print(
    "adjustment set == all common causes:",
    set(est.get_adjustment_set() or []) == set(cov),
    flush=True,
)
