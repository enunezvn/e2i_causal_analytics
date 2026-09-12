"""T4 step 8: wall-time of the node's negative-control fit + interval (the REAL
`_fit_negative_control`, LinearDML with production RF nuisances) on the seed-21
HETEROGENEOUS Remibrutinib frame at n = 1500, for the three registry pairs.
Run: PYTHONPATH=<worktree> .venv/bin/python docs/demos/results/2026-09-11_negative_control_disproof/nc_fit_timing.py
"""

import asyncio
import time
import warnings

warnings.filterwarnings("ignore")
import src  # noqa: E402

assert "lane-g-2007" in src.__file__, src.__file__
from src.agents.causal_impact.nodes.refutation import (  # noqa: E402
    _build_dowhy_estimate,
    _fit_negative_control,
    _negative_control_interval,
)
from src.ml.synthetic.config import Brand, DGPType  # noqa: E402
from src.ml.synthetic.dgp.treatment_arm import ARM_REGISTRY  # noqa: E402
from src.ml.synthetic.generators import GeneratorConfig, PatientGenerator  # noqa: E402

PAIRS = [
    ("copay_support", "treatment_initiated"),
    ("psp_enrolled", "treatment_initiated"),
    ("rep_detailing_high", "persistent_180d"),
]


async def main() -> None:
    df = PatientGenerator(
        GeneratorConfig(
            seed=21, n_records=1500, brand=Brand.REMIBRUTINIB, dgp_type=DGPType.HETEROGENEOUS
        )
    ).generate()
    est = {"method": "LinearDML", "selected_estimator": "linear_dml"}
    for arm, nc in PAIRS:
        covs = list(ARM_REGISTRY[arm].confounders)
        # the arm's planted (primary) outcome rides in the frame like the live passthrough
        primary = next(o for o in df.attrs["true_ate_by_arm"][arm] if o != nc)
        frame = df[[arm, primary] + covs].dropna().copy()
        nc_df = df[[nc]].loc[frame.index]
        # (1) the pieces: build + interval, timed separately
        t0 = time.perf_counter()
        _, _, estimate, method = _build_dowhy_estimate(
            data=frame.join(nc_df),
            treatment=arm,
            outcome=nc,
            common_causes=covs,
            estimation_result=dict(est),
        )
        t1 = time.perf_counter()
        interval = _negative_control_interval(estimate, method)
        t2 = time.perf_counter()
        # (2) the node function end to end (alignment + pooled fit)
        t3 = time.perf_counter()
        result, reason = await _fit_negative_control(
            refutation_data=frame,
            negative_control_data=nc_df,
            treatment=arm,
            nc_outcome=nc,
            common_causes=covs,
            estimation_result=dict(est),
            deadline=None,
        )
        t4 = time.perf_counter()
        print(
            f"{arm:>18} -> {nc:<20} n={len(frame)} covs={covs} | build {t1 - t0:.2f}s interval {t2 - t1:.3f}s"
            f" | _fit_negative_control {t4 - t3:.2f}s -> {result if result else reason}"
            f" | value={float(estimate.value):+.5f} interval_point={interval[0] if interval else None}"
        )


if __name__ == "__main__":
    asyncio.run(main())
