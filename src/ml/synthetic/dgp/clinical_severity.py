"""UAS7 follows the latent disease severity, so the severity tier means what it says.

WHY: until 2026-09-16 Remibrutinib's ``urticaria_severity_uas7`` was drawn uniformly
on 16..42 independently of ``disease_severity``, while ``segment_assignment``
(low/medium/high_severity) is cut from ``disease_severity``. Live, the mean UAS7 was
28.9 / 28.9 / 29.2 across high / medium / low tiers (corr ~0.01): a "high severity"
CSU patient was no more likely to have uncontrolled urticaria than a "low severity"
one, and the chat served breakdowns by a tier the clinical score contradicted
(session_1789548670222_fcscf3u).

HOW: a Gaussian copula that REUSES the existing ``integers(16, 43)`` draw as its
noise term, so the generator consumes exactly the same RNG stream (every other
column is byte-identical) and UAS7 keeps its uniform 16..42 marginal, so
P(UAS7 >= 28), the uncontrolled-CSU axis prevalence, stays ~15/27::

    u = (draw - 16 + 0.5) / 27
    z = rho * (severity - 5) / 2 + sqrt(1 - rho**2) * PHI^-1(u)
    uas7 = 16 + floor(27 * PHI(z))            (clipped to 16..42)

At rho = 0 this is the identity (``floor(u * 27) + 16 == draw``).

WHY RHO = 0.4 (measured 2026-09-16, docs/demos/results/2026-09-16_remi_share_axis/):
the planted UAS7 >= 28 -> persistent_180d effect (+0.152) was designed assuming
independence; correlating the axis with severity makes severity a real confounder.
On the live Remibrutinib cohort (n=8,863) the CausalForestDML estimate adjusted for
severity stays on the planted value for rho 0 / 0.4 / 0.6 (+0.159 / +0.154 / +0.150),
while the naive contrast collapses (+0.154 / +0.072 / +0.031), and at rho 0.8 the high
tier loses its controls (P(UAS7 >= 28) = 0.985) and the interval explodes. On the
faithful gold-standard persistence path (seed 42, n=20000) Remibrutinib AUC is 0.7647 /
0.7556 / 0.7539 / 0.7545 at rho 0 / 0.4 / 0.5 / 0.6: the cost is flat past 0.4, so 0.4
buys the clinical gradient (mean UAS7 33.7 / 29.7 / 25.5 by tier) with the widest
positivity margin (high tier 78% uncontrolled, vs 89% at 0.6).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

UAS7_MIN = 16
UAS7_MAX = 42
_UAS7_LEVELS = UAS7_MAX - UAS7_MIN + 1  # 27

#: Latent disease_severity is N(5, 2) clipped to [0, 10] (_generate_confounders).
_SEVERITY_MEAN = 5.0
_SEVERITY_SD = 2.0

UAS7_SEVERITY_RHO = 0.4


def uas7_from_severity(
    draw: np.ndarray | int,
    disease_severity: np.ndarray | float,
    rho: float = UAS7_SEVERITY_RHO,
) -> np.ndarray:
    """Map the uniform ``integers(16, 43)`` draw to a UAS7 that rises with severity.

    Vectorised; returns an int64 array shaped like the broadcast inputs.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must be in [0, 1), got {rho}")
    d = np.asarray(draw, dtype=float)
    sev = np.asarray(disease_severity, dtype=float)
    if not np.all(np.isfinite(d)) or np.any((d < UAS7_MIN) | (d > UAS7_MAX)):
        raise ValueError("UAS7 draw outside 16..42")
    if not np.all(np.isfinite(sev)):
        raise ValueError("disease_severity must be finite")
    u = (d - UAS7_MIN + 0.5) / _UAS7_LEVELS
    z = rho * (sev - _SEVERITY_MEAN) / _SEVERITY_SD + np.sqrt(1.0 - rho**2) * norm.ppf(u)
    out = UAS7_MIN + np.floor(_UAS7_LEVELS * norm.cdf(z))
    result: np.ndarray = np.clip(out, UAS7_MIN, UAS7_MAX).astype(np.int64)
    return result
