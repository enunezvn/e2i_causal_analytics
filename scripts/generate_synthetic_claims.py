#!/usr/bin/env python3
"""Generate the 6 raw CSU/Remibrutinib-only claim parquets (Shard 10 P1.3).

Emits exactly the files ``scripts/convert_optum_rwd.py`` consumes —
``{demographics, medication, procedure, lab, inpatientdata, provider}.parquet``
— scoped to CSU/Remibrutinib so the EXISTING converter -> tier-0 pipeline
recovers an embedded DGP at an honest ``val_AUC`` in [0.62, 0.68].

This script is DB-independent: it only calls ``DataFrame.to_parquet``; there is
no DB write. Each frame is streamed to disk independently (droplet OOM
discipline — never materialise the full join in memory).

Usage::

    python scripts/generate_synthetic_claims.py --out /tmp/syn_raw --n 2000 --seed 42
    python scripts/convert_optum_rwd.py --input /tmp/syn_raw --output /tmp/syn_out --cohort all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make ``src`` importable when run as a bare script (the test imports the
# module directly, so this only matters for the CLI entrypoint).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ml.synthetic.claims.claims_events import (  # noqa: E402
    emit_lab,
    emit_medication,
    emit_procedure,
    emit_provider,
)
from src.ml.synthetic.claims.config import ClaimsDGPConfig  # noqa: E402
from src.ml.synthetic.claims.hcp_network import make_npi_assigner  # noqa: E402
from src.ml.synthetic.claims.patient_state import (  # noqa: E402
    emit_inpatient,
    generate_patients,
)

# Latent helper columns that drive the DGP but must NOT reach demographics.parquet
# (the converter ignores them and they would mislead a downstream reader).
_LATENT_COLS = (
    "severity",
    "tx_burden",
    "response_propensity",
    "adherence_propensity",
    "claim_index",
)

# Demographics columns the converter actually reads (drop everything else).
_DEMO_OUT_COLS = (
    "patid",
    "eligeff",
    "eligend",
    "diagcode",
    "age",
    "gdr_cd",
    "zipcode_5",
    "bus",
    "product",
    "health_exch",
    "lis_dual",
    "continuous_enrollment",
)


def generate_to(out_dir, n_patients: int = 2000, seed: int = 42, **cfg_kwargs) -> None:
    """Generate and write the 6 raw claim parquets to ``out_dir``.

    A single seeded RNG drives the whole DGP so the output is deterministic for
    a given (n_patients, seed). med/proc NPI assignment shares a per-patient HCP
    map so the converter's med.npi ∪ proc.npi shared-patient graph is coherent.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ClaimsDGPConfig(n_patients=n_patients, seed=seed, **cfg_kwargs)
    rng = np.random.default_rng(seed)

    # Patient latent-state + demographics.
    pats = generate_patients(rng, cfg)

    # Exogenous HCP network -> per-patient NPI assigner (P1b). Built BEFORE any
    # adoption so centrality is exogenous (no circularity).
    npi_for = make_npi_assigner(rng, pats, cfg)

    # Inpatient (claim-dated index anchor + comorbidities).
    inpatient = emit_inpatient(rng, pats, cfg)
    _write(inpatient, out_dir / "inpatientdata.parquet")
    del inpatient

    # Medication (prior therapy + biologic initiation/sequence).
    med = emit_medication(rng, pats, cfg, npi_for=npi_for)
    _write(med, out_dir / "medication.parquet")

    # Procedure (office visits; shares the HCP graph).
    proc = emit_procedure(rng, pats, cfg, npi_for=npi_for)
    _write(proc, out_dir / "procedure.parquet")

    # Provider (npi -> taxonomy1; derived from the med/proc NPIs actually used).
    prov = emit_provider(rng, med, proc, cfg)
    _write(prov, out_dir / "provider.parquet")
    del med, proc, prov

    # Lab (claims-plausible LOINC results only).
    lab = emit_lab(rng, pats, cfg)
    _write(lab, out_dir / "lab.parquet")
    del lab

    # Demographics — drop latent + phantom columns last.
    demo = pats.drop(columns=[c for c in _LATENT_COLS if c in pats.columns])
    demo = demo[[c for c in _DEMO_OUT_COLS if c in demo.columns]]
    _write(demo, out_dir / "demographics.parquet")


def _write(df, path: Path) -> None:
    df.to_parquet(path, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate 6 raw CSU/Remibrutinib synthetic claim parquets."
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--n", type=int, default=2000, help="Number of patients")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--signal-scale",
        type=float,
        default=1.0,
        help="Honest-band tuning knob (scales latent effect on initiation/adherence).",
    )
    args = parser.parse_args(argv)
    generate_to(
        out_dir=args.out,
        n_patients=args.n,
        seed=args.seed,
        signal_scale=args.signal_scale,
    )
    print(f"Wrote 6 claim parquets to {args.out} (n={args.n}, seed={args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
