#!/usr/bin/env python3
"""
Train + persist a digital-twin model (#705 H4).

A reachable, low-risk ops trigger for the twin train/persist/load pipeline that
does NOT depend on the dark ``worker_heavy`` service. Run it inside the API
container (which has the MLflow + Supabase env) to create a real, loadable model
so ``POST /digital-twin/simulate`` stops returning 503:

    docker exec -i e2i_api python scripts/train_twin_model.py \
        --twin-type hcp --brand Remibrutinib --synthetic

With real data instead of synthetic:

    docker exec -i e2i_api python scripts/train_twin_model.py \
        --twin-type hcp --brand Kisqali \
        --data-source /data/hcp_cohort.parquet --target-column prescribing_change

This is a thin CLI over ``src.digital_twin.training_job.train_and_persist_twin``
(the same code path the ``src.tasks.train_twin_model`` Celery task uses); all the
real logic + tests live there.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def _main(args: argparse.Namespace) -> int:
    from src.digital_twin.models.twin_models import Brand, TwinType
    from src.digital_twin.training_job import train_and_persist_twin
    from src.digital_twin.twin_repository import TwinRepository
    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    repo = TwinRepository(supabase_client=client)

    result = await train_and_persist_twin(
        twin_type=TwinType(args.twin_type),
        brand=Brand(args.brand),
        repo=repo,
        data_source=args.data_source,
        target_column=args.target_column,
        algorithm=args.algorithm,
        synthetic=args.synthetic,
        n_rows=args.n_rows,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train + persist a digital-twin model.")
    parser.add_argument("--twin-type", required=True, choices=["hcp", "patient", "territory"])
    parser.add_argument("--brand", required=True, help="e.g. Remibrutinib, Fabhalta, Kisqali")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="train on a synthetic frame")
    src.add_argument("--data-source", help="path to a .csv/.parquet training file (RWD)")
    parser.add_argument("--target-column", default="outcome")
    parser.add_argument(
        "--algorithm", default="random_forest", choices=["random_forest", "gradient_boosting"]
    )
    parser.add_argument("--n-rows", type=int, default=2000, help="rows for synthetic frames")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
