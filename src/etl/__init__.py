"""ETL package for E2I Causal Analytics.

Hosts batch-style data-transformation jobs that read from canonical
PostgreSQL tables (Supabase) and write back into rollup tables consumed by
Feast feature views and downstream agents. Distinct from ``src.tasks`` which
holds the lighter Celery glue / API-side coordinators.

Modules in this package register their own Celery tasks; the eager imports
below mirror the ``src.tasks.__init__`` pattern so ``celery_app.autodiscover_tasks``
in ``src.workers.celery_app`` picks them up at worker startup. (Celery's default
``related_name='tasks'`` lookup expects ``<package>.tasks``; this package uses
one module per ETL instead, so eager imports are how the @celery_app.task
decorators get a chance to register.)
"""

# Import ETL modules so their @celery_app.task decorators register on worker
# startup. Re-exports are intentional: callers that need the entrypoints can
# import them from ``src.etl`` directly.
from src.etl.business_metrics_per_hcp_etl import run_per_hcp_rollup
from src.etl.patient_adherence_etl import run_patient_adherence_rollup
from src.etl.territory_metrics_etl import run_territory_rollup

__all__ = [
    "run_per_hcp_rollup",
    "run_patient_adherence_rollup",
    "run_territory_rollup",
]
