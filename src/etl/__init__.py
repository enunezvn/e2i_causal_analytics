"""ETL package for E2I Causal Analytics.

Hosts batch-style data-transformation jobs that read from canonical
PostgreSQL tables (Supabase) and write back into rollup tables consumed by
Feast feature views and downstream agents. Distinct from ``src.tasks`` which
holds the lighter Celery glue / API-side coordinators.

Modules in this package register their own Celery tasks; see
``src.workers.celery_app`` for the autodiscovery and beat-schedule wiring.
"""
