"""CohortProfiler: Tier 0 population-profiling agent for chat cohort queries.

Answers "size / define a cohort of ... patients" chat queries with REAL
per-segment prescribing counts (severity tier + line-of-therapy), reusing the
mig-105 KPI breakdown path. Companion to cohort_constructor (which materializes
the actual patient list for the ML pipeline and cannot run from chat).

Since #1356 (ratified `extend:cohort_profiler` ruling, 2026-07-29) the ask's
parameters BIND into the profile: brand + servable inclusion criteria on the
patient path (with an honest accounting of criteria the data model cannot
serve), and HCP-entity cohorts with quantitative KPI thresholds over an
explicit time window ("HCPs who prescribed >50 TRx last quarter").
"""

from .agent import CohortProfilerAgent
from .ask import CohortAsk, parse_cohort_ask

__all__ = ["CohortAsk", "CohortProfilerAgent", "parse_cohort_ask"]
