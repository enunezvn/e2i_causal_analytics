"""CohortProfiler: Tier 0 population-profiling agent for chat cohort queries.

Answers "size / define a cohort of ... patients" chat queries with REAL
per-segment prescribing counts (severity tier + line-of-therapy), reusing the
mig-105 KPI breakdown path. Companion to cohort_constructor (which materializes
the actual patient list for the ML pipeline and cannot run from chat).
"""

from .agent import CohortProfilerAgent

__all__ = ["CohortProfilerAgent"]
