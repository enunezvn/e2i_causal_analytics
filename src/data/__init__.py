"""Data layer utilities.

This package contains foundational data primitives that span multiple agents
and converters. Modules here MUST be deterministic and side-effect-free; they
are loaded by tests, the CSU/Optum converters, and the ML pipeline.
"""
