"""E2I Security utilities.

This package owns deterministic security tooling — pattern scanners,
provenance enforcement contracts, and audit harnesses — that runs
OUTSIDE the request path (offline audit, migrations, hardening tests).

Modules:
    phi_scanner: Regex-based PHI/PII detection for crystal narrative
        + LLM-prompt audit. See ``scripts/audit_phi_in_crystal_narratives.py``.
"""
