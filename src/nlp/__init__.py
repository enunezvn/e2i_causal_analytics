"""
NLP module for E2I Causal Analytics.

Provides natural language processing capabilities:
- Typo correction with fastText subword embeddings
- Domain vocabulary training
"""

from src.nlp.typo_handler import (
    TypoHandler,
    CorrectionResult,
    correct_term,
    correct_query,
    get_typo_handler,
    CANONICAL_VOCABULARY,
    ABBREVIATION_EXPANSIONS,
)

__all__ = [
    "TypoHandler",
    "CorrectionResult",
    "correct_term",
    "correct_query",
    "get_typo_handler",
    "CANONICAL_VOCABULARY",
    "ABBREVIATION_EXPANSIONS",
]
