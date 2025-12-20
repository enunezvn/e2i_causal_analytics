"""
Query optimization with domain context.

Expands queries with domain knowledge:
- Add synonyms from domain_vocabulary.yaml
- Include related KPIs
- Add temporal context
"""

from typing import Dict, List, Optional


class QueryOptimizer:
    """
    Expand queries with domain knowledge for better retrieval.

    Example:
        "TRx for Kisqali" → "Total prescriptions TRx Kisqali breast cancer HR+ conversion"
    """

    def __init__(self, vocabulary_path: Optional[str] = None):
        """
        Initialize with domain vocabulary.

        Args:
            vocabulary_path: Path to domain_vocabulary.yaml
        """
        self.vocabulary_path = vocabulary_path
        self._synonyms: Dict[str, List[str]] = {}
        self._kpi_relations: Dict[str, List[str]] = {}
        self._load_vocabulary()

    def _load_vocabulary(self) -> None:
        """Load domain vocabulary from YAML file."""
        # Default pharmaceutical commercial domain synonyms
        self._synonyms = {
            "trx": ["total prescriptions", "total rx", "prescription volume"],
            "nrx": ["new prescriptions", "new rx", "new starts"],
            "kisqali": ["ribociclib", "hr+ breast cancer", "her2-"],
            "fabhalta": ["iptacopan", "pnh", "paroxysmal nocturnal hemoglobinuria"],
            "remibrutinib": ["csu", "chronic spontaneous urticaria"],
            "hcp": ["healthcare provider", "physician", "prescriber"],
            "conversion": ["conversion rate", "rx conversion", "switch rate"],
            "market share": ["share of voice", "sov", "competitive share"],
        }

        self._kpi_relations = {
            "trx": ["nrx", "conversion_rate", "market_share"],
            "nrx": ["trx", "new_patient_starts", "time_to_first_rx"],
            "conversion_rate": ["trx", "nrx", "abandonment_rate"],
        }

    def expand(self, query) -> str:
        """
        Expand query with domain knowledge.

        Args:
            query: ParsedQuery or string

        Returns:
            Expanded query string
        """
        query_text = query.text if hasattr(query, 'text') else str(query)
        query_lower = query_text.lower()

        expansions = []

        # Add synonyms for recognized terms
        for term, synonyms in self._synonyms.items():
            if term in query_lower:
                expansions.extend(synonyms[:2])  # Limit expansions

        # Add related KPIs
        for kpi, related in self._kpi_relations.items():
            if kpi in query_lower:
                expansions.extend(related[:2])

        # Combine original query with expansions
        if expansions:
            return f"{query_text} {' '.join(expansions)}"

        return query_text

    def add_temporal_context(self, query: str, time_range: Optional[str] = None) -> str:
        """
        Add temporal context to query.

        Args:
            query: Query text
            time_range: Optional time range (e.g., "last 30 days")

        Returns:
            Query with temporal context
        """
        if time_range:
            return f"{query} {time_range}"
        return query
