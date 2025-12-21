# src/e2i/agents/orchestrator/classifier/dependency_detector.py
"""
Stage 3: Detect dependencies between sub-questions.

This module analyzes multi-part queries to detect data dependencies
between sub-questions. Dependencies indicate that later parts of a
query require outputs from earlier parts, which determines whether
parallel execution is possible or Tool Composer is needed.

Dependency Types:
- REFERENCE_CHAIN: Pronouns like "that", "those" referring back
- CONDITIONAL: "if X then Y" logical structure
- LOGICAL_SEQUENCE: Natural ordering (cause → effect → intervention)
- ENTITY_TRANSFORMATION: Entity filtered/transformed by earlier step
"""

import re
from typing import Optional

from anthropic import Anthropic

from .prompts import DEPENDENCY_DETECTION_PROMPT
from .schemas import (
    Dependency,
    DependencyAnalysis,
    DependencyType,
    Domain,
    DomainMapping,
    ExtractedFeatures,
    SubQuestion,
)


class DependencyDetector:
    """
    Detects dependencies between sub-questions in multi-part queries.
    Uses both rule-based patterns and LLM for complex cases.
    """

    # Reference words that indicate dependency
    REFERENCE_PRONOUNS = {"that", "those", "this", "these", "it", "them"}
    REFERENCE_PHRASES = {
        "the result",
        "the finding",
        "the effect",
        "the impact",
        "the segment",
        "the group",
        "the responders",
    }

    CONDITIONAL_PATTERNS = [
        r"if\s+.+\s*,?\s*(then|what)",
        r"assuming\s+.+\s*,",
        r"given\s+(that|the)\s+",
        r"based on\s+(that|the|what)",
    ]

    SEQUENCE_PATTERNS = [
        r"(first|then|after|next|finally)",
        r"once\s+.+\s*,",
        r"after\s+determining",
    ]

    def __init__(self, llm_client: Optional[Anthropic] = None):
        """
        Initialize detector.

        Args:
            llm_client: Anthropic client for complex dependency detection
        """
        self.llm_client = llm_client

    # =========================================================================
    # MAIN DETECTION METHOD
    # =========================================================================

    async def detect(
        self,
        query: str,
        features: ExtractedFeatures,
        domain_mapping: DomainMapping,
        use_llm: bool = False,
    ) -> DependencyAnalysis:
        """
        Detect dependencies in query.

        Args:
            query: Original query
            features: Extracted features
            domain_mapping: Domain mapping results
            use_llm: Whether to use LLM for complex detection

        Returns:
            DependencyAnalysis with sub-questions and dependencies
        """
        # Step 1: Decompose query into sub-questions
        sub_questions = self._decompose_query(query, domain_mapping)

        if len(sub_questions) <= 1:
            # Single question, no dependencies possible
            return DependencyAnalysis(
                sub_questions=sub_questions,
                dependencies=[],
                has_dependencies=False,
                is_parallelizable=True,
                dependency_depth=0,
            )

        # Step 2: Detect dependencies (rule-based first)
        dependencies = self._detect_rule_based(query, sub_questions)

        # Step 3: Use LLM for complex cases if enabled
        if (
            use_llm
            and self.llm_client
            and self._needs_llm_analysis(query, sub_questions, dependencies)
        ):
            llm_dependencies = await self._detect_with_llm(query, sub_questions)
            # Merge LLM findings with rule-based (prefer LLM for conflicts)
            dependencies = self._merge_dependencies(dependencies, llm_dependencies)

        # Step 4: Calculate dependency depth
        depth = self._calculate_depth(sub_questions, dependencies)

        return DependencyAnalysis(
            sub_questions=sub_questions,
            dependencies=dependencies,
            has_dependencies=len(dependencies) > 0,
            is_parallelizable=len(dependencies) == 0,
            dependency_depth=depth,
        )

    # =========================================================================
    # QUERY DECOMPOSITION
    # =========================================================================

    def _decompose_query(self, query: str, domain_mapping: DomainMapping) -> list[SubQuestion]:
        """
        Decompose query into sub-questions.
        Simple heuristic: split on ", and" or "?" patterns.
        """
        # Common split patterns
        patterns = [
            r"[,;]\s*and\s+(?=what|which|how|why|who|where)",
            r"\?\s*(?:and\s+)?(?=what|which|how|why|who|where)",
            r"[,;]\s*(?=what would|what if)",
        ]

        # Try to split
        parts = [query]
        for pattern in patterns:
            new_parts = []
            for part in parts:
                splits = re.split(pattern, part, flags=re.IGNORECASE)
                new_parts.extend([s.strip() for s in splits if s.strip()])
            parts = new_parts

        # Create SubQuestion objects
        sub_questions = []
        for i, part in enumerate(parts):
            # Simple domain assignment based on keywords
            # (In production, would use domain mapper on each part)
            domains = self._infer_domains_for_part(part)

            sub_questions.append(
                SubQuestion(
                    id=f"Q{i + 1}",
                    text=part,
                    domains=domains,
                    primary_domain=domains[0] if domains else Domain.EXPLANATION,
                )
            )

        return sub_questions

    def _infer_domains_for_part(self, text: str) -> list[Domain]:
        """Infer domains for a sub-question text."""
        text_lower = text.lower()
        domains = []

        if any(kw in text_lower for kw in ["impact", "effect", "response"]):
            domains.append(Domain.CAUSAL_ANALYSIS)
        if any(kw in text_lower for kw in ["which", "segment", "best"]):
            domains.append(Domain.HETEROGENEITY)
        if any(kw in text_lower for kw in ["would", "if", "predict", "extend"]):
            domains.append(Domain.PREDICTION)
        if any(kw in text_lower for kw in ["gap", "opportunity", "region"]):
            domains.append(Domain.GAP_ANALYSIS)
        if any(kw in text_lower for kw in ["test", "experiment", "design"]):
            domains.append(Domain.EXPERIMENTATION)

        return domains if domains else [Domain.EXPLANATION]

    # =========================================================================
    # RULE-BASED DETECTION
    # =========================================================================

    def _detect_rule_based(self, query: str, sub_questions: list[SubQuestion]) -> list[Dependency]:
        """
        Detect dependencies using rule-based patterns.
        """
        dependencies = []
        query.lower()

        for i, sq in enumerate(sub_questions[1:], start=1):
            text_lower = sq.text.lower()
            prev_sq = sub_questions[i - 1]

            # Check for reference pronouns
            if any(pron in text_lower for pron in self.REFERENCE_PRONOUNS):
                dependencies.append(
                    Dependency(
                        **{
                            "from": prev_sq.id,
                            "to": sq.id,
                        },
                        dependency_type=DependencyType.REFERENCE_CHAIN,
                        reason=f"Pronoun reference in '{sq.text[:50]}...'",
                    )
                )
                continue

            # Check for reference phrases
            if any(phrase in text_lower for phrase in self.REFERENCE_PHRASES):
                dependencies.append(
                    Dependency(
                        **{
                            "from": prev_sq.id,
                            "to": sq.id,
                        },
                        dependency_type=DependencyType.REFERENCE_CHAIN,
                        reason=f"Reference phrase in '{sq.text[:50]}...'",
                    )
                )
                continue

            # Check for conditional patterns
            for pattern in self.CONDITIONAL_PATTERNS:
                if re.search(pattern, text_lower):
                    dependencies.append(
                        Dependency(
                            **{
                                "from": prev_sq.id,
                                "to": sq.id,
                            },
                            dependency_type=DependencyType.CONDITIONAL,
                            reason="Conditional structure detected",
                        )
                    )
                    break

        return dependencies

    # =========================================================================
    # LLM-BASED DETECTION
    # =========================================================================

    def _needs_llm_analysis(
        self,
        query: str,
        sub_questions: list[SubQuestion],
        rule_based_deps: list[Dependency],
    ) -> bool:
        """Determine if LLM analysis is needed."""
        # Use LLM if:
        # 1. Multiple sub-questions but no dependencies found
        # 2. Query is complex (high word count, multiple domains)
        # 3. Implicit dependencies likely (semantic rather than syntactic)

        if len(sub_questions) > 2 and len(rule_based_deps) == 0:
            return True
        if len(query.split()) > 30:
            return True
        return False

    async def _detect_with_llm(
        self, query: str, sub_questions: list[SubQuestion]
    ) -> list[Dependency]:
        """Use LLM to detect semantic dependencies."""
        if not self.llm_client:
            return []

        # Format sub-questions for prompt
        sq_text = "\n".join([f"{sq.id}: {sq.text}" for sq in sub_questions])

        prompt = DEPENDENCY_DETECTION_PROMPT.format(
            query=query,
            sub_questions=sq_text,
        )

        try:
            self.llm_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse LLM response (expecting JSON)
            # Implementation would parse the response and create Dependency objects
            # For now, return empty list as placeholder
            return []

        except Exception as e:
            # Log error and fall back to rule-based only
            print(f"LLM dependency detection failed: {e}")
            return []

    def _merge_dependencies(
        self,
        rule_based: list[Dependency],
        llm_based: list[Dependency],
    ) -> list[Dependency]:
        """Merge dependencies, preferring LLM for conflicts."""
        # Simple merge: combine unique dependencies
        seen = set()
        merged = []

        for dep in llm_based + rule_based:
            key = (dep.from_id, dep.to_id)
            if key not in seen:
                seen.add(key)
                merged.append(dep)

        return merged

    def _calculate_depth(
        self,
        sub_questions: list[SubQuestion],
        dependencies: list[Dependency],
    ) -> int:
        """Calculate maximum dependency chain depth."""
        if not dependencies:
            return 0

        # Build adjacency list
        adj: dict[str, list[str]] = {sq.id: [] for sq in sub_questions}
        for dep in dependencies:
            adj[dep.from_id].append(dep.to_id)

        # Find longest path (DFS)
        def dfs(node: str, visited: set) -> int:
            if node in visited:
                return 0
            visited.add(node)
            max_depth = 0
            for neighbor in adj.get(node, []):
                max_depth = max(max_depth, 1 + dfs(neighbor, visited.copy()))
            return max_depth

        max_depth = 0
        for sq in sub_questions:
            max_depth = max(max_depth, dfs(sq.id, set()))

        return max_depth
