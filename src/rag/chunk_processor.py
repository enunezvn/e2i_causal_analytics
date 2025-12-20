"""
Semantic chunking for agent outputs.

Processes agent outputs into retrievable chunks for the RAG system.
"""

from typing import List, Optional
from src.rag.models.insight_models import Chunk


class ChunkProcessor:
    """
    Process agent outputs into retrievable chunks.

    Chunk types:
    - Analysis summaries (from agent_activities)
    - Causal findings (from causal_paths)
    - KPI snapshots (from business_metrics)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize chunk processor.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_agent_output(
        self,
        output,  # AgentActivity
        chunk_size: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Chunk agent activity output for indexing.

        Args:
            output: AgentActivity record
            chunk_size: Override default chunk size

        Returns:
            List of Chunk objects for indexing
        """
        chunk_size = chunk_size or self.chunk_size
        chunks = []

        # Extract content based on output structure
        if hasattr(output, 'analysis_results'):
            content = output.analysis_results
        elif hasattr(output, 'content'):
            content = output.content
        else:
            content = str(output)

        # Split into semantic chunks
        raw_chunks = self._semantic_split(content, chunk_size)

        for i, text in enumerate(raw_chunks):
            chunk = Chunk(
                content=text,
                source_type="agent_analysis",
                embedding=None,  # Generated during indexing
                metadata={
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                    "agent_id": getattr(output, 'agent_id', None),
                    "activity_id": getattr(output, 'id', None),
                },
            )
            chunks.append(chunk)

        return chunks

    def chunk_causal_path(self, path) -> List[Chunk]:
        """
        Chunk causal path for indexing.

        Args:
            path: CausalPath record

        Returns:
            List of Chunk objects
        """
        # Causal paths are typically single chunks
        content = path.path_description if hasattr(path, 'path_description') else str(path)

        return [
            Chunk(
                content=content,
                source_type="causal_path",
                embedding=None,
                metadata={
                    "path_id": getattr(path, 'id', None),
                    "cause": getattr(path, 'cause', None),
                    "effect": getattr(path, 'effect', None),
                },
            )
        ]

    def chunk_kpi_snapshot(self, snapshot) -> List[Chunk]:
        """
        Chunk KPI snapshot for indexing.

        Args:
            snapshot: BusinessMetric or KPI view record

        Returns:
            List of Chunk objects
        """
        # KPI snapshots are typically single chunks
        if hasattr(snapshot, 'to_dict'):
            content = str(snapshot.to_dict())
        else:
            content = str(snapshot)

        return [
            Chunk(
                content=content,
                source_type="kpi_snapshot",
                embedding=None,
                metadata={
                    "kpi_name": getattr(snapshot, 'kpi_name', None),
                    "timestamp": getattr(snapshot, 'timestamp', None),
                },
            )
        ]

    def _semantic_split(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into semantic chunks.

        Tries to split on sentence boundaries when possible.

        Args:
            text: Text to split
            chunk_size: Target chunk size

        Returns:
            List of text chunks
        """
        if not text:
            return []

        # Simple sentence-based splitting
        # In production, use a proper tokenizer
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence.split())

            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                # Keep overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else []
                current_chunk = overlap_sentences
                current_size = sum(len(s.split()) for s in current_chunk)

            current_chunk.append(sentence)
            current_size += sentence_size

        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks
