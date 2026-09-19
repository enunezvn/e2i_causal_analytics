"""RAGAS dependency imports and the Vertex AI compatibility shim."""

from __future__ import annotations

import sys
import types
from typing import Any


class RagasDependencyError(RuntimeError):
    """Raised when the RAGAS dependency tree is broken or incompatible."""


def ensure_ragas_vertexai_compat() -> None:
    """Supply unused Vertex classes removed from modern langchain-community.

    RAGAS 0.4.x imports both classes unconditionally even though this project
    judges with OpenAI. Real classes win when installed; otherwise lightweight
    stubs keep the unrelated OpenAI path importable (issue #491).
    """
    try:
        import langchain_community  # noqa: F401
    except ImportError:
        return

    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
    except ImportError:
        stub = types.ModuleType("langchain_community.chat_models.vertexai")
        stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
        sys.modules["langchain_community.chat_models.vertexai"] = stub

    try:
        from langchain_community.llms import VertexAI  # noqa: F401
    except ImportError:
        import langchain_community.llms as llms

        if not hasattr(llms, "VertexAI"):
            llms.VertexAI = type("VertexAI", (), {})  # type: ignore[attr-defined]


def import_ragas_components() -> dict[str, Any]:
    """Import the exact components shared by the smoke and judged paths."""
    try:
        ensure_ragas_vertexai_compat()
        import openai
        from datasets import Dataset
        from ragas import aevaluate
        from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RagasDependencyError(
            "RAGAS evaluation dependencies are broken or incompatible "
            f"({exc}). The langchain stack in requirements-ragas.txt likely "
            "drifted; see issue #491."
        ) from exc

    return {
        "openai": openai,
        "Dataset": Dataset,
        "aevaluate": aevaluate,
        "OpenAIEmbeddings": RagasOpenAIEmbeddings,
        "llm_factory": llm_factory,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
