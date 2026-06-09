"""Offline guard: the live chatbot caller actually enables multi-hop (audit F6).

The audit found the live caller hardcoded ``enable_multi_hop=False`` so the loop
was dead. This CI-collectable guard asserts the caller now forwards the
``CHATBOT_RAG_MULTI_HOP`` flag (default-on) into ``cognitive_rag_retrieve``, so a
regression back to a hardcoded False is caught without needing live backends.
"""

from pathlib import Path

GRAPH = Path(__file__).resolve().parents[3] / "src" / "api" / "routes" / "chatbot_graph.py"


def test_caller_forwards_multihop_flag():
    src = GRAPH.read_text()
    assert 'CHATBOT_RAG_MULTI_HOP_ENABLED = os.getenv("CHATBOT_RAG_MULTI_HOP", "true")' in src, (
        "multi-hop flag missing/renamed; default must stay on"
    )
    assert "enable_multi_hop=CHATBOT_RAG_MULTI_HOP_ENABLED" in src, (
        "live caller must forward the flag (regressed to hardcoded?)"
    )
    assert "enable_multi_hop=False" not in src, "live caller still hardcodes single-hop"
