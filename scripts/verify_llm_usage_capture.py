"""One-shot faithful verification of both LLM usage capture hooks (spec
2026-07-12). Run ON THE DROPLET from the repo root with the prod .env:

    PYTHONPATH=. .venv/bin/python scripts/verify_llm_usage_capture.py

Makes one tiny LangChain call via llm_factory (STREAMED — the copilotkit
path) and one tiny dspy/litellm call (~$0.001 total), then asserts both
landed in llm_usage_events with nonzero tokens. Exits non-zero on failure.
"""

import asyncio
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from src.api.dependencies.supabase_client import get_supabase  # noqa: E402
from src.utils.litellm_usage_logger import register_litellm_usage_logger  # noqa: E402


async def _stream_langchain_call() -> str:
    from src.utils.llm_factory import get_chat_llm

    llm = get_chat_llm(model_tier="fast", max_tokens=16)
    chunks = []
    async for chunk in llm.astream("Reply with the single word OK."):
        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            chunks.append(content)
    return "".join(chunks)


def main() -> int:
    client = get_supabase()
    if client is None:
        print("FAIL: no Supabase client (check .env)")
        return 1
    before = client.table("llm_usage_events").select("id", count="exact").execute().count or 0

    print("langchain (streamed):", asyncio.run(_stream_langchain_call()))

    register_litellm_usage_logger()
    import dspy

    from src.optimization.dspy_lm import get_default_dspy_model

    lm = dspy.LM(get_default_dspy_model(), max_tokens=16, cache=False)
    print("dspy:", lm("Reply with the single word OK."))

    time.sleep(8)  # background flusher polls every 2s

    after = client.table("llm_usage_events").select("id", count="exact").execute().count or 0
    new_count = after - before
    rows = (
        client.table("llm_usage_events")
        .select("provider, model, input_tokens, output_tokens, surface, user_id")
        .order("id", desc=True)
        .limit(max(new_count, 1))
        .execute()
        .data
        or []
    )
    print(f"rows before={before} after={after}")
    for row in rows:
        print(row)

    if new_count < 2:
        print(f"FAIL: expected >=2 new llm_usage_events rows, got {new_count}")
        return 1
    zero_rows = [r for r in rows if not (r["input_tokens"] or r["output_tokens"])]
    if zero_rows:
        print(f"FAIL: rows with zero tokens: {zero_rows}")
        return 1
    print("PASS: both capture hooks recorded real token usage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
