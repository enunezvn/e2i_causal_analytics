"""llm_attribution: per-run contextvar both capture hooks read, plus the
drain-on-persist token accumulator (drain = read-and-reset so sums across a
session's assistant rows never double-count)."""

from src.utils.llm_attribution import (
    ANONYMOUS_USER_ID,
    clear_attribution,
    drain_run_usage,
    get_attribution,
    record_usage,
    set_chat_attribution,
    set_platform_attribution,
    user_id_from_session,
)

USER = "11111111-1111-1111-1111-111111111111"


def setup_function(_fn):
    clear_attribution()


def test_user_id_from_session_shapes():
    assert user_id_from_session(f"{USER}~abc-123") == USER
    assert user_id_from_session(f"{ANONYMOUS_USER_ID}~abc") is None  # honest NULL
    assert user_id_from_session("not-a-uuid~abc") is None
    assert user_id_from_session("no-tilde-here") is None
    assert user_id_from_session(None) is None


def test_chat_attribution_set_and_get():
    set_chat_attribution(f"{USER}~s1", request_id="run-9")
    attr = get_attribution()
    assert attr is not None
    assert attr.user_id == USER
    assert attr.session_id == f"{USER}~s1"
    assert attr.surface == "chat"
    assert attr.request_id == "run-9"


def test_platform_attribution():
    set_platform_attribution("insights", component="ExecutiveBrief")
    attr = get_attribution()
    assert attr.user_id is None
    assert attr.surface == "insights"
    assert attr.component == "ExecutiveBrief"


def test_record_usage_noop_without_attribution():
    record_usage("gpt-4o", 10, 5)  # must not raise
    assert drain_run_usage() is None


def test_record_and_drain_resets():
    set_chat_attribution(f"{USER}~s1")
    record_usage("gpt-4o", 10, 5)
    record_usage("claude-sonnet-4-6", 100, 50)
    drained = drain_run_usage()
    assert drained.input_tokens == 110
    assert drained.output_tokens == 55
    assert drained.last_model == "claude-sonnet-4-6"
    # drained: second read is empty — no double-counting across persists
    assert drain_run_usage() is None
