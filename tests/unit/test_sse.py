"""Unit tests for deepseek4free.client.sse.parse_sse_line - no network needed.

Each test feeds one raw SSE line (as the bytes DeepSeek's stream actually
sends) through parse_sse_line and checks the normalized chunk(s) yielded.
state is a fresh dict per test unless a test explicitly wants to check
cross-line behaviour (see test_append_reuses_last_path_from_prior_line).
"""

import json

from deepseek4free.client.sse import parse_sse_line


def _line(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode()


def test_ignores_non_data_non_finish_lines() -> None:
    state: dict = {}
    assert list(parse_sse_line(b": keep-alive", state)) == []


def test_event_finish_yields_stop() -> None:
    state: dict = {}
    chunks = list(parse_sse_line(b"event: finish", state))
    assert chunks == [{"type": "text", "content": "", "finish_reason": "stop"}]


def test_malformed_json_is_ignored() -> None:
    state: dict = {}
    assert list(parse_sse_line(b"data: {not json", state)) == []


def test_message_ids_yield_meta_chunk() -> None:
    state: dict = {}
    line = _line({"request_message_id": 1, "response_message_id": 2})
    chunks = list(parse_sse_line(line, state))
    assert chunks == [{"type": "meta", "parent_message_id": 2}]
    assert state["_parent_message_id"] == 2


def test_full_snapshot_with_content_and_thinking_yields_both() -> None:
    """Regression case explicitly called out in sse.py's docstring: a short
    reply can arrive entirely inside one snapshot frame with no subsequent
    APPEND - skipping this would silently drop the whole reply."""
    state: dict = {}
    line = _line({"v": {"response": {"content": "Hi", "thinking_content": "thinking..."}}})
    chunks = list(parse_sse_line(line, state))
    assert {"type": "text", "content": "Hi", "finish_reason": None} in chunks
    assert {"type": "thinking", "content": "thinking...", "finish_reason": None} in chunks
    assert state["response"] == {"content": "Hi", "thinking_content": "thinking..."}


def test_full_snapshot_with_empty_content_yields_nothing() -> None:
    state: dict = {}
    line = _line({"v": {"response": {"content": "", "thinking_content": ""}}})
    assert list(parse_sse_line(line, state)) == []


def test_response_status_finished_yields_stop() -> None:
    state: dict = {}
    line = _line({"p": "response/status", "v": "FINISHED"})
    chunks = list(parse_sse_line(line, state))
    assert chunks == [{"type": "text", "content": "", "finish_reason": "stop"}]


def test_append_content_with_explicit_path() -> None:
    state: dict = {}
    line = _line({"p": "response/content", "o": "APPEND", "v": "Hello"})
    chunks = list(parse_sse_line(line, state))
    assert chunks == [{"type": "text", "content": "Hello", "finish_reason": None}]


def test_append_thinking_with_explicit_path() -> None:
    state: dict = {}
    line = _line({"p": "response/thinking_content", "o": "APPEND", "v": "reasoning..."})
    chunks = list(parse_sse_line(line, state))
    assert chunks == [{"type": "thinking", "content": "reasoning...", "finish_reason": None}]


def test_append_reuses_last_path_from_prior_line() -> None:
    """DeepSeek's stream omits `p` on follow-up APPEND frames for the same
    path - parse_sse_line must remember it via `state['_last_path']`."""
    state: dict = {}
    first = _line({"p": "response/content", "o": "APPEND", "v": "Hel"})
    second = _line({"o": "APPEND", "v": "lo"})  # no "p" this time

    first_chunks = list(parse_sse_line(first, state))
    second_chunks = list(parse_sse_line(second, state))

    assert first_chunks == [{"type": "text", "content": "Hel", "finish_reason": None}]
    assert second_chunks == [{"type": "text", "content": "lo", "finish_reason": None}]


def test_append_with_empty_string_yields_nothing() -> None:
    state: dict = {}
    line = _line({"p": "response/content", "o": "APPEND", "v": ""})
    assert list(parse_sse_line(line, state)) == []


def test_no_path_and_no_state_yields_nothing() -> None:
    state: dict = {}
    line = _line({"o": "APPEND", "v": "orphaned"})
    assert list(parse_sse_line(line, state)) == []
