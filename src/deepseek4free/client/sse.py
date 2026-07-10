"""Server-Sent-Events parsing for DeepSeek's streaming chat completion endpoint.

Extracted from the old DeepSeekAPI._parse_sse so it can be unit-tested with
plain bytes/dict fixtures, without spinning up curl_cffi or a real network
call. Parsing logic is UNCHANGED from the original.
"""

import json
from collections.abc import Generator
from typing import Any


def parse_sse_line(line: bytes, state: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
    """Parses one raw SSE line from DeepSeek's stream, yielding zero or more
    normalized chunks: {"type": "text"|"thinking"|"meta", "content": ..., "finish_reason": ...}.

    `state` is mutated across calls in the same stream (caller passes the
    same dict for every line in one stream) to track the last-seen path and,
    for full snapshot frames, the current response object - DeepSeek's
    stream isn't fully self-describing per individual line.
    """
    try:
        if line.startswith(b"data: "):
            data = json.loads(line[6:])
        elif line.startswith(b"event: finish"):
            yield {"type": "text", "content": "", "finish_reason": "stop"}
            return
        else:
            return
    except json.JSONDecodeError:
        return

    path = data.get("p")
    op = data.get("o")
    val = data.get("v")

    if path:
        state["_last_path"] = path

    # Message IDs: {"request_message_id":1,"response_message_id":2,...}
    if "request_message_id" in data and "response_message_id" in data:
        state["_parent_message_id"] = data["response_message_id"]
        yield {"type": "meta", "parent_message_id": data["response_message_id"]}
        return

    # Full state snapshot: {"v": {"response": {...}}}. This can carry the
    # *first* piece of content/thinking_content (not just metadata) - e.g. a
    # short reply can arrive entirely inside this single snapshot frame with
    # no subsequent APPEND at all. Not yielding here would silently drop the
    # opening character(s) of every reply (or the whole reply, for short
    # ones).
    if isinstance(val, dict) and "response" in val:
        state["response"] = val["response"]
        snapshot_content = val["response"].get("content")
        if isinstance(snapshot_content, str) and snapshot_content:
            yield {"type": "text", "content": snapshot_content, "finish_reason": None}
        snapshot_thinking = val["response"].get("thinking_content")
        if isinstance(snapshot_thinking, str) and snapshot_thinking:
            yield {"type": "thinking", "content": snapshot_thinking, "finish_reason": None}
        return

    # Status finished
    if path == "response/status" and val == "FINISHED":
        yield {"type": "text", "content": "", "finish_reason": "stop"}
        return

    path = path or state.get("_last_path")
    if not path:
        return

    if op == "APPEND" or (op is None and path != "response/status"):
        if path == "response/content" and isinstance(val, str):
            if val:
                yield {"type": "text", "content": val, "finish_reason": None}
        elif path == "response/thinking_content" and isinstance(val, str):
            if val:
                yield {"type": "thinking", "content": val, "finish_reason": None}
