"""Unit tests for deepseek4free.server.ollama_compat.routes - no network,
no FastAPI TestClient/httpx needed.

Route functions are called directly as plain Python functions rather than
through an HTTP client: httpx/TestClient are not in this project's
dependencies (checked via rg_search - zero matches), and adding httpx as a
new dependency just to test route wiring would be disproportionate when the
route functions themselves are directly callable (the `Depends(...)`
defaults in their signatures are only resolved by FastAPI's own request
machinery - calling the function directly with explicit `manager=`/`cache=`
keyword arguments bypasses that machinery entirely and exercises the same
body).

Uses the same FakeDeepSeekAPI-backed real SessionManager pattern as
test_session_manager.py, wired through a real OllamaSessionCache, so the
full mapping -> session-cache -> SessionManager.send_message() path is
exercised exactly as it would be in production, just without real network
calls.
"""

import json
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from deepseek4free.server.ollama_compat.routes import (
    ollama_chat,
    ollama_generate,
    ollama_ps,
    ollama_show,
    ollama_tags,
    ollama_version,
)
from deepseek4free.server.ollama_compat.schemas import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaMessage,
    OllamaShowRequest,
)
from deepseek4free.server.ollama_compat.session_cache import OllamaSessionCache
from deepseek4free.server.session_manager import SessionManager


class FakeDeepSeekAPI:
    """Same minimal fake as test_session_manager.py: only the method surface
    SessionManager actually calls, with scripted chunks instead of real
    network calls."""

    def __init__(self) -> None:
        self.created_session_ids = iter(f"session-{i}" for i in range(1, 1000))
        self.next_chunks: list[dict[str, Any]] = []
        self.chat_completion_calls: list[dict[str, Any]] = []

    def create_chat_session(self) -> str:
        return next(self.created_session_ids)

    def chat_completion(
        self,
        chat_session_id: str,
        prompt: str,
        parent_message_id: str | None = None,
        thinking_enabled: bool = True,
        search_enabled: bool = False,
        file_ids: list[str] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        self.chat_completion_calls.append({"chat_session_id": chat_session_id, "prompt": prompt})
        yield from self.next_chunks


def _manager_and_cache():
    fake_api = FakeDeepSeekAPI()
    manager = SessionManager(fake_api)  # type: ignore[arg-type]
    return manager, fake_api, OllamaSessionCache()


def _json_body(response: JSONResponse) -> dict[str, Any]:
    return json.loads(response.body)


# --------------------------------------------------------------------------
# GET /api/tags
# --------------------------------------------------------------------------


def test_tags_returns_two_known_models() -> None:
    result = ollama_tags()
    names = {m.name for m in result.models}
    assert names == {"deepseek-chat:latest", "deepseek-reasoner:latest"}


def test_tags_entries_have_deterministic_digest() -> None:
    # Same digest across two independent calls - not random per request,
    # since Ollama clients may compare/cache this field.
    first = {m.name: m.digest for m in ollama_tags().models}
    second = {m.name: m.digest for m in ollama_tags().models}
    assert first == second


# --------------------------------------------------------------------------
# POST /api/show
# --------------------------------------------------------------------------


def test_show_unknown_model_returns_404_with_error_body() -> None:
    response = ollama_show(OllamaShowRequest(model="llama3"))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 404
    body = _json_body(response)
    assert "error" in body
    assert "llama3" in body["error"]


def test_show_known_model_returns_capabilities() -> None:
    response = ollama_show(OllamaShowRequest(model="deepseek-reasoner"))
    assert response["capabilities"] == ["completion", "thinking"]


def test_show_deepseek_chat_has_no_thinking_capability() -> None:
    response = ollama_show(OllamaShowRequest(model="deepseek-chat"))
    assert response["capabilities"] == ["completion"]


def test_show_requires_model_or_name() -> None:
    response = ollama_show(OllamaShowRequest())
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400


def test_show_accepts_legacy_name_field() -> None:
    response = ollama_show(OllamaShowRequest(name="deepseek-chat"))
    assert response["capabilities"] == ["completion"]


# --------------------------------------------------------------------------
# GET /api/version, GET /api/ps
# --------------------------------------------------------------------------


def test_version_returns_a_version_string() -> None:
    result = ollama_version()
    assert isinstance(result.version, str)
    assert result.version  # non-empty


def test_ps_reports_both_known_models_as_loaded() -> None:
    result = ollama_ps()
    names = {m.name for m in result.models}
    assert names == {"deepseek-chat:latest", "deepseek-reasoner:latest"}
    for entry in result.models:
        assert entry.expires_at
        assert entry.size_vram == 0


# --------------------------------------------------------------------------
# POST /api/chat (non-streaming)
# --------------------------------------------------------------------------


def test_chat_non_stream_returns_message_content() -> None:
    manager, fake_api, cache = _manager_and_cache()
    fake_api.next_chunks = [{"type": "text", "content": "Hello there", "finish_reason": "stop"}]

    req = OllamaChatRequest(
        model="deepseek-chat",
        messages=[OllamaMessage(role="user", content="hi")],
        stream=False,
    )
    result = ollama_chat(req, manager=manager, cache=cache)

    assert result["done"] is True
    assert result["done_reason"] == "stop"
    assert result["message"]["role"] == "assistant"
    assert result["message"]["content"] == "Hello there"
    assert result["model"] == "deepseek-chat:latest"
    # Timing/eval-count fields must be present (even if approximate) since
    # Ollama clients read them - see mapping/timing docstrings in routes.py.
    assert result["eval_count"] >= 1
    assert result["total_duration"] >= 0


def test_chat_unknown_model_returns_404() -> None:
    manager, _fake_api, cache = _manager_and_cache()
    req = OllamaChatRequest(
        model="llama3",
        messages=[OllamaMessage(role="user", content="hi")],
        stream=False,
    )
    response = ollama_chat(req, manager=manager, cache=cache)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 404


def test_chat_empty_messages_returns_400() -> None:
    manager, _fake_api, cache = _manager_and_cache()
    req = OllamaChatRequest(model="deepseek-chat", messages=[], stream=False)
    response = ollama_chat(req, manager=manager, cache=cache)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400


def test_chat_second_call_with_same_history_reuses_cached_session() -> None:
    """Exercises the session_cache hit path end-to-end: same history prefix
    -> same underlying DeepSeek session id reused, only one new session
    created overall (not two)."""
    manager, fake_api, cache = _manager_and_cache()
    fake_api.next_chunks = [{"type": "text", "content": "first reply", "finish_reason": "stop"}]

    first_req = OllamaChatRequest(
        model="deepseek-chat",
        messages=[OllamaMessage(role="user", content="hello")],
        stream=False,
    )
    ollama_chat(first_req, manager=manager, cache=cache)
    assert manager.session_count() == 1

    fake_api.next_chunks = [{"type": "text", "content": "second reply", "finish_reason": "stop"}]
    second_req = OllamaChatRequest(
        model="deepseek-chat",
        messages=[
            OllamaMessage(role="user", content="hello"),
            OllamaMessage(role="assistant", content="first reply"),
            OllamaMessage(role="user", content="follow up"),
        ],
        stream=False,
    )
    ollama_chat(second_req, manager=manager, cache=cache)

    # Same history prefix (the first exchange) -> cache hit -> no second
    # DeepSeek session created.
    assert manager.session_count() == 1


# --------------------------------------------------------------------------
# POST /api/chat (streaming) - async because StreamingResponse wraps the
# sync generator via Starlette's iterate_in_threadpool, which needs a
# running event loop to drive.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_stream_yields_ndjson_lines_ending_in_done() -> None:
    manager, fake_api, cache = _manager_and_cache()
    fake_api.next_chunks = [
        {"type": "text", "content": "Hel", "finish_reason": None},
        {"type": "text", "content": "lo", "finish_reason": "stop"},
    ]

    req = OllamaChatRequest(
        model="deepseek-chat",
        messages=[OllamaMessage(role="user", content="hi")],
        stream=True,
    )
    response = ollama_chat(req, manager=manager, cache=cache)

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "application/x-ndjson"

    raw_lines = [chunk async for chunk in response.body_iterator]
    parsed = [json.loads(line) for line in raw_lines]

    # At least one non-final content line plus the final done:true line.
    assert len(parsed) >= 2
    assert all(p["done"] is False for p in parsed[:-1])
    assert parsed[-1]["done"] is True
    assert parsed[-1]["done_reason"] == "stop"
    # Concatenating the streamed content pieces reproduces the full reply.
    streamed_text = "".join(p["message"]["content"] for p in parsed[:-1])
    assert streamed_text == "Hello"


# --------------------------------------------------------------------------
# POST /api/generate
# --------------------------------------------------------------------------


def test_generate_non_stream_returns_response_field() -> None:
    manager, fake_api, _cache = _manager_and_cache()
    fake_api.next_chunks = [{"type": "text", "content": "42", "finish_reason": "stop"}]

    req = OllamaGenerateRequest(model="deepseek-chat", prompt="what is the answer", stream=False)
    result = ollama_generate(req, manager=manager)

    assert result["done"] is True
    assert result["response"] == "42"
    assert "message" not in result  # /api/generate uses response, not message


def test_generate_always_creates_a_new_session_even_with_same_prompt() -> None:
    """/api/generate is stateless by Ollama's own semantics - unlike
    /api/chat it must never consult session_cache, so two calls (even
    identical ones) always produce two DeepSeek sessions."""
    manager, fake_api, _cache = _manager_and_cache()
    fake_api.next_chunks = [{"type": "text", "content": "ok", "finish_reason": "stop"}]

    req = OllamaGenerateRequest(model="deepseek-chat", prompt="same prompt", stream=False)
    ollama_generate(req, manager=manager)
    fake_api.next_chunks = [{"type": "text", "content": "ok again", "finish_reason": "stop"}]
    ollama_generate(req, manager=manager)

    assert manager.session_count() == 2


def test_generate_unknown_model_returns_404() -> None:
    manager, _fake_api, _cache = _manager_and_cache()
    req = OllamaGenerateRequest(model="llama3", prompt="hi", stream=False)
    response = ollama_generate(req, manager=manager)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 404


def test_generate_empty_prompt_returns_400() -> None:
    manager, _fake_api, _cache = _manager_and_cache()
    req = OllamaGenerateRequest(model="deepseek-chat", prompt="", stream=False)
    response = ollama_generate(req, manager=manager)
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
