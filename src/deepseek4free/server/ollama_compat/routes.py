"""Ollama-compatible endpoints: /api/chat, /api/generate, /api/tags,
/api/show, /api/version, /api/ps, plus explicit 501s for the most commonly
auto-probed unsupported endpoints (/api/pull, /api/embed, /api/embeddings).

This router is mounted on its own FastAPI app (see ollama_compat/app.py) on
a separate port from the project's native REST API (server/app.py) - it is
NOT included in that app's router list. Both apps share the same
SessionManager/DeepSeekAPI singleton via server/dependencies.py, so sessions
created through this compat layer show up in the native /sessions endpoints
too (and vice versa) - there's only ever one underlying DeepSeek client.

Response bodies for /api/chat and /api/generate are built as plain dicts,
not pydantic response models - see schemas.py's module docstring for why
(non-final streamed chunks and the final chunk have different required
fields).
"""

import hashlib
import importlib.metadata
import json
import logging
import time
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse

from deepseek4free.server.dependencies import get_session_manager
from deepseek4free.server.ollama_compat.dependencies import get_ollama_session_cache
from deepseek4free.server.ollama_compat.mapping import (
    KNOWN_MODELS,
    history_prefix_key,
    last_user_message,
    resolve_model,
)
from deepseek4free.server.ollama_compat.schemas import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaModelDetails,
    OllamaModelEntry,
    OllamaPsModelEntry,
    OllamaPsResponse,
    OllamaShowRequest,
    OllamaTagsResponse,
    OllamaVersionResponse,
)
from deepseek4free.server.ollama_compat.session_cache import OllamaSessionCache
from deepseek4free.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ollama-compat"])

NDJSON_MEDIA_TYPE = "application/x-ndjson"


def _model_digest(name: str) -> str:
    """Deterministic fake digest - DeepSeek has no real model blobs/files to
    hash, but Ollama clients display/compare this field, so it needs to be
    stable across calls for the same name rather than random per request."""
    return "sha256:" + hashlib.sha256(name.encode("utf-8")).hexdigest()


def _model_entry(name: str, thinking_enabled: bool) -> OllamaModelEntry:
    return OllamaModelEntry(
        name=f"{name}:latest",
        model=f"{name}:latest",
        modified_at="2026-01-01T00:00:00Z",
        size=0,
        digest=_model_digest(name),
        details=OllamaModelDetails(
            family="deepseek",
            families=["deepseek"],
        ),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _error_response(status_code: int, message: str) -> JSONResponse:
    # Ollama clients read the `error` field specifically (not FastAPI's own
    # default `detail`), so error bodies from this router always use that
    # shape rather than relying on server/errors.py's global handlers (which
    # are registered on the *other* FastAPI app in server/app.py and never
    # even run for this one).
    return JSONResponse(status_code=status_code, content={"error": message})


def _collect_response(
    manager: SessionManager,
    session: Any,
    prompt: str,
    thinking_enabled: bool,
) -> dict[str, Any]:
    """Drains manager.send_message() fully and returns the combined text +
    thinking + finish_reason. Used by the non-streaming branches of both
    /api/chat and /api/generate."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    for chunk in manager.send_message(session, prompt, thinking_enabled, False):
        if chunk.get("type") == "text" and chunk.get("content"):
            text_parts.append(chunk["content"])
        elif chunk.get("type") == "thinking" and chunk.get("content"):
            thinking_parts.append(chunk["content"])
    return {"content": "".join(text_parts), "thinking": "".join(thinking_parts) or None}


def _timing_fields(total_duration_s: float, prompt_text: str, response_text: str) -> dict[str, int]:
    """Builds Ollama's timing/eval-count fields as nanosecond ints.

    IMPORTANT: eval_count/prompt_eval_count here are a crude word-count
    approximation, NOT real token counts - DeepSeekAPI doesn't expose a
    tokenizer or usage/token accounting anywhere in its response stream, so
    there is no way to report true token counts. Likewise prompt_eval_duration
    is reported as 0 and eval_duration as the full total_duration, since
    DeepSeek's API gives no separate timing breakdown between "processing the
    prompt" and "generating the reply" - only a single end-to-end stream.
    Any client that treats these as authoritative token/perf metrics will be
    misled; they exist only so response bodies have the fields Ollama clients
    expect to find, not because the numbers are meaningful for capacity
    planning or billing.
    """
    total_ns = int(total_duration_s * 1e9)
    return {
        "total_duration": total_ns,
        "load_duration": 0,
        "prompt_eval_count": max(1, len(prompt_text.split())),
        "prompt_eval_duration": 0,
        "eval_count": max(1, len(response_text.split())),
        "eval_duration": total_ns,
    }


@router.get("/")
async def root_ping():
    # Возвращаем строго то, что ждет любой Ollama-клиент
    return Response(content="Ollama is running", media_type="text/plain")

@router.post("/api/pull")
async def stub_pull():
    # Клиенты ждут потоковый JSON с прогрессом
    async def generate():
        yield json.dumps({"status": "pulling manifest"}) + "\n"
        yield json.dumps({"status": "success"}) + "\n"
    return StreamingResponse(generate(), media_type="application/x-ndjson")

@router.post("/api/embed")
@router.post("/api/embeddings")
async def stub_embed():
    # Возвращаем фейковый вектор, чтобы не ломать индексацию Continue/Langchain, 
    # либо отдаем 501 Not Implemented, но в формате JSON, а не 404
    return {"embedding": [0.0] * 1536}

# --------------------------------------------------------------------------
# POST /api/chat
# --------------------------------------------------------------------------


@router.post("/api/chat")
def ollama_chat(
    req: OllamaChatRequest,
    manager: SessionManager = Depends(get_session_manager),
    cache: OllamaSessionCache = Depends(get_ollama_session_cache),
):
    try:
        canonical_model, thinking_enabled = resolve_model(req.model)
    except ValueError as e:
        return _error_response(404, str(e))

    try:
        prompt = last_user_message(req.messages)
    except ValueError as e:
        return _error_response(400, str(e))

    cache_key = history_prefix_key(canonical_model, req.messages)
    cached = cache.get(cache_key)
    if cached is not None:
        # Cache hit: reuse the existing DeepSeek session. get_session()
        # raises HTTPException(404) if the session was somehow dropped from
        # SessionManager (e.g. process restarted, losing in-memory state,
        # while this cache entry - also in-memory but with its own TTL -
        # happened to still be considered fresh) - treat that the same as a
        # cache miss instead of propagating a confusing 404 to the Ollama
        # client, since from its point of view it just sent a normal chat
        # request, not a request for a specific session id it's tracking.
        try:
            session = manager.get_session(cached.deepseek_session_id)
            session.parent_message_id = cached.parent_message_id
        except HTTPException:
            session = manager.create_session()
    else:
        session = manager.create_session()

    model_name = f"{canonical_model}:latest"
    start = time.perf_counter()

    if req.stream:

        def event_stream() -> Generator[str, None, None]:
            text_parts: list[str] = []
            thinking_parts: list[str] = []
            try:
                for chunk in manager.send_message(session, prompt, thinking_enabled, False):
                    content = chunk.get("content", "")
                    if chunk.get("type") == "text" and content:
                        text_parts.append(content)
                        payload = {
                            "model": model_name,
                            "created_at": _now_iso(),
                            "message": {"role": "assistant", "content": content},
                            "done": False,
                        }
                        yield json.dumps(payload, ensure_ascii=False) + "\n"
                    elif chunk.get("type") == "thinking" and content:
                        thinking_parts.append(content)
                        payload = {
                            "model": model_name,
                            "created_at": _now_iso(),
                            "message": {"role": "assistant", "content": "", "thinking": content},
                            "done": False,
                        }
                        yield json.dumps(payload, ensure_ascii=False) + "\n"

                cache.put(cache_key, session.deepseek_session_id, session.parent_message_id)
                elapsed = time.perf_counter() - start
                final_payload = {
                    "model": model_name,
                    "created_at": _now_iso(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    **_timing_fields(elapsed, prompt, "".join(text_parts)),
                }
                yield json.dumps(final_payload, ensure_ascii=False) + "\n"
            except Exception as e:
                # Mirrors server/routes/messages.py's approach: a streaming
                # response has already sent a 200 status line by the time an
                # error occurs mid-stream, so it can only be signaled
                # in-band. Ollama itself doesn't define a standard in-band
                # NDJSON error frame, so this uses the same {"error": ...}
                # shape as this router's non-streaming error responses for
                # consistency, plus done:true so clients stop reading.
                logger.exception("Error while streaming Ollama-compat /api/chat response")
                yield json.dumps({"error": str(e), "done": True}, ensure_ascii=False) + "\n"

        return StreamingResponse(event_stream(), media_type=NDJSON_MEDIA_TYPE)

    # Non-streaming: collect the full response and return one JSON body.
    result = _collect_response(manager, session, prompt, thinking_enabled)
    cache.put(cache_key, session.deepseek_session_id, session.parent_message_id)
    elapsed = time.perf_counter() - start
    message: dict[str, Any] = {"role": "assistant", "content": result["content"]}
    if result["thinking"]:
        message["thinking"] = result["thinking"]
    return {
        "model": model_name,
        "created_at": _now_iso(),
        "message": message,
        "done": True,
        "done_reason": "stop",
        **_timing_fields(elapsed, prompt, result["content"]),
    }


# --------------------------------------------------------------------------
# POST /api/generate
# --------------------------------------------------------------------------


@router.post("/api/generate")
def ollama_generate(
    req: OllamaGenerateRequest,
    manager: SessionManager = Depends(get_session_manager),
):
    try:
        canonical_model, thinking_enabled = resolve_model(req.model)
    except ValueError as e:
        return _error_response(404, str(e))

    if not req.prompt:
        return _error_response(400, "prompt must not be empty")

    # /api/generate is a single-shot, stateless call by Ollama's own
    # semantics (a completion for one prompt, not a chat with history) -
    # unlike /api/chat there is no messages[] to hash into a cache key, so
    # this always creates a fresh DeepSeek session rather than consulting
    # session_cache.py. A caller wanting multi-turn behavior should use
    # /api/chat instead, exactly as with real Ollama.
    session = manager.create_session()
    model_name = f"{canonical_model}:latest"
    start = time.perf_counter()

    if req.stream:

        def event_stream() -> Generator[str, None, None]:
            text_parts: list[str] = []
            try:
                for chunk in manager.send_message(session, req.prompt, thinking_enabled, False):
                    content = chunk.get("content", "")
                    if chunk.get("type") == "text" and content:
                        text_parts.append(content)
                        payload = {
                            "model": model_name,
                            "created_at": _now_iso(),
                            "response": content,
                            "done": False,
                        }
                        yield json.dumps(payload, ensure_ascii=False) + "\n"

                elapsed = time.perf_counter() - start
                final_payload = {
                    "model": model_name,
                    "created_at": _now_iso(),
                    "response": "",
                    "done": True,
                    "done_reason": "stop",
                    **_timing_fields(elapsed, req.prompt, "".join(text_parts)),
                }
                yield json.dumps(final_payload, ensure_ascii=False) + "\n"
            except Exception as e:
                logger.exception("Error while streaming Ollama-compat /api/generate response")
                yield json.dumps({"error": str(e), "done": True}, ensure_ascii=False) + "\n"

        return StreamingResponse(event_stream(), media_type=NDJSON_MEDIA_TYPE)

    result = _collect_response(manager, session, req.prompt, thinking_enabled)
    elapsed = time.perf_counter() - start
    return {
        "model": model_name,
        "created_at": _now_iso(),
        "response": result["content"],
        "done": True,
        "done_reason": "stop",
        **_timing_fields(elapsed, req.prompt, result["content"]),
    }


# --------------------------------------------------------------------------
# GET /api/tags, GET /api/ps, POST /api/show, GET /api/version
# --------------------------------------------------------------------------


@router.get("/api/tags", response_model=OllamaTagsResponse)
def ollama_tags() -> OllamaTagsResponse:
    return OllamaTagsResponse(
        models=[
            _model_entry(name, thinking)
            for name, thinking in KNOWN_MODELS.items()
            if name != "deepseek-r1"
        ]
    )


@router.post("/api/show")
def ollama_show(req: OllamaShowRequest):
    requested = req.model or req.name
    if not requested:
        return _error_response(400, "either 'model' or 'name' must be provided")

    try:
        canonical_model, thinking_enabled = resolve_model(requested)
    except ValueError as e:
        return _error_response(404, str(e))

    entry = _model_entry(canonical_model, thinking_enabled)
    capabilities = ["completion"] + (["thinking"] if thinking_enabled else [])
    return {
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": entry.details.model_dump(),
        "model_info": {},
        "capabilities": capabilities,
    }


@router.get("/api/version", response_model=OllamaVersionResponse)
def ollama_version() -> OllamaVersionResponse:
    try:
        version = importlib.metadata.version("deepseek4free")
    except importlib.metadata.PackageNotFoundError:
        # Mirrors the fallback style already used elsewhere in this project
        # (e.g. config.py's own defensive try/except patterns) for the case
        # where the package is imported directly from source without being
        # pip-installed (e.g. some test/dev setups).
        version = "0.0.0-dev"
    return OllamaVersionResponse(version=version)


@router.get("/api/ps", response_model=OllamaPsResponse)
def ollama_ps() -> OllamaPsResponse:
    # There's no real model loading/unloading here - DeepSeek is a remote
    # API, not a local weights file - so every known model is always
    # reported as "loaded" with a far-future expires_at rather than modeling
    # real VRAM-based expiry.
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    models = []
    for name, thinking in KNOWN_MODELS.items():
        if name == "deepseek-r1":
            continue
        base = _model_entry(name, thinking)
        models.append(
            OllamaPsModelEntry(
                **base.model_dump(),
                expires_at=expires_at,
                size_vram=0,
            )
        )
    return OllamaPsResponse(models=models)


# --------------------------------------------------------------------------
# Explicit stubs for the most commonly auto-probed unsupported endpoints.
#
# Real Ollama has ~15 more endpoints (pull/push/create/copy/delete/blobs/
# embed/embeddings) that this compat layer intentionally does not
# implement - DeepSeek has no local model files to pull/push/create/copy/
# delete, and DeepSeekAPI exposes no embeddings endpoint at all. Rather than
# leaving these to FastAPI's generic 404 (indistinguishable from a typo'd
# path), the three most commonly auto-probed ones by autodetect-style
# clients get an explicit 501 with a clear reason.
# --------------------------------------------------------------------------

_NOT_SUPPORTED_MESSAGE = (
    "not supported by deepseek4free's Ollama-compat layer: DeepSeek has no "
    "local model files or embeddings endpoint to back this operation"
)


@router.post("/api/pull")
def ollama_pull_unsupported():
    return _error_response(501, _NOT_SUPPORTED_MESSAGE)


@router.post("/api/embed")
def ollama_embed_unsupported():
    return _error_response(501, _NOT_SUPPORTED_MESSAGE)


@router.post("/api/embeddings")
def ollama_embeddings_unsupported():
    return _error_response(501, _NOT_SUPPORTED_MESSAGE)
