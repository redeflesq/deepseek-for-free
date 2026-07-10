"""Model-name mapping and history-to-prompt helpers for the Ollama-compatible layer.

DeepSeek has no real per-request model choice the way Ollama does - the only
thing that varies is whether "thinking" (reasoning) mode is on. This module
maps Ollama-style model names onto that single boolean, and provides the two
small pure functions routes.py needs to turn an Ollama chat request (full
message history every call) into a single DeepSeekAPI prompt + a cache key
for server/ollama_compat/session_cache.py's history-based session reuse.
"""

import hashlib
import json

from deepseek4free.server.ollama_compat.schemas import OllamaMessage

# name -> thinking_enabled. Keys are canonical (already normalized - no
# ":latest"/"-latest" suffix). "deepseek-r1" is accepted as an alias for
# "deepseek-reasoner" since that's the name real Ollama users are used to
# pulling (see docs/api.md research: `ollama pull deepseek-r1:32b` is the
# real-world command clients are configured against).
KNOWN_MODELS: dict[str, bool] = {
    "deepseek-chat": False,
    "deepseek-reasoner": True,
    "deepseek-r1": True,
}

_LATEST_SUFFIXES = (":latest", "-latest")


def normalize_model_name(name: str) -> str:
    """Strips a trailing ":latest"/"-latest" tag, matching how Ollama treats
    an untagged model name as implicitly ":latest" (see docs/api.md: "The tag
    is optional and, if not provided, will default to latest.")."""
    for suffix in _LATEST_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def resolve_model(name: str) -> tuple[str, bool]:
    """Maps an Ollama-style model name to (canonical_name, thinking_enabled).

    Raises ValueError (not looked up in KNOWN_MODELS) if the name doesn't
    match one of our two known models. Callers in routes.py must catch this
    themselves and return an explicit HTTP 404 with an {"error": ...} body -
    NOT let it propagate to server/errors.py's global ValueError handler,
    which maps ValueError to 422. Real Ollama returns 404 for an unknown
    model (that's the status code clients like Continue.dev actually check
    for before attempting `ollama pull` - see docs/api.md research), so a
    422 here would silently break that client-side fallback behavior.
    """
    canonical = normalize_model_name(name)
    if canonical not in KNOWN_MODELS:
        known = ", ".join(sorted(KNOWN_MODELS))
        raise ValueError(f'model "{name}" not found, try one of: {known}')
    return canonical, KNOWN_MODELS[canonical]


def last_user_message(messages: list[OllamaMessage]) -> str:
    """Returns the content of the last message, which must have role=='user'.

    Ollama clients always send the new user turn as the last message in a
    real chat exchange - DeepSeekAPI.chat_completion() only accepts a single
    new prompt string (not an arbitrary message list), so that's the only
    piece of the request actually sent onward; everything before it is used
    only to compute the session-cache key (see history_prefix_key below),
    not resent to DeepSeek.

    Raises ValueError if messages is empty or the last message isn't role
    'user' - routes.py maps this to an HTTP 400, since sending an empty or
    non-user-terminated history is a malformed request, not a "model not
    found" case.
    """
    if not messages:
        raise ValueError("messages must contain at least one message")
    last = messages[-1]
    if last.role != "user":
        raise ValueError("last message must have role=user")
    return last.content


def history_prefix_key(model: str, messages: list[OllamaMessage]) -> str:
    """Hashes (model, messages[:-1]) into a stable cache key for
    session_cache.py.

    messages[:-1] is everything except the new user turn being sent right
    now - two requests share a key iff they represent "the same conversation
    so far, plus one new message", which is exactly when reusing the
    DeepSeek-side session (and its parent_message_id thread) is valid.
    json.dumps(..., sort_keys=True) makes the key independent of dict key
    order (pydantic's model_dump() order is stable per-model already, but
    sort_keys=True removes any dependency on that implementation detail).
    """
    prefix = [m.model_dump() for m in messages[:-1]]
    payload = model + "|" + json.dumps(prefix, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
