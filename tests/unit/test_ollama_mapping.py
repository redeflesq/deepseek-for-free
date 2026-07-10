"""Unit tests for deepseek4free.server.ollama_compat.mapping - no network needed.

Covers model-name resolution (deepseek-chat/-reasoner/-r1 plus :latest/-latest
suffix normalization and the unknown-model error path), last-message
extraction, and the history-prefix cache-key helper used by session_cache.py.
"""

import pytest

from deepseek4free.server.ollama_compat.mapping import (
    history_prefix_key,
    last_user_message,
    normalize_model_name,
    resolve_model,
)
from deepseek4free.server.ollama_compat.schemas import OllamaMessage

# --------------------------------------------------------------------------
# normalize_model_name / resolve_model
# --------------------------------------------------------------------------


def test_normalize_strips_colon_latest_suffix() -> None:
    assert normalize_model_name("deepseek-chat:latest") == "deepseek-chat"


def test_normalize_strips_dash_latest_suffix() -> None:
    assert normalize_model_name("deepseek-chat-latest") == "deepseek-chat"


def test_normalize_leaves_untagged_name_unchanged() -> None:
    assert normalize_model_name("deepseek-chat") == "deepseek-chat"


def test_resolve_deepseek_chat_maps_to_thinking_disabled() -> None:
    canonical, thinking = resolve_model("deepseek-chat")
    assert canonical == "deepseek-chat"
    assert thinking is False


def test_resolve_deepseek_chat_with_latest_tag() -> None:
    canonical, thinking = resolve_model("deepseek-chat:latest")
    assert canonical == "deepseek-chat"
    assert thinking is False


def test_resolve_deepseek_reasoner_maps_to_thinking_enabled() -> None:
    canonical, thinking = resolve_model("deepseek-reasoner")
    assert canonical == "deepseek-reasoner"
    assert thinking is True


def test_resolve_deepseek_r1_alias_maps_to_thinking_enabled() -> None:
    canonical, thinking = resolve_model("deepseek-r1")
    assert canonical == "deepseek-r1"
    assert thinking is True


def test_resolve_unknown_model_raises_value_error_listing_known_models() -> None:
    with pytest.raises(ValueError) as exc_info:
        resolve_model("llama3")
    message = str(exc_info.value)
    assert "llama3" in message
    assert "deepseek-chat" in message
    assert "deepseek-reasoner" in message


# --------------------------------------------------------------------------
# last_user_message
# --------------------------------------------------------------------------


def test_last_user_message_returns_final_user_content() -> None:
    messages = [
        OllamaMessage(role="system", content="be helpful"),
        OllamaMessage(role="user", content="first"),
        OllamaMessage(role="assistant", content="reply"),
        OllamaMessage(role="user", content="second"),
    ]
    assert last_user_message(messages) == "second"


def test_last_user_message_raises_on_empty_list() -> None:
    with pytest.raises(ValueError):
        last_user_message([])


def test_last_user_message_raises_if_last_role_not_user() -> None:
    messages = [
        OllamaMessage(role="user", content="hi"),
        OllamaMessage(role="assistant", content="hello"),
    ]
    with pytest.raises(ValueError):
        last_user_message(messages)


# --------------------------------------------------------------------------
# history_prefix_key
# --------------------------------------------------------------------------


def test_history_prefix_key_same_history_same_key() -> None:
    messages_a = [
        OllamaMessage(role="user", content="hi"),
        OllamaMessage(role="assistant", content="hello"),
        OllamaMessage(role="user", content="how are you"),
    ]
    messages_b = [
        OllamaMessage(role="user", content="hi"),
        OllamaMessage(role="assistant", content="hello"),
        OllamaMessage(role="user", content="a different new message"),
    ]
    # Same prefix (messages[:-1]) in both - only the final new user turn
    # differs, which history_prefix_key deliberately excludes.
    assert history_prefix_key("deepseek-chat", messages_a) == history_prefix_key("deepseek-chat", messages_b)


def test_history_prefix_key_different_history_different_key() -> None:
    messages_a = [
        OllamaMessage(role="user", content="hi"),
        OllamaMessage(role="user", content="new"),
    ]
    messages_b = [
        OllamaMessage(role="user", content="bye"),
        OllamaMessage(role="user", content="new"),
    ]
    assert history_prefix_key("deepseek-chat", messages_a) != history_prefix_key("deepseek-chat", messages_b)


def test_history_prefix_key_different_model_different_key() -> None:
    messages = [
        OllamaMessage(role="user", content="hi"),
        OllamaMessage(role="user", content="new"),
    ]
    assert history_prefix_key("deepseek-chat", messages) != history_prefix_key("deepseek-reasoner", messages)
