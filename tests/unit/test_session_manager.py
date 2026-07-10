"""Unit tests for deepseek4free.server.session_manager - no network needed.

Uses a minimal fake in place of DeepSeekAPI (matching only the four methods
SessionManager actually calls: create_chat_session, upload_file,
fetch_file_status, chat_completion) instead of importing the real
DeepSeekAPI, since constructing a real one requires a valid cookies path /
auth token and would make these "unit" tests secretly depend on filesystem
state.
"""

from collections.abc import Generator
from typing import Any

import pytest
from fastapi import HTTPException

from deepseek4free.server.session_manager import SessionManager


class FakeDeepSeekAPI:
    """Stands in for DeepSeekAPI: same method surface SessionManager calls,
    scripted responses instead of real network calls."""

    def __init__(self) -> None:
        self.created_session_ids = iter(f"session-{i}" for i in range(1, 1000))
        self.chat_completion_calls: list[dict[str, Any]] = []
        self.next_chunks: list[dict[str, Any]] = []

    def create_chat_session(self) -> str:
        return next(self.created_session_ids)

    def upload_file(self, file_path: str) -> dict[str, Any]:
        return {"id": "file-abc", "status": "PENDING", "file_name": "x.txt", "file_size": 3}

    def fetch_file_status(self, file_ids: list[str]) -> list[dict[str, Any]]:
        return [{"id": fid, "status": "SUCCESS"} for fid in file_ids]

    def chat_completion(
        self,
        chat_session_id: str,
        prompt: str,
        parent_message_id: str | None = None,
        thinking_enabled: bool = True,
        search_enabled: bool = False,
        file_ids: list[str] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        self.chat_completion_calls.append(
            {
                "chat_session_id": chat_session_id,
                "prompt": prompt,
                "parent_message_id": parent_message_id,
                "file_ids": file_ids,
            }
        )
        yield from self.next_chunks


def _manager() -> tuple[SessionManager, FakeDeepSeekAPI]:
    fake_api = FakeDeepSeekAPI()
    return SessionManager(fake_api), fake_api  # type: ignore[arg-type]


def test_create_and_get_session_round_trip() -> None:
    manager, _ = _manager()
    session = manager.create_session()
    fetched = manager.get_session(session.deepseek_session_id)
    assert fetched is session


def test_get_missing_session_raises_404() -> None:
    manager, _ = _manager()
    with pytest.raises(HTTPException) as exc_info:
        manager.get_session("does-not-exist")
    assert exc_info.value.status_code == 404


def test_list_sessions_reflects_message_count() -> None:
    manager, fake_api = _manager()
    session = manager.create_session()
    fake_api.next_chunks = [{"type": "text", "content": "hi", "finish_reason": "stop"}]
    list(manager.send_message(session, "hello", True, False))

    infos = manager.list_sessions()
    assert len(infos) == 1
    # one user message + one assistant message appended by send_message
    assert infos[0].message_count == 2


def test_delete_session_removes_it() -> None:
    manager, _ = _manager()
    session = manager.create_session()
    manager.delete_session(session.deepseek_session_id)
    with pytest.raises(HTTPException):
        manager.get_session(session.deepseek_session_id)


def test_delete_missing_session_raises_404() -> None:
    manager, _ = _manager()
    with pytest.raises(HTTPException) as exc_info:
        manager.delete_session("nope")
    assert exc_info.value.status_code == 404


def test_send_message_appends_user_then_assistant_history() -> None:
    manager, fake_api = _manager()
    session = manager.create_session()
    fake_api.next_chunks = [
        {"type": "meta", "parent_message_id": "msg-2"},
        {"type": "text", "content": "Hello ", "finish_reason": None},
        {"type": "text", "content": "world", "finish_reason": "stop"},
    ]

    chunks = list(manager.send_message(session, "hi there", True, False))

    assert len(chunks) == 3
    assert session.parent_message_id == "msg-2"
    assert len(session.history) == 2
    assert session.history[0].role == "user"
    assert session.history[0].content == "hi there"
    assert session.history[1].role == "assistant"
    assert session.history[1].content == "Hello world"


def test_send_message_saves_partial_history_on_stream_error() -> None:
    """Regression case explicitly called out in session_manager.py's
    docstring: history must be saved even if the stream is cut off by an
    exception mid-way (the `finally` block)."""

    def failing_chat_completion(*args: Any, **kwargs: Any) -> Generator[dict[str, Any], None, None]:
        yield {"type": "text", "content": "partial", "finish_reason": None}
        raise RuntimeError("connection dropped")

    manager, fake_api = _manager()
    fake_api.chat_completion = failing_chat_completion  # type: ignore[method-assign]
    session = manager.create_session()

    with pytest.raises(RuntimeError):
        list(manager.send_message(session, "hi", True, False))

    assert len(session.history) == 2
    assert session.history[1].role == "assistant"
    assert session.history[1].content == "partial"


def test_send_message_passes_file_ids_through() -> None:
    manager, fake_api = _manager()
    session = manager.create_session()
    fake_api.next_chunks = [{"type": "text", "content": "ok", "finish_reason": "stop"}]

    list(manager.send_message(session, "hi", True, False, file_ids=["file-1", "file-2"]))

    assert fake_api.chat_completion_calls[0]["file_ids"] == ["file-1", "file-2"]


def test_upload_file_and_fetch_file_status_delegate_to_api() -> None:
    manager, _ = _manager()
    record = manager.upload_file("/tmp/whatever.txt")
    assert record["id"] == "file-abc"

    statuses = manager.fetch_file_status(["file-abc"])
    assert statuses == [{"id": "file-abc", "status": "SUCCESS"}]
