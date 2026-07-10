"""In-memory chat session state on top of DeepSeekAPI.

ChatSession/SessionManager moved 1:1 out of old/example.py. Locking
semantics are unchanged: SessionManager._global_lock guards the sessions
dict itself, while each ChatSession.lock serializes concurrent
send_message() calls against *that* session (DeepSeek threads messages via
parent_message_id, so two concurrent requests into the same session would
race on which message becomes the parent of the next one). DeepSeekAPI is
itself thread-safe (see client/transport.py Transport.lock) - this class
only protects its own dict and per-session history from concurrent HTTP
requests, it does not rely on the lower layer also doing so.

Persistence is intentionally in-memory only: a process restart drops all
sessions. Adding Redis/DB-backed persistence would be infrastructure the
current single-process deployment does not need; if that changes later, the
SessionManager interface (create/get/list/delete) is the natural seam to
swap in a persistent backend.
"""

import threading
import time
from collections.abc import Generator
from typing import Any

from fastapi import HTTPException

from deepseek4free.client.api import DeepSeekAPI
from deepseek4free.server.schemas import ChatMessage, SessionInfo


class ChatSession:
    """Local state for one chat session: the DeepSeek-side id,
    parent_message_id for threading, and display history."""

    def __init__(self, deepseek_session_id: str) -> None:
        self.deepseek_session_id = deepseek_session_id
        self.parent_message_id: str | None = None
        self.history: list[ChatMessage] = []
        self.created_at = time.time()
        self.lock = threading.Lock()


class SessionManager:
    """Thread-safe in-memory store of chat sessions."""

    def __init__(self, api: DeepSeekAPI) -> None:
        self._api = api
        self._sessions: dict[str, ChatSession] = {}
        self._global_lock = threading.Lock()

    def create_session(self, character_id: str | None = None) -> ChatSession:
        deepseek_session_id = self._api.create_chat_session()
        session = ChatSession(deepseek_session_id)
        with self._global_lock:
            self._sessions[deepseek_session_id] = session
        return session

    def get_session(self, session_id: str) -> ChatSession:
        with self._global_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        return session

    def list_sessions(self) -> list[SessionInfo]:
        with self._global_lock:
            sessions = list(self._sessions.values())
        return [
            SessionInfo(
                session_id=s.deepseek_session_id,
                created_at=s.created_at,
                message_count=len(s.history),
            )
            for s in sessions
        ]

    def session_count(self) -> int:
        with self._global_lock:
            return len(self._sessions)

    def delete_session(self, session_id: str) -> None:
        with self._global_lock:
            if session_id not in self._sessions:
                raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
            del self._sessions[session_id]

    def upload_file(self, file_path: str) -> dict[str, Any]:
        """Uploads via the base DeepSeekAPI - not tied to a specific chat
        session on DeepSeek's side (files are only attached to a message via
        ref_file_ids at send time)."""
        return self._api.upload_file(file_path)

    def fetch_file_status(self, file_ids: list[str]) -> list[dict[str, Any]]:
        return self._api.fetch_file_status(file_ids)

    def send_message(
        self,
        session: ChatSession,
        prompt: str,
        thinking_enabled: bool,
        search_enabled: bool,
        file_ids: list[str] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Yields response chunks. One in-flight request per session at a
        time (session.lock), since DeepSeek threads parent_message_id
        sequentially and concurrent requests into one session would race on
        message order."""
        with session.lock:
            session.history.append(
                ChatMessage(role="user", content=prompt, created_at=time.time())
            )

            text_parts: list[str] = []
            thinking_parts: list[str] = []

            try:
                for chunk in self._api.chat_completion(
                    session.deepseek_session_id,
                    prompt,
                    parent_message_id=session.parent_message_id,
                    thinking_enabled=thinking_enabled,
                    search_enabled=search_enabled,
                    file_ids=file_ids,
                ):
                    if chunk.get("type") == "meta" and chunk.get("parent_message_id"):
                        session.parent_message_id = chunk["parent_message_id"]
                    elif chunk.get("type") == "text" and chunk.get("content"):
                        text_parts.append(chunk["content"])
                    elif chunk.get("type") == "thinking" and chunk.get("content"):
                        thinking_parts.append(chunk["content"])

                    yield chunk
            finally:
                # Save whatever was received even if the stream was cut off
                # or errored mid-way.
                if text_parts or thinking_parts:
                    session.history.append(
                        ChatMessage(
                            role="assistant",
                            content="".join(text_parts),
                            thinking="".join(thinking_parts) or None,
                            created_at=time.time(),
                        )
                    )
