"""Pydantic request/response models for the chat HTTP server.

Moved 1:1 out of old/example.py (previously all endpoint models, ChatSession
and SessionManager lived in a single 450-line file). Only the models live
here; ChatSession/SessionManager moved to server/session_manager.py so this
module can be imported by both the routes and the session manager without a
circular dependency (session_manager.py needs ChatMessage from here).
"""

from typing import Any

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    character_id: str | None = None


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: float


class SessionInfo(BaseModel):
    session_id: str
    created_at: float
    message_count: int


class SendMessageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User message text")
    thinking_enabled: bool = True
    search_enabled: bool = False
    stream: bool = False
    file_ids: list[str] = Field(
        default_factory=list,
        description="Ids of files previously uploaded via POST /sessions/{id}/files "
        "to attach to this message (DeepSeek ref_file_ids).",
    )


class FileUploadResponse(BaseModel):
    file_id: str
    status: str
    file_name: str | None = None
    file_size: int | None = None


class FileStatusResponse(BaseModel):
    files: list[dict[str, Any]]


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    thinking: str | None = None
    created_at: float


class SendMessageResponse(BaseModel):
    session_id: str
    content: str
    thinking: str | None = None


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    auth_token_configured: bool
    cookies_loaded: bool
    active_sessions: int
    detail: str | None = None
