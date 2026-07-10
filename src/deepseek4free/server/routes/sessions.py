"""POST/GET /sessions, GET /sessions/{id}/history, DELETE /sessions/{id}."""


from fastapi import APIRouter, Depends

from deepseek4free.server.dependencies import get_session_manager
from deepseek4free.server.schemas import (
    ChatMessage,
    CreateSessionRequest,
    CreateSessionResponse,
    SessionInfo,
)
from deepseek4free.server.session_manager import SessionManager

router = APIRouter(tags=["sessions"])


@router.post("/sessions", response_model=CreateSessionResponse)
def create_session(
    _: CreateSessionRequest = CreateSessionRequest(),
    manager: SessionManager = Depends(get_session_manager),
) -> CreateSessionResponse:
    session = manager.create_session()
    return CreateSessionResponse(session_id=session.deepseek_session_id, created_at=session.created_at)


@router.get("/sessions", response_model=list[SessionInfo])
def list_sessions(manager: SessionManager = Depends(get_session_manager)) -> list[SessionInfo]:
    return manager.list_sessions()


@router.get("/sessions/{session_id}/history", response_model=list[ChatMessage])
def get_history(
    session_id: str, manager: SessionManager = Depends(get_session_manager)
) -> list[ChatMessage]:
    session = manager.get_session(session_id)
    return session.history


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str, manager: SessionManager = Depends(get_session_manager)
) -> dict[str, str]:
    manager.delete_session(session_id)
    return {"status": "deleted", "session_id": session_id}
