"""POST /sessions/{id}/messages - send a chat message, streaming or not."""

import json
import logging
from collections.abc import Generator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from deepseek4free.server.dependencies import get_session_manager
from deepseek4free.server.schemas import SendMessageRequest, SendMessageResponse
from deepseek4free.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["messages"])


@router.post("/sessions/{session_id}/messages")
def send_message(
    session_id: str,
    req: SendMessageRequest,
    manager: SessionManager = Depends(get_session_manager),
):
    session = manager.get_session(session_id)

    if req.stream:

        def event_stream() -> Generator[str, None, None]:
            try:
                for chunk in manager.send_message(
                    session, req.prompt, req.thinking_enabled, req.search_enabled, req.file_ids
                ):
                    payload = {
                        "type": chunk.get("type"),
                        "content": chunk.get("content", ""),
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    if chunk.get("finish_reason") == "stop":
                        break
                yield "event: done\ndata: {}\n\n"
            except Exception as e:
                # A streaming response has already sent a 200 status line by
                # the time an error can occur mid-stream, so it can't be
                # turned into an HTTP error status anymore - the only option
                # is to signal it in-band as an SSE `error` event, same as
                # the old code did.
                logger.exception("Error while streaming message for session %s", session_id)
                error_payload = {"error": str(e)}
                yield f"event: error\ndata: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming: collect the full response and return one JSON body.
    # Exceptions here propagate to the global exception handlers
    # (server/errors.py) instead of being caught locally, since the
    # response hasn't started yet and a normal HTTP error status is still
    # possible.
    text_parts = []
    thinking_parts = []
    for chunk in manager.send_message(
        session, req.prompt, req.thinking_enabled, req.search_enabled, req.file_ids
    ):
        if chunk.get("type") == "text" and chunk.get("content"):
            text_parts.append(chunk["content"])
        elif chunk.get("type") == "thinking" and chunk.get("content"):
            thinking_parts.append(chunk["content"])

    return SendMessageResponse(
        session_id=session_id,
        content="".join(text_parts),
        thinking="".join(thinking_parts) or None,
    )
