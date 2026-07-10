"""POST /sessions/{id}/files (upload), GET /files/status."""

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from deepseek4free.config import get_settings
from deepseek4free.server.dependencies import get_session_manager
from deepseek4free.server.schemas import FileStatusResponse, FileUploadResponse
from deepseek4free.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["files"])


@router.post("/sessions/{session_id}/files", response_model=FileUploadResponse)
async def upload_file(
    session_id: str,
    file: UploadFile = File(...),
    manager: SessionManager = Depends(get_session_manager),
) -> FileUploadResponse:
    """Uploads a file to DeepSeek and returns its file_id for later use in
    file_ids when POSTing /sessions/{session_id}/messages.

    session_id in the path only verifies the session exists (to match the
    REST style of the other /sessions/{id}/... endpoints) - the file itself
    is not tied to a session on DeepSeek's side.

    The uploaded file may come back with status 'PENDING' (DeepSeek is still
    parsing it asynchronously) - poll GET /files/status before including it
    in a message's file_ids until status becomes 'SUCCESS'.
    """
    manager.get_session(session_id)  # raises 404 if the session doesn't exist

    # DeepSeekAPI.upload_file() opens a file by path for multipart re-upload
    # to DeepSeek, while FastAPI hands us the uploaded file as an async
    # stream with no guaranteed on-disk path - so it's saved to a temp file
    # first to get a real path for the underlying client, and removed in
    # finally regardless of outcome.
    max_upload_bytes = get_settings().max_upload_bytes
    suffix = Path(file.filename).suffix if file.filename else ""
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            total_bytes = 0
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {max_upload_bytes} byte upload limit",
                    )
                tmp.write(chunk)

        record = manager.upload_file(tmp_path)
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.warning("Failed to remove temp upload file %s: %s", tmp_path, e)

    return FileUploadResponse(
        file_id=record["id"],
        status=record.get("status", "PENDING"),
        file_name=record.get("file_name"),
        file_size=record.get("file_size"),
    )


@router.get("/files/status", response_model=FileStatusResponse)
def get_file_status(
    file_ids: str, manager: SessionManager = Depends(get_session_manager)
) -> FileStatusResponse:
    """Checks parsing status of previously uploaded files.

    file_ids is a comma-separated list, e.g. ?file_ids=file-aaa,file-bbb.
    Status of each file is PENDING (still parsing) | SUCCESS | FAILED.
    """
    ids = [f.strip() for f in file_ids.split(",") if f.strip()]
    if not ids:
        raise HTTPException(status_code=422, detail="file_ids must contain at least one non-empty id")
    files = manager.fetch_file_status(ids)
    return FileStatusResponse(files=files)
