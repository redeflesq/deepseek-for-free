"""High-level DeepSeek chat client.

Thin façade over Transport (HTTP/retries/cookies) + DeepSeekPOW (proof of
work) + client.sse (stream parsing). Only business methods live here:
session creation, file upload/status, and streaming chat completion -
everything about *how* a request is sent (retries, Cloudflare detection,
cookie refresh) belongs to Transport instead.
"""

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

from curl_cffi import CurlMime, requests

from deepseek4free.client.sse import parse_sse_line
from deepseek4free.client.transport import (
    BASE_URL,
    FILE_UPLOAD_TIMEOUT,
    STREAMING_REQUEST_TIMEOUT,
    Transport,
)
from deepseek4free.exceptions import (
    APIError,
    AuthenticationError,
    NetworkError,
    RateLimitError,
)
from deepseek4free.pow.solver import DeepSeekPOW

logger = logging.getLogger(__name__)


class DeepSeekAPI:
    """Public entry point - same surface as the old dsk.api.DeepSeekAPI:
    create_chat_session(), upload_file(), fetch_file_status(), chat_completion().
    """

    def __init__(self, auth_token: str, cookies_path: Path | None = None) -> None:
        if cookies_path is None:
            # Imported lazily (not at module top) to avoid a hard import-time
            # dependency on Settings for callers who construct DeepSeekAPI
            # with an explicit cookies_path (e.g. tests) and don't want
            # pydantic-settings to read the environment at all.
            from deepseek4free.config import get_settings
            cookies_path = get_settings().cookies_path

        self._transport = Transport(auth_token, cookies_path)
        self.pow_solver = DeepSeekPOW()

    @property
    def cookies(self) -> dict[str, str]:
        return self._transport.cookies

    def _get_pow_challenge_locked(self, target_path: str) -> dict[str, Any]:
        """Must be called with self._transport.lock already held."""
        response = self._transport.request_locked(
            "POST", "/chat/create_pow_challenge", {"target_path": target_path}
        )
        try:
            return response["data"]["biz_data"]["challenge"]
        except (KeyError, TypeError) as e:
            raise APIError(f"Invalid challenge response format from server: {e}") from e

    def create_chat_session(self) -> str:
        """Creates a new chat session and returns the session ID."""
        response = self._transport.request(
            "POST", "/chat_session/create", {"character_id": None}
        )
        try:
            return response["data"]["biz_data"]["id"]
        except (KeyError, TypeError) as e:
            raise APIError(f"Invalid session creation response format from server: {e}") from e

    def upload_file(self, file_path: str) -> dict[str, Any]:
        """Uploads a file to DeepSeek so it can be referenced in a later
        chat_completion() call via file_ids.

        Mirrors the real web client's POST /api/v0/file/upload_file, which
        requires a PoW challenge solved against that specific target_path
        (different from the chat completion challenge) plus an
        x-file-size header, and returns the file record while DeepSeek is
        still parsing it server-side (status == 'PENDING').

        Callers that need to reference this file in chat_completion()
        should poll fetch_file_status() until status is 'SUCCESS' - DeepSeek
        rejects ref_file_ids that are still PENDING.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {file_path}")

        file_size = path.stat().st_size

        # Everything below touches self._transport.cookies/pow_solver and
        # issues a real network call - hold the lock for the whole
        # operation so a concurrent upload_file()/chat_completion() from
        # another thread can't interleave a cookie refresh mid-way through
        # this one.
        with self._transport.lock:
            pow_response = self.pow_solver.solve_challenge(
                self._get_pow_challenge_locked(target_path="/api/v0/file/upload_file")
            )

            headers = self._transport.get_headers(pow_response=pow_response)
            # content-type is set by curl_cffi per multipart part when using
            # `multipart=`; sending our own default JSON content-type header
            # would conflict with the multipart boundary, so drop it here.
            headers.pop("content-type", None)
            headers["x-file-size"] = str(file_size)

            mime = CurlMime()
            mime.addpart(name="file", filename=path.name, local_path=str(path))

            try:
                response = requests.post(
                    f"{BASE_URL}/file/upload_file",
                    headers=headers,
                    multipart=mime,
                    cookies=self._transport.cookies,
                    impersonate="chrome120",
                    timeout=FILE_UPLOAD_TIMEOUT,
                )
            except requests.exceptions.RequestException as e:
                raise NetworkError(f"Network error occurred during file upload: {e}") from e

            self._transport.raise_for_status(response)
            data = self._transport.parse_json(response)
            try:
                return data["data"]["biz_data"]
            except (KeyError, TypeError) as e:
                raise APIError(
                    f"Invalid file upload response format from server: {e} "
                    f"(body: {response.text[:500]!r})"
                ) from e

    def fetch_file_status(self, file_ids: list[str]) -> list[dict[str, Any]]:
        """Queries parsing status for one or more previously uploaded files.

        Note: file_ids is sent as repeated query parameters
        (file_ids=a&file_ids=b), not as a single comma-joined value -
        DeepSeek's server rejects the latter with a deserialize error.
        """
        if not file_ids:
            raise ValueError("file_ids must be a non-empty list")

        with self._transport.lock:
            headers = self._transport.get_headers()
            try:
                response = requests.get(
                    f"{BASE_URL}/file/fetch_files",
                    headers=headers,
                    params={"file_ids": file_ids},
                    cookies=self._transport.cookies,
                    impersonate="chrome120",
                    timeout=60.0,
                )
            except requests.exceptions.RequestException as e:
                raise NetworkError(f"Network error occurred while fetching file status: {e}") from e

            self._transport.raise_for_status(response)
            data = self._transport.parse_json(response)
            try:
                return data["data"]["biz_data"]["files"]
            except (KeyError, TypeError) as e:
                raise APIError(
                    f"Invalid file status response format from server: {e} "
                    f"(body: {response.text[:500]!r})"
                ) from e

    def chat_completion(
        self,
        chat_session_id: str,
        prompt: str,
        parent_message_id: str | None = None,
        thinking_enabled: bool = True,
        search_enabled: bool = False,
        file_ids: list[str] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Sends a message and yields streaming response chunks.

        Args:
            chat_session_id: The ID of the chat session.
            prompt: The message to send.
            parent_message_id: ID of the parent message for threading.
            thinking_enabled: Whether to show the thinking process.
            search_enabled: Whether to enable web search.
            file_ids: Ids of files previously uploaded via upload_file() to
                attach to this message (DeepSeek's ref_file_ids). Files
                should be in 'SUCCESS' status before being referenced here.

        Yields:
            Dict[str, Any]: chunks with 'type' ('text' | 'thinking' | 'meta'),
            'content', and 'finish_reason'.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string")
        if not chat_session_id or not isinstance(chat_session_id, str):
            raise ValueError("Chat session ID must be a non-empty string")

        json_data = {
            "chat_session_id": chat_session_id,
            "parent_message_id": parent_message_id,
            "prompt": prompt,
            "ref_file_ids": file_ids or [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
        }

        # Only the setup (PoW challenge + solving it) needs the lock, since
        # it reads self._transport.cookies/pow_solver. The actual streaming
        # HTTP call below is issued with a snapshot of headers/cookies
        # already baked in, so we don't hold the lock for the (potentially
        # long) duration of the whole stream - that would block every other
        # request (uploads, other sessions' messages) for as long as this
        # one keeps streaming, defeating the purpose of the lock (races on
        # shared state) by turning it into an unrelated throughput cap.
        with self._transport.lock:
            pow_response = self.pow_solver.solve_challenge(
                self._get_pow_challenge_locked(target_path="/api/v0/chat/completion")
            )
            headers = self._transport.get_headers(pow_response=pow_response)
            cookies_snapshot = dict(self._transport.cookies)

        try:
            response = requests.post(
                f"{BASE_URL}/chat/completion",
                headers=headers,
                json=json_data,
                cookies=cookies_snapshot,
                impersonate="chrome120",
                stream=True,
                timeout=STREAMING_REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                error_text = next(response.iter_lines(), b"").decode("utf-8", "ignore")
                if response.status_code == 401:
                    raise AuthenticationError("Invalid or expired authentication token")
                if response.status_code == 429:
                    raise RateLimitError("API rate limit exceeded")
                raise APIError(f"API request failed: {error_text}", response.status_code)

            state: dict[str, Any] = {}
            for line in response.iter_lines():
                if not line:
                    continue
                for parsed in parse_sse_line(line, state):
                    if parsed:
                        yield parsed
                        if parsed.get("finish_reason") == "stop":
                            return

        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error occurred during streaming: {e}") from e
