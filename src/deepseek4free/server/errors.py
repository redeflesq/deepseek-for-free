"""Exception -> HTTPException translation for the chat server.

Replaces old/example.py's _translate_exception(), which callers had to
remember to invoke manually (`raise _translate_exception(e)`) at every call
site that could raise a DeepSeekError. That's easy to forget at a new call
site and was in fact already inconsistently applied in the old code (some
routes wrapped calls in try/except, others didn't and relied on FastAPI's
default 500).

Here the same mapping is registered as FastAPI exception_handlers, so it
applies uniformly to every route without each one needing its own
try/except - a route can simply let a DeepSeekError/FileNotFoundError/
ValueError propagate and get the exact same status code/detail mapping as
before. Handlers are registered in server/app.py via register_exception_handlers(app).
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from deepseek4free.exceptions import (
    APIError,
    AuthenticationError,
    CloudflareError,
    NetworkError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


def _json_error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


async def _handle_authentication_error(request: Request, exc: AuthenticationError) -> JSONResponse:
    logger.warning("AuthenticationError: %s", exc)
    return _json_error(401, str(exc))


async def _handle_rate_limit_error(request: Request, exc: RateLimitError) -> JSONResponse:
    logger.warning("RateLimitError: %s", exc)
    return _json_error(429, str(exc))


async def _handle_cloudflare_error(request: Request, exc: CloudflareError) -> JSONResponse:
    logger.warning("CloudflareError: %s", exc)
    return _json_error(503, str(exc))


async def _handle_network_error(request: Request, exc: NetworkError) -> JSONResponse:
    logger.warning("NetworkError: %s", exc)
    return _json_error(502, str(exc))


async def _handle_api_error(request: Request, exc: APIError) -> JSONResponse:
    logger.warning("APIError (status=%s): %s", exc.status_code, exc)
    return _json_error(exc.status_code or 502, str(exc))


async def _handle_file_not_found_error(request: Request, exc: FileNotFoundError) -> JSONResponse:
    logger.warning("FileNotFoundError: %s", exc)
    return _json_error(404, str(exc))


async def _handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
    logger.warning("ValueError: %s", exc)
    return _json_error(422, str(exc))


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # Truly unexpected - log with full traceback so it's diagnosable from
    # container logs, but still return a clean JSON body instead of letting
    # it propagate as a bare ASGI-level 500.
    logger.exception("Unexpected error in DeepSeek request handling")
    return _json_error(500, f"Unexpected error: {exc}")


def register_exception_handlers(app: FastAPI) -> None:
    """Registers every DeepSeek-domain exception handler on the given app.

    Order matters for FastAPI's Starlette-based dispatch only in the sense
    that more specific exception types should be registered - here each
    type is distinct (no shared subclassing among AuthenticationError/
    RateLimitError/CloudflareError/NetworkError/APIError beyond their common
    DeepSeekError base, which is NOT registered separately to avoid masking
    the more specific handlers above), so registration order does not
    actually matter here, but is kept in the same order as the old
    if/elif chain for readability.
    """
    app.add_exception_handler(AuthenticationError, _handle_authentication_error)
    app.add_exception_handler(RateLimitError, _handle_rate_limit_error)
    app.add_exception_handler(CloudflareError, _handle_cloudflare_error)
    app.add_exception_handler(NetworkError, _handle_network_error)
    app.add_exception_handler(APIError, _handle_api_error)
    app.add_exception_handler(FileNotFoundError, _handle_file_not_found_error)
    app.add_exception_handler(ValueError, _handle_value_error)
    # HTTPException raised explicitly by route code (e.g. SessionManager's
    # 404s) must keep FastAPI's own default handling - do NOT override it
    # here, otherwise those intentional 404s would be swallowed by
    # _handle_unexpected_error since HTTPException is also an Exception
    # subclass evaluated after all the specific ones above.
    app.add_exception_handler(Exception, _handle_unexpected_error)
