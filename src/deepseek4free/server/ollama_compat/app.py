"""FastAPI application factory for the Ollama-compatible HTTP layer.

Mirrors server/app.py's create_app() pattern exactly (factory function +
module-level `app` object for `uvicorn ...:app` invocation + main() entry
point), but builds a SEPARATE FastAPI application, mounted on a separate
port (settings.ollama_compat_port, default 11434) via a separate uvicorn
process. This is NOT a router registered on the native server's app in
server/app.py - the two apps are independent ASGI applications that happen
to share the same underlying SessionManager/DeepSeekAPI singleton (via
server/dependencies.py's module-level lazy init), so sessions created
through either one are visible to both.
"""

import logging
import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from deepseek4free.config import get_settings
from deepseek4free.exceptions import (
    APIError,
    AuthenticationError,
    CloudflareError,
    NetworkError,
    RateLimitError,
)
from deepseek4free.server.ollama_compat.routes import router as ollama_router

logger = logging.getLogger(__name__)

# server/errors.py's register_exception_handlers() is deliberately NOT reused
# here as-is: it produces {"detail": ...} bodies (FastAPI's own convention),
# while every other error response in this router - see routes.py's
# _error_response() - uses Ollama's {"error": ...} shape instead, since
# that's the field Ollama clients actually read. Without SOME handler for
# these exception types, a DeepSeek-side failure (expired token, Cloudflare
# block, network error) raised deep inside manager.send_message() during a
# non-streaming /api/chat or /api/generate call would propagate unhandled
# past routes.py's narrow try/excepts (which only catch resolve_model's and
# last_user_message's ValueError) all the way to Starlette's default 500
# handler - a bare "Internal Server Error" with no JSON body at all, not
# even FastAPI's usual {"detail": ...}. The streaming branches already guard
# against this with their own try/except around the generator; this covers
# the non-streaming branches and any other route in this router.


def _ollama_json_error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


async def _handle_authentication_error(request: Request, exc: AuthenticationError) -> JSONResponse:
    logger.warning("AuthenticationError: %s", exc)
    return _ollama_json_error(401, str(exc))


async def _handle_rate_limit_error(request: Request, exc: RateLimitError) -> JSONResponse:
    logger.warning("RateLimitError: %s", exc)
    return _ollama_json_error(429, str(exc))


async def _handle_cloudflare_error(request: Request, exc: CloudflareError) -> JSONResponse:
    logger.warning("CloudflareError: %s", exc)
    return _ollama_json_error(503, str(exc))


async def _handle_network_error(request: Request, exc: NetworkError) -> JSONResponse:
    logger.warning("NetworkError: %s", exc)
    return _ollama_json_error(502, str(exc))


async def _handle_api_error(request: Request, exc: APIError) -> JSONResponse:
    logger.warning("APIError (status=%s): %s", exc.status_code, exc)
    return _ollama_json_error(exc.status_code or 502, str(exc))


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unexpected error in Ollama-compat request handling")
    return _ollama_json_error(500, f"Unexpected error: {exc}")


def _register_ollama_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AuthenticationError, _handle_authentication_error)
    app.add_exception_handler(RateLimitError, _handle_rate_limit_error)
    app.add_exception_handler(CloudflareError, _handle_cloudflare_error)
    app.add_exception_handler(NetworkError, _handle_network_error)
    app.add_exception_handler(APIError, _handle_api_error)
    # HTTPException raised explicitly by route/dependency code (e.g.
    # get_session_manager()'s 500 when DEEPSEEK_AUTH_TOKEN is unset) keeps
    # FastAPI's own default handling - not overridden here, for the same
    # reason server/errors.py doesn't override it either.
    app.add_exception_handler(Exception, _handle_unexpected_error)


def create_ollama_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    app = FastAPI(
        title="DeepSeek4Free Ollama-Compatible API",
        description=(
            "Ollama-compatible HTTP endpoints (/api/chat, /api/generate, /api/tags, "
            "/api/show, /api/version, /api/ps) on top of the same DeepSeekAPI/"
            "SessionManager used by the native chat server (see server/app.py). "
            "Intended for Ollama-speaking clients such as Continue.dev, Open WebUI, "
            "or langchain's ChatOllama."
        ),
        version="1.1.0",
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_ollama_exception_handlers(app)
    app.include_router(ollama_router)

    return app


# Module-level ASGI app object for `uvicorn deepseek4free.server.ollama_compat.app:app`,
# matching server/app.py's own invocation style. Built unconditionally at
# import time even when ENABLE_OLLAMA_API=false - constructing the app object
# itself (router registration) has no side effects worth gating; only main()
# actually binds a port, and that's where the enable check happens.
app = create_ollama_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    if not settings.enable_ollama_api:
        # Exits cleanly (not an error) so entrypoint.sh can launch this
        # process unconditionally in Docker without needing its own
        # ENABLE_OLLAMA_API check duplicated in shell - the process starts,
        # logs why it's not listening, and exits 0 instead of occupying a
        # port nobody asked for.
        print("Ollama-compat API disabled via ENABLE_OLLAMA_API=false")
        sys.exit(0)

    try:
        uvicorn.run(app, host="0.0.0.0", port=settings.ollama_compat_port)
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
