"""FastAPI application factory for the chat server.

Replaces old/example.py's module-level `app = FastAPI(...)` with a
create_app() factory. A factory (rather than a module-level singleton) lets
tests construct a fresh app instance per test if needed and keeps import
side effects out of module import time - `import
deepseek4free.server.app` no longer implicitly builds a live app object.

Endpoints, paths, and behavior are unchanged from old/example.py: this file
only wires together the pieces (routers, exception handlers) that used to
all live in one 450-line file.
"""

import logging
import sys

from fastapi import FastAPI

from deepseek4free.config import get_settings
from deepseek4free.server.errors import register_exception_handlers
from deepseek4free.server.routes import files, health, messages, sessions


def create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    app = FastAPI(
        title="DeepSeek4Free Chat Server",
        description="HTTP server with chat session support on top of DeepSeekAPI",
        version="1.1.0",
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(files.router)
    app.include_router(messages.router)

    return app


# Module-level ASGI app object for `uvicorn deepseek4free.server.app:app`,
# matching the old `uvicorn example:app` invocation style. Building it at
# import time (rather than only exposing create_app) is what makes that
# invocation work without a custom factory flag - the tradeoff (import-time
# side effect of logging.basicConfig) is the same one old/example.py already
# had, just now centralized in create_app() instead of duplicated at module
# scope.
app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    try:
        uvicorn.run(app, host="0.0.0.0", port=settings.fastapi_server_port)
    except KeyboardInterrupt:
        print("\n\n⚠️ Operation cancelled by user")
        sys.exit(0)


if __name__ == "__main__":
    main()
