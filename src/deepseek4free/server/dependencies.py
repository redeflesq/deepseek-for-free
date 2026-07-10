"""Lazy singleton wiring for DeepSeekAPI / SessionManager.

Factored out of old/example.py's module-level `_api`/`_session_manager`
globals + `get_session_manager()` function so routes can depend on it via
FastAPI's `Depends()` instead of importing a global from the main app
module (which would create a routes -> app -> routes import cycle once
routes are split into their own files).

Initialization stays lazy and double-checked-locked, exactly as before:
the first request that needs DeepSeekAPI constructs it (reading the auth
token from Settings instead of a raw os.getenv call), and every
subsequent call reuses the same instance. This also means a server that
never receives a request never has to have a valid token configured -
useful for e.g. importing the app in tests that only check schema shapes.
"""

import logging
import threading

from fastapi import HTTPException

from deepseek4free.client.api import DeepSeekAPI
from deepseek4free.config import get_settings
from deepseek4free.exceptions import AuthenticationError
from deepseek4free.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

_api: DeepSeekAPI | None = None
_session_manager: SessionManager | None = None
_init_lock = threading.Lock()


def get_deepseek_api() -> DeepSeekAPI | None:
    """Returns the initialized DeepSeekAPI instance, or None if
    get_session_manager() has never successfully run. Used by health()
    to inspect cookie state without forcing initialization itself."""
    return _api


def get_session_manager() -> SessionManager:
    global _api, _session_manager
    if _session_manager is not None:
        return _session_manager
    with _init_lock:
        # Re-check inside the lock: another thread may have finished
        # initialization while this one was waiting on _init_lock.
        if _session_manager is not None:
            return _session_manager
        settings = get_settings()
        if not settings.deepseek_auth_token:
            raise HTTPException(
                status_code=500,
                detail="DEEPSEEK_AUTH_TOKEN is not set in the environment",
            )
        try:
            _api = DeepSeekAPI(settings.deepseek_auth_token, cookies_path=settings.cookies_path)
        except AuthenticationError as e:
            raise HTTPException(status_code=500, detail=f"Authentication Error: {e}") from e
        _session_manager = SessionManager(_api)
        logger.info("DeepSeekAPI initialized, cookies_loaded=%s", bool(_api.cookies))
    return _session_manager


def reset_dependencies_for_testing() -> None:
    """Clears the singleton so tests can re-initialize with different
    Settings/mocked DeepSeekAPI between test cases. Not used by production
    code paths."""
    global _api, _session_manager
    with _init_lock:
        _api = None
        _session_manager = None
