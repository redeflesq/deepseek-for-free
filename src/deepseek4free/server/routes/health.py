"""GET /health - liveness/readiness check."""

from fastapi import APIRouter, HTTPException

from deepseek4free.config import get_settings
from deepseek4free.server.dependencies import get_deepseek_api, get_session_manager
from deepseek4free.server.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness check.

    Unlike a bare {"status": "ok"}, this actually checks whether
    DEEPSEEK_AUTH_TOKEN is configured and whether cookies were loaded from
    disk (cf_clearance). It does NOT make a network call to DeepSeek - this
    is intentionally a cheap, fast check, not a full auth probe (ordinary
    requests already surface 401/503 via the exception handlers in
    server/errors.py if the token/cookies ultimately don't work).

    status == "degraded" means the service is up and responding, but is
    very likely unable to serve real DeepSeek requests without manual
    intervention (e.g. no cookies.json yet because the Cloudflare bypass
    flow hasn't been run).
    """
    settings = get_settings()
    auth_token_configured = bool(settings.deepseek_auth_token)

    try:
        manager = get_session_manager()
    except HTTPException as e:
        return HealthResponse(
            status="degraded",
            auth_token_configured=auth_token_configured,
            cookies_loaded=False,
            active_sessions=0,
            detail=str(e.detail),
        )

    api = get_deepseek_api()
    cookies_loaded = bool(api.cookies) if api is not None else False
    status = "ok" if (auth_token_configured and cookies_loaded) else "degraded"
    detail = None
    if not auth_token_configured:
        detail = "DEEPSEEK_AUTH_TOKEN is not set"
    elif not cookies_loaded:
        detail = (
            "cookies.json is missing or empty - run "
            "`python -m deepseek4free.cloudflare.cookie_refresher` to obtain cf_clearance"
        )

    return HealthResponse(
        status=status,
        auth_token_configured=auth_token_configured,
        cookies_loaded=cookies_loaded,
        active_sessions=manager.session_count(),
        detail=detail,
    )
