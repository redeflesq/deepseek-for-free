"""Real-network smoke test against the live chat.deepseek.com API.

Skipped by default: requires DEEPSEEK_AUTH_TOKEN to be set in the
environment AND a valid cookies.json already present at the configured
data_dir (see deepseek4free.config.Settings.cookies_path) - this test does
NOT attempt a Cloudflare bypass itself, that's cloudflare/cookie_refresher.py's
job and is out of scope for a fast CI-friendly smoke test.

Run explicitly with:
    DEEPSEEK_AUTH_TOKEN=... pytest tests/integration -m integration

CI (see .github/workflows/ci.yml) intentionally does NOT set
DEEPSEEK_AUTH_TOKEN, so this file is collected but every test inside is
skipped rather than failing the pipeline on missing credentials.
"""

import pytest

from deepseek4free.client.api import DeepSeekAPI
from deepseek4free.config import get_settings

pytestmark = pytest.mark.integration


def _skip_if_not_configured() -> DeepSeekAPI:
    settings = get_settings()
    if not settings.deepseek_auth_token:
        pytest.skip("DEEPSEEK_AUTH_TOKEN not set - skipping real-network smoke test")
    if not settings.cookies_path.is_file():
        pytest.skip(
            f"{settings.cookies_path} not found - run "
            "`python -m deepseek4free.cloudflare.cookie_refresher` first"
        )
    return DeepSeekAPI(settings.deepseek_auth_token, cookies_path=settings.cookies_path)


def test_create_chat_session_returns_a_session_id() -> None:
    api = _skip_if_not_configured()
    session_id = api.create_chat_session()
    assert isinstance(session_id, str)
    assert session_id


def test_chat_completion_streams_a_non_empty_reply() -> None:
    api = _skip_if_not_configured()
    session_id = api.create_chat_session()

    text_parts = []
    for chunk in api.chat_completion(session_id, "Reply with exactly the word: pong"):
        if chunk.get("type") == "text" and chunk.get("content"):
            text_parts.append(chunk["content"])
        if chunk.get("finish_reason") == "stop":
            break

    assert "".join(text_parts).strip() != ""
