"""Obtains and saves cf_clearance cookies for chat.deepseek.com.

This replaces THREE overlapping scripts from the old layout:

- dsk/bypass.py            - automated: launches dsk.server, hits /cookies, saves.
- dsk/run_and_get_cookies.py - near-duplicate of bypass.py (fewer retries, no
                                SERVER_PORT env support) - dropped, no unique
                                behaviour worth keeping separately.
- dsk/get_cookies_nodriver.py - manual: opens a real (visible) browser via
                                 `nodriver`, waits for a human to log in, then
                                 reads cookies + the userToken from
                                 localStorage directly.

Both real *use cases* survive as two functions here instead of two files:

- refresh_via_bypass_server(): the automated path used by
  Transport.refresh_cookies() (spawns cloudflare.bypass_server as a
  subprocess) and by docker/entrypoint.sh at container startup.
- refresh_via_manual_login(): the interactive path for a developer who
  needs a *fresh* userToken (not just a fresh cf_clearance) - e.g. first
  setup, or after the account's token itself expired. Run explicitly via
  `python -m deepseek4free.cloudflare.cookie_refresher --manual`.

Running this module directly (`python -m deepseek4free.cloudflare.cookie_refresher`)
defaults to the automated path, matching the previous dsk.bypass behaviour
that Transport.refresh_cookies() and docker/entrypoint.sh both rely on.
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time

import requests

from deepseek4free.config import get_settings

logger = logging.getLogger(__name__)


def _save_cookies(cookies: dict, user_agent: str) -> None:
    settings = get_settings()
    settings.cookies_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.cookies_path, "w", encoding="utf-8") as f:
        json.dump({"cookies": cookies, "user_agent": user_agent}, f, indent=2, ensure_ascii=False)
    logger.info("Cookies saved to %s", settings.cookies_path)


def _fetch_cookies_from_bypass_server(server_url: str, max_retries: int = 5) -> dict | None:
    for attempt in range(max_retries):
        try:
            response = requests.get(server_url, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.warning("Connection error on attempt %d/%d: %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(5)
    return None


def refresh_via_bypass_server() -> bool:
    """Automated cf_clearance refresh: launches cloudflare.bypass_server as a
    background subprocess, asks it to solve the Cloudflare challenge for
    chat.deepseek.com, saves the resulting cookies, then tears the
    subprocess down. Returns True on success.

    This is what Transport.refresh_cookies() runs (via
    `python -m deepseek4free.cloudflare.cookie_refresher`) whenever a live
    request hits a Cloudflare challenge mid-session.
    """
    settings = get_settings()

    process = subprocess.Popen(
        [sys.executable, "-m", "deepseek4free.cloudflare.bypass_server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    try:
        # Fixed settle delay before the first request, mirroring the old
        # bypass.py - the bypass server needs a moment to bind its port
        # before /cookies can be queried at all.
        time.sleep(10)
        server_url = f"http://localhost:{settings.server_port}/cookies?url=https://chat.deepseek.com"
        cookies_data = _fetch_cookies_from_bypass_server(server_url, max_retries=5)

        if cookies_data is None:
            logger.error("Failed to obtain valid cookies from bypass server after all attempts")
            return False

        _save_cookies(cookies_data.get("cookies", {}), cookies_data.get("user_agent", ""))
        return True
    finally:
        process.terminate()


async def _refresh_via_manual_login_async() -> None:
    import nodriver as uc

    browser = await uc.start(browser_executable_path="/usr/bin/brave", headless=False)
    page = await browser.get("https://chat.deepseek.com")
    await asyncio.to_thread(
        input, "Press Enter after logging in to chat.deepseek.com in the browser window..."
    )

    raw_cookies = await browser.cookies.get_all(requests_cookie_format=False)
    cookies_dict = {c.name: c.value for c in raw_cookies}

    user_token_raw = await page.evaluate('localStorage.getItem("userToken")')
    if user_token_raw:
        try:
            user_token = json.loads(user_token_raw).get("value", user_token_raw)
        except json.JSONDecodeError:
            user_token = user_token_raw
        cookies_dict["userToken"] = user_token

    user_agent = await page.evaluate("navigator.userAgent")
    browser.stop()

    _save_cookies(cookies_dict, user_agent)
    print(f"Cookies saved. Keys: {list(cookies_dict.keys())}")


def refresh_via_manual_login() -> None:
    """Interactive cf_clearance + userToken refresh: opens a real, visible
    browser, waits for a human to log in to chat.deepseek.com, then reads
    both cookies and the userToken straight from localStorage.

    Use this for first-time setup or when the auth token itself (not just
    cf_clearance) has expired - refresh_via_bypass_server() only refreshes
    the Cloudflare cookie, not the account token.
    """
    asyncio.run(_refresh_via_manual_login_async())


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    parser = argparse.ArgumentParser(description="Refresh chat.deepseek.com cookies")
    parser.add_argument(
        "--manual", action="store_true",
        help="Interactive login flow via a visible browser (also captures a fresh userToken)",
    )
    args = parser.parse_args()

    if args.manual:
        refresh_via_manual_login()
        return

    if not refresh_via_bypass_server():
        sys.exit(1)


if __name__ == "__main__":
    main()
