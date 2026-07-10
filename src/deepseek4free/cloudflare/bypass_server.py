"""Local FastAPI service that drives a real (headless, virtual-display)
Chromium browser to solve Cloudflare's Turnstile challenge and hand back
`cf_clearance` cookies + user-agent for a given URL.

This is a separate long-lived process from the main chat server
(server/app.py) - it owns a browser, not just HTTP connections - started by
docker/entrypoint.sh (or manually via `python -m
deepseek4free.cloudflare.bypass_server`) and queried by cookie_refresher.py.

Logic UNCHANGED from the old dsk/server.py; only imports/config moved to
the new package layout.
"""

import argparse
import atexit
import json
import re
import time
from urllib.parse import urlparse

import uvicorn
from DrissionPage import ChromiumOptions, ChromiumPage
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from pyvirtualdisplay import Display

from deepseek4free.cloudflare.bypasser import CloudflareBypasser
from deepseek4free.config import get_settings

settings = get_settings()

# Chromium launch arguments - unchanged from the original, these disable
# a handful of Chrome features that otherwise leak "this is automation"
# signals or slow down repeated headless launches.
CHROMIUM_ARGUMENTS = [
    "-no-first-run",
    "-force-color-profile=srgb",
    "-metrics-recording-only",
    "-password-store=basic",
    "-use-mock-keychain",
    "-export-tagged-pdf",
    "-no-default-browser-check",
    "-disable-background-mode",
    "-enable-features=NetworkService,NetworkServiceInProcess,LoadCryptoTokenExtension,PermuteTLSExtensions",
    "-disable-features=FlashDeprecationWarning,EnablePasswordsAccountStorage",
    "-deny-permission-prompts",
    "-disable-gpu",
    "-accept-lang=en-US",
]

BROWSER_PATH = "/usr/bin/google-chrome"

app = FastAPI(title="deepseek4free Cloudflare bypass service")

# Whether logging is enabled for CloudflareBypasser - set from CLI args in
# main(), read by the /cookies and /html handlers. Module-level because
# FastAPI route handlers can't easily receive argparse Namespace otherwise;
# this mirrors the original script's use of a bare global `log`.
_log_enabled = True


class CookieResponse(BaseModel):
    cookies: dict[str, str]
    user_agent: str


def is_safe_url(url: str) -> bool:
    """Rejects localhost/private-network/file:// targets - this endpoint
    would otherwise let any caller make our browser fetch arbitrary internal
    URLs (SSRF via a `curl -sf http://127.0.0.1:8000/cookies?url=...`)."""
    parsed_url = urlparse(url)
    ip_pattern = re.compile(
        r"^(127\.0\.0\.1|localhost|0\.0\.0\.0|::1|10\.\d+\.\d+\.\d+|"
        r"172\.1[6-9]\.\d+\.\d+|172\.2[0-9]\.\d+\.\d+|172\.3[0-1]\.\d+\.\d+|192\.168\.\d+\.\d+)$"
    )
    hostname = parsed_url.hostname
    if (hostname and ip_pattern.match(hostname)) or parsed_url.scheme == "file":
        return False
    return True


def verify_page_loaded(driver: ChromiumPage) -> bool:
    """Verify the page has loaded properly (has a body with real content)."""
    try:
        body = driver.ele("tag:body", timeout=10)
        return len(body.html) > 100
    except Exception:
        return False


def bypass_cloudflare(url: str, retries: int, log: bool, proxy: str | None = None) -> ChromiumPage:
    max_load_retries = 3

    for load_attempt in range(max_load_retries):
        options = ChromiumOptions().auto_port()
        if settings.docker_mode:
            options.set_argument("--auto-open-devtools-for-tabs", "true")
            options.set_argument("--remote-debugging-port=9222")
            options.set_argument("--no-sandbox")  # Necessary for Docker
            options.set_argument("--disable-gpu")
            options.set_paths(browser_path=BROWSER_PATH).headless(False)
        else:
            options.set_paths(browser_path=BROWSER_PATH).headless(False)

        if proxy:
            options.set_proxy(proxy)

        driver = ChromiumPage(addr_or_opts=options)
        try:
            driver.get(url)
            time.sleep(5)  # let the initial page settle before checking

            if not verify_page_loaded(driver):
                driver.quit()
                if load_attempt < max_load_retries - 1:
                    time.sleep(3)
                    continue
                raise Exception("Failed to load page properly after multiple attempts")

            cf_bypasser = CloudflareBypasser(driver, retries, log)
            cf_bypasser.bypass()
            return driver
        except Exception:
            driver.quit()
            if load_attempt < max_load_retries - 1:
                time.sleep(3)
                continue
            raise

    raise RuntimeError("unreachable")  # loop always returns or raises


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Lightweight liveness endpoint - lets docker/entrypoint.sh and external
    health checks confirm this process is up without triggering a real
    (slow, browser-driven) Cloudflare bypass."""
    return {"status": "ok"}


@app.get("/cookies", response_model=CookieResponse)
async def get_cookies(url: str, retries: int = 5, proxy: str | None = None):
    if not is_safe_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")
    try:
        driver = bypass_cloudflare(url, retries, _log_enabled, proxy)
        cookies = {c.get("name", ""): c.get("value", " ") for c in driver.cookies()}
        user_agent = driver.user_agent
        driver.quit()
        return CookieResponse(cookies=cookies, user_agent=user_agent)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/html")
async def get_html(url: str, retries: int = 5, proxy: str | None = None):
    if not is_safe_url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")
    try:
        driver = bypass_cloudflare(url, retries, _log_enabled, proxy)
        html = driver.html
        cookies_json = {c.get("name", ""): c.get("value", " ") for c in driver.cookies()}
        response = Response(content=html, media_type="text/html")
        response.headers["cookies"] = json.dumps(cookies_json)
        response.headers["user_agent"] = driver.user_agent
        driver.quit()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def main() -> None:
    global _log_enabled

    parser = argparse.ArgumentParser(description="Cloudflare bypass service")
    parser.add_argument("--nolog", action="store_true", help="Disable logging")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()

    display: Display | None = None
    if args.headless or settings.docker_mode:
        display = Display(visible=0, size=(1920, 1080))
        display.start()

        def cleanup_display() -> None:
            if display:
                display.stop()

        atexit.register(cleanup_display)

    _log_enabled = not args.nolog

    uvicorn.run(app, host="0.0.0.0", port=settings.server_port)


if __name__ == "__main__":
    main()
