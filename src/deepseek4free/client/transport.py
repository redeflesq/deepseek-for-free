"""Low-level HTTP transport for chat.deepseek.com: cookie jar, retries, and
Cloudflare-challenge detection.

Split out of the old monolithic DeepSeekAPI (dsk/api.py) so the retry/
locking machinery can be reasoned about - and unit-tested - independently of
the higher-level domain methods (create_chat_session, chat_completion, ...)
that live in client/api.py. Behaviour is unchanged from the original; only
the responsibilities are separated.
"""

import json
import logging
import subprocess
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

from curl_cffi import requests

from deepseek4free.exceptions import (
    APIError,
    AuthenticationError,
    CloudflareError,
    NetworkError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://chat.deepseek.com/api/v0"

# curl_cffi has no built-in per-request timeout guarantee when timeout=None,
# and DeepSeek's own connection can silently hang (proxy stall, half-open
# TCP) - a real ceiling here turns an indefinite hang into a NetworkError
# the caller can handle, instead of a request that never returns.
DEFAULT_REQUEST_TIMEOUT = 60.0          # simple JSON endpoints
STREAMING_REQUEST_TIMEOUT = 300.0       # chat_completion, which can run long
FILE_UPLOAD_TIMEOUT = 120.0             # large file uploads

REQUIRED_CURL_CFFI_VERSION = "0.8.1b9"


def _check_curl_cffi_version() -> None:
    try:
        found = _pkg_version("curl-cffi")
        if found != REQUIRED_CURL_CFFI_VERSION:
            logger.warning(
                "DeepSeek API requires curl-cffi version %s, found %s. "
                "Install the pinned version: pip install curl-cffi==%s",
                REQUIRED_CURL_CFFI_VERSION, found, REQUIRED_CURL_CFFI_VERSION,
            )
    except PackageNotFoundError:
        logger.warning(
            "curl-cffi not found. Install the pinned version: pip install curl-cffi==%s",
            REQUIRED_CURL_CFFI_VERSION,
        )


class Transport:
    """Thread-safe request layer shared by every DeepSeekAPI instance method.

    Thread-safety note: a single DeepSeekAPI/Transport instance is shared
    across all FastAPI requests in server/app.py (one process-wide
    SessionManager). FastAPI runs sync `def` endpoints in a threadpool, so
    concurrent HTTP requests (e.g. uploading several files back-to-back)
    really do call into this class from multiple OS threads at once.
    `self.cookies` used to be mutated by `refresh_cookies()` without any
    synchronization, and every request read `self.cookies` mid-flight - a
    classic read/write race that could hand one request a half-updated
    cookie jar. `self.lock` below serializes all outgoing calls through this
    instance; given DeepSeek rate-limits a single account hard anyway, there
    is no real throughput to lose by not parallelizing requests from one
    account.
    """

    def __init__(self, auth_token: str, cookies_path: Path) -> None:
        if not auth_token or not isinstance(auth_token, str):
            raise AuthenticationError("Invalid auth token provided")

        _check_curl_cffi_version()

        self.auth_token = auth_token
        self.cookies_path = cookies_path

        # Guards self.cookies (read + read-modify-write in refresh_cookies)
        # and serializes calls to the underlying curl_cffi requests, which
        # are not guaranteed thread-safe across concurrent impersonate
        # sessions sharing the same cookie jar. Public (not `_lock`) because
        # client/api.py needs to hold it around multi-step operations
        # (PoW challenge + solve + request) that live outside this class.
        self.lock = threading.RLock()

        self.cookies: dict[str, str] = self._load_cookies_from_disk()

    def _load_cookies_from_disk(self) -> dict[str, str]:
        """Reads cookies.json fresh from disk. Never raises - falls back to
        an empty cookie jar with a logged warning, matching the previous
        behaviour that callers already rely on (auth still deferred to
        DeepSeek's 401 response rather than a hard failure at construction).
        """
        try:
            with open(self.cookies_path, encoding="utf-8") as f:
                cookie_data = json.load(f)
            return cookie_data.get("cookies", {})
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Could not load cookies from %s: %s", self.cookies_path, e)
            return {}

    def get_headers(self, pow_response: str | None = None) -> dict[str, str]:
        headers = {
            "accept": "*/*",
            "accept-language": "en,fr-FR;q=0.9,fr;q=0.8,es-ES;q=0.7,es;q=0.6,en-US;q=0.5,am;q=0.4,de;q=0.3",
            "authorization": f"Bearer {self.auth_token}",
            "content-type": "application/json",
            "origin": "https://chat.deepseek.com",
            "referer": "https://chat.deepseek.com/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
            "x-app-version": "20241129.1",
            "x-client-locale": "en_US",
            "x-client-platform": "web",
            "x-client-version": "1.0.0-always",
        }
        if pow_response:
            headers["x-ds-pow-response"] = pow_response
        return headers

    def refresh_cookies(self) -> None:
        """Runs the cookie refresher (drives a real headless browser through
        Cloudflare, in a separate process) and reloads cookies from disk.

        Caller must hold self.lock - this both triggers a write to the
        shared cookies.json file (in a separate process) and mutates
        self.cookies, so two threads racing here could interleave a
        half-written file read with another thread's write.
        """
        try:
            subprocess.run(
                [sys.executable, "-m", "deepseek4free.cloudflare.cookie_refresher"],
                check=True,
                timeout=90,
                capture_output=True,
            )
            # cookie_refresher writes cookies.json itself; small settle delay
            # in case of slow disk flush before we re-read it.
            time.sleep(1)
            self.cookies = self._load_cookies_from_disk()
        except subprocess.TimeoutExpired:
            logger.warning("Cookie refresh timed out after 90s")
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", "ignore") if e.stderr else ""
            logger.warning("Cookie refresh failed (exit %s): %s", e.returncode, stderr[-2000:])
        except Exception as e:
            logger.warning("Failed to refresh cookies: %s", e)

    def request_locked(self, method: str, endpoint: str, json_data: dict[str, Any],
                        timeout: float = DEFAULT_REQUEST_TIMEOUT) -> Any:
        """Minimal single-attempt request used only from within a call that
        already holds self.lock (avoids re-entering the retry/Cloudflare
        logic in `request()` and risking recursive lock issues). RLock makes
        plain re-entry safe, but this keeps simple lookups (like the PoW
        challenge fetch) side-effect-free with respect to cookie refresh.
        """
        url = f"{BASE_URL}{endpoint}"
        headers = self.get_headers()
        try:
            response = requests.request(
                method=method, url=url, headers=headers, json=json_data,
                cookies=self.cookies, impersonate="chrome120", timeout=timeout,
            )
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Network error occurred: {e}") from e

        self.raise_for_status(response)
        return self.parse_json(response)

    def request(self, method: str, endpoint: str, json_data: dict[str, Any],
                timeout: float = DEFAULT_REQUEST_TIMEOUT, max_retries: int = 2) -> Any:
        """Retrying request with Cloudflare-challenge detection. Caller must
        hold self.lock for the whole retry loop, since a Cloudflare hit
        triggers refresh_cookies(), which mutates shared state.

        No pow_response parameter here on purpose: the only current caller
        of this method (create_chat_session) hits a PoW-free endpoint, and
        the two endpoints that do need PoW (upload_file, chat_completion)
        bypass this retry loop entirely and call requests directly - see
        client/api.py. A pow_response param was carried over from an
        intermediate refactor but was never actually passed by any caller;
        removed as dead code rather than kept "just in case".
        """
        url = f"{BASE_URL}{endpoint}"
        retry_count = 0
        last_error: Exception | None = None

        while retry_count < max_retries:
            try:
                headers = self.get_headers()
                response = requests.request(
                    method=method, url=url, headers=headers, json=json_data,
                    cookies=self.cookies, impersonate="chrome120", timeout=timeout,
                )

                if "<!DOCTYPE html>" in response.text and "Just a moment" in response.text:
                    logger.warning("Cloudflare protection detected, refreshing cookies and retrying")
                    if retry_count < max_retries - 1:
                        self.refresh_cookies()
                        retry_count += 1
                        continue
                    raise CloudflareError(
                        "Cloudflare is still blocking requests after a cookie refresh attempt"
                    )

                self.raise_for_status(response)
                return self.parse_json(response)

            except requests.exceptions.RequestException as e:
                # Network-level failure (timeout, connection reset, DNS,
                # TLS) - worth one retry since DeepSeek's edge is flaky, but
                # don't loop forever.
                last_error = e
                retry_count += 1
                if retry_count >= max_retries:
                    raise NetworkError(f"Network error occurred: {e}") from e
                time.sleep(1)
                continue
            except (AuthenticationError, RateLimitError, APIError, CloudflareError):
                # Already a typed, actionable error - propagate as-is
                # instead of falling through to the generic failure below.
                raise

        # Loop exhausted without an explicit return/raise above - this only
        # happens if every retry hit the Cloudflare branch. Surface the
        # concrete cause instead of a generic "bypass failed" message.
        if last_error is not None:
            raise NetworkError(f"Network error occurred: {last_error}") from last_error
        raise CloudflareError("Failed to bypass Cloudflare protection after multiple attempts")

    @staticmethod
    def raise_for_status(response: Any) -> None:
        if response.status_code == 401:
            raise AuthenticationError("Invalid or expired authentication token")
        if response.status_code == 429:
            raise RateLimitError("API rate limit exceeded")
        if response.status_code >= 500:
            raise APIError(f"Server error occurred: {response.text}", response.status_code)
        if response.status_code != 200:
            raise APIError(f"API request failed: {response.text}", response.status_code)

    @staticmethod
    def parse_json(response: Any) -> Any:
        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise APIError(
                f"Invalid JSON response from server (status {response.status_code}): "
                f"{response.text[:500]!r}"
            ) from e
