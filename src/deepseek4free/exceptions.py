"""Typed exception hierarchy shared by the DeepSeek client and the chat server.

Kept as a single flat module (not split per-layer) on purpose: both
deepseek4free.client.api and deepseek4free.server.errors need every one of
these types, and splitting them across two files would just require two
imports everywhere instead of one, for no organizational benefit at six
classes. Logic/semantics are unchanged from the old dsk/api.py - only the
location moved so client and server code don't have to import from a
client-specific module for something that's really a cross-cutting type.
"""



class DeepSeekError(Exception):
    """Base exception for all DeepSeek API errors."""


class AuthenticationError(DeepSeekError):
    """Raised when authentication fails (invalid or expired token)."""


class RateLimitError(DeepSeekError):
    """Raised when DeepSeek's API rate limit is exceeded."""


class NetworkError(DeepSeekError):
    """Raised when the underlying HTTP transport fails (timeout, DNS, TLS, connection reset)."""


class CloudflareError(DeepSeekError):
    """Raised when Cloudflare blocks the request and a cookie refresh did not resolve it."""


class APIError(DeepSeekError):
    """Raised when DeepSeek's API returns an error response."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
