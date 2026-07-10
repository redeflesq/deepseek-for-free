"""deepseek4free: unofficial Python client + local HTTP chat server for chat.deepseek.com.

Public surface re-exported here on purpose (matches the old flat `dsk.api`
import habit users already have) so `from deepseek4free import DeepSeekAPI`
works without knowing the internal client/pow/cloudflare/server package split.
"""

from deepseek4free.client.api import DeepSeekAPI
from deepseek4free.exceptions import (
    APIError,
    AuthenticationError,
    CloudflareError,
    DeepSeekError,
    NetworkError,
    RateLimitError,
)

__version__ = "2.0.0"

__all__ = [
    "DeepSeekAPI",
    "DeepSeekError",
    "AuthenticationError",
    "RateLimitError",
    "NetworkError",
    "CloudflareError",
    "APIError",
    "__version__",
]
