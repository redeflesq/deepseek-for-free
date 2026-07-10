"""Process-wide singleton wiring for the Ollama-compat layer's session cache.

Deliberately a separate, much simpler module than server/dependencies.py:
that module's double-checked-locked lazy init exists because constructing
DeepSeekAPI needs a validated auth token and can raise AuthenticationError.
OllamaSessionCache has no such dependency - it's just an in-memory dict, so
a plain module-level singleton (built on first access, no locking needed
for the construction itself since OllamaSessionCache's own methods are
already thread-safe) is enough here.
"""

from deepseek4free.server.ollama_compat.session_cache import OllamaSessionCache

_session_cache: OllamaSessionCache = OllamaSessionCache()


def get_ollama_session_cache() -> OllamaSessionCache:
    return _session_cache


def reset_ollama_session_cache_for_testing() -> None:
    """Replaces the singleton with a fresh, empty cache. Not used by
    production code paths - mirrors server/dependencies.py's
    reset_dependencies_for_testing() so tests can isolate cache state
    between cases."""
    global _session_cache
    _session_cache = OllamaSessionCache()
