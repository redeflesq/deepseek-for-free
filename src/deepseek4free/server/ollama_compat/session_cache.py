"""LRU+TTL cache mapping "conversation history so far" to a DeepSeek session.

Ollama's /api/chat is stateless on the wire - every call resends the full
message history, there's no session_id. DeepSeekAPI is the opposite: it
threads messages via parent_message_id within a server-side chat session.
This cache bridges the two: routes.py hashes (model, messages[:-1]) via
mapping.history_prefix_key() and looks up whether that exact conversation
prefix was already seen, reusing the same DeepSeek session + parent_message_id
if so, or creating a new DeepSeek session (via the existing SessionManager)
otherwise.

This is a supplementary mapping layer only - it does NOT replace
SessionManager or duplicate ChatSession's responsibilities. The actual
session state (history, per-session lock, DeepSeek session id) still lives
in SessionManager exactly as before; this cache only remembers which
DeepSeek session id corresponds to a given (model, history-prefix) hash, plus
DeepSeek's current parent_message_id for continuing that thread.
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

DEFAULT_MAX_SIZE = 200
DEFAULT_TTL_SECONDS = 2 * 60 * 60  # 2 hours


@dataclass
class CacheEntry:
    deepseek_session_id: str
    parent_message_id: str | None
    last_used: float


class OllamaSessionCache:
    """Thread-safe LRU+TTL cache, keyed by mapping.history_prefix_key()'s
    output.

    Expired entries are swept lazily on access (inside get()) rather than by
    a background thread/timer - avoids adding a second thread to the process
    just to prune a cache that's already bounded in size by max_size, and
    keeps all cache mutation under the same lock without needing to
    coordinate a separate sweeper against concurrent get()/put() calls.
    """

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if time.time() - entry.last_used >= self._ttl_seconds:
                # Expired - drop it now instead of returning stale data that
                # would make routes.py reuse a DeepSeek session id that may
                # no longer even exist in the (also in-memory, also
                # restart-losing) SessionManager.
                del self._entries[key]
                return None
            self._entries.move_to_end(key)  # LRU touch
            return entry

    def put(self, key: str, deepseek_session_id: str, parent_message_id: str | None) -> None:
        with self._lock:
            if key not in self._entries and len(self._entries) >= self._max_size:
                self._entries.popitem(last=False)  # evict oldest
            self._entries[key] = CacheEntry(
                deepseek_session_id=deepseek_session_id,
                parent_message_id=parent_message_id,
                last_used=time.time(),
            )
            self._entries.move_to_end(key)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
