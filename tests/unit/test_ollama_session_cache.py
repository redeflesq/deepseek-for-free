"""Unit tests for deepseek4free.server.ollama_compat.session_cache - no network,
no real time.sleep() needed (TTL expiry is tested via ttl_seconds=0 rather than
actually waiting, keeping the suite fast and deterministic).
"""

from deepseek4free.server.ollama_compat.session_cache import OllamaSessionCache


def test_get_on_empty_cache_is_miss() -> None:
    cache = OllamaSessionCache()
    assert cache.get("nope") is None


def test_put_then_get_is_a_hit() -> None:
    cache = OllamaSessionCache()
    cache.put("key-1", "session-abc", "parent-1")
    entry = cache.get("key-1")
    assert entry is not None
    assert entry.deepseek_session_id == "session-abc"
    assert entry.parent_message_id == "parent-1"


def test_put_allows_none_parent_message_id() -> None:
    cache = OllamaSessionCache()
    cache.put("key-1", "session-abc", None)
    entry = cache.get("key-1")
    assert entry is not None
    assert entry.parent_message_id is None


def test_put_overwrites_existing_key() -> None:
    cache = OllamaSessionCache()
    cache.put("key-1", "session-abc", "parent-1")
    cache.put("key-1", "session-xyz", "parent-2")
    entry = cache.get("key-1")
    assert entry is not None
    assert entry.deepseek_session_id == "session-xyz"
    assert entry.parent_message_id == "parent-2"
    assert len(cache) == 1


def test_expired_entry_is_treated_as_miss_and_removed() -> None:
    # ttl_seconds=0: any elapsed time (even microseconds between put() and
    # get()) satisfies "now - last_used >= ttl_seconds", so this is a
    # deterministic way to force expiry without sleeping in the test.
    cache = OllamaSessionCache(ttl_seconds=0)
    cache.put("key-1", "session-abc", "parent-1")
    assert cache.get("key-1") is None
    assert len(cache) == 0  # confirms it was actually evicted, not just hidden


def test_lru_eviction_drops_oldest_when_max_size_exceeded() -> None:
    cache = OllamaSessionCache(max_size=2)
    cache.put("key-1", "session-1", None)
    cache.put("key-2", "session-2", None)
    cache.put("key-3", "session-3", None)  # should evict key-1 (oldest)

    assert cache.get("key-1") is None
    assert cache.get("key-2") is not None
    assert cache.get("key-3") is not None
    assert len(cache) == 2


def test_get_touches_lru_order_so_recently_used_key_survives_eviction() -> None:
    cache = OllamaSessionCache(max_size=2)
    cache.put("key-1", "session-1", None)
    cache.put("key-2", "session-2", None)
    cache.get("key-1")  # touch key-1, making key-2 the least-recently-used
    cache.put("key-3", "session-3", None)  # should evict key-2, not key-1

    assert cache.get("key-1") is not None
    assert cache.get("key-2") is None
    assert cache.get("key-3") is not None


def test_len_reflects_current_entry_count() -> None:
    cache = OllamaSessionCache()
    assert len(cache) == 0
    cache.put("key-1", "session-1", None)
    assert len(cache) == 1
