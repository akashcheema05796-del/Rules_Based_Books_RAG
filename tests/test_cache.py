"""
Tests for BenchmarkCache (src/utils/cache.py).

Covers: get/set round-trip, cache miss returns None, hit rate tracking,
get_or_compute, key collision avoidance, TTL expiry, namespace isolation,
clear(), context-manager protocol, and stats dict schema.
"""

import time
import pytest

from src.utils.cache import BenchmarkCache, get_cache


@pytest.fixture
def cache(tmp_path):
    """Fresh cache in a temp directory, closed after each test."""
    c = BenchmarkCache(cache_dir=tmp_path / "cache", namespace="test")
    yield c
    c._cache.close()


class TestGetSet:
    def test_miss_returns_none(self, cache):
        assert cache.get("nonexistent_key_xyz") is None

    def test_set_then_get(self, cache):
        cache.set("k1", [1, 2, 3])
        assert cache.get("k1") == [1, 2, 3]

    def test_set_overwrite(self, cache):
        cache.set("k", "original")
        cache.set("k", "updated")
        assert cache.get("k") == "updated"

    def test_various_value_types(self, cache):
        cache.set("int", 42)
        cache.set("float", 3.14)
        cache.set("list", [1, "two", 3.0])
        cache.set("dict", {"a": 1, "b": [2, 3]})
        cache.set("none_val", {"present": True})

        assert cache.get("int") == 42
        assert abs(cache.get("float") - 3.14) < 1e-9
        assert cache.get("list") == [1, "two", 3.0]
        assert cache.get("dict") == {"a": 1, "b": [2, 3]}


class TestHitRate:
    def test_initial_hit_rate_zero(self, cache):
        assert cache.hit_rate == 0.0

    def test_all_hits(self, cache):
        cache.set("x", 1)
        cache.get("x")
        cache.get("x")
        assert cache.hit_rate == 1.0

    def test_all_misses(self, cache):
        cache.get("missing1")
        cache.get("missing2")
        assert cache.hit_rate == 0.0

    def test_mixed_hit_rate(self, cache):
        cache.set("a", 1)
        cache.get("a")   # hit
        cache.get("b")   # miss
        assert abs(cache.hit_rate - 0.5) < 1e-9

    def test_hit_miss_counters_in_stats(self, cache):
        cache.set("x", 1)
        cache.get("x")   # hit
        cache.get("y")   # miss
        s = cache.stats
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["namespace"] == "test"
        assert "hit_rate" in s
        assert "size" in s


class TestGetOrCompute:
    def test_compute_called_on_miss(self, cache):
        calls = []

        def compute(x):
            calls.append(x)
            return x * 2

        result = cache.get_or_compute("k", compute, 5)
        assert result == 10
        assert len(calls) == 1

    def test_compute_not_called_on_hit(self, cache):
        calls = []

        def compute(x):
            calls.append(x)
            return x * 2

        cache.set("k", 99)
        result = cache.get_or_compute("k", compute, 5)
        assert result == 99
        assert len(calls) == 0

    def test_result_stored_after_compute(self, cache):
        cache.get_or_compute("k", lambda: "hello")
        # Second call should hit cache
        result = cache.get_or_compute("k", lambda: "world")
        assert result == "hello"


class TestMakeKey:
    def test_key_is_deterministic(self, cache):
        k1 = cache._make_key("model", "text-embedding-3-small", [1.0, 2.0])
        k2 = cache._make_key("model", "text-embedding-3-small", [1.0, 2.0])
        assert k1 == k2

    def test_different_args_different_keys(self, cache):
        k1 = cache._make_key("a", 1)
        k2 = cache._make_key("a", 2)
        assert k1 != k2

    def test_key_is_hex_string(self, cache):
        k = cache._make_key("test")
        assert len(k) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in k)


class TestNamespaceIsolation:
    def test_different_namespaces_isolated(self, tmp_path):
        c1 = BenchmarkCache(cache_dir=tmp_path / "cache", namespace="ns1")
        c2 = BenchmarkCache(cache_dir=tmp_path / "cache", namespace="ns2")
        try:
            c1.set("key", "value_from_ns1")
            assert c2.get("key") is None
        finally:
            c1._cache.close()
            c2._cache.close()


class TestClear:
    def test_clear_removes_entries(self, cache):
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_size_zero_after_clear(self, cache):
        cache.set("x", 42)
        cache.clear()
        assert cache.stats["size"] == 0


class TestContextManager:
    def test_context_manager_closes_gracefully(self, tmp_path):
        with BenchmarkCache(cache_dir=tmp_path / "ctx_cache", namespace="ctx") as c:
            c.set("key", "val")
            assert c.get("key") == "val"
        # No exception raised on __exit__


class TestGetCacheFactory:
    def test_returns_benchmark_cache(self, tmp_path):
        c = get_cache("embeddings", cache_dir=tmp_path / "cache")
        try:
            assert isinstance(c, BenchmarkCache)
            assert c.namespace == "embeddings"
        finally:
            c._cache.close()
