"""
Disk-based caching using diskcache.

Caches: embeddings, contextual blurbs, HyDE outputs, reranker scores.
Logs cache hit rate per session.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

import diskcache

logger = logging.getLogger(__name__)


class BenchmarkCache:
    """Persistent disk cache with hit rate tracking.

    Supports namespaced caches for different data types
    (embeddings, contextual, hyde, reranker).
    """

    def __init__(self, cache_dir: str | Path = ".cache", namespace: str = "default"):
        """Initialize cache.

        Args:
            cache_dir: Root directory for cache storage.
            namespace: Cache namespace (e.g., 'embeddings', 'contextual').
        """
        self.cache_dir = Path(cache_dir) / namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(self.cache_dir))
        self.namespace = namespace
        self._hits = 0
        self._misses = 0

    def _make_key(self, *args) -> str:
        """Create a deterministic cache key from arguments."""
        raw = json.dumps(args, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found.
        """
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result

    def set(self, key: str, value: Any, expire: Optional[float] = None) -> None:
        """Store value in cache.

        Args:
            key: Cache key.
            value: Value to store.
            expire: Optional TTL in seconds.
        """
        self._cache.set(key, value, expire=expire)

    def get_or_compute(self, key: str, compute_fn, *args, **kwargs) -> Any:
        """Get from cache or compute and store.

        Args:
            key: Cache key.
            compute_fn: Function to call if cache miss.
            *args: Arguments for compute_fn.
            **kwargs: Keyword arguments for compute_fn.

        Returns:
            Cached or computed value.
        """
        result = self.get(key)
        if result is not None:
            return result

        result = compute_fn(*args, **kwargs)
        self.set(key, result)
        return result

    @property
    def hit_rate(self) -> float:
        """Current cache hit rate as a fraction."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    @property
    def stats(self) -> dict:
        """Cache statistics."""
        return {
            "namespace": self.namespace,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "size": len(self._cache),
        }

    def log_stats(self) -> None:
        """Log cache statistics."""
        s = self.stats
        logger.info(
            f"Cache[{s['namespace']}]: hits={s['hits']}, misses={s['misses']}, "
            f"hit_rate={s['hit_rate']:.2%}, size={s['size']}"
        )

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.info(f"Cache[{self.namespace}] cleared")

    def close(self) -> None:
        """Close the cache."""
        self.log_stats()
        self._cache.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# Convenience factory
def get_cache(namespace: str, cache_dir: str | Path = ".cache") -> BenchmarkCache:
    """Get a namespaced cache instance.

    Args:
        namespace: Cache namespace.
        cache_dir: Root cache directory.

    Returns:
        BenchmarkCache instance.
    """
    return BenchmarkCache(cache_dir=cache_dir, namespace=namespace)
