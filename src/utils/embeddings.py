"""
OpenAI embedding client with caching and batching.
Uses text-embedding-3-small (1536-dim) as specified in §2.
"""

import hashlib
import logging
from typing import Optional

import numpy as np
from openai import OpenAI

from src.utils.backoff import with_backoff
from src.utils.cache import get_cache
from src.utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Cached embedding client using OpenAI text-embedding-3-small."""

    def __init__(self, model="text-embedding-3-small", dimensions=1536,
                 batch_size=256, cache_dir=".cache", cost_tracker=None):
        self.client = OpenAI()
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.cache = get_cache("embeddings", cache_dir)
        self.cost_tracker = cost_tracker

    def _cache_key(self, text):
        return hashlib.sha256(f"{self.model}:{self.dimensions}:{text}".encode()).hexdigest()

    @with_backoff(max_retries=6, base_wait=1.0, max_wait=60.0)
    def _call_api(self, texts):
        response = self.client.embeddings.create(
            model=self.model, input=texts, dimensions=self.dimensions,
        )
        if self.cost_tracker:
            self.cost_tracker.record(self.model, response.usage.total_tokens, 0,
                                     f"embed {len(texts)} texts")
        return [item.embedding for item in response.data]

    def embed_single(self, text):
        key = self._cache_key(text)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        result = self._call_api([text])[0]
        self.cache.set(key, result)
        return result

    def embed_batch(self, texts):
        results = [None] * len(texts)
        uncached_indices, uncached_texts = [], []
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            cached = self.cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        if not uncached_texts:
            return results
        all_embeddings = []
        for start in range(0, len(uncached_texts), self.batch_size):
            batch = uncached_texts[start:start + self.batch_size]
            all_embeddings.extend(self._call_api(batch))
        for idx, emb in zip(uncached_indices, all_embeddings):
            results[idx] = emb
            self.cache.set(self._cache_key(texts[idx]), emb)
        return results

    def embed_as_numpy(self, texts):
        return np.array(self.embed_batch(texts), dtype=np.float32)

    def close(self):
        self.cache.close()
