"""
System metrics (spec §8.4).

Chunking time/cost, indexing time/size, query latency p50/p95,
chunk count, token distribution.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.corpus.models import ChunkMetadata

logger = logging.getLogger(__name__)


def chunk_token_distribution(chunks: list[ChunkMetadata]) -> dict:
    """Compute token distribution statistics for chunks.

    Returns mean, p50, p95, max, and coefficient of variation.
    """
    if not chunks:
        return {"count": 0}

    tokens = np.array([c.token_count for c in chunks])

    return {
        "count": len(chunks),
        "mean": round(float(np.mean(tokens)), 1),
        "median_p50": round(float(np.median(tokens)), 1),
        "p95": round(float(np.percentile(tokens, 95)), 1),
        "max": int(np.max(tokens)),
        "min": int(np.min(tokens)),
        "std": round(float(np.std(tokens)), 1),
        "cv": round(float(np.std(tokens) / np.mean(tokens)), 4) if np.mean(tokens) > 0 else 0,
        "total_tokens": int(np.sum(tokens)),
    }


def index_size_on_disk(index_dir: Path) -> dict:
    """Compute index size on disk.

    Args:
        index_dir: Path to index directory.

    Returns:
        Dict with size in bytes and human-readable size.
    """
    total = 0
    file_count = 0
    if index_dir.exists():
        for f in index_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
                file_count += 1

    return {
        "total_bytes": total,
        "total_mb": round(total / (1024 * 1024), 2),
        "file_count": file_count,
    }


def compute_latency_stats(latencies_ms: list[float]) -> dict:
    """Compute latency statistics.

    Args:
        latencies_ms: List of latency measurements in milliseconds.

    Returns:
        Dict with p50 and p95 latencies.
    """
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0}

    arr = np.array(latencies_ms)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
    }


class LatencyTracker:
    """Context manager to track query latency."""

    def __init__(self):
        self.latencies = []
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = (time.perf_counter() - self._start) * 1000
        self.latencies.append(elapsed)

    @property
    def stats(self):
        return compute_latency_stats(self.latencies)
