"""
Retrieval metrics (spec §8.1).

Recall@k, MRR, nDCG@k, Hit@k — computed against reference_contexts
with >= 50% span overlap threshold.
"""

import logging
import math
from typing import Optional

import numpy as np

from src.corpus.models import GoldEntry, RetrievedChunk

logger = logging.getLogger(__name__)


def span_overlap_fraction(
    retrieved_start: int, retrieved_end: int,
    ref_start: int, ref_end: int,
) -> float:
    """Compute overlap fraction between two spans.

    Returns fraction of the shorter span that overlaps.
    """
    overlap_start = max(retrieved_start, ref_start)
    overlap_end = min(retrieved_end, ref_end)
    overlap = max(0, overlap_end - overlap_start)

    shorter = min(retrieved_end - retrieved_start, ref_end - ref_start)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def is_relevant(
    chunk: RetrievedChunk,
    gold: GoldEntry,
    overlap_threshold: float = 0.5,
) -> bool:
    """Check if a retrieved chunk is relevant to a gold entry.

    Uses >= 50% intersection of shorter span (spec §8.1).
    Falls back to content matching if no char offsets available.
    """
    # If gold has reference_contexts with char offsets
    for ref_ctx in gold.reference_contexts:
        ref_start = ref_ctx.get("char_start", 0)
        ref_end = ref_ctx.get("char_end", 0)
        if ref_start > 0 and ref_end > ref_start:
            chunk_start = chunk.metadata.get("char_start", 0)
            chunk_end = chunk.metadata.get("char_end", 0)
            if chunk_start > 0 and chunk_end > chunk_start:
                frac = span_overlap_fraction(chunk_start, chunk_end, ref_start, ref_end)
                if frac >= overlap_threshold:
                    return True

    # Fallback: check if reference answer appears in chunk content
    if gold.reference_answer and gold.reference_answer.lower() in chunk.content.lower():
        return True

    return False


def recall_at_k(
    retrieved: list[RetrievedChunk],
    gold: GoldEntry,
    k: int,
    overlap_threshold: float = 0.5,
) -> float:
    """Recall@k: fraction of relevant reference contexts found in top-k."""
    if not gold.reference_contexts:
        # Fallback: binary — did any chunk contain the answer?
        for chunk in retrieved[:k]:
            if is_relevant(chunk, gold, overlap_threshold):
                return 1.0
        return 0.0

    found = 0
    total = len(gold.reference_contexts)
    for ref_ctx in gold.reference_contexts:
        for chunk in retrieved[:k]:
            if is_relevant(chunk, gold, overlap_threshold):
                found += 1
                break

    return found / total if total > 0 else 0.0


def mrr(
    retrieved: list[RetrievedChunk],
    gold: GoldEntry,
    overlap_threshold: float = 0.5,
) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant chunk."""
    for i, chunk in enumerate(retrieved):
        if is_relevant(chunk, gold, overlap_threshold):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved: list[RetrievedChunk],
    gold: GoldEntry,
    k: int,
    overlap_threshold: float = 0.5,
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Graded by overlap fraction with reference contexts.
    """
    # Compute relevance scores (graded)
    relevances = []
    for chunk in retrieved[:k]:
        max_rel = 0.0
        for ref_ctx in gold.reference_contexts:
            ref_start = ref_ctx.get("char_start", 0)
            ref_end = ref_ctx.get("char_end", 0)
            chunk_start = chunk.metadata.get("char_start", 0)
            chunk_end = chunk.metadata.get("char_end", 0)
            if ref_start > 0 and ref_end > ref_start and chunk_start > 0:
                frac = span_overlap_fraction(chunk_start, chunk_end, ref_start, ref_end)
                max_rel = max(max_rel, frac)
        # Fallback
        if max_rel == 0.0 and is_relevant(chunk, gold, overlap_threshold):
            max_rel = 1.0
        relevances.append(max_rel)

    if not relevances:
        return 0.0

    # DCG
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))

    # Ideal DCG
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_at_k(
    retrieved: list[RetrievedChunk],
    gold: GoldEntry,
    k: int,
    overlap_threshold: float = 0.5,
) -> float:
    """Hit@k: binary recall — 1 if any relevant chunk in top-k, else 0."""
    for chunk in retrieved[:k]:
        if is_relevant(chunk, gold, overlap_threshold):
            return 1.0
    return 0.0


def compute_retrieval_metrics(
    retrieved: list[RetrievedChunk],
    gold: GoldEntry,
    k_values: list[int] = None,
    overlap_threshold: float = 0.5,
) -> dict:
    """Compute all retrieval metrics for a single query.

    Returns dict with metrics for each k value.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    metrics = {"mrr": mrr(retrieved, gold, overlap_threshold)}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved, gold, k, overlap_threshold)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, gold, k, overlap_threshold)
        metrics[f"hit@{k}"] = hit_at_k(retrieved, gold, k, overlap_threshold)

    return metrics
