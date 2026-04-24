"""
Hybrid Dense + BM25 with RRF (spec §6.3).
Reciprocal Rank Fusion with k=60.
"""

import logging
from collections import defaultdict
from pathlib import Path

from omegaconf import DictConfig

from src.retrieval import BaseRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.corpus.models import RetrievedChunk

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
) -> list[RetrievedChunk]:
    """Fuse multiple ranked lists using RRF.

    Args:
        ranked_lists: List of ranked result lists.
        k: RRF constant (default 60).

    Returns:
        Fused and re-ranked results.
    """
    scores = defaultdict(float)
    chunk_map = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, 1):
            rrf_score = 1.0 / (k + rank)
            scores[chunk.chunk_id] += rrf_score
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

    # Sort by fused score
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results = []
    for rank, chunk_id in enumerate(sorted_ids, 1):
        chunk = chunk_map[chunk_id]
        results.append(RetrievedChunk(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            score=scores[chunk_id],
            rank=rank,
            metadata=chunk.metadata,
        ))

    return results


class HybridRRFRetriever(BaseRetriever):
    """Hybrid Dense + BM25 with Reciprocal Rank Fusion."""

    method_name = "hybrid_rrf"

    def __init__(self, cfg, project_root, chunking_strategy):
        super().__init__(cfg, project_root, chunking_strategy)
        self.dense = DenseRetriever(cfg, project_root, chunking_strategy)
        self.bm25 = BM25Retriever(cfg, project_root, chunking_strategy)
        self.rrf_k = cfg.fusion.rrf_k

    def load_index(self):
        self.dense.load_index()
        self.bm25.load_index()

    def retrieve(self, query, k=10):
        # Retrieve more from each to ensure good fusion
        n_retrieve = max(k * 3, 50)
        dense_results = self.dense.retrieve(query, k=n_retrieve)
        bm25_results = self.bm25.retrieve(query, k=n_retrieve)

        fused = reciprocal_rank_fusion(
            [dense_results, bm25_results], k=self.rrf_k
        )
        return fused[:k]
