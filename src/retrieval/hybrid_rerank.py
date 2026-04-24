"""
Hybrid + Rerank — Three-Stage Pipeline (spec §6.4).
BM25 + Dense → RRF → bge-reranker-v2-m3 cross-encoder.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig

from src.retrieval import BaseRetriever
from src.retrieval.hybrid_rrf import HybridRRFRetriever
from src.corpus.models import RetrievedChunk

logger = logging.getLogger(__name__)


class HybridRerankRetriever(BaseRetriever):
    """Three-stage: hybrid RRF retrieval → cross-encoder reranking."""

    method_name = "hybrid_rerank"

    def __init__(self, cfg, project_root, chunking_strategy):
        super().__init__(cfg, project_root, chunking_strategy)
        self.hybrid = HybridRRFRetriever(cfg, project_root, chunking_strategy)
        self.initial_top_n = cfg.reranker.top_n_input
        self.reranker = None
        self.reranker_model = cfg.reranker.model

    def load_index(self):
        self.hybrid.load_index()
        # Lazy load reranker
        try:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(
                self.reranker_model,
                use_fp16=self.cfg.reranker.use_fp16,
            )
            logger.info(f"Loaded reranker: {self.reranker_model}")
        except ImportError:
            logger.warning("FlagEmbedding not available, falling back to hybrid-only")

    def retrieve(self, query, k=10):
        # Stage 1-2: Hybrid retrieval with RRF
        candidates = self.hybrid.retrieve(query, k=self.initial_top_n)

        if not self.reranker or not candidates:
            return candidates[:k]

        # Stage 3: Cross-encoder reranking
        pairs = [[query, c.content] for c in candidates]

        # Batch scoring
        batch_size = self.cfg.reranker.batch_size
        all_scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            scores = self.reranker.compute_score(batch)
            if isinstance(scores, (int, float)):
                scores = [scores]
            all_scores.extend(scores)

        # Re-rank by reranker score
        scored = list(zip(candidates, all_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (chunk, score) in enumerate(scored[:k], 1):
            results.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                score=float(score),
                rank=rank,
                metadata=chunk.metadata,
            ))
        return results
