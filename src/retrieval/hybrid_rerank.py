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
        self._reranker_backend = None

    def load_index(self):
        self.hybrid.load_index()
        # Lazy load reranker — prefer sentence-transformers CrossEncoder (compatible with
        # transformers 5.x); fall back to FlagEmbedding if available, then hybrid-only.
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(self.reranker_model)
            self._reranker_backend = "sentence_transformers"
            logger.info(f"Loaded reranker (CrossEncoder): {self.reranker_model}")
            return
        except Exception as st_err:
            logger.debug(f"sentence_transformers CrossEncoder failed: {st_err}")

        try:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker(
                self.reranker_model,
                use_fp16=self.cfg.reranker.use_fp16,
            )
            self._reranker_backend = "flag_embedding"
            logger.info(f"Loaded reranker (FlagEmbedding): {self.reranker_model}")
        except Exception:
            logger.warning("No reranker backend available, falling back to hybrid-only")
            self._reranker_backend = None

    def retrieve(self, query, k=10):
        # Stage 1-2: Hybrid retrieval with RRF
        candidates = self.hybrid.retrieve(query, k=self.initial_top_n)

        if not self.reranker or not candidates:
            return candidates[:k]

        # Stage 3: Cross-encoder reranking
        pairs = [[query, c.content] for c in candidates]

        # Batch scoring — API differs per backend
        batch_size = self.cfg.reranker.batch_size
        all_scores = []
        if getattr(self, "_reranker_backend", None) == "sentence_transformers":
            import numpy as np
            raw = self.reranker.predict(pairs, batch_size=batch_size)
            all_scores = [float(s) for s in np.atleast_1d(raw)]
        else:
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
