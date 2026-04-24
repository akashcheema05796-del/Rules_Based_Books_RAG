"""
BM25-only Retrieval (spec §6.2).
Sparse keyword retrieval via rank_bm25.
"""

import logging
import pickle
import re
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from src.retrieval import BaseRetriever
from src.corpus.models import RetrievedChunk

logger = logging.getLogger(__name__)


def _tokenize_query(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r'\b\w+\b', text)


class BM25Retriever(BaseRetriever):
    """BM25 sparse keyword retrieval."""

    method_name = "bm25"

    def __init__(self, cfg, project_root, chunking_strategy):
        super().__init__(cfg, project_root, chunking_strategy)
        self.bm25 = None
        self.chunk_ids = []
        self.chunk_contents = []

    def load_index(self):
        index_path = (self.project_root / self.cfg.paths.bm25_index /
                      self.chunking_strategy / "bm25_index.pkl")
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunk_ids = data["chunk_ids"]
        self.chunk_contents = data["chunk_contents"]
        logger.info(f"Loaded BM25 index: {self.chunking_strategy} "
                     f"({len(self.chunk_ids)} chunks)")

    def retrieve(self, query, k=10):
        tokenized = _tokenize_query(query)
        scores = self.bm25.get_scores(tokenized)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]

        chunks = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                chunks.append(RetrievedChunk(
                    chunk_id=self.chunk_ids[idx],
                    content=self.chunk_contents[idx],
                    score=float(scores[idx]),
                    rank=rank + 1,
                ))
        return chunks
