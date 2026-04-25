"""
Dense-only Retrieval (spec §6.1).
Cosine similarity via ChromaDB.
"""

import json
import logging
from pathlib import Path

import chromadb
from omegaconf import DictConfig

from src.retrieval import BaseRetriever
from src.corpus.models import RetrievedChunk
from src.utils.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """Dense retrieval using ChromaDB cosine similarity."""

    method_name = "dense"

    def __init__(self, cfg, project_root, chunking_strategy):
        super().__init__(cfg, project_root, chunking_strategy)
        self.collection = None
        self.embed_client = EmbeddingClient(
            model=cfg.embedding.model,
            dimensions=cfg.embedding.dimensions,
            cache_dir=str(project_root / cfg.paths.cache),
        )

    def load_index(self):
        persist_dir = self.project_root / self.cfg.paths.vector_store / self.chunking_strategy
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = client.get_collection(name=self.chunking_strategy)
        logger.info(f"Loaded dense index: {self.chunking_strategy} "
                     f"({self.collection.count()} chunks)")

    def retrieve(self, query, k=10):
        query_embedding = self.embed_client.embed_single(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append(RetrievedChunk(
                chunk_id=results["ids"][0][i],
                content=results["documents"][0][i],
                score=1.0 - results["distances"][0][i],  # cosine distance → similarity
                rank=i + 1,
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
            ))
        return chunks
