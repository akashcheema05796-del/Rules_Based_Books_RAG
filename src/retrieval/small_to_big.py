"""
Small-to-Big Retrieval (spec §6.6).
Chunk small (200 tok), retrieve small, return parent section (2000 tok).
"""

import json
import logging
import pickle
from pathlib import Path

import chromadb
from omegaconf import DictConfig

from src.retrieval import BaseRetriever
from src.corpus.models import RetrievedChunk
from src.utils.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


class SmallToBigRetriever(BaseRetriever):
    """Retrieve small chunks for precision, return parent for context breadth."""

    method_name = "small_to_big"

    def __init__(self, cfg, project_root, chunking_strategy):
        super().__init__(cfg, project_root, chunking_strategy)
        self.embed_client = EmbeddingClient(
            model=cfg.embedding.model, dimensions=cfg.embedding.dimensions,
            cache_dir=str(project_root / cfg.paths.cache))
        self.collection = None
        self.parent_map = {}  # chunk_id -> parent content

    def load_index(self):
        persist_dir = self.project_root / self.cfg.paths.vector_store / self.chunking_strategy
        client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = client.get_collection(name=self.chunking_strategy)

        # Build parent map from chunks
        chunks_path = (self.project_root / self.cfg.paths.processed /
                       self.chunking_strategy / "chunks.jsonl")
        if chunks_path.exists():
            from src.corpus.models import ChunkMetadata
            chunks = []
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chunks.append(ChunkMetadata(**json.loads(line)))
            # Group adjacent chunks as "parents" (2000 tok windows)
            self._build_parent_map(chunks)

    def _build_parent_map(self, chunks):
        """Create parent context for each chunk by grouping adjacent chunks."""
        from src.utils.tokenizer import count_tokens
        chunks_sorted = sorted(chunks, key=lambda c: c.char_start)
        for i, chunk in enumerate(chunks_sorted):
            parent_parts = []
            parent_tokens = 0
            # Expand window around current chunk
            for j in range(max(0, i - 5), min(len(chunks_sorted), i + 6)):
                c = chunks_sorted[j]
                if c.book_index != chunk.book_index:
                    continue
                t = count_tokens(c.content)
                if parent_tokens + t > 2000:
                    break
                parent_parts.append(c.content)
                parent_tokens += t
            self.parent_map[chunk.chunk_id] = "\n\n".join(parent_parts)

    def retrieve(self, query, k=10):
        query_emb = self.embed_client.embed_single(query)
        # Retrieve more small chunks
        results = self.collection.query(
            query_embeddings=[query_emb], n_results=k * 3,
            include=["documents", "distances", "metadatas"])

        seen_parents = set()
        chunks = []
        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            parent_content = self.parent_map.get(chunk_id, results["documents"][0][i])
            parent_key = parent_content[:200]  # dedup by prefix
            if parent_key in seen_parents:
                continue
            seen_parents.add(parent_key)
            chunks.append(RetrievedChunk(
                chunk_id=chunk_id, content=parent_content,
                score=1.0 - results["distances"][0][i], rank=len(chunks) + 1,
                metadata=results["metadatas"][0][i] if results["metadatas"] else {}))
            if len(chunks) >= k:
                break
        return chunks
