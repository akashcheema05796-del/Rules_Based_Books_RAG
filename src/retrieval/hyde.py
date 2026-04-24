"""
HyDE — Hypothetical Document Embeddings (spec §6.7).
LLM generates hypothetical answer → embed answer → retrieve.
"""

import logging
from pathlib import Path
from omegaconf import DictConfig
from src.retrieval import BaseRetriever
from src.retrieval.dense import DenseRetriever
from src.corpus.models import RetrievedChunk
from src.utils.llm import LLMClient

logger = logging.getLogger(__name__)

HYDE_PROMPT = """You are a knowledgeable AD&D 2nd Edition expert.
Given the following question, write a detailed passage that would answer it,
as if it were found in an AD&D 2e rulebook. Write in a factual, reference style.

Question: {query}

Hypothetical passage:"""


class HyDERetriever(BaseRetriever):
    """Hypothetical document embedding retrieval."""
    method_name = "hyde"

    def __init__(self, cfg, project_root, chunking_strategy):
        super().__init__(cfg, project_root, chunking_strategy)
        self.dense = DenseRetriever(cfg, project_root, chunking_strategy)
        self.llm = LLMClient(model=cfg.llm.model, temperature=0, max_tokens=300,
                             cache_dir=str(project_root / ".cache"))

    def load_index(self):
        self.dense.load_index()

    def retrieve(self, query, k=10):
        # Generate hypothetical document
        hyp_doc = self.llm.generate(
            system_prompt="You are an AD&D 2e rulebook.",
            user_prompt=HYDE_PROMPT.format(query=query),
            max_tokens=300, cache_key=f"hyde:{query}")
        # Embed the hypothetical document
        hyp_emb = self.dense.embed_client.embed_single(hyp_doc)
        results = self.dense.collection.query(
            query_embeddings=[hyp_emb], n_results=k,
            include=["documents", "distances", "metadatas"])
        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append(RetrievedChunk(
                chunk_id=results["ids"][0][i], content=results["documents"][0][i],
                score=1.0 - results["distances"][0][i], rank=i + 1,
                metadata=results["metadatas"][0][i] if results["metadatas"] else {}))
        return chunks
