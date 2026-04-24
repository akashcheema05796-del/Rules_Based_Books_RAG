"""
Metadata Filter + Dense Retrieval (spec §6.5).
LLM parses query into (filter, semantic_query), then dense retrieval within filtered set.
"""

import json
import logging
from pathlib import Path
from omegaconf import DictConfig
from src.retrieval import BaseRetriever
from src.retrieval.dense import DenseRetriever
from src.corpus.models import RetrievedChunk
from src.utils.llm import LLMClient

logger = logging.getLogger(__name__)

FILTER_PROMPT = """Analyze this query and extract metadata filters.
Query: {query}
Return JSON with optional fields: is_table (bool), book_title (str), semantic_query (str).
Return ONLY valid JSON."""


class MetadataFilterRetriever(BaseRetriever):
    method_name = "metadata_filter"

    def __init__(self, cfg, project_root, chunking_strategy):
        super().__init__(cfg, project_root, chunking_strategy)
        self.dense = DenseRetriever(cfg, project_root, chunking_strategy)
        self.llm = LLMClient(model=cfg.llm.model, temperature=0, max_tokens=200,
                             cache_dir=str(project_root / ".cache"))

    def load_index(self):
        self.dense.load_index()

    def retrieve(self, query, k=10):
        filters = self._parse_query(query)
        semantic_query = filters.get("semantic_query", query)
        query_emb = self.dense.embed_client.embed_single(semantic_query)
        where = {}
        if filters.get("is_table"):
            where["is_table"] = True
        if filters.get("book_title"):
            where["book_title"] = filters["book_title"]
        try:
            kw = {"query_embeddings": [query_emb], "n_results": k,
                  "include": ["documents", "distances", "metadatas"]}
            if where:
                kw["where"] = where
            results = self.dense.collection.query(**kw)
        except Exception as e:
            logger.warning(f"Filtered query failed ({e}), falling back")
            results = self.dense.collection.query(
                query_embeddings=[query_emb], n_results=k,
                include=["documents", "distances", "metadatas"])
        chunks = []
        for i in range(len(results["ids"][0])):
            chunks.append(RetrievedChunk(
                chunk_id=results["ids"][0][i], content=results["documents"][0][i],
                score=1.0 - results["distances"][0][i], rank=i + 1,
                metadata=results["metadatas"][0][i] if results["metadatas"] else {}))
        return chunks

    def _parse_query(self, query):
        try:
            resp = self.llm.generate("Return only valid JSON.",
                                     FILTER_PROMPT.format(query=query),
                                     max_tokens=200, cache_key=f"filter:{query}")
            return json.loads(resp.strip())
        except Exception:
            return {"semantic_query": query}
