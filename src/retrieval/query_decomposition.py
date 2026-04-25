"""
Query Decomposition Retrieval (spec §6.8).
LLM splits multi-hop query into sub-queries; retrieve per sub-query; merge via RRF.
"""

import json
import logging
from pathlib import Path
from omegaconf import DictConfig
from src.retrieval import BaseRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid_rrf import reciprocal_rank_fusion
from src.corpus.models import RetrievedChunk
from src.utils.llm import LLMClient

logger = logging.getLogger(__name__)

DECOMPOSE_PROMPT = """Break this complex question into simpler sub-questions that can
each be answered independently. Return a JSON array of sub-question strings.
Return ONLY the JSON array.

Question: {query}"""


class QueryDecompositionRetriever(BaseRetriever):
    """Decompose multi-hop queries into sub-queries, retrieve and merge."""
    method_name = "query_decomposition"

    def __init__(self, cfg, project_root, chunking_strategy):
        super().__init__(cfg, project_root, chunking_strategy)
        self.dense = DenseRetriever(cfg, project_root, chunking_strategy)
        self.llm = LLMClient(model=cfg.llm.model, temperature=0, max_tokens=300,
                             cache_dir=str(project_root / cfg.paths.cache),
                             provider=cfg.llm.get("provider", None))

    def load_index(self):
        self.dense.load_index()

    def retrieve(self, query, k=10):
        sub_queries = self._decompose(query)
        if not sub_queries or len(sub_queries) <= 1:
            return self.dense.retrieve(query, k)
        # Retrieve per sub-query
        all_results = []
        for sq in sub_queries[:4]:  # max 4 sub-queries
            results = self.dense.retrieve(sq, k=k)
            all_results.append(results)
        # Merge via RRF
        fused = reciprocal_rank_fusion(all_results, k=60)
        return fused[:k]

    def _decompose(self, query):
        try:
            resp = self.llm.generate(
                "Return only a JSON array of strings.",
                DECOMPOSE_PROMPT.format(query=query),
                max_tokens=300, cache_key=f"decompose:{query}")
            return json.loads(resp.strip())
        except Exception:
            return [query]
