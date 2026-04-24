"""
Retrieval methods (spec §6).

Base ABC and factory for all 8 retrieval methods.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from omegaconf import DictConfig

from src.corpus.models import RetrievedChunk

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for all retrieval methods."""

    method_name: str = "base"

    def __init__(self, cfg: DictConfig, project_root: Path, chunking_strategy: str):
        self.cfg = cfg
        self.project_root = project_root
        self.chunking_strategy = chunking_strategy

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """Retrieve top-k chunks for a query.

        Args:
            query: Search query.
            k: Number of results to return.

        Returns:
            List of RetrievedChunk objects, ranked by relevance.
        """
        ...

    def load_index(self) -> None:
        """Load pre-built indices. Called before retrieval."""
        pass


def get_retriever(
    method_name: str,
    cfg: DictConfig,
    project_root: Path,
    chunking_strategy: str,
) -> BaseRetriever:
    """Factory function to get a retriever by method name."""
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.hybrid_rrf import HybridRRFRetriever
    from src.retrieval.hybrid_rerank import HybridRerankRetriever
    from src.retrieval.metadata_filter import MetadataFilterRetriever
    from src.retrieval.small_to_big import SmallToBigRetriever
    from src.retrieval.hyde import HyDERetriever
    from src.retrieval.query_decomposition import QueryDecompositionRetriever

    retrievers = {
        "dense": DenseRetriever,
        "bm25": BM25Retriever,
        "hybrid_rrf": HybridRRFRetriever,
        "hybrid_rerank": HybridRerankRetriever,
        "metadata_filter": MetadataFilterRetriever,
        "small_to_big": SmallToBigRetriever,
        "hyde": HyDERetriever,
        "query_decomposition": QueryDecompositionRetriever,
    }

    if method_name not in retrievers:
        raise ValueError(f"Unknown retrieval method: {method_name}. "
                         f"Available: {list(retrievers.keys())}")

    retriever = retrievers[method_name](cfg, project_root, chunking_strategy)
    retriever.load_index()
    return retriever
