"""
Chunking strategies (spec §5).

Base ABC and factory function for all 5 strategies.
All chunkers consume AST nodes and produce ChunkMetadata lists.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig

from src.corpus.models import ASTNode, BookSpan, ChunkMetadata
from src.corpus.book_detector import get_chapter_path, build_heading_index

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    strategy_name: str = "base"

    def __init__(self, cfg: DictConfig, project_root: Path):
        self.cfg = cfg
        self.project_root = project_root

    @abstractmethod
    def chunk_book(
        self,
        book: BookSpan,
        nodes: list[ASTNode],
        full_text: str,
    ) -> list[ChunkMetadata]:
        """Chunk a single book's content.

        Args:
            book: Book boundary information.
            nodes: AST nodes for this book.
            full_text: Full corpus text (for offset lookups).

        Returns:
            List of ChunkMetadata objects.
        """
        ...

    def chunk_corpus(
        self,
        all_nodes: list[ASTNode],
    ) -> list[ChunkMetadata]:
        """Chunk the entire corpus as one unit.

        Args:
            all_nodes: All AST nodes from the corpus.

        Returns:
            List of all ChunkMetadata objects.
        """
        full_text_path = self.project_root / self.cfg.paths.raw_corpus
        full_text = full_text_path.read_text(encoding="utf-8")

        # Treat the whole corpus as one BookSpan
        corpus_span = BookSpan(
            book_title="corpus",
            book_index=0,
            start_offset=0,
            end_offset=len(full_text),
        )

        # Pre-build heading index once — avoids O(n) sort inside every
        # _build_metadata call (critical for 60K+ node corpora).
        self._heading_index = build_heading_index(all_nodes)

        logger.info(f"  Chunking full corpus ({len(full_text):,} chars) with {self.strategy_name}")
        chunks = self.chunk_book(corpus_span, all_nodes, full_text)
        logger.info(f"Total chunks ({self.strategy_name}): {len(chunks)}")
        return chunks

    def _build_metadata(
        self,
        content: str,
        book: BookSpan,
        all_nodes: list[ASTNode],
        char_start: int,
        char_end: int,
        **kwargs,
    ) -> ChunkMetadata:
        """Build ChunkMetadata for a chunk."""
        from src.utils.tokenizer import count_tokens

        # Use pre-built index if available (set by chunk_corpus); else build ad-hoc.
        heading_index = getattr(self, "_heading_index", None)
        chapter_path = get_chapter_path(all_nodes, char_start, _heading_index=heading_index)

        return ChunkMetadata(
            strategy=self.strategy_name,
            content=content,
            book_title=book.book_title,
            book_index=book.book_index,
            chapter_path=chapter_path,
            char_start=char_start,
            char_end=char_end,
            token_count=count_tokens(content),
            is_table=kwargs.get("is_table", False),
            contains_table=kwargs.get("contains_table", "|" in content and content.count("|") > 4),
            contains_code=kwargs.get("contains_code", "```" in content),
            parent_chunk_id=kwargs.get("parent_chunk_id"),
            contextual_prefix=kwargs.get("contextual_prefix"),
        )


def get_chunker(strategy_name: str, cfg: DictConfig, project_root: Path) -> BaseChunker:
    """Factory function to get a chunker by strategy name.

    Args:
        strategy_name: One of: recursive, markdown_hierarchical, contextual,
                       table_aware, adaptive.
        cfg: Hydra configuration.
        project_root: Project root path.

    Returns:
        BaseChunker instance.
    """
    from src.chunkers.recursive import RecursiveChunker
    from src.chunkers.markdown_hierarchical import MarkdownHierarchicalChunker
    from src.chunkers.contextual import ContextualChunker
    from src.chunkers.table_aware import TableAwareChunker
    from src.chunkers.adaptive import AdaptiveChunker

    chunkers = {
        "recursive": RecursiveChunker,
        "markdown_hierarchical": MarkdownHierarchicalChunker,
        "contextual": ContextualChunker,
        "table_aware": TableAwareChunker,
        "adaptive": AdaptiveChunker,
    }

    if strategy_name not in chunkers:
        raise ValueError(f"Unknown chunking strategy: {strategy_name}. "
                         f"Available: {list(chunkers.keys())}")

    return chunkers[strategy_name](cfg, project_root)
