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
from src.corpus.book_detector import get_chapter_path

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
        books: list[BookSpan],
        all_nodes: list[ASTNode],
    ) -> list[ChunkMetadata]:
        """Chunk the entire corpus by iterating over books.

        Args:
            books: List of book boundaries.
            all_nodes: All AST nodes from the corpus.

        Returns:
            List of all ChunkMetadata objects.
        """
        full_text_path = self.project_root / self.cfg.paths.raw_corpus
        full_text = full_text_path.read_text(encoding="utf-8")
        all_chunks = []

        for book in books:
            # Filter nodes belonging to this book
            book_nodes = [
                n for n in all_nodes
                if book.start_offset <= n.char_start < book.end_offset
            ]
            logger.info(f"  Chunking '{book.book_title}': {len(book_nodes)} nodes")

            chunks = self.chunk_book(book, book_nodes, full_text)
            all_chunks.extend(chunks)
            logger.info(f"  → {len(chunks)} chunks")

        logger.info(f"Total chunks ({self.strategy_name}): {len(all_chunks)}")
        return all_chunks

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

        chapter_path = get_chapter_path(all_nodes, char_start)

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
