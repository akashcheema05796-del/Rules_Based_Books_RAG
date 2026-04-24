"""
Table-Aware Chunking (spec §5.4).

Each table = one chunk with ~200 tok preceding context prepended.
Non-table content falls through to recursive chunking.
"""

import logging
from pathlib import Path

from omegaconf import DictConfig

from src.chunkers import BaseChunker
from src.chunkers.recursive import RecursiveChunker
from src.corpus.models import ASTNode, BookSpan, ChunkMetadata
from src.utils.tokenizer import count_tokens, truncate_to_tokens

logger = logging.getLogger(__name__)


class TableAwareChunker(BaseChunker):
    """Tables as atomic chunks with context; non-table falls to recursive."""

    strategy_name = "table_aware"

    def __init__(self, cfg: DictConfig, project_root: Path):
        super().__init__(cfg, project_root)
        self.context_tokens = 200
        self.recursive = RecursiveChunker(cfg, project_root)
        self.recursive.strategy_name = "table_aware"

    def chunk_book(self, book, nodes, full_text):
        book_text = full_text[book.start_offset:book.end_offset]
        chunks = []

        # Separate table nodes from non-table nodes
        table_nodes = [n for n in nodes if n.node_type == "table"]
        table_ranges = set()
        for tn in table_nodes:
            table_ranges.add((tn.char_start, tn.char_end))

        # Process tables as atomic chunks with preceding context
        for table_node in table_nodes:
            # Get preceding context
            preceding_start = max(book.start_offset, table_node.char_start - 1000)
            preceding_text = full_text[preceding_start:table_node.char_start].strip()

            if preceding_text:
                preceding_text = truncate_to_tokens(
                    preceding_text, self.context_tokens
                )
                # Take the last context_tokens worth of text
                lines = preceding_text.split("\n")
                context = "\n".join(lines[-10:]) if len(lines) > 10 else preceding_text
            else:
                context = ""

            # Build chunk: context + table
            if context:
                chunk_content = f"{context}\n\n{table_node.raw_content}"
            else:
                chunk_content = table_node.raw_content

            chunk = self._build_metadata(
                content=chunk_content,
                book=book,
                all_nodes=nodes,
                char_start=table_node.char_start,
                char_end=table_node.char_end,
                is_table=True,
                contains_table=True,
            )
            chunks.append(chunk)

        # Process non-table content with recursive chunking
        non_table_text = self._extract_non_table_text(
            full_text, book, table_ranges
        )
        if non_table_text.strip():
            # Create temp text for recursive splitting
            recursive_chunks = self.recursive.splitter.split_text(non_table_text)
            for rc_text in recursive_chunks:
                # Find approximate position
                pos = book_text.find(rc_text[:80])
                char_start = book.start_offset + (pos if pos >= 0 else 0)
                char_end = char_start + len(rc_text)

                chunk = self._build_metadata(
                    content=rc_text, book=book, all_nodes=nodes,
                    char_start=char_start, char_end=char_end,
                    is_table=False,
                )
                chunks.append(chunk)

        # Sort by position
        chunks.sort(key=lambda c: c.char_start)
        return chunks

    def _extract_non_table_text(self, full_text, book, table_ranges):
        """Extract text from book excluding table regions."""
        book_text = full_text[book.start_offset:book.end_offset]
        # Adjust ranges to be relative to book start
        relative_ranges = sorted([
            (max(0, s - book.start_offset), min(len(book_text), e - book.start_offset))
            for s, e in table_ranges
            if s < book.end_offset and e > book.start_offset
        ])

        parts = []
        pos = 0
        for start, end in relative_ranges:
            if pos < start:
                parts.append(book_text[pos:start])
            pos = end
        if pos < len(book_text):
            parts.append(book_text[pos:])

        return "\n\n".join(parts)
