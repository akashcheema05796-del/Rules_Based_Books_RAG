"""
Markdown Hierarchical Chunking (spec §5.2).

Split on h2/h3/h4 headers. Oversized chunks (>2000 tokens) post-split with recursive.
"""

import bisect
import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from omegaconf import DictConfig

from src.chunkers import BaseChunker
from src.corpus.models import ASTNode, BookSpan, ChunkMetadata
from src.utils.tokenizer import count_tokens, token_length_function

logger = logging.getLogger(__name__)


class MarkdownHierarchicalChunker(BaseChunker):
    """Split on markdown headers with recursive fallback for oversized sections."""

    strategy_name = "markdown_hierarchical"

    def __init__(self, cfg: DictConfig, project_root: Path):
        super().__init__(cfg, project_root)
        self.oversized_threshold = 2000

        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=77,
            length_function=token_length_function,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True,
        )

    def chunk_book(self, book, nodes, full_text):
        book_text = full_text[book.start_offset:book.end_offset]

        # Group nodes into sections by h2/h3/h4 headers
        sections = self._split_by_headers(nodes, full_text, book)

        chunks = []
        for section_text, section_start, section_nodes in sections:
            tokens = count_tokens(section_text)

            if tokens > self.oversized_threshold:
                # Post-split oversized sections with recursive
                sub_splits = self.fallback_splitter.split_text(section_text)
                search_pos = 0
                for sub_text in sub_splits:
                    pos = section_text.find(sub_text[:80], search_pos)
                    if pos == -1:
                        pos = search_pos
                    abs_start = section_start + pos
                    abs_end = abs_start + len(sub_text)
                    search_pos = max(pos + 1, search_pos)

                    chunk = self._build_metadata(
                        content=sub_text, book=book, all_nodes=nodes,
                        char_start=abs_start, char_end=abs_end,
                    )
                    chunks.append(chunk)
            else:
                chunk = self._build_metadata(
                    content=section_text, book=book, all_nodes=nodes,
                    char_start=section_start,
                    char_end=section_start + len(section_text),
                )
                chunks.append(chunk)

        return chunks

    def _split_by_headers(self, nodes, full_text, book):
        """Split book content into sections at h2/h3/h4 boundaries.

        Uses binary search to avoid O(sections × nodes) scan.
        """
        # Find header positions
        header_positions = sorted(
            node.char_start for node in nodes
            if (node.node_type == "heading" and node.level in (2, 3, 4)
                and book.start_offset <= node.char_start < book.end_offset)
        )

        if not header_positions:
            text = full_text[book.start_offset:book.end_offset]
            return [(text, book.start_offset, nodes)]

        # Pre-sort nodes by char_start for binary-search slicing.
        sorted_nodes = sorted(nodes, key=lambda n: n.char_start)
        node_starts = [n.char_start for n in sorted_nodes]

        def nodes_in_range(lo, hi):
            l = bisect.bisect_left(node_starts, lo)
            r = bisect.bisect_left(node_starts, hi)
            return sorted_nodes[l:r]

        sections = []
        if header_positions[0] > book.start_offset:
            pre_text = full_text[book.start_offset:header_positions[0]]
            if pre_text.strip():
                sections.append((pre_text, book.start_offset, []))

        for i, pos in enumerate(header_positions):
            end = header_positions[i + 1] if i + 1 < len(header_positions) else book.end_offset
            section_text = full_text[pos:end]
            if section_text.strip():
                sections.append((section_text, pos, nodes_in_range(pos, end)))

        return sections
