"""
Recursive Character Text Splitting (spec §5.1).

512 tokens, 15% overlap, markdown separators.
Uses langchain RecursiveCharacterTextSplitter adapted for token counting.
"""

import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from omegaconf import DictConfig

from src.chunkers import BaseChunker
from src.corpus.models import ASTNode, BookSpan, ChunkMetadata
from src.utils.tokenizer import token_length_function

logger = logging.getLogger(__name__)

# Markdown-aware separator hierarchy
MARKDOWN_SEPARATORS = [
    "\n## ", "\n### ", "\n#### ",
    "\n\n", "\n", " ", "",
]


class RecursiveChunker(BaseChunker):
    """Recursive character text splitting with markdown separators."""

    strategy_name = "recursive"

    def __init__(self, cfg: DictConfig, project_root: Path):
        super().__init__(cfg, project_root)
        self.chunk_size = 512
        self.overlap = 77  # 15% of 512

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            length_function=token_length_function,
            separators=MARKDOWN_SEPARATORS,
            keep_separator=True,
            is_separator_regex=False,
        )

    def chunk_book(self, book, nodes, full_text):
        # Extract book text
        book_text = full_text[book.start_offset:book.end_offset]

        # Split using langchain
        splits = self.splitter.split_text(book_text)

        chunks = []
        search_start = 0
        for split_text in splits:
            # Find position in book text
            pos = book_text.find(split_text[:100], search_start)
            if pos == -1:
                pos = search_start

            char_start = book.start_offset + pos
            char_end = char_start + len(split_text)
            search_start = max(pos + 1, search_start)

            chunk = self._build_metadata(
                content=split_text,
                book=book,
                all_nodes=nodes,
                char_start=char_start,
                char_end=char_end,
            )
            chunks.append(chunk)

        return chunks
