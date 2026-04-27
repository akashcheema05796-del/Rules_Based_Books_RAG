"""
Fixed-Size Chunking (spec §5.2 variant).

Splits text into fixed token-size windows with overlap.
No awareness of sentence/paragraph/markdown boundaries —
purely mechanical, serves as a baseline.

Config: 512 tokens, 64-token overlap (~12.5%).
"""

import logging
from pathlib import Path

from omegaconf import DictConfig

from src.chunkers import BaseChunker
from src.corpus.models import ASTNode, BookSpan, ChunkMetadata
from src.utils.tokenizer import count_tokens

import tiktoken

logger = logging.getLogger(__name__)

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64


class FixedSizeChunker(BaseChunker):
    """Pure fixed-size token window chunking with overlap."""

    strategy_name = "fixed_size"

    def __init__(self, cfg: DictConfig, project_root: Path):
        super().__init__(cfg, project_root)
        self.chunk_tokens = CHUNK_TOKENS
        self.overlap_tokens = OVERLAP_TOKENS
        self._enc = tiktoken.get_encoding("cl100k_base")

    def chunk_book(self, book, nodes, full_text):
        book_text = full_text[book.start_offset:book.end_offset]

        # Encode entire book to token IDs
        token_ids = self._enc.encode(book_text)
        total_tokens = len(token_ids)

        if total_tokens == 0:
            return []

        step = self.chunk_tokens - self.overlap_tokens  # stride between windows
        chunks = []
        token_pos = 0

        while token_pos < total_tokens:
            end_pos = min(token_pos + self.chunk_tokens, total_tokens)
            window_ids = token_ids[token_pos:end_pos]
            chunk_text = self._enc.decode(window_ids)

            # Map token position back to char offset (approximate via re-search)
            # Use the first 80 chars of the decoded chunk to locate position
            search_hint = chunk_text[:80]
            char_pos = book_text.find(search_hint)
            if char_pos == -1:
                char_pos = 0  # fallback

            char_start = book.start_offset + char_pos
            char_end = char_start + len(chunk_text)

            chunk = self._build_metadata(
                content=chunk_text,
                book=book,
                all_nodes=nodes,
                char_start=char_start,
                char_end=char_end,
            )
            chunks.append(chunk)

            token_pos += step

        return chunks
