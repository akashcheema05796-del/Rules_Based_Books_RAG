"""
Adaptive Chunking (spec §5.5).

Sentence-level embeddings → merge adjacent sentences while cosine sim > τ (0.75)
AND token count < 512. Prepend deterministic micro-header [{book} › {chapter_path}].
"""

import logging
import re
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from src.chunkers import BaseChunker
from src.corpus.models import ASTNode, BookSpan, ChunkMetadata
from src.corpus.book_detector import get_chapter_path, build_heading_index
from src.utils.tokenizer import count_tokens
from src.utils.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using rule-based approach."""
    # Split on sentence-ending punctuation followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    # Filter empty
    return [s.strip() for s in sentences if s.strip()]


class AdaptiveChunker(BaseChunker):
    """Similarity-based adaptive chunking with micro-headers."""

    strategy_name = "adaptive"

    def __init__(self, cfg: DictConfig, project_root: Path):
        super().__init__(cfg, project_root)
        self.cosine_threshold = 0.75
        self.max_chunk_tokens = 512
        self.embed_client = EmbeddingClient(
            model=cfg.embedding.model,
            dimensions=cfg.embedding.dimensions,
            batch_size=cfg.embedding.batch_size,
            cache_dir=str(project_root / cfg.paths.cache),
        )

    def chunk_book(self, book, nodes, full_text):
        book_text = full_text[book.start_offset:book.end_offset]

        # Split into sentences
        sentences = _split_sentences(book_text)
        if not sentences:
            return []

        logger.info(f"  Adaptive: {len(sentences)} sentences to process")

        # Get sentence embeddings
        embeddings = self.embed_client.embed_as_numpy(sentences)

        # Merge adjacent sentences based on cosine similarity
        merged_chunks = self._merge_sentences(sentences, embeddings)

        # Pre-build heading index once for this chunk_book call.
        heading_index = getattr(self, "_heading_index", None) or build_heading_index(nodes)

        # Build chunk metadata with micro-headers
        chunks = []
        search_pos = 0
        for chunk_text in merged_chunks:
            # Find position in book text
            first_sentence = chunk_text.split("\n")[0][:60]
            pos = book_text.find(first_sentence, search_pos)
            if pos == -1:
                pos = search_pos

            char_start = book.start_offset + pos
            char_end = char_start + len(chunk_text)
            search_pos = max(pos + 1, search_pos)

            # Build micro-header
            chapter_path = get_chapter_path(nodes, char_start, _heading_index=heading_index)
            micro_header = f"[{book.book_title} › {' › '.join(chapter_path)}]"

            # Prepend micro-header
            full_content = f"{micro_header}\n{chunk_text}"

            chunk = self._build_metadata(
                content=full_content,
                book=book,
                all_nodes=nodes,
                char_start=char_start,
                char_end=char_end,
            )
            chunks.append(chunk)

        return chunks

    def _merge_sentences(self, sentences, embeddings):
        """Merge adjacent sentences while similarity > threshold and under token limit."""
        if len(sentences) == 0:
            return []

        chunks = []
        current = [sentences[0]]
        current_tokens = count_tokens(sentences[0])

        for i in range(1, len(sentences)):
            # Compute cosine similarity with previous sentence
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            new_tokens = count_tokens(sentences[i])

            if sim > self.cosine_threshold and (current_tokens + new_tokens) < self.max_chunk_tokens:
                current.append(sentences[i])
                current_tokens += new_tokens
            else:
                chunks.append("\n".join(current))
                current = [sentences[i]]
                current_tokens = new_tokens

        if current:
            chunks.append("\n".join(current))

        return chunks

    @staticmethod
    def _cosine_similarity(a, b):
        """Compute cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)
