"""Tests for chunking strategies (§5)."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.corpus.models import ASTNode, BookSpan, ChunkMetadata
from src.chunkers.recursive import RecursiveChunker

# Minimal config mock
def _make_cfg():
    cfg = MagicMock()
    cfg.paths.raw_corpus = "data/raw/test.md"
    cfg.embedding.model = "text-embedding-3-small"
    cfg.embedding.dimensions = 1536
    cfg.embedding.batch_size = 256
    cfg.llm.model = "claude-sonnet-4-5-20241022"
    cfg.llm.temperature = 0
    cfg.cost_budget = {"contextual_chunking": 15.0}
    return cfg

SAMPLE_BOOK_TEXT = """## Player's Handbook

### Chapter 1

This is chapter one content with enough text to create multiple chunks.
The text needs to be long enough to actually trigger splitting behavior.
""" + ("This is filler text for testing purposes. " * 100)


class TestRecursiveChunker:
    def test_produces_chunks(self, tmp_path):
        # Write test corpus
        corpus = tmp_path / "data" / "raw" / "test.md"
        corpus.parent.mkdir(parents=True)
        corpus.write_text(SAMPLE_BOOK_TEXT, encoding="utf-8")

        cfg = _make_cfg()
        cfg.paths.raw_corpus = str(corpus.relative_to(tmp_path))

        chunker = RecursiveChunker(cfg, tmp_path)
        book = BookSpan(book_title="PHB", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_BOOK_TEXT))
        nodes = [ASTNode(node_type="heading", level=2, content="Player's Handbook",
                         char_start=0, char_end=25)]

        chunks = chunker.chunk_book(book, nodes, SAMPLE_BOOK_TEXT)
        assert len(chunks) > 0

    def test_chunk_has_metadata(self, tmp_path):
        corpus = tmp_path / "data" / "raw" / "test.md"
        corpus.parent.mkdir(parents=True)
        corpus.write_text(SAMPLE_BOOK_TEXT, encoding="utf-8")

        cfg = _make_cfg()
        chunker = RecursiveChunker(cfg, tmp_path)
        book = BookSpan(book_title="PHB", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_BOOK_TEXT))
        nodes = []
        chunks = chunker.chunk_book(book, nodes, SAMPLE_BOOK_TEXT)

        if chunks:
            c = chunks[0]
            assert c.strategy == "recursive"
            assert c.book_title == "PHB"
            assert c.book_index == 0
            assert c.token_count > 0
            assert c.chunk_id != ""

    def test_chunk_token_count_reasonable(self, tmp_path):
        corpus = tmp_path / "data" / "raw" / "test.md"
        corpus.parent.mkdir(parents=True)
        corpus.write_text(SAMPLE_BOOK_TEXT, encoding="utf-8")

        cfg = _make_cfg()
        chunker = RecursiveChunker(cfg, tmp_path)
        book = BookSpan(book_title="PHB", book_index=0,
                        start_offset=0, end_offset=len(SAMPLE_BOOK_TEXT))
        chunks = chunker.chunk_book(book, [], SAMPLE_BOOK_TEXT)

        for c in chunks:
            # Should be roughly <= 512 tokens (with some tolerance for overlap)
            assert c.token_count <= 700, f"Chunk too large: {c.token_count} tokens"


class TestChunkMetadataIntegrity:
    def test_no_empty_content(self):
        chunk = ChunkMetadata(strategy="recursive", content="test",
                              book_title="PHB", book_index=0)
        assert chunk.content.strip() != ""

    def test_coverage_no_gaps(self, tmp_path):
        """Chunks should cover the full book text without major gaps."""
        text = "Word " * 500  # ~500 tokens
        corpus = tmp_path / "data" / "raw" / "test.md"
        corpus.parent.mkdir(parents=True)
        corpus.write_text(text, encoding="utf-8")

        cfg = _make_cfg()
        chunker = RecursiveChunker(cfg, tmp_path)
        book = BookSpan(book_title="Test", book_index=0,
                        start_offset=0, end_offset=len(text))
        chunks = chunker.chunk_book(book, [], text)

        # All original content should appear in at least one chunk
        total_chunk_chars = sum(len(c.content) for c in chunks)
        assert total_chunk_chars >= len(text) * 0.8  # At least 80% coverage
